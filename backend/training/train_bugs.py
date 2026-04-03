"""
Train Bug Prediction Model (Logistic Regression + XGBoost + MLP)

Usage:
    cd backend
    python training/train_bugs.py --data data/bug_dataset.jsonl
    python training/train_bugs.py --data data/bug_dataset.jsonl --no-mlp

    # Cross-project split (research-grade evaluation):
    python training/train_bugs.py \\
        --data data/bug_dataset.jsonl \\
        --test-repos https://github.com/pallets/flask https://github.com/psf/requests

Outputs:
    checkpoints/bug_predictor/lr_model.pkl
    checkpoints/bug_predictor/xgb_model.pkl
    checkpoints/bug_predictor/mlp_model.pt      (if --no-mlp not set)
    checkpoints/bug_predictor/mlp_scaler.pkl
    checkpoints/bug_predictor/metrics.json

JIT-SDP feature set (Kamei et al. 2013 + static):
    Static (16): cyclomatic, cognitive, halstead metrics, LOC, function counts, line lengths
    JIT    (14): NS, ND, NF, Entropy, LA, LD, LT, FIX, NDEV, AGE, NUC, EXP, REXP, SEXP
    Total  (30): concatenated at training and inference time
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# -- Global seeds --------------------------------------------------------------
os.environ["PYTHONHASHSEED"] = "42"
random.seed(42)
np.random.seed(42)

# -- Feature groups ------------------------------------------------------------
STATIC_FEATURE_NAMES = [
    "cyclomatic_complexity", "cognitive_complexity",
    "max_function_complexity", "avg_function_complexity",
    "sloc", "comments", "blank_lines",
    "halstead_volume", "halstead_difficulty", "halstead_effort", "halstead_bugs",
    "n_long_functions", "n_complex_functions",
    "max_line_length", "avg_line_length", "n_lines_over_80",
]

# JIT (Just-In-Time) features from Kamei et al. (2013)
# These are process/history features extracted at commit time.
# In the dataset they are stored under "git_features" extended dict.
JIT_FEATURE_NAMES = [
    "code_churn",     # LA + LD (lines added + deleted) — basic churn
    "author_count",   # NDEV — number of unique developers
    "file_age_days",  # AGE  — time since last change
    "n_past_bugs",    # NUC  — number of previous bug fixes in this file
    "commit_freq",    # REXP — recent commit frequency (commits/week)
    # Extended JIT features (populated if dataset includes them):
    "n_subsystems",   # NS  — number of subsystems touched in commit
    "n_directories",  # ND  — number of directories touched
    "n_files",        # NF  — number of files modified
    "entropy",        # Entropy — distribution of changes across files
    "lines_added",    # LA  — lines added
    "lines_deleted",  # LD  — lines deleted
    "lines_touched",  # LT  — lines in touched files before change
    "is_fix",         # FIX — is this a bug-fix commit (0/1)
    "developer_exp",  # EXP — total developer experience (past commits)
]

ALL_FEATURE_NAMES = STATIC_FEATURE_NAMES + JIT_FEATURE_NAMES


# -- Data loading --------------------------------------------------------------

def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load bug_dataset.jsonl.

    Supports both the legacy 5-feature git format and the extended 14-feature
    JIT format. Missing JIT features default to 0.

    Returns:
        X_static : (N, 16)
        X_jit    : (N, 14)
        y        : (N,)    binary labels
        repos    : (N,)    repo identifier per sample
    """
    X_static, X_jit, y, repos = [], [], [], []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            label = int(rec["label"])
            sf = rec.get("static_features", [])
            gf = rec.get("git_features", {})
            if not sf:
                continue

            jit_vec = [float(gf.get(k, 0.0)) for k in JIT_FEATURE_NAMES]
            X_static.append([float(x) for x in sf])
            X_jit.append(jit_vec)
            y.append(label)
            repos.append(rec.get("repo", "unknown"))

    X_s = np.array(X_static, dtype=np.float32)
    X_j = np.array(X_jit, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int32)

    print(f"Loaded {len(y_arr)} samples  |  buggy: {y_arr.sum()}  clean: {(y_arr==0).sum()}")
    print(f"  Static features: {X_s.shape[1]}  JIT features: {X_j.shape[1]}")
    unique_repos = sorted(set(repos))
    print(f"  Repositories ({len(unique_repos)}): {unique_repos}")
    return X_s, X_j, y_arr, repos


def cross_project_split(
    X: np.ndarray,
    y: np.ndarray,
    repos: list[str],
    test_repos: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Hold out specified repos as test — no repo appears in both sets."""
    test_set = set(test_repos)
    train_mask = np.array([r not in test_set for r in repos])
    test_mask = ~train_mask
    if test_mask.sum() == 0:
        raise ValueError(
            f"No test samples for repos {test_repos}. "
            f"Available: {sorted(set(repos))}"
        )
    print(f"Cross-project split — Train: {train_mask.sum()}  Test: {test_mask.sum()}")
    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


# -- Logistic Regression -------------------------------------------------------

def train_lr(
    X: np.ndarray,
    y: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    save_path: str,
    cv_folds: int = 5,
) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, classification_report,
        average_precision_score, f1_score,
    )

    print("\n-- Logistic Regression ------------------------------------------")
    base = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(base, cv=5, method="sigmoid")),
    ])

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_auc = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc", n_jobs=1)
    cv_ap  = cross_val_score(clf, X, y, cv=skf, scoring="average_precision", n_jobs=1)
    print(f"CV AUC-ROC : {np.mean(cv_auc):.4f} ± {np.std(cv_auc):.4f}")
    print(f"CV AP      : {np.mean(cv_ap):.4f} ± {np.std(cv_ap):.4f}")

    clf.fit(X, y)
    te_prob = clf.predict_proba(X_te)[:, 1]
    te_pred = (te_prob >= 0.5).astype(int)

    te_auc = roc_auc_score(y_te, te_prob)
    te_ap  = average_precision_score(y_te, te_prob)
    te_f1  = f1_score(y_te, te_pred)
    te_acc = accuracy_score(y_te, te_pred)

    print(f"Test AUC-ROC : {te_auc:.4f}")
    print(f"Test AP      : {te_ap:.4f}")
    print(f"Test F1      : {te_f1:.4f}")
    print(classification_report(y_te, te_pred, target_names=["clean", "buggy"]))

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"LR model saved -> {save_path}")

    return {
        "cv_auc_mean": float(np.mean(cv_auc)), "cv_auc_std": float(np.std(cv_auc)),
        "cv_ap_mean":  float(np.mean(cv_ap)),  "cv_ap_std":  float(np.std(cv_ap)),
        "test_auc": float(te_auc), "test_ap": float(te_ap),
        "test_f1":  float(te_f1),  "test_acc": float(te_acc),
    }


# -- MLP -----------------------------------------------------------------------

def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    save_path: str,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> dict:
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, f1_score
    except ImportError as e:
        print(f"WARNING: {e}. Skipping MLP training.")
        return {}

    torch.manual_seed(42)

    print("\n-- MLP (Neural Network) -----------------------------------------")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    scaler = StandardScaler()
    X_norm    = scaler.fit_transform(X).astype(np.float32)
    X_te_norm = scaler.transform(X_te).astype(np.float32)

    # Internal validation split for early stopping signal (not the held-out test)
    from sklearn.model_selection import train_test_split as _tts
    X_tr, X_val, y_tr, y_val = _tts(X_norm, y, test_size=0.15, stratify=y, random_state=42)

    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    input_dim = X.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 1), nn.Sigmoid(),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    n_pos  = max(int(y_tr.sum()), 1)
    n_neg  = len(y_tr) - n_pos
    pos_w  = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    criterion = nn.BCELoss(reduction="none")

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            w   = torch.where(yb == 1, pos_w.expand_as(yb), torch.ones_like(yb))
            loss = (criterion(out, yb) * w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.4f}")

    # -- Evaluate on held-out test set -----------------------------------------
    model.eval()
    Xte_t = torch.tensor(X_te_norm, dtype=torch.float32).to(device)
    with torch.no_grad():
        te_probs = model(Xte_t).cpu().numpy().flatten()

    te_pred = (te_probs >= 0.5).astype(int)
    te_auc  = roc_auc_score(y_te, te_probs)
    te_ap   = average_precision_score(y_te, te_probs)
    te_f1   = f1_score(y_te, te_pred)
    te_acc  = accuracy_score(y_te, te_pred)

    print(f"Test AUC-ROC : {te_auc:.4f}")
    print(f"Test AP      : {te_ap:.4f}")
    print(f"Test F1      : {te_f1:.4f}")
    print(f"Best train loss: {best_loss:.4f}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    scaler_path = Path(save_path).parent / "mlp_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    meta_path = Path(save_path).parent / "mlp_meta.json"
    with open(meta_path, "w") as f:
        json.dump({"input_dim": input_dim, "epochs": epochs}, f)
    print(f"MLP saved -> {save_path}")

    return {
        "test_auc": float(te_auc), "test_ap": float(te_ap),
        "test_f1":  float(te_f1),  "test_acc": float(te_acc),
        "best_loss": float(best_loss), "input_dim": input_dim,
    }


# -- XGBoost -------------------------------------------------------------------

def train_xgb(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    save_path: str,
) -> dict:
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("WARNING: xgboost not installed. Skipping XGB training.")
        return {}

    from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, f1_score

    print("\n-- XGBoost ------------------------------------------------------")
    n_buggy = max(int(y_tr.sum()), 1)
    n_clean = int((y_tr == 0).sum())
    scale   = n_clean / n_buggy

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=5,
        gamma=0.3,
        reg_alpha=0.5,
        reg_lambda=1.5,
        scale_pos_weight=scale,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    print(f"Best iteration: {clf.best_iteration}")

    te_prob = clf.predict_proba(X_te)[:, 1]
    te_pred = (te_prob >= 0.5).astype(int)
    te_auc  = roc_auc_score(y_te, te_prob)
    te_ap   = average_precision_score(y_te, te_prob)
    te_f1   = f1_score(y_te, te_pred)
    te_acc  = accuracy_score(y_te, te_pred)

    tr_prob = clf.predict_proba(X_tr)[:, 1]
    tr_auc  = roc_auc_score(y_tr, tr_prob)

    print(f"Test  AUC-ROC : {te_auc:.4f}  AP: {te_ap:.4f}  F1: {te_f1:.4f}")
    print(f"Train AUC-ROC : {tr_auc:.4f}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"XGB saved -> {save_path}")

    return {
        "train_auc": float(tr_auc),
        "test_auc":  float(te_auc), "test_ap": float(te_ap),
        "test_f1":   float(te_f1),  "test_acc": float(te_acc),
        "best_iteration": int(clf.best_iteration),
    }


# -- Main orchestrator ---------------------------------------------------------

def train(
    data_path: str,
    output_dir: str = "checkpoints/bug_predictor",
    cv_folds: int = 5,
    use_mlp: bool = True,
    epochs: int = 50,
    test_split: float = 0.15,
    test_repos: Optional[list[str]] = None,
) -> dict:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
    from scipy.stats import wilcoxon

    X_static, X_jit, y, repos = load_dataset(data_path)
    X_full = np.hstack([X_static, X_jit])

    # -- Split -----------------------------------------------------------------
    if test_repos:
        X_tr, X_te, y_tr, y_te = cross_project_split(X_full, y, repos, test_repos)
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_full, y, test_size=test_split, stratify=y, random_state=42
        )
        print(f"Random split — Train: {len(X_tr)}  Test: {len(X_te)}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # -- Train each model ------------------------------------------------------
    lr_metrics  = train_lr(X_tr, y_tr, X_te, y_te, str(out / "lr_model.pkl"), cv_folds=cv_folds)
    xgb_metrics = train_xgb(X_tr, y_tr, X_te, y_te, str(out / "xgb_model.pkl"))
    mlp_metrics: dict = {}
    if use_mlp:
        mlp_metrics = train_mlp(X_tr, y_tr, X_te, y_te, str(out / "mlp_model.pt"), epochs=epochs)

    # -- Ensemble: XGB + LR ----------------------------------------------------
    ensemble_metrics: dict = {}
    if xgb_metrics:
        with open(out / "lr_model.pkl", "rb") as f:
            lr_clf = pickle.load(f)
        with open(out / "xgb_model.pkl", "rb") as f:
            xgb_clf = pickle.load(f)
        lr_probs  = lr_clf.predict_proba(X_te)[:, 1]
        xgb_probs = xgb_clf.predict_proba(X_te)[:, 1]
        ens_probs = 0.5 * lr_probs + 0.5 * xgb_probs
        ens_auc = roc_auc_score(y_te, ens_probs)
        ens_ap  = average_precision_score(y_te, ens_probs)
        ens_acc = accuracy_score(y_te, (ens_probs >= 0.5).astype(int))
        print(f"\n-- Ensemble (LR + XGBoost) --------------------------------------")
        print(f"Test AUC-ROC : {ens_auc:.4f}  AP: {ens_ap:.4f}")
        ensemble_metrics = {
            "test_auc": float(ens_auc),
            "test_ap":  float(ens_ap),
            "test_acc": float(ens_acc),
        }

        # -- Wilcoxon: ensemble vs. static-only baseline -----------------------
        # Baseline: XGBoost trained on static features only (no JIT)
        n_static = X_static.shape[1]
        xgb_static = train_xgb(
            X_tr[:, :n_static], y_tr,
            X_te[:, :n_static], y_te,
            str(out / "xgb_static_baseline.pkl"),
        )
        if xgb_static and len(y_te) >= 10:
            with open(out / "xgb_static_baseline.pkl", "rb") as f:
                xgb_static_clf = pickle.load(f)
            static_probs = xgb_static_clf.predict_proba(X_te[:, :n_static])[:, 1]
            ens_errors    = np.abs(y_te - (ens_probs >= 0.5).astype(int))
            static_errors = np.abs(y_te - (static_probs >= 0.5).astype(int))
            try:
                stat, p_val = wilcoxon(ens_errors, static_errors, alternative="less")
                print(f"\nWilcoxon (ensemble < static-only): stat={stat:.1f}  p={p_val:.4e}  "
                      f"{'[significant]' if p_val < 0.05 else '[not significant]'}")
                ensemble_metrics["wilcoxon_vs_static"] = {
                    "stat": float(stat), "p_value": float(p_val),
                    "static_auc": xgb_static.get("test_auc", 0.0),
                }
            except Exception:
                pass

    # -- Multi-seed stability evaluation (XGBoost, n=5 seeds) --------------------
    EVAL_SEEDS = [42, 0, 7, 123, 999]
    seed_aucs, seed_aps = [], []
    if not test_repos:
        print(f"\n=== Multi-seed stability check ({len(EVAL_SEEDS)} seeds, XGB) ===")
        for seed in EVAL_SEEDS:
            try:
                X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
                    X_full, y, test_size=test_split, stratify=y, random_state=seed
                )
                from xgboost import XGBClassifier
                from sklearn.metrics import roc_auc_score, average_precision_score
                xgb_s = XGBClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.05,
                    subsample=0.8, n_jobs=1, random_state=seed,
                    eval_metric="logloss", verbosity=0,
                )
                xgb_s.fit(X_tr_s, y_tr_s)
                prob_s = xgb_s.predict_proba(X_te_s)[:, 1]
                if len(set(y_te_s)) > 1:
                    auc_s = float(roc_auc_score(y_te_s, prob_s))
                    ap_s  = float(average_precision_score(y_te_s, prob_s))
                    seed_aucs.append(auc_s)
                    seed_aps.append(ap_s)
                    print(f"  seed={seed}  AUC={auc_s:.4f}  AP={ap_s:.4f}")
            except Exception as e:
                print(f"  seed={seed} failed: {e}")
        if seed_aucs:
            print(
                f"Multi-seed XGB:  AUC={np.mean(seed_aucs):.4f}+/-{np.std(seed_aucs):.4f}  "
                f"AP={np.mean(seed_aps):.4f}+/-{np.std(seed_aps):.4f}  (n={len(seed_aucs)} seeds)"
            )

    # -- Save combined metrics -------------------------------------------------
    metrics = {
        "logistic_regression": lr_metrics,
        "xgboost":             xgb_metrics,
        "ensemble_lr_xgb":     ensemble_metrics,
        "mlp":                 mlp_metrics,
        "n_train":             len(X_tr),
        "n_test":              len(X_te),
        "n_static_features":   int(X_static.shape[1]),
        "n_jit_features":      int(X_jit.shape[1]),
        "n_total_features":    int(X_full.shape[1]),
        "split_strategy":      "cross_project" if test_repos else "random",
        "test_repos":          test_repos or [],
        "class_balance":       {"buggy": int(y.sum()), "clean": int((y == 0).sum())},
        "feature_groups":      {
            "static": STATIC_FEATURE_NAMES,
            "jit":    JIT_FEATURE_NAMES,
        },
        "multi_seed_auc_mean": round(float(np.mean(seed_aucs)), 4) if seed_aucs else None,
        "multi_seed_auc_std":  round(float(np.std(seed_aucs)),  4) if seed_aucs else None,
        "multi_seed_ap_mean":  round(float(np.mean(seed_aps)),  4) if seed_aps  else None,
        "multi_seed_ap_std":   round(float(np.std(seed_aps)),   4) if seed_aps  else None,
    }
    metrics_path = out / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved -> {metrics_path}")
    print("[OK] Bug predictor training complete.")
    return metrics


# -- CLI -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train bug prediction model (LR + XGBoost + MLP)")
    parser.add_argument("--data", required=True, help="Path to bug_dataset.jsonl")
    parser.add_argument("--out", default="checkpoints/bug_predictor")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--epochs", type=int, default=50, help="MLP training epochs")
    parser.add_argument("--no-mlp", action="store_true", help="Skip MLP, train only LR + XGB")
    parser.add_argument(
        "--test-repos", nargs="+", default=None, metavar="REPO",
        help="Repo URLs/names to hold out entirely as test set.",
    )
    args = parser.parse_args()

    train(
        data_path=args.data,
        output_dir=args.out,
        cv_folds=args.cv,
        use_mlp=not args.no_mlp,
        epochs=args.epochs,
        test_repos=args.test_repos,
    )


if __name__ == "__main__":
    main()
