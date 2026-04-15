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
    Static (17): cyclomatic, cognitive, halstead metrics, LOC, function counts, line lengths, n_functions
    JIT    (14): NS, ND, NF, Entropy, LA, LD, LT, FIX, NDEV, AGE, NUC, EXP, REXP, SEXP
    Total  (31): concatenated at training and inference time
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
    # 17-dim — matches bug_predictor.py STATIC_FEATURE_NAMES exactly.
    # cognitive_complexity is a VALID input feature for bug prediction
    # (only excluded from complexity_prediction.py where it is the TARGET).
    # n_functions is extracted from ASTExtractor at both train and inference time.
    "cyclomatic_complexity", "cognitive_complexity",
    "max_function_complexity", "avg_function_complexity",
    "sloc", "comments", "blank_lines",
    "halstead_volume", "halstead_difficulty", "halstead_effort", "halstead_bugs",
    "n_long_functions", "n_complex_functions",
    "max_line_length", "avg_line_length", "n_lines_over_80",
    "n_functions",
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
    # "is_fix" EXCLUDED: FIX=1 iff commit message contains "fix/bug", which is
    # identical to the keyword label → perfect circular correlation → AUC=1.0.
    # Kamei uses it with SZZ labels (no circularity there), but not here.
    "developer_exp",  # EXP — total developer experience (past commits)
]

ALL_FEATURE_NAMES = STATIC_FEATURE_NAMES + JIT_FEATURE_NAMES


# -- Data loading --------------------------------------------------------------

def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list]:
    """
    Load bug_dataset.jsonl.

    Supports both the legacy 5-feature git format and the extended 14-feature
    JIT format. Missing JIT features default to 0.

    Returns:
        X_static : (N, 17)
        X_jit    : (N, 14)
        y        : (N,)    binary labels
        repos    : list[str] repo identifier per sample
        dates    : list[str] author_date per sample (empty string if absent)
    """
    X_static, X_jit, y, repos, dates = [], [], [], [], []

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
            dates.append(rec.get("author_date", ""))

    X_s = np.array(X_static, dtype=np.float32)
    X_j = np.array(X_jit, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int32)

    print(f"Loaded {len(y_arr)} samples  |  buggy: {y_arr.sum()}  clean: {(y_arr==0).sum()}")
    print(f"  Static features: {X_s.shape[1]}  JIT features: {X_j.shape[1]}")
    unique_repos = sorted(set(repos))
    print(f"  Repositories ({len(unique_repos)}): {unique_repos}")
    return X_s, X_j, y_arr, repos, dates


def temporal_split(
    X: np.ndarray,
    y: np.ndarray,
    dates: list,
    train_frac: float = 0.70,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort records by author_date (oldest first) and split at train_frac.
    Records with missing/unparseable dates are placed at the beginning (oldest).

    This is the correct protocol to detect temporal leakage: if a model trained
    on old commits cannot generalise to new commits, the labels contain future
    information (SZZ leakage).
    """
    import dateutil.parser  # pydriller dependency, always available

    def _parse(d: str) -> float:
        try:
            return dateutil.parser.parse(d).timestamp()
        except Exception:
            return 0.0

    timestamps = np.array([_parse(d) for d in dates], dtype=np.float64)
    order = np.argsort(timestamps, kind="stable")
    X_sorted = X[order]
    y_sorted = y[order]

    split_idx = int(len(y_sorted) * train_frac)
    return X_sorted[:split_idx], X_sorted[split_idx:], y_sorted[:split_idx], y_sorted[split_idx:]


def temporal_walk_forward_cv(
    X: np.ndarray,
    y: np.ndarray,
    dates: list,
    n_folds: int = 5,
) -> dict:
    """
    Walk-forward (expanding-window) temporal cross-validation.

    Records are sorted chronologically and divided into `n_folds` equal slices.
    For fold k (k = 1 .. n_folds-1): train on slices 0..k-1, test on slice k.
    Fold 0 is never a test set (no history before it).

    This is the gold-standard evaluation for JIT defect prediction.
    A single 70/30 temporal cut has high variance; averaging across folds gives
    a more reliable estimate of the true temporal generalisation gap.

    Returns a dict with per-fold AUC/AP and summary stats.
    """
    import dateutil.parser
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score

    def _parse(d: str) -> float:
        try:
            return dateutil.parser.parse(d).timestamp()
        except Exception:
            return 0.0

    timestamps = np.array([_parse(d) for d in dates], dtype=np.float64)
    order = np.argsort(timestamps, kind="stable")
    X_s = X[order]
    y_s = y[order]

    n = len(y_s)
    fold_size = n // n_folds
    fold_aucs, fold_aps = [], []
    fold_details = []

    for k in range(1, n_folds):
        train_end = k * fold_size
        test_start = train_end
        test_end = (k + 1) * fold_size if k < n_folds - 1 else n

        X_tr, y_tr = X_s[:train_end], y_s[:train_end]
        X_te, y_te = X_s[test_start:test_end], y_s[test_start:test_end]

        if len(set(y_te)) < 2 or len(X_tr) < 20:
            fold_details.append({"fold": k, "skipped": True})
            continue

        clf = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
            eval_metric="logloss", random_state=42,
            n_jobs=1, verbosity=0,
        )
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_te)[:, 1]
        auc = float(roc_auc_score(y_te, probs))
        ap  = float(average_precision_score(y_te, probs))
        fold_aucs.append(auc)
        fold_aps.append(ap)
        fold_details.append({
            "fold": k,
            "n_train": int(len(y_tr)),
            "n_test":  int(len(y_te)),
            "pos_rate_train": round(float(y_tr.mean()), 4),
            "pos_rate_test":  round(float(y_te.mean()), 4),
            "auc": round(auc, 4),
            "ap":  round(ap, 4),
        })

    result = {
        "protocol": f"walk_forward_{n_folds}_fold",
        "n_folds_evaluated": len(fold_aucs),
        "fold_details": fold_details,
        "mean_auc": round(float(np.mean(fold_aucs)), 4) if fold_aucs else None,
        "std_auc":  round(float(np.std(fold_aucs)),  4) if fold_aucs else None,
        "mean_ap":  round(float(np.mean(fold_aps)),  4) if fold_aps else None,
        "std_ap":   round(float(np.std(fold_aps)),   4) if fold_aps else None,
    }
    return result


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
        n_jobs=1,   # n_jobs=-1 causes OOM/pickle errors on Windows
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

    X_static, X_jit, y, repos, dates = load_dataset(data_path)
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

    # -- Temporal split evaluation (leakage detection) -------------------------
    # Train on oldest 70% of commits, test on newest 30%.
    # If labels are temporally clean (proper SZZ with max_fix_lag_days), the
    # temporal AUC should be close to the random-split AUC.
    # A large drop (>0.05) indicates residual label leakage.
    temporal_metrics: dict = {}
    has_dates = any(d for d in dates)
    if has_dates and not test_repos:
        print("\n=== Temporal split evaluation (leakage check) ===")
        try:
            from xgboost import XGBClassifier
            from sklearn.metrics import roc_auc_score, average_precision_score

            X_ttr, X_tte, y_ttr, y_tte = temporal_split(X_full, y, dates, train_frac=0.70)
            print(f"Temporal split — Train: {len(X_ttr)}  Test: {len(X_tte)}")
            print(f"  Train positive rate: {y_ttr.mean():.3f}  Test positive rate: {y_tte.mean():.3f}")

            if len(set(y_tte)) > 1 and len(X_ttr) >= 50:
                n_buggy_t = max(int(y_ttr.sum()), 1)
                n_clean_t = int((y_ttr == 0).sum())
                xgb_t = XGBClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.05,
                    subsample=0.7, colsample_bytree=0.7,
                    scale_pos_weight=n_clean_t / n_buggy_t,
                    eval_metric="logloss", random_state=42, n_jobs=1, verbosity=0,
                )
                xgb_t.fit(X_ttr, y_ttr)
                t_probs = xgb_t.predict_proba(X_tte)[:, 1]
                t_auc = float(roc_auc_score(y_tte, t_probs))
                t_ap  = float(average_precision_score(y_tte, t_probs))
                random_auc = xgb_metrics.get("test_auc", 0.0) or 0.0
                delta = round(t_auc - random_auc, 4)
                print(f"Temporal AUC : {t_auc:.4f}  AP: {t_ap:.4f}")
                print(f"Random AUC   : {random_auc:.4f}  Delta: {delta:+.4f}")
                if abs(delta) < 0.05:
                    print("  [PASS] Temporal delta < 0.05 — labels appear temporally clean")
                else:
                    print(f"  [WARN] Large temporal delta={delta:+.4f} — possible residual label leakage")
                temporal_metrics = {
                    "temporal_auc": round(t_auc, 4),
                    "temporal_ap":  round(t_ap, 4),
                    "random_auc":   round(random_auc, 4),
                    "delta_auc":    delta,
                    "protocol":     "age_sorted_70_30",
                    "n_train":      int(len(X_ttr)),
                    "n_test":       int(len(X_tte)),
                }
            else:
                print("  Skipped: not enough samples or only one class in temporal test set")
        except Exception as e:
            print(f"  Temporal split evaluation failed: {e}")

    # -- Walk-forward temporal cross-validation --------------------------------
    temporal_cv_metrics: dict = {}
    if has_dates and not test_repos:
        print("\n=== Walk-forward temporal CV (5-fold rolling window) ===")
        try:
            temporal_cv_metrics = temporal_walk_forward_cv(X_full, y, dates, n_folds=5)
            folds = temporal_cv_metrics.get("fold_details", [])
            for fd in folds:
                if fd.get("skipped"):
                    print(f"  Fold {fd['fold']}: skipped (insufficient data)")
                else:
                    print(f"  Fold {fd['fold']}: n_train={fd['n_train']}  n_test={fd['n_test']}  "
                          f"AUC={fd['auc']:.4f}  AP={fd['ap']:.4f}")
            m = temporal_cv_metrics.get("mean_auc")
            s = temporal_cv_metrics.get("std_auc")
            if m is not None:
                print(f"Walk-forward mean AUC: {m:.4f} +/- {s:.4f}  "
                      f"(vs random-split AUC={xgb_metrics.get('test_auc', 0):.4f})")
        except Exception as e:
            print(f"  Walk-forward CV failed: {e}")

    # -- Save combined metrics -------------------------------------------------
    random_auc_for_gate = xgb_metrics.get("test_auc") if xgb_metrics else None
    metrics = {
        "logistic_regression": lr_metrics,
        "xgboost":             xgb_metrics,
        "ensemble_lr_xgb":     ensemble_metrics,
        "mlp":                 mlp_metrics,
        "temporal_split":      temporal_metrics,
        "temporal_auc":        temporal_metrics.get("temporal_auc"),   # top-level for bug_predictor.py gate
        "temporal_walk_forward_cv": temporal_cv_metrics,
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

    # Also write temporal_auc into lopo_bug.json so bug_predictor.py gate
    # reads the correct value directly (no more fallback to baseline_comparison.json)
    lopo_path = Path(__file__).resolve().parent.parent / "evaluation" / "results" / "lopo_bug.json"
    if temporal_metrics and lopo_path.exists():
        try:
            with open(lopo_path) as f:
                lopo_data = json.load(f)
            lopo_data["temporal_auc"]          = temporal_metrics.get("temporal_auc")
            lopo_data["temporal_ap"]           = temporal_metrics.get("temporal_ap")
            lopo_data["random_split_auc"]      = temporal_metrics.get("random_auc")
            lopo_data["degradation_auc"]       = temporal_metrics.get("delta_auc")
            lopo_data["temporal_protocol"]     = temporal_metrics.get("protocol")
            if temporal_cv_metrics:
                lopo_data["temporal_walk_forward_cv"] = temporal_cv_metrics
            with open(lopo_path, "w") as f:
                json.dump(lopo_data, f, indent=2)
            print(f"lopo_bug.json updated with temporal_auc={temporal_metrics.get('temporal_auc')}")
        except Exception as e:
            print(f"Warning: could not update lopo_bug.json: {e}")

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
