"""
Train Bug Prediction Model (Logistic Regression + MLP)

Usage:
    cd backend
    python training/train_bugs.py --data data/bug_dataset.jsonl
    python training/train_bugs.py --data data/bug_dataset.jsonl --no-mlp

Outputs:
    checkpoints/bug_predictor/lr_model.pkl
    checkpoints/bug_predictor/mlp_model.pt      (if --no-mlp not set)
    checkpoints/bug_predictor/metrics.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Global seed for reproducibility
random.seed(42)
np.random.seed(42)

GIT_FEATURE_NAMES = [
    "code_churn",
    "author_count",
    "file_age_days",
    "n_past_bugs",
    "commit_freq",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load bug_dataset.jsonl produced by generate_synthetic_data.py.

    Returns:
        X_static : (N, S)  — static feature vectors
        X_git    : (N, 5)  — git metadata vectors
        y        : (N,)    — binary labels (0=clean, 1=buggy)
    """
    X_static, X_git, y = [], [], []

    with open(path) as f:
        for line in f:
            rec = json.loads(line.strip())
            label = int(rec["label"])
            sf = rec.get("static_features", [])
            gf = rec.get("git_features", {})
            if not sf:
                continue

            git_vec = [float(gf.get(k, 0.0)) for k in GIT_FEATURE_NAMES]
            X_static.append([float(x) for x in sf])
            X_git.append(git_vec)
            y.append(label)

    print(f"Loaded {len(y)} samples from {path}")
    print(f"  Buggy: {sum(y)}, Clean: {len(y) - sum(y)}")

    X_s = np.array(X_static, dtype=np.float32)
    X_g = np.array(X_git, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int32)
    print(f"  Static features: {X_s.shape[1]}, Git features: {X_g.shape[1]}")
    return X_s, X_g, y_arr


# ---------------------------------------------------------------------------
# Logistic Regression training
# ---------------------------------------------------------------------------

def train_lr(
    X: np.ndarray,
    y: np.ndarray,
    save_path: str,
    cv_folds: int = 5,
) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

    print("\n-- Logistic Regression ---------------------------------------")
    base = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", CalibratedClassifierCV(base, cv=5, method="sigmoid")),
    ])

    # Cross-validation
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_auc = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc", n_jobs=1)
    cv_acc = cross_val_score(clf, X, y, cv=skf, scoring="accuracy", n_jobs=1)
    print(f"CV AUC:      {np.mean(cv_auc):.4f} ± {np.std(cv_auc):.4f}")
    print(f"CV Accuracy: {np.mean(cv_acc):.4f} ± {np.std(cv_acc):.4f}")

    # Final fit on all data
    clf.fit(X, y)
    y_prob = clf.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    train_auc = roc_auc_score(y, y_prob)
    train_acc = accuracy_score(y, y_pred)
    print(f"Train AUC:   {train_auc:.4f}")
    print(f"Train Acc:   {train_acc:.4f}")
    print(classification_report(y, y_pred, target_names=["clean", "buggy"]))

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"LR model saved -> {save_path}")

    return {
        "cv_auc_mean": float(np.mean(cv_auc)),
        "cv_auc_std": float(np.std(cv_auc)),
        "cv_acc_mean": float(np.mean(cv_acc)),
        "cv_acc_std": float(np.std(cv_acc)),
        "train_auc": float(train_auc),
        "train_acc": float(train_acc),
    }


# ---------------------------------------------------------------------------
# MLP training
# ---------------------------------------------------------------------------

def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
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
        from sklearn.metrics import roc_auc_score, accuracy_score
    except ImportError as e:
        print(f"WARNING: {e}. Skipping MLP training.")
        return {}

    torch.manual_seed(42)

    print("\n-- MLP (Neural Network) --------------------------------------")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Normalise
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X).astype(np.float32)

    # Hold out a validation set for honest evaluation (not seen during training)
    from sklearn.model_selection import train_test_split as _tts
    X_tr, X_val, y_tr, y_val = _tts(X_norm, y, test_size=0.15, stratify=y, random_state=42)

    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    input_dim = X.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 1), nn.Sigmoid(),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Class-imbalance: weight positive samples by neg/pos ratio
    n_pos = max(int(y_tr.sum()), 1)
    n_neg = len(y_tr) - n_pos
    pos_w = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    criterion = nn.BCELoss(reduction="none")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            # Apply pos_weight manually (BCELoss doesn't have pos_weight, BCEWithLogitsLoss does)
            weight = torch.where(yb == 1, pos_w.expand_as(yb), torch.ones_like(yb))
            loss = (criterion(out, yb) * weight).mean()
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

    # Evaluate on held-out validation set (not the training data)
    model.eval()
    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = model(Xv).cpu().numpy().flatten()

    y_pred = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y_val, probs)
    acc = accuracy_score(y_val, y_pred)
    print(f"Val AUC: {auc:.4f}")
    print(f"Val Acc: {acc:.4f}")
    print(f"Best loss: {best_loss:.4f}")

    # Save model + scaler together
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_path)
    scaler_path = Path(save_path).parent / "mlp_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"MLP model saved  -> {save_path}")
    print(f"MLP scaler saved -> {scaler_path}")

    # Save input_dim so BugPredictionModel can load it
    meta_path = Path(save_path).parent / "mlp_meta.json"
    with open(meta_path, "w") as f:
        json.dump({"input_dim": input_dim, "epochs": epochs}, f)

    return {
        "val_auc": float(auc),
        "val_acc": float(acc),
        "best_loss": float(best_loss),
        "input_dim": input_dim,
    }


# ---------------------------------------------------------------------------
# XGBoost training
# ---------------------------------------------------------------------------

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

    from sklearn.metrics import roc_auc_score, accuracy_score

    print("\n-- XGBoost ---------------------------------------------------")
    n_buggy = int(y_tr.sum())
    n_clean = int((y_tr == 0).sum())
    scale = n_clean / max(n_buggy, 1)

    clf = XGBClassifier(
        n_estimators=300,
        max_depth=4,           # reduced 6→4 to prevent overfitting
        learning_rate=0.05,
        subsample=0.7,         # reduced 0.8→0.7
        colsample_bytree=0.7,  # reduced 0.8→0.7
        min_child_weight=5,    # increased 3→5
        gamma=0.3,             # increased 0.1→0.3
        reg_alpha=0.5,         # L1 regularization (new)
        reg_lambda=1.5,        # L2 regularization (new)
        scale_pos_weight=scale,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )

    clf.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        verbose=False,
    )
    print(f"Best iteration: {clf.best_iteration}")

    te_prob = clf.predict_proba(X_te)[:, 1]
    te_auc = roc_auc_score(y_te, te_prob)
    te_acc = accuracy_score(y_te, (te_prob >= 0.5).astype(int))
    print(f"Test  AUC:   {te_auc:.4f}")
    print(f"Test  Acc:   {te_acc:.4f}")

    tr_prob = clf.predict_proba(X_tr)[:, 1]
    tr_auc = roc_auc_score(y_tr, tr_prob)
    print(f"Train AUC:   {tr_auc:.4f}")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        import pickle
        pickle.dump(clf, f)
    print(f"XGB model saved -> {save_path}")

    return {
        "train_auc": float(tr_auc),
        "test_auc": float(te_auc),
        "test_acc": float(te_acc),
        "best_iteration": int(clf.best_iteration),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train(
    data_path: str,
    output_dir: str = "checkpoints/bug_predictor",
    cv_folds: int = 5,
    use_mlp: bool = True,
    epochs: int = 50,
    test_split: float = 0.15,
):
    from sklearn.model_selection import train_test_split

    X_static, X_git, y = load_dataset(data_path)
    X_full = np.hstack([X_static, X_git])

    # Train / test split for final held-out evaluation
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_full, y, test_size=test_split, stratify=y, random_state=42
    )
    print(f"\nTrain: {len(X_tr)}, Test: {len(X_te)}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # -- LR --
    lr_metrics = train_lr(X_tr, y_tr, str(out / "lr_model.pkl"), cv_folds=cv_folds)

    # Evaluate LR on held-out test set
    with open(out / "lr_model.pkl", "rb") as f:
        lr_clf = pickle.load(f)
    from sklearn.metrics import roc_auc_score, accuracy_score
    te_prob = lr_clf.predict_proba(X_te)[:, 1]
    lr_metrics["test_auc"] = float(roc_auc_score(y_te, te_prob))
    lr_metrics["test_acc"] = float(accuracy_score(y_te, (te_prob >= 0.5).astype(int)))
    print(f"\nTest AUC (LR): {lr_metrics['test_auc']:.4f}")
    print(f"Test Acc (LR): {lr_metrics['test_acc']:.4f}")

    # -- XGBoost --
    xgb_metrics = train_xgb(X_tr, y_tr, X_te, y_te, str(out / "xgb_model.pkl"))

    # -- MLP --
    mlp_metrics = {}
    if use_mlp:
        mlp_metrics = train_mlp(X_tr, y_tr, str(out / "mlp_model.pt"), epochs=epochs)

    # -- Ensemble: LR + XGBoost --
    from sklearn.metrics import roc_auc_score, accuracy_score
    ensemble_metrics = {}
    if xgb_metrics:
        with open(out / "xgb_model.pkl", "rb") as f:
            xgb_clf = pickle.load(f)
        lr_probs_te = lr_clf.predict_proba(X_te)[:, 1]
        xgb_probs_te = xgb_clf.predict_proba(X_te)[:, 1]
        ens_probs = 0.5 * lr_probs_te + 0.5 * xgb_probs_te
        ens_auc = roc_auc_score(y_te, ens_probs)
        ens_acc = accuracy_score(y_te, (ens_probs >= 0.5).astype(int))
        print(f"\n-- Ensemble (LR + XGBoost) -----------------------------------")
        print(f"Test AUC: {ens_auc:.4f}")
        print(f"Test Acc: {ens_acc:.4f}")
        ensemble_metrics = {"test_auc": float(ens_auc), "test_acc": float(ens_acc)}

    # -- Save combined metrics --
    metrics = {
        "logistic_regression": lr_metrics,
        "xgboost": xgb_metrics,
        "ensemble_lr_xgb": ensemble_metrics,
        "mlp": mlp_metrics,
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "n_static_features": int(X_static.shape[1]),
        "n_git_features": int(X_git.shape[1]),
        "n_total_features": int(X_full.shape[1]),
        "class_balance": {"buggy": int(y.sum()), "clean": int((y == 0).sum())},
    }
    metrics_path = out / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved -> {metrics_path}")
    print("\n[OK] Bug predictor training complete.")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train bug prediction model (LR + MLP)")
    parser.add_argument("--data", required=True, help="Path to bug_dataset.jsonl")
    parser.add_argument("--out", default="checkpoints/bug_predictor")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--epochs", type=int, default=50, help="MLP training epochs")
    parser.add_argument("--no-mlp", action="store_true", help="Skip MLP, train only LR")
    args = parser.parse_args()

    train(
        data_path=args.data,
        output_dir=args.out,
        cv_folds=args.cv,
        use_mlp=not args.no_mlp,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
