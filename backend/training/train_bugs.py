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
import math
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    cv_auc = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    cv_acc = cross_val_score(clf, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
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

    print("\n-- MLP (Neural Network) --------------------------------------")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Normalise
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X).astype(np.float32)

    Xt = torch.tensor(X_norm, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

    input_dim = X.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 1), nn.Sigmoid(),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        avg_loss = total_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        probs = model(Xt.to(device)).cpu().numpy().flatten()

    y_pred = (probs >= 0.5).astype(int)
    auc = roc_auc_score(y, probs)
    acc = accuracy_score(y, y_pred)
    print(f"Train AUC: {auc:.4f}")
    print(f"Train Acc: {acc:.4f}")
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
        "train_auc": float(auc),
        "train_acc": float(acc),
        "best_loss": float(best_loss),
        "input_dim": input_dim,
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

    # -- MLP --
    mlp_metrics = {}
    if use_mlp:
        mlp_metrics = train_mlp(X_tr, y_tr, str(out / "mlp_model.pt"), epochs=epochs)

    # -- Save combined metrics --
    metrics = {
        "logistic_regression": lr_metrics,
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
