"""
Train Complexity Prediction Model (XGBoost)

Usage:
    cd backend
    python training/train_complexity.py --data data/complexity_dataset.jsonl
    python training/train_complexity.py --data data/complexity_dataset.jsonl --cv 5

Outputs:
    checkpoints/complexity/model.pkl
    checkpoints/complexity/metrics.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Global seed for reproducibility
random.seed(42)
np.random.seed(42)


def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load JSONL dataset produced by dataset_builder.py."""
    X, y = [], []
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            feats = record["features"]
            target = record["target"]
            if feats and target is not None and 0 <= target <= 100:
                X.append(feats)
                y.append(target)

    print(f"Loaded {len(X)} samples from {path}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train(
    data_path: str,
    output_dir: str = "checkpoints/complexity",
    cv_folds: int = 5,
    test_split: float = 0.15,
    **xgb_kwargs,
):
    try:
        import xgboost as xgb
    except ImportError:
        print("ERROR: xgboost required. pip install xgboost")
        sys.exit(1)

    from sklearn.model_selection import train_test_split, KFold, cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import math

    X, y = load_dataset(data_path)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=42
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Hyper-parameters
    params = {
        "n_estimators": xgb_kwargs.get("n_estimators", 500),
        "max_depth": xgb_kwargs.get("max_depth", 6),
        "learning_rate": xgb_kwargs.get("learning_rate", 0.05),
        "subsample": xgb_kwargs.get("subsample", 0.8),
        "colsample_bytree": xgb_kwargs.get("colsample_bytree", 0.8),
        "reg_lambda": xgb_kwargs.get("reg_lambda", 1.0),
        "reg_alpha": xgb_kwargs.get("reg_alpha", 0.1),
        "min_child_weight": xgb_kwargs.get("min_child_weight", 5),
        "gamma": xgb_kwargs.get("gamma", 0.1),
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "rmse",
    }

    # --- Cross-validation ---
    print(f"\nRunning {cv_folds}-fold cross validation...")
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_rmses, cv_r2s = [], []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), 1):
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train[tr_idx], y_train[tr_idx],
            eval_set=[(X_train[val_idx], y_train[val_idx])],
            verbose=False,
        )
        preds = model.predict(X_train[val_idx])
        rmse = math.sqrt(mean_squared_error(y_train[val_idx], preds))
        r2 = r2_score(y_train[val_idx], preds)
        cv_rmses.append(rmse)
        cv_r2s.append(r2)
        print(f"  Fold {fold}: RMSE={rmse:.3f}, R²={r2:.4f}")

    print(f"\nCV Mean RMSE: {np.mean(cv_rmses):.3f} ± {np.std(cv_rmses):.3f}")
    print(f"CV Mean R²:   {np.mean(cv_r2s):.4f} ± {np.std(cv_r2s):.4f}")

    # --- Final training on full train set ---
    print("\nTraining final model on full training set...")
    final_model = xgb.XGBRegressor(**params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100,
    )

    # --- Evaluation on held-out test set ---
    y_pred = final_model.predict(X_test)
    test_rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)

    print(f"\nTest  RMSE: {test_rmse:.3f}")
    print(f"Test  MAE:  {test_mae:.3f}")
    print(f"Test  R²:   {test_r2:.4f}")

    # --- Feature importance ---
    feature_names = [
        "cyclomatic_complexity", "cognitive_complexity", "max_func_cc",
        "avg_func_cc", "sloc", "comments", "blank_lines",
        "halstead_volume", "halstead_difficulty", "halstead_effort",
        "bugs_delivered",
        "n_long_functions", "n_complex_functions",
        "max_line_length", "avg_line_length", "n_lines_over_80",
    ]
    importances = final_model.feature_importances_
    feat_imp = sorted(
        zip(feature_names[:len(importances)], importances),
        key=lambda x: x[1], reverse=True
    )
    print("\nTop Feature Importances:")
    for name, imp in feat_imp[:10]:
        print(f"  {name:40s} {imp:.4f}")

    # --- Save model ---
    import pickle
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    print(f"\nModel saved to {model_path}")

    # --- Save metrics ---
    metrics = {
        "cv_rmse_mean": float(np.mean(cv_rmses)),
        "cv_rmse_std": float(np.std(cv_rmses)),
        "cv_r2_mean": float(np.mean(cv_r2s)),
        "cv_r2_std": float(np.std(cv_r2s)),
        "test_rmse": float(test_rmse),
        "test_mae": float(test_mae),
        "test_r2": float(test_r2),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_importances": [(n, float(i)) for n, i in feat_imp],
        "hyperparams": params,
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train complexity prediction XGBoost model")
    parser.add_argument("--data", required=True, help="Path to JSONL dataset")
    parser.add_argument("--out", default="checkpoints/complexity")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()

    train(
        data_path=args.data,
        output_dir=args.out,
        cv_folds=args.cv,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
