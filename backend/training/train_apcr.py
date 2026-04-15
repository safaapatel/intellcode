"""
Train Asymmetric Complexity Regressor (APCR)
=============================================
Trains the custom XGBoost regressor with asymmetric pinball loss on the
complexity dataset (same JSONL format as train_complexity.py).

The model penalises underestimating complexity 3x more than overestimating,
encoding the real-world asymmetry where missed complex functions cause bugs.

Usage:
    cd backend
    python training/train_apcr.py --data data/complexity_dataset.jsonl
    python training/train_apcr.py --data data/complexity_dataset.jsonl --alpha 0.8

Outputs:
    checkpoints/apcr/apcr.pkl
    checkpoints/apcr/metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Index of cognitive complexity in the 16-dim training vector
COG_IDX = 1


def load_dataset(data_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load complexity dataset JSONL.

    Each record: {"features": [16-dim vector], "target": float, "repo": str}
    X: rows are the 15 input features (cognitive_complexity removed — it IS the target)
    y: cognitive complexity values extracted from index COG_IDX

    Returns (X, y).
    """
    X_list, y_list = [], []

    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                feat = rec.get("features", [])
                if len(feat) < 16:
                    continue
                cog = float(feat[COG_IDX])
                # Build 15-dim input from first 16 elements, remove COG_IDX
                # (mirrors train_complexity.py exactly)
                x = [feat[i] for i in range(16) if i != COG_IDX]
                if len(x) != 15:
                    continue
                X_list.append(x)
                y_list.append(cog)
            except Exception:
                continue

    if not X_list:
        raise ValueError(f"No valid records found in {data_path}")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    logger.info("Loaded %d samples from %s  (target range: %.1f–%.1f)",
                len(X), data_path, y.min(), y.max())
    return X, y


def train(
    data_path: str,
    output_dir: str = "checkpoints/apcr",
    alpha: float = 0.75,
    test_split: float = 0.20,
) -> dict:
    """Train APCR and evaluate on a held-out split."""
    from models.asymmetric_complexity_regressor import AsymmetricComplexityRegressor

    X, y = load_dataset(data_path)

    # Chronological split (last 20% as test, preserving temporal order)
    n = len(X)
    n_test = max(1, int(n * test_split))
    X_tr, X_te = X[:-n_test], X[-n_test:]
    y_tr, y_te = y[:-n_test], y[-n_test:]

    logger.info("Split: %d train / %d test", len(X_tr), len(X_te))

    model = AsymmetricComplexityRegressor(alpha=alpha)
    train_metrics = model.fit(X_tr, y_tr, output_dir=output_dir)

    # Evaluation on test set
    import xgboost as xgb
    dm_te = xgb.DMatrix(X_te)
    preds_asym = model._model.predict(dm_te)
    preds_sym  = model._symmetric_model.predict(dm_te)

    rmse_asym = float(np.sqrt(np.mean((y_te - preds_asym) ** 2)))
    rmse_sym  = float(np.sqrt(np.mean((y_te - preds_sym) ** 2)))
    mae_asym  = float(np.mean(np.abs(y_te - preds_asym)))

    # Pinball loss on test set
    def _pinball(y_true, y_pred, a):
        r = y_true - y_pred
        return float(np.mean(np.where(r >= 0, a * r, (a - 1) * r)))

    pinball_asym = _pinball(y_te, preds_asym, alpha)
    pinball_sym  = _pinball(y_te, preds_sym,  alpha)

    # Underestimate rate: fraction where prediction < truth
    under_rate_asym = float(np.mean(preds_asym < y_te))
    under_rate_sym  = float(np.mean(preds_sym  < y_te))

    # Spearman correlation
    from scipy.stats import spearmanr
    rho_asym = float(spearmanr(y_te, preds_asym).correlation)
    rho_sym  = float(spearmanr(y_te, preds_sym).correlation)

    eval_metrics = {
        **train_metrics,
        "test_rmse_apcr":        round(rmse_asym, 4),
        "test_rmse_symmetric":   round(rmse_sym, 4),
        "test_mae_apcr":         round(mae_asym, 4),
        "test_pinball_apcr":     round(pinball_asym, 4),
        "test_pinball_symmetric":round(pinball_sym, 4),
        "test_underestimate_rate_apcr": round(under_rate_asym, 4),
        "test_underestimate_rate_sym":  round(under_rate_sym, 4),
        "test_spearman_apcr":    round(rho_asym, 4),
        "test_spearman_sym":     round(rho_sym, 4),
        "n_train": len(X_tr),
        "n_test": len(X_te),
    }

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)

    logger.info(
        "APCR evaluation:  RMSE=%.3f (sym=%.3f)  pinball=%.4f  under_rate=%.2f  rho=%.3f",
        rmse_asym, rmse_sym, pinball_asym, under_rate_asym, rho_asym,
    )
    logger.info(
        "Asymmetric loss reduces underestimation from %.1f%% to %.1f%%",
        under_rate_sym * 100, under_rate_asym * 100,
    )
    return eval_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train Asymmetric Complexity Regressor (APCR)"
    )
    parser.add_argument("--data", default="data/complexity_dataset.jsonl",
                        help="Path to complexity JSONL dataset")
    parser.add_argument("--alpha", type=float, default=0.75,
                        help="Pinball alpha (fraction of loss on underestimation), default 0.75")
    parser.add_argument("--out", default="checkpoints/apcr",
                        help="Checkpoint output directory")
    parser.add_argument("--test-split", type=float, default=0.20,
                        help="Fraction of data to hold out for evaluation (default 0.20)")
    args = parser.parse_args()

    if not Path(args.data).exists():
        logger.error("Dataset not found: %s", args.data)
        sys.exit(1)

    try:
        import xgboost
    except ImportError:
        logger.error("xgboost not installed: pip install xgboost")
        sys.exit(1)

    metrics = train(
        data_path=args.data,
        output_dir=args.out,
        alpha=args.alpha,
        test_split=args.test_split,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
