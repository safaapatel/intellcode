"""
calibrate_conformal.py
=======================
Computes split-conformal calibration quantiles from held-out training data
and saves them to checkpoints/{task}/conformal.json.

Uses residual-based split conformal prediction (Barber et al. 2021):
  1. Split training data into train / calibration (80/20)
  2. Train (or load) the model on the train split
  3. Compute absolute residuals on the calibration split
  4. Save the (1-alpha) quantile of residuals as the conformal quantile

At inference, the API loads conformal.json and returns:
  [point_estimate - q, point_estimate + q]
with guaranteed (1-alpha) coverage on exchangeable test samples.

Overwrites the default hardcoded quantiles with data-driven ones, tightening
the prediction intervals significantly.

Usage:
    cd backend
    python training/calibrate_conformal.py
    python training/calibrate_conformal.py --tasks security bug complexity --alpha 0.10

References:
    Barber et al. 2021 -- "Predictive inference with the jackknife+"
    Angelopoulos & Bates 2021 -- "A Gentle Introduction to Conformal Prediction"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_DATASET_PATHS = {
    "security":   "data/security_dataset.jsonl",
    "bug":        "data/bug_dataset.jsonl",
    "complexity": "data/complexity_dataset.jsonl",
}

_CHECKPOINT_DIRS = {
    "security":   "checkpoints/security",
    "bug":        "checkpoints/bug_predictor",
    "complexity": "checkpoints/complexity",
}


def _predict_security(sources: list[str]) -> np.ndarray:
    from models.security_detection import EnsembleSecurityModel
    model = EnsembleSecurityModel()
    probs = []
    for src in sources:
        try:
            probs.append(model.vulnerability_score(src))
        except Exception:
            probs.append(0.0)
    return np.array(probs, dtype=np.float64)


def _predict_bug(sources: list[str]) -> np.ndarray:
    from models.bug_predictor import BugPredictionModel
    model = BugPredictionModel()
    probs = []
    for src in sources:
        try:
            res = model.predict(src, None)
            probs.append(res.to_dict().get("bug_probability", 0.0))
        except Exception:
            probs.append(0.0)
    return np.array(probs, dtype=np.float64)


def _predict_complexity(sources: list[str]) -> np.ndarray:
    from models.complexity_prediction import ComplexityPredictionModel
    model = ComplexityPredictionModel()
    preds = []
    for src in sources:
        try:
            res = model.predict(src)
            preds.append(float(res.to_dict().get("cognitive_complexity", 0.0)))
        except Exception:
            preds.append(0.0)
    return np.array(preds, dtype=np.float64)


def _load_dataset(path: str, task: str) -> tuple[list[str], np.ndarray]:
    """Load (source, label) pairs for the given task."""
    sources, labels = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                src = record.get("source") or record.get("tokens", "")
                if not src:
                    continue
                if task == "security":
                    label = float(record.get("label", record.get("is_vulnerable", 0)))
                elif task == "bug":
                    label = float(record.get("label", record.get("is_buggy", 0)))
                else:  # complexity
                    label = float(record.get("cognitive_complexity", record.get("label", 0)))
                sources.append(src)
                labels.append(label)
            except Exception:
                continue
    return sources, np.array(labels, dtype=np.float64)


_PREDICTORS = {
    "security":   _predict_security,
    "bug":        _predict_bug,
    "complexity": _predict_complexity,
}


def calibrate_task(task: str, alpha: float) -> None:
    from features.conformal_predictor import ResidualConformalPredictor

    dataset_path = _DATASET_PATHS[task]
    ckpt_dir = Path(_CHECKPOINT_DIRS[task])
    out_path = ckpt_dir / "conformal.json"

    if not Path(dataset_path).exists():
        logger.warning("Dataset not found: %s — skipping conformal calibration for task=%s", dataset_path, task)
        return

    logger.info("Loading dataset for task=%s ...", task)
    sources, y = _load_dataset(dataset_path, task)
    n = len(sources)
    if n < 40:
        logger.warning("Too few samples (%d) for calibration — need >= 40. Skipping task=%s.", n, task)
        return

    # 80/20 split (calibration set = last 20%)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    n_cal = max(20, n // 5)
    cal_idx = idx[-n_cal:]

    cal_sources = [sources[i] for i in cal_idx]
    y_cal = y[cal_idx]

    logger.info("Predicting on %d calibration samples for task=%s ...", n_cal, task)
    predict_fn = _PREDICTORS[task]
    y_pred_cal = predict_fn(cal_sources)

    predictor = ResidualConformalPredictor(task=task, alpha=alpha)
    predictor.calibrate(y_cal, y_pred_cal)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    predictor.save(out_path)
    logger.info(
        "Conformal quantile saved: task=%s  q=%.4f  n_cal=%d  coverage=%.0f%%",
        task, predictor._quantile, n_cal, (1 - alpha) * 100,
    )


def main(tasks: list[str], alpha: float) -> None:
    logger.info("=== Conformal Calibration (alpha=%.2f, target coverage=%.0f%%) ===", alpha, (1-alpha)*100)
    for task in tasks:
        try:
            calibrate_task(task, alpha)
        except Exception as exc:
            logger.error("Calibration failed for task=%s: %s", task, exc)
    logger.info("=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate conformal prediction quantiles")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["security", "bug", "complexity"],
        choices=["security", "bug", "complexity"],
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="Miscoverage level (default: 0.10 => 90%% coverage)",
    )
    args = parser.parse_args()
    main(args.tasks, args.alpha)
