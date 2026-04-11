"""
train_ood_detector.py
======================
Fits OOD (Out-of-Distribution) detectors on training feature vectors and
saves them to checkpoints/{task}/ood_detector.pkl.

The OOD detector uses Mahalanobis distance from the training distribution
(Lee et al. 2018). At inference, samples with distance > 3.5 sigma trigger
abstention; 2.0-3.5 sigma reduces confidence by 50%.

Run this script once after training the main models to generate the
ood_detector.pkl files that the API loads at startup.

Usage:
    cd backend
    python training/train_ood_detector.py
    python training/train_ood_detector.py --tasks security bug complexity
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


def _load_security_features(dataset_path: str) -> np.ndarray:
    from features.code_metrics import compute_all_metrics, metrics_to_feature_vector
    rows = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                source = record.get("source") or record.get("tokens", "")
                if not source:
                    continue
                metrics = compute_all_metrics(source)
                rows.append(metrics_to_feature_vector(metrics))
            except Exception:
                continue
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 15), dtype=np.float32)


def _load_bug_features(dataset_path: str) -> np.ndarray:
    from features.code_metrics import compute_all_metrics, metrics_to_feature_vector
    rows = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                source = record.get("source", "")
                if not source:
                    continue
                metrics = compute_all_metrics(source)
                rows.append(metrics_to_feature_vector(metrics))
            except Exception:
                continue
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 15), dtype=np.float32)


def _load_complexity_features(dataset_path: str) -> np.ndarray:
    from features.code_metrics import compute_all_metrics, metrics_to_feature_vector
    rows = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                source = record.get("source", "")
                if not source:
                    continue
                metrics = compute_all_metrics(source)
                rows.append(metrics_to_feature_vector(metrics))
            except Exception:
                continue
    return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 15), dtype=np.float32)


_LOADERS = {
    "security":   _load_security_features,
    "bug":        _load_bug_features,
    "complexity": _load_complexity_features,
}


def train_ood_detector(task: str) -> None:
    from features.ood_detector import OODDetector

    dataset_path = _DATASET_PATHS[task]
    ckpt_dir = _CHECKPOINT_DIRS[task]
    out_path = Path(ckpt_dir) / "ood_detector.pkl"

    if not Path(dataset_path).exists():
        logger.warning("Dataset not found: %s — skipping OOD for task=%s", dataset_path, task)
        return

    logger.info("Loading features for task=%s from %s ...", task, dataset_path)
    X = _LOADERS[task](dataset_path)

    if X.shape[0] < 20:
        logger.warning("Too few samples (%d) for OOD fitting — skipping task=%s", X.shape[0], task)
        return

    logger.info("Fitting OOD detector on %d samples (dim=%d) ...", X.shape[0], X.shape[1])
    detector = OODDetector()
    detector.fit(X)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    detector.save(str(out_path))
    logger.info("OOD detector saved to %s (1-sigma=%.2f)", out_path, detector._sigma_unit)


def main(tasks: list[str]) -> None:
    logger.info("=== OOD Detector Training ===")
    for task in tasks:
        try:
            train_ood_detector(task)
        except Exception as exc:
            logger.error("OOD training failed for task=%s: %s", task, exc)
    logger.info("=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OOD detectors for all tasks")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["security", "bug", "complexity"],
        choices=["security", "bug", "complexity"],
        help="Which tasks to train OOD detectors for (default: all)",
    )
    args = parser.parse_args()
    main(args.tasks)
