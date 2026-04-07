"""
Dynamic Hard Negative Miner
============================
Runs the trained security model over a corpus of clean (label=0) code files,
collects false positives (predicted vulnerable but labelled clean), and adds
them as high-weight hard negatives to the training set.

This breaks the static negative sampling loop:
  - Static negatives are drawn at random -> model never sees "safe" uses of
    dangerous APIs (e.g. parameterized queries, properly escaped output).
  - The model learns only "dangerous API present -> vulnerability", which is
    causally wrong and causes high false-positive rates on real codebases.
  - Mining hard negatives rebalances the training signal toward causal features
    (taint flow) rather than API co-occurrence.

Usage:
    python hard_negative_miner.py \\
        --clean-dirs path/to/clean/repos \\
        --dataset    backend/data/security_dataset.jsonl \\
        --output     backend/data/security_dataset_hn.jsonl \\
        --threshold  0.5 \\
        --max-hn     500

The output file is a superset of the input dataset with hard negatives appended.
Run train_security.py on the output file to retrain with hard negatives.

Iteration:
    For best results repeat the mine->retrain loop 2-3 times until the
    false-positive rate on a held-out clean set stops decreasing.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Ensure backend is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Hard-negative weight assigned in the retrained dataset
# ---------------------------------------------------------------------------
_HARD_NEGATIVE_WEIGHT = 3.0


# ---------------------------------------------------------------------------
# Feature extraction (mirrors dataset_builder.py for consistency)
# ---------------------------------------------------------------------------

def _safe_read(path: Path) -> Optional[str]:
    """Read a Python file, returning None on decode errors."""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def _is_valid_python(source: str) -> bool:
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False


def _is_hard_negative_candidate(source: str, threshold: float, model) -> bool:
    """Return True if the model predicts vulnerability above threshold."""
    try:
        score = model.vulnerability_score(source)
        return score >= threshold
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Mine loop
# ---------------------------------------------------------------------------

def mine_hard_negatives(
    clean_dirs: list[str],
    dataset_path: str,
    output_path: str,
    threshold: float = 0.5,
    max_hn: int = 500,
    checkpoint_dir: str = "checkpoints/security",
) -> int:
    """
    Mine hard negatives from clean code directories.

    Args:
        clean_dirs:    List of directories containing clean Python files.
        dataset_path:  Existing JSONL dataset (will be copied to output).
        output_path:   Path to write the augmented dataset.
        threshold:     Security model score above which a file is a false positive.
        max_hn:        Maximum number of hard negatives to add.
        checkpoint_dir: Where to load the security model checkpoints from.

    Returns:
        Number of hard negatives added.
    """
    # Load model
    try:
        from models.security_detection import EnsembleSecurityModel
        model = EnsembleSecurityModel(checkpoint_dir=checkpoint_dir)
    except Exception as e:
        logger.error("Failed to load security model: %s", e)
        return 0

    if not model._rf_ready and not model._cnn_ready:
        logger.error("No trained model found in %s — train first.", checkpoint_dir)
        return 0

    # Collect candidate files
    py_files: list[Path] = []
    for d in clean_dirs:
        p = Path(d)
        if not p.is_dir():
            logger.warning("Skipping non-directory: %s", d)
            continue
        py_files.extend(p.rglob("*.py"))

    logger.info("Scanning %d Python files for hard negatives...", len(py_files))

    hard_negatives: list[dict] = []
    scanned = 0

    for fpath in py_files:
        if len(hard_negatives) >= max_hn:
            break
        source = _safe_read(fpath)
        if source is None or len(source) < 50:
            continue
        if not _is_valid_python(source):
            continue

        scanned += 1
        if scanned % 500 == 0:
            logger.info(
                "  Scanned %d files, found %d hard negatives so far...",
                scanned, len(hard_negatives)
            )

        if _is_hard_negative_candidate(source, threshold, model):
            hard_negatives.append({
                "source": source,
                "label": 0,              # clean — this IS a hard negative
                "is_hard_negative": True,
                "sample_weight": _HARD_NEGATIVE_WEIGHT,
                "file": str(fpath),
            })

    logger.info(
        "Found %d hard negatives in %d scanned files (%.1f%% FP rate).",
        len(hard_negatives), scanned,
        100.0 * len(hard_negatives) / max(1, scanned),
    )

    # Copy existing dataset + append hard negatives
    out = Path(output_path)
    existing_lines: list[str] = []
    if Path(dataset_path).exists():
        with open(dataset_path, encoding="utf-8") as f:
            existing_lines = f.readlines()

    with open(out, "w", encoding="utf-8") as f:
        for line in existing_lines:
            f.write(line)
        for rec in hard_negatives:
            f.write(json.dumps(rec) + "\n")

    logger.info(
        "Wrote %d original + %d hard negatives to %s",
        len(existing_lines), len(hard_negatives), output_path,
    )
    return len(hard_negatives)


# ---------------------------------------------------------------------------
# ECE under shift: calibration breakdown (in-dist vs LOPO vs OOD)
# ---------------------------------------------------------------------------

def compute_ece_breakdown(
    probs: np.ndarray,
    labels: np.ndarray,
    ood_flags: Optional[np.ndarray] = None,
    n_bins: int = 10,
) -> dict:
    """
    Compute ECE (Expected Calibration Error) for three splits:
      - in_dist: samples flagged as in-distribution (ood_flags=False)
      - ood: samples flagged as out-of-distribution (ood_flags=True)
      - overall: all samples

    Args:
        probs:     Predicted probabilities (N,)
        labels:    Ground truth labels 0/1 (N,)
        ood_flags: Boolean array, True = OOD sample. If None, overall only.
        n_bins:    Number of calibration bins.

    Returns:
        dict with keys: overall_ece, in_dist_ece, ood_ece
    """
    def _ece(p: np.ndarray, y: np.ndarray) -> float:
        if len(p) == 0:
            return float("nan")
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        ece_sum = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (p >= lo) & (p < hi)
            if mask.sum() == 0:
                continue
            bin_conf = p[mask].mean()
            bin_acc  = y[mask].mean()
            ece_sum += mask.sum() * abs(bin_conf - bin_acc)
        return float(ece_sum / len(p))

    result = {"overall_ece": _ece(probs, labels)}

    if ood_flags is not None:
        in_mask  = ~ood_flags
        ood_mask = ood_flags
        result["in_dist_ece"] = _ece(probs[in_mask],  labels[in_mask])
        result["ood_ece"]     = _ece(probs[ood_mask], labels[ood_mask])
    else:
        result["in_dist_ece"] = result["overall_ece"]
        result["ood_ece"]     = float("nan")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Mine hard negatives for security model retraining.")
    p.add_argument("--clean-dirs", nargs="+", required=True,
                   help="Directories with clean Python files to scan.")
    p.add_argument("--dataset", required=True,
                   help="Existing security dataset JSONL (will be preserved).")
    p.add_argument("--output", required=True,
                   help="Output JSONL path for augmented dataset.")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Model score threshold to classify a file as false positive.")
    p.add_argument("--max-hn", type=int, default=500,
                   help="Maximum hard negatives to add.")
    p.add_argument("--checkpoint-dir", default="checkpoints/security",
                   help="Security model checkpoint directory.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    n_added = mine_hard_negatives(
        clean_dirs=args.clean_dirs,
        dataset_path=args.dataset,
        output_path=args.output,
        threshold=args.threshold,
        max_hn=args.max_hn,
        checkpoint_dir=args.checkpoint_dir,
    )
    sys.exit(0 if n_added >= 0 else 1)
