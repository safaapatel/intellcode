"""
Temporal Train/Test Split Utilities
=====================================
Provides strict past->future splits for defect prediction evaluation.

Motivation: Random splits allow future bug-fix commits to appear in the
training set. SZZ labels a commit as bug-introducing based on the fix
commit that comes *after* it. If the fix appears in training, the model
learns to recognise already-fixed code patterns -- a form of target leakage
that inflates AUC by 0.10-0.15 on random splits vs temporal splits.

Bug temporal split results (Mar 2026 audit):
    Random split AUC = 0.618  -- overstated due to leakage
    Temporal split AUC = 0.460 -- true generalisation performance

All bug/security model training MUST use temporal_split() not train_test_split().

References:
    Kamei et al. 2013 -- "A Large-Scale Empirical Study of JIT Quality Assurance"
    Ni et al. 2022 -- "Just-In-Time Defect Prediction on JavaScript Projects"
    Rodiguez-Perez et al. 2020 -- "How bugs are born: SZZ accuracy issues"
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class TemporalSplit:
    """Result of a temporal train/test split."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_test:  np.ndarray
    y_test:  np.ndarray
    # Metadata
    cutoff_timestamp: float          # Unix timestamp of the split boundary
    n_train: int
    n_test: int
    train_positive_rate: float       # class balance in train
    test_positive_rate: float        # class balance in test
    train_repos: set                 # repos present in training
    test_repos: set                  # repos present in test (may overlap)


# ---------------------------------------------------------------------------
# Core splitter
# ---------------------------------------------------------------------------

def temporal_split(
    records: list[dict],
    feature_key: str = "features",
    label_key: str = "label",
    timestamp_key: str = "timestamp",
    test_ratio: float = 0.20,
    min_test_positives: int = 20,
) -> TemporalSplit:
    """
    Split dataset records in strict chronological order.

    Records are sorted by timestamp (Unix seconds). The last `test_ratio`
    fraction forms the test set; everything before forms training. No record
    in the test set was seen during training.

    Args:
        records:            List of dicts, each with feature_key, label_key,
                            timestamp_key, and optionally "repo".
        feature_key:        Key in each record holding the feature vector (list).
        label_key:          Key holding the binary label (0/1).
        timestamp_key:      Key holding Unix timestamp of the commit.
        test_ratio:         Fraction of records to use as test set (by time).
        min_test_positives: Raise ValueError if test set has fewer positive
                            examples than this threshold.

    Returns:
        TemporalSplit dataclass.

    Raises:
        ValueError: If fewer than min_test_positives bug-introducing commits
                    fall in the test window.
    """
    # Sort chronologically
    sorted_recs = sorted(records, key=lambda r: r.get(timestamp_key, 0))

    n_total = len(sorted_recs)
    n_test = max(1, int(n_total * test_ratio))
    n_train = n_total - n_test

    train_recs = sorted_recs[:n_train]
    test_recs  = sorted_recs[n_train:]

    cutoff = test_recs[0].get(timestamp_key, 0) if test_recs else 0

    def _to_arrays(recs):
        X = np.array([r[feature_key] for r in recs], dtype=np.float32)
        y = np.array([int(r[label_key]) for r in recs], dtype=np.int32)
        return X, y

    X_train, y_train = _to_arrays(train_recs)
    X_test,  y_test  = _to_arrays(test_recs)

    n_test_pos = int(y_test.sum())
    if n_test_pos < min_test_positives:
        logger.warning(
            "Temporal test set has only %d positive samples (threshold=%d). "
            "Consider a wider time window or a larger dataset.",
            n_test_pos, min_test_positives,
        )

    train_pos_rate = float(y_train.mean()) if len(y_train) else 0.0
    test_pos_rate  = float(y_test.mean())  if len(y_test)  else 0.0

    train_repos = {r.get("repo", "") for r in train_recs}
    test_repos  = {r.get("repo", "") for r in test_recs}

    logger.info(
        "Temporal split: train=%d (pos=%.1f%%), test=%d (pos=%.1f%%), cutoff=%s",
        n_train, train_pos_rate * 100,
        n_test,  test_pos_rate  * 100,
        _unix_to_iso(cutoff),
    )

    return TemporalSplit(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        cutoff_timestamp=cutoff,
        n_train=n_train,
        n_test=n_test,
        train_positive_rate=train_pos_rate,
        test_positive_rate=test_pos_rate,
        train_repos=train_repos,
        test_repos=test_repos,
    )


def temporal_split_jsonl(
    path: str | Path,
    feature_key: str = "features",
    label_key: str = "label",
    timestamp_key: str = "timestamp",
    test_ratio: float = 0.20,
) -> TemporalSplit:
    """
    Convenience wrapper: load records from a JSONL file and apply temporal_split().
    """
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return temporal_split(
        records,
        feature_key=feature_key,
        label_key=label_key,
        timestamp_key=timestamp_key,
        test_ratio=test_ratio,
    )


# ---------------------------------------------------------------------------
# Cross-temporal validation (walk-forward)
# ---------------------------------------------------------------------------

def walk_forward_splits(
    records: list[dict],
    n_splits: int = 5,
    feature_key: str = "features",
    label_key: str = "label",
    timestamp_key: str = "timestamp",
    min_train_size: int = 100,
) -> list[TemporalSplit]:
    """
    Walk-forward temporal cross-validation.

    Divides the timeline into (n_splits + 1) equal windows. For split i:
      - Train on windows 0..i
      - Test on window i+1

    This simulates sequential deployment: the model is always evaluated on
    data it has never seen, from a future time period.

    Args:
        records:        Chronologically-ordered dataset records.
        n_splits:       Number of walk-forward folds.
        min_train_size: Minimum training examples required for a valid fold.

    Returns:
        List of TemporalSplit objects, one per fold.
    """
    sorted_recs = sorted(records, key=lambda r: r.get(timestamp_key, 0))
    n = len(sorted_recs)
    window_size = n // (n_splits + 1)

    splits = []
    for fold in range(n_splits):
        train_end = (fold + 1) * window_size
        test_end  = (fold + 2) * window_size
        if test_end > n:
            test_end = n

        train_recs = sorted_recs[:train_end]
        test_recs  = sorted_recs[train_end:test_end]

        if len(train_recs) < min_train_size or not test_recs:
            continue

        def _arr(recs, key, dtype):
            return np.array([r[key] for r in recs], dtype=dtype)

        X_tr = _arr(train_recs, feature_key, np.float32)
        y_tr = _arr(train_recs, label_key,   np.int32)
        X_te = _arr(test_recs,  feature_key, np.float32)
        y_te = _arr(test_recs,  label_key,   np.int32)

        cutoff = test_recs[0].get(timestamp_key, 0)
        splits.append(TemporalSplit(
            X_train=X_tr, y_train=y_tr,
            X_test=X_te,  y_test=y_te,
            cutoff_timestamp=cutoff,
            n_train=len(train_recs), n_test=len(test_recs),
            train_positive_rate=float(y_tr.mean()) if len(y_tr) else 0.0,
            test_positive_rate=float(y_te.mean())  if len(y_te) else 0.0,
            train_repos={r.get("repo", "") for r in train_recs},
            test_repos={r.get("repo", "") for r in test_recs},
        ))
        logger.info(
            "WF fold %d/%d: train=%d test=%d cutoff=%s",
            fold + 1, n_splits,
            len(train_recs), len(test_recs),
            _unix_to_iso(cutoff),
        )

    return splits


def evaluate_walk_forward(
    model,
    splits: list[TemporalSplit],
    fit_fn=None,
) -> dict:
    """
    Run walk-forward evaluation given a list of TemporalSplit folds.

    Args:
        model:   An sklearn-compatible estimator (has fit/predict_proba).
        splits:  Output of walk_forward_splits().
        fit_fn:  Optional custom fit callable: fit_fn(model, X_train, y_train).
                 If None, calls model.fit(X_train, y_train).

    Returns:
        dict with per-fold and aggregate AUC, PofB20, Precision@10.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    fold_results = []
    for i, split in enumerate(splits):
        if fit_fn is not None:
            fit_fn(model, split.X_train, split.y_train)
        else:
            model.fit(split.X_train, split.y_train)

        probas = model.predict_proba(split.X_test)[:, 1]
        auc = float(roc_auc_score(split.y_test, probas))
        ap  = float(average_precision_score(split.y_test, probas))

        # PofB20: fraction of bugs found in top-20% of ranked files
        n_test = len(split.y_test)
        top20  = max(1, int(n_test * 0.20))
        order  = np.argsort(probas)[::-1]
        pofb20 = float(split.y_test[order[:top20]].sum() / max(1, split.y_test.sum()))

        fold_results.append({
            "fold": i + 1,
            "auc": round(auc, 4),
            "ap": round(ap, 4),
            "pofb20": round(pofb20, 4),
            "n_train": split.n_train,
            "n_test": split.n_test,
        })
        logger.info("  Fold %d: AUC=%.3f  AP=%.3f  PofB20=%.3f", i + 1, auc, ap, pofb20)

    aucs   = [r["auc"]   for r in fold_results]
    pofbs  = [r["pofb20"] for r in fold_results]

    return {
        "folds": fold_results,
        "mean_auc":    round(float(np.mean(aucs)),  4),
        "std_auc":     round(float(np.std(aucs)),   4),
        "mean_pofb20": round(float(np.mean(pofbs)), 4),
        "std_pofb20":  round(float(np.std(pofbs)),  4),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unix_to_iso(ts: float) -> str:
    """Convert Unix timestamp to ISO-8601 date string."""
    import datetime
    try:
        return datetime.datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
    except Exception:
        return str(ts)
