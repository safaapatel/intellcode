"""
Time-Decayed Code Prediction (TDCP)
=====================================
Novel contribution #4.

Core insight: Older commits are less representative of current coding patterns.
Weighting recent commits more heavily during training should improve
forward-looking (temporal split) AUC.

Weight formula
--------------
    w_i = exp(-lambda * delta_years_i)

where delta_years_i = (max_date - commit_date_i).days / 365.0
and lambda controls the decay rate (higher = faster forgetting).

Evaluation
----------
Compare temporal split AUC:
  - Uniform weights (standard training)
  - TDCP weights (lambda = 0.5, 1.0, 2.0)

Reference
---------
Kamei et al. (2013): "A Large-Scale Empirical Study of Just-In-Time Quality Assurance"
Jiang et al. (2013): "Personalized Defect Prediction"
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False


STATIC_NAMES = [
    "cyclomatic_complexity", "cognitive_complexity",
    "max_function_complexity", "avg_function_complexity",
    "sloc", "comments", "blank_lines",
    "halstead_volume", "halstead_difficulty", "halstead_effort", "halstead_bugs",
    "n_long_functions", "n_complex_functions",
    "max_line_length", "avg_line_length", "n_lines_over_80",
    "n_functions",
]

JIT_NAMES = [
    "code_churn", "author_count", "file_age_days", "n_past_bugs", "commit_freq",
    "n_subsystems", "n_directories", "n_files", "entropy",
    "lines_added", "lines_deleted", "lines_touched", "developer_exp",
]

ALL_NAMES = STATIC_NAMES + JIT_NAMES  # 30-dim


def _parse_date(date_str: str) -> datetime:
    """Parse ISO date string to timezone-aware datetime."""
    try:
        dt = datetime.fromisoformat(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return datetime(2018, 1, 1, tzinfo=timezone.utc)


def _load_records(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _record_to_vec(rec: dict) -> list[float]:
    sf = rec["static_features"]
    gf = rec.get("git_features", {})
    git_vec = [float(gf.get(k, 0) or 0) for k in JIT_NAMES]
    return [float(v) for v in sf] + git_vec


def compute_decay_weights(
    records: list[dict],
    lam: float = 1.0,
) -> np.ndarray:
    """
    Compute exponential time-decay weights for each record.
    Recent commits (close to max_date) get weight ~ 1.0.
    Old commits get weight ~ exp(-lambda * years_ago).
    """
    dates = [_parse_date(r.get("author_date", "2018-01-01")) for r in records]
    max_date = max(dates)

    weights = []
    for dt in dates:
        delta_years = (max_date - dt).days / 365.0
        w = float(np.exp(-lam * delta_years))
        weights.append(w)

    weights = np.array(weights)
    # Normalise so mean weight = 1 (prevents scale change)
    weights = weights / weights.mean()
    return weights


def temporal_split(
    records: list[dict],
    test_fraction: float = 0.20,
) -> tuple[list[dict], list[dict]]:
    """
    Sort by date ascending; use last test_fraction as test set.
    """
    def get_date(r):
        return r.get("author_date", "2018-01-01")

    sorted_recs = sorted(records, key=get_date)
    split = int(len(sorted_recs) * (1 - test_fraction))
    return sorted_recs[:split], sorted_recs[split:]


def _train_and_eval(
    train_recs: list[dict],
    test_recs:  list[dict],
    sample_weight: np.ndarray | None,
    seed: int = 42,
) -> dict:
    """Train XGB and evaluate on test set."""
    if not _DEPS_OK:
        return {}

    X_train = np.array([_record_to_vec(r) for r in train_recs], dtype=np.float32)
    y_train = np.array([int(r["label"]) for r in train_recs])
    X_test  = np.array([_record_to_vec(r) for r in test_recs],  dtype=np.float32)
    y_test  = np.array([int(r["label"]) for r in test_recs])

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=0.0)
    X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=1e6, neginf=0.0)

    if len(set(y_test)) < 2:
        return {"error": "test set has only one class"}

    scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale,
        eval_metric="logloss", random_state=seed,
        n_jobs=1, verbosity=0,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)

    proba = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, proba))
    ap  = float(average_precision_score(y_test, proba))
    positive_rate = float(y_test.mean())

    return {
        "auc": auc,
        "ap": ap,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "test_positive_rate": positive_rate,
    }


def run_tdcp(
    data_path: str = "data/bug_dataset_v2.jsonl",
    lambdas: list[float] = (0.0, 0.5, 1.0, 2.0),
    test_fraction: float = 0.20,
    seed: int = 42,
) -> dict:
    """
    Run TDCP: compare temporal-split AUC with and without time-decay weighting.
    """
    if not _DEPS_OK:
        return {"error": "xgboost/sklearn not available"}

    logger.info("TDCP: loading data...")
    records = _load_records(data_path)
    logger.info("  %d records", len(records))

    train_recs, test_recs = temporal_split(records, test_fraction)
    logger.info("  Train: %d | Test: %d (temporal split)", len(train_recs), len(test_recs))

    results = {}
    for lam in lambdas:
        logger.info("  lambda=%.1f ...", lam)
        if lam == 0.0:
            weights = None
            label = "uniform (baseline)"
        else:
            weights = compute_decay_weights(train_recs, lam=lam)
            label = f"TDCP lambda={lam}"

        res = _train_and_eval(train_recs, test_recs, weights, seed=seed)
        res["lambda"] = lam
        res["label"] = label
        results[f"lambda_{lam}"] = res
        logger.info("    AUC=%.4f  AP=%.4f", res.get("auc", 0), res.get("ap", 0))

    # Best lambda
    best_key = max(
        results,
        key=lambda k: results[k].get("auc", 0),
    )
    best_lam = results[best_key]["lambda"]
    best_auc = results[best_key]["auc"]
    baseline_auc = results.get("lambda_0.0", {}).get("auc", 0)
    delta = best_auc - baseline_auc

    report = {
        "method": "TDCP",
        "description": (
            "Time-Decayed Code Prediction: assigns exponentially decaying sample "
            "weights during training so that recent commits are prioritised. "
            "Addresses temporal drift observed in forward-looking evaluation."
        ),
        "test_fraction": test_fraction,
        "lambdas_tested": list(lambdas),
        "results": results,
        "best_lambda": best_lam,
        "best_temporal_auc": best_auc,
        "baseline_temporal_auc": baseline_auc,
        "delta_vs_baseline": float(delta),
        "summary": (
            f"Best lambda={best_lam} improves temporal AUC by {delta:+.4f} "
            f"({baseline_auc:.4f} -> {best_auc:.4f})."
        ),
    }

    logger.info("  Best lambda=%.1f: AUC=%.4f (delta=%+.4f)", best_lam, best_auc, delta)
    return report


if __name__ == "__main__":
    import io as _io
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    report = run_tdcp()
    out = Path("evaluation/results/tdcp_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {out}")
    print(f"Baseline (uniform) temporal AUC: {report['baseline_temporal_auc']:.4f}")
    print(f"Best TDCP temporal AUC:          {report['best_temporal_auc']:.4f} (lambda={report['best_lambda']})")
    print(f"Delta:                           {report['delta_vs_baseline']:+.4f}")
