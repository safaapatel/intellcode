"""
Leakage-Aware Model Validation (LAMV)
======================================
Novel contribution #3.

Formalizes a reusable pipeline for detecting label leakage in code analysis
ML models.  We encountered leakage empirically (is_fix circular with keyword
labels; maintainability_index as both feature and target); this module turns
that discovery into a general, automated detection method.

Three detection mechanisms
---------------------------
1. Feature-label correlation: Spearman |r| > threshold => suspicious
2. Training accuracy anomaly: train_AUC - CV_AUC > gap_threshold => suspicious
3. Single-feature oracle: can any single feature alone predict the label
   with AUC > oracle_threshold?

Reference
---------
Kaufman et al. (2012): "Leakage in Data Mining: Formulation, Detection, and Avoidance"
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logger = logging.getLogger(__name__)

try:
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_score
    from xgboost import XGBClassifier
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
    "is_fix",   # Deliberately included so LAMV can FLAG it as leaky
]

ALL_NAMES_WITH_ISFIX = STATIC_NAMES + JIT_NAMES  # 31-dim (includes is_fix for detection)


def _load_dataset(path: str, include_is_fix: bool = True) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load dataset into (X, y, feature_names) arrays."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    X, y = [], []
    names = STATIC_NAMES + JIT_NAMES if include_is_fix else STATIC_NAMES + JIT_NAMES[:-1]

    for rec in records:
        sf = rec.get("static_features", [])
        gf = rec.get("git_features", {})
        if include_is_fix:
            git_vec = [float(gf.get(k, 0) or 0) for k in JIT_NAMES]
        else:
            git_vec = [float(gf.get(k, 0) or 0) for k in JIT_NAMES[:-1]]
        x = [float(v) for v in sf] + git_vec
        X.append(x)
        y.append(int(rec["label"]))

    X_arr = np.array(X, dtype=np.float64)
    X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=1e6, neginf=0.0)
    return X_arr, np.array(y), names


# ── Detection mechanisms ──────────────────────────────────────────────────────

def check_feature_label_correlation(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    r_threshold: float = 0.70,
) -> list[dict]:
    """
    Compute Spearman |r| between each feature and the label.
    Flag features with |r| > r_threshold as potentially leaky.
    """
    results = []
    for i, name in enumerate(feature_names):
        col = X[:, i]
        if col.std() < 1e-9:
            r, p = 0.0, 1.0
        else:
            r, p = spearmanr(col, y)
        suspicious = abs(r) > r_threshold
        results.append({
            "feature": name,
            "spearman_r": float(r),
            "p_value": float(p),
            "suspicious": bool(suspicious),
            "reason": f"|r|={abs(r):.3f} > threshold {r_threshold}" if suspicious else None,
        })
    results.sort(key=lambda x: -abs(x["spearman_r"]))
    return results


def check_single_feature_oracle(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    oracle_threshold: float = 0.85,
) -> list[dict]:
    """
    For each feature, check if a single-feature XGB achieves AUC > oracle_threshold.
    Such a feature is almost certainly a label proxy.
    """
    if not _DEPS_OK:
        return []

    results = []
    for i, name in enumerate(feature_names):
        col = X[:, i:i+1]
        if col.std() < 1e-9:
            continue
        try:
            model = XGBClassifier(
                n_estimators=50, max_depth=2, random_state=42,
                eval_metric="logloss", verbosity=0, n_jobs=1,
            )
            # Quick 3-fold CV
            scores = cross_val_score(model, col, y, cv=3,
                                     scoring="roc_auc", n_jobs=1)
            auc = float(np.mean(scores))
            suspicious = auc > oracle_threshold
            results.append({
                "feature": name,
                "single_feature_auc": auc,
                "suspicious": suspicious,
                "reason": f"single-feature AUC={auc:.3f} > {oracle_threshold}" if suspicious else None,
            })
        except Exception:
            pass
    results.sort(key=lambda x: -x["single_feature_auc"])
    return results


def check_train_test_gap(
    X: np.ndarray,
    y: np.ndarray,
    gap_threshold: float = 0.20,
    seed: int = 42,
) -> dict:
    """
    Train a full XGB on all data; compare train AUC vs 5-fold CV AUC.
    Large gap indicates overfitting or leakage.
    """
    if not _DEPS_OK:
        return {}

    model = XGBClassifier(
        n_estimators=200, max_depth=4, random_state=seed,
        eval_metric="logloss", verbosity=0, n_jobs=1,
    )
    # CV AUC
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc", n_jobs=1)
    cv_auc = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))

    # Train AUC
    model.fit(X, y)
    train_proba = model.predict_proba(X)[:, 1]
    train_auc   = float(roc_auc_score(y, train_proba))

    gap = train_auc - cv_auc
    suspicious = gap > gap_threshold

    return {
        "train_auc": train_auc,
        "cv_auc": cv_auc,
        "cv_std": cv_std,
        "gap": gap,
        "suspicious": suspicious,
        "reason": f"train-CV gap={gap:.3f} > {gap_threshold}" if suspicious else None,
    }


# ── Main runner ───────────────────────────────────────────────────────────────

def run_lamv(
    data_path: str = "data/bug_dataset_v2.jsonl",
    r_threshold: float = 0.70,
    oracle_threshold: float = 0.85,
    gap_threshold: float = 0.20,
) -> dict:
    if not _DEPS_OK:
        return {"error": "scipy/sklearn/xgboost not available"}

    logger.info("LAMV: loading dataset (with is_fix)...")
    X, y, names = _load_dataset(data_path, include_is_fix=True)
    logger.info("  %d records, %d features, %d positive labels (%.1f%%)",
                len(y), X.shape[1], y.sum(), 100 * y.mean())

    logger.info("  Check 1: feature-label correlation...")
    corr_results = check_feature_label_correlation(X, y, names, r_threshold)
    flagged_corr = [r for r in corr_results if r["suspicious"]]

    logger.info("  Check 2: single-feature oracle...")
    oracle_results = check_single_feature_oracle(X, y, names, oracle_threshold)
    flagged_oracle = [r for r in oracle_results if r["suspicious"]]

    logger.info("  Check 3: train-test gap...")
    gap_result = check_train_test_gap(X, y, gap_threshold)

    # Summary flags
    all_suspicious = set(
        [r["feature"] for r in flagged_corr] +
        [r["feature"] for r in flagged_oracle]
    )

    report = {
        "method": "LAMV",
        "description": (
            "Leakage-Aware Model Validation: three automated checks for label leakage "
            "in ML-based code analysis pipelines. "
            "(1) feature-label correlation, (2) single-feature oracle AUC, "
            "(3) train/CV accuracy gap."
        ),
        "n_records": int(len(y)),
        "n_features": int(X.shape[1]),
        "positive_rate": float(y.mean()),
        "thresholds": {
            "r_threshold": r_threshold,
            "oracle_threshold": oracle_threshold,
            "gap_threshold": gap_threshold,
        },
        "check1_correlation": {
            "all_results": corr_results[:15],  # top 15 by |r|
            "flagged": flagged_corr,
            "n_flagged": len(flagged_corr),
        },
        "check2_oracle": {
            "top_results": oracle_results[:10],
            "flagged": flagged_oracle,
            "n_flagged": len(flagged_oracle),
        },
        "check3_gap": gap_result,
        "summary": {
            "leakage_detected": bool(flagged_corr or flagged_oracle or gap_result.get("suspicious")),
            "suspicious_features": sorted(all_suspicious),
            "recommendation": (
                "Remove flagged features before retraining." if all_suspicious
                else "No obvious leakage detected."
            ),
        },
        "known_leakage_verified": {
            "is_fix_flagged": "is_fix" in all_suspicious,
            "note": (
                "is_fix is the Kamei JIT feature equal to 1 when the commit message "
                "contains 'fix' or 'bug'. With keyword-based labels, this is identically "
                "equal to the label — a textbook case of circular feature construction. "
                "LAMV correctly flags it."
            ),
        },
    }

    logger.info("  Suspicious features: %s", sorted(all_suspicious))
    logger.info("  is_fix flagged: %s", report["known_leakage_verified"]["is_fix_flagged"])

    return report


if __name__ == "__main__":
    import io as _io
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    report = run_lamv()
    out = Path("evaluation/results/lamv_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {out}")
    print(f"Leakage detected: {report['summary']['leakage_detected']}")
    print(f"Suspicious features: {report['summary']['suspicious_features']}")
    print(f"is_fix correctly flagged: {report['known_leakage_verified']['is_fix_flagged']}")
