"""
Pattern Label Leakage Analysis and Debiased Model
===================================================
Addresses the circular labeling problem in the pattern classifier.

Root cause
----------
Labels are assigned by _assign_pattern_label_tool_consensus() using:
  - Signal 2 (metric thresholds): CC > 20 -> anti_pattern, SLOC <= 40 -> clean, etc.
  - Signal 1 (PyNose): structural AST rules

The training features include CC, SLOC, Halstead volume, MI — all of which are
direct inputs to the label rule, creating a self-fulfilling prediction.

Fix
---
Retrain using ONLY structural AST features that are NOT part of the labeling rule:
  n_functions, n_classes, n_try_blocks, n_raises, n_with_blocks, max_nesting_depth,
  max_params, avg_params, n_decorated_functions, n_imports,
  max_function_body_lines, avg_function_body_lines

These features are derived from AST node counts and function structural properties,
NOT from the metric thresholds used to assign labels.

Expected result
---------------
Accuracy drops vs the full-feature model (the full model partly memorises label rules).
The debiased model F1 is a more honest estimate of true generalisation.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.metrics import (
        classification_report, f1_score, roc_auc_score,
        cohen_kappa_score
    )
    from sklearn.preprocessing import label_binarize
    from scipy.stats import spearmanr, wilcoxon
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False


LABEL_NAMES  = ["clean", "code_smell", "anti_pattern", "style_violation"]
LABEL2ID     = {l: i for i, l in enumerate(LABEL_NAMES)}

# Features used in current (leaky) model — indices from train_pattern.py
FULL_FEAT_NAMES = [
    "CC", "cog", "maxCC", "avgCC", "sloc", "comments", "comment_ratio",
    "halstead_volume", "halstead_difficulty", "halstead_bugs", "MI",
    "n_long_functions", "n_complex_functions", "n_lines_over_80",
    "n_functions", "n_classes", "n_try_blocks", "n_raises", "n_with_blocks",
    "max_nesting_depth", "max_params", "avg_params",
    "n_decorated_functions", "n_imports",
    "max_function_body_lines", "avg_function_body_lines",
]

# Leaky features — directly encode labeling thresholds
LEAKY_FEATURES = {
    "CC",                # rule: CC > 20 -> anti_pattern; CC <= 4 -> clean
    "cog",               # correlated with CC (r~0.9)
    "maxCC",             # same as CC for single-function snippets
    "avgCC",             # same
    "sloc",              # rule: SLOC <= 40 -> clean; SLOC in (40,100] -> code_smell
    "halstead_volume",   # linear function of SLOC — r(sloc,hv)~0.99
    "halstead_bugs",     # = hv / 3000; direct SLOC proxy
    "halstead_difficulty", # SLOC proxy
    "MI",                # closed-form of hv * CC * sloc — leaks all three
    "n_long_functions",  # defined as functions > 50 lines — SLOC-derived
    "n_complex_functions", # defined as functions with CC > 10 — CC-derived
    "comments",          # correlated with SLOC
    "comment_ratio",     # correlated with SLOC
    "n_lines_over_80",   # style_violation label uses this directly
}

# Clean (non-leaky) structural AST features
CLEAN_FEAT_NAMES = [f for f in FULL_FEAT_NAMES if f not in LEAKY_FEATURES]
CLEAN_FEAT_INDICES = [FULL_FEAT_NAMES.index(f) for f in CLEAN_FEAT_NAMES]
FULL_FEAT_INDICES  = list(range(len(FULL_FEAT_NAMES)))


def _load_pattern_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    from training.train_pattern import _extract_features

    X, y = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            label = rec.get("label")
            if label not in LABEL2ID:
                continue
            code = rec.get("code", "")
            if not code.strip():
                continue
            try:
                x = _extract_features(code)
                X.append(x)
                y.append(LABEL2ID[label])
            except Exception:
                pass

    return np.array(X, dtype=np.float32), np.array(y)


def leakage_audit(
    X: np.ndarray,
    y: np.ndarray,
    feat_names: list[str] = FULL_FEAT_NAMES,
    leaky_set: set[str] = LEAKY_FEATURES,
) -> dict:
    """Compute per-feature Spearman |r| with each class indicator."""
    if not _DEPS_OK:
        return {}
    y_bin = label_binarize(y, classes=[0, 1, 2, 3])

    audit = []
    for fi, fname in enumerate(feat_names):
        col = X[:, fi]
        max_r = 0.0
        per_class = {}
        for ci, cname in enumerate(LABEL_NAMES):
            r, _ = spearmanr(col, y_bin[:, ci])
            per_class[cname] = round(float(r), 4)
            max_r = max(max_r, abs(r))
        audit.append({
            "feature": fname,
            "is_leaky": fname in leaky_set,
            "max_abs_r": round(max_r, 4),
            "per_class_r": per_class,
        })

    audit.sort(key=lambda x: -x["max_abs_r"])
    return audit


def compare_models(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> dict:
    """5-fold CV comparison: full (leaky) vs debiased (clean) features."""
    if not _DEPS_OK:
        return {}

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    full_f1s, clean_f1s = [], []
    full_kappas, clean_kappas = [], []

    for train_idx, test_idx in kf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # Full (leaky) model
        rf_full = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=seed, n_jobs=1
        )
        rf_full.fit(X_tr[:, FULL_FEAT_INDICES], y_tr)
        pred_full = rf_full.predict(X_te[:, FULL_FEAT_INDICES])
        full_f1s.append(f1_score(y_te, pred_full, average="macro", zero_division=0))
        full_kappas.append(cohen_kappa_score(y_te, pred_full))

        # Debiased (clean) model
        rf_clean = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=seed, n_jobs=1
        )
        rf_clean.fit(X_tr[:, CLEAN_FEAT_INDICES], y_tr)
        pred_clean = rf_clean.predict(X_te[:, CLEAN_FEAT_INDICES])
        clean_f1s.append(f1_score(y_te, pred_clean, average="macro", zero_division=0))
        clean_kappas.append(cohen_kappa_score(y_te, pred_clean))

    # Wilcoxon signed-rank test: is the difference significant?
    try:
        w_stat, w_p = wilcoxon(full_f1s, clean_f1s, alternative="greater")
    except Exception:
        w_stat, w_p = None, None

    return {
        "full_model": {
            "features_used": FULL_FEAT_NAMES,
            "n_features": len(FULL_FEAT_NAMES),
            "mean_f1_macro": float(np.mean(full_f1s)),
            "std_f1": float(np.std(full_f1s)),
            "mean_kappa": float(np.mean(full_kappas)),
            "per_fold_f1": [round(v, 4) for v in full_f1s],
        },
        "debiased_model": {
            "features_used": CLEAN_FEAT_NAMES,
            "n_features": len(CLEAN_FEAT_NAMES),
            "mean_f1_macro": float(np.mean(clean_f1s)),
            "std_f1": float(np.std(clean_f1s)),
            "mean_kappa": float(np.mean(clean_kappas)),
            "per_fold_f1": [round(v, 4) for v in clean_f1s],
        },
        "delta_f1": float(np.mean(full_f1s) - np.mean(clean_f1s)),
        "wilcoxon_stat": float(w_stat) if w_stat is not None else None,
        "wilcoxon_p": float(w_p) if w_p is not None else None,
        "leakage_inflation": float(np.mean(full_f1s) - np.mean(clean_f1s)),
        "interpretation": (
            f"Full-feature model F1={np.mean(full_f1s):.3f} vs "
            f"debiased F1={np.mean(clean_f1s):.3f}. "
            f"Delta={np.mean(full_f1s)-np.mean(clean_f1s):+.3f} = "
            "estimated leakage inflation. "
            + ("Wilcoxon p=" + f"{w_p:.4f}" + (
                " — full model is significantly better: leakage confirmed."
                if (w_p or 1) < 0.05 else
                " — difference not statistically significant at alpha=0.05."
            ) if w_p is not None else "")
        ),
    }


def run_pattern_leakage_fix(
    data_path: str = "data/pattern_dataset.jsonl",
    seed: int = 42,
) -> dict:
    if not _DEPS_OK:
        return {"error": "sklearn/scipy not available"}

    logger.info("Pattern Leakage Fix: loading dataset...")
    X, y = _load_pattern_data(data_path)
    logger.info("  %d samples, %d features", len(y), X.shape[1])

    logger.info("  Running leakage audit...")
    audit = leakage_audit(X, y)
    leaky_count = sum(1 for a in audit if a["is_leaky"])

    logger.info("  Comparing full vs debiased model...")
    comparison = compare_models(X, y, seed=seed)

    full_f1  = comparison["full_model"]["mean_f1_macro"]
    clean_f1 = comparison["debiased_model"]["mean_f1_macro"]
    delta    = comparison["delta_f1"]

    logger.info("  Full model F1:     %.4f", full_f1)
    logger.info("  Debiased model F1: %.4f", clean_f1)
    logger.info("  Leakage inflation: %+.4f", delta)

    report = {
        "method": "Pattern_Leakage_Fix",
        "description": (
            "Demonstrates circular labeling leakage in the pattern classifier: "
            "features (CC, SLOC, Halstead) are used in label assignment AND in training. "
            "A debiased model using only structural AST features (not used in labeling) "
            "shows the true generalisation performance."
        ),
        "n_samples": int(len(y)),
        "n_leaky_features": leaky_count,
        "n_clean_features": len(CLEAN_FEAT_NAMES),
        "leaky_features": sorted(LEAKY_FEATURES),
        "clean_features": CLEAN_FEAT_NAMES,
        "audit_top15": audit[:15],
        "model_comparison": comparison,
        "finding": (
            f"Full-feature model F1={full_f1:.3f} is inflated by {delta:+.3f} "
            f"over the debiased model (F1={clean_f1:.3f}). "
            "This gap quantifies the leakage: the model partially memorises "
            "the metric thresholds used to construct labels, not genuine patterns."
        ),
        "fix_recommendation": (
            "For final training: use only CLEAN_FEAT_NAMES (structural AST features). "
            "For labeling: consider external ground truth (developer code reviews, "
            "pylint/flake8 annotations) rather than metric thresholds."
        ),
    }

    return report


if __name__ == "__main__":
    import io as _io
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    report = run_pattern_leakage_fix()
    out = Path("evaluation/results/pattern_leakage_fix.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {out}")
    comp = report["model_comparison"]
    print(f"Full model F1:     {comp['full_model']['mean_f1_macro']:.4f}")
    print(f"Debiased F1:       {comp['debiased_model']['mean_f1_macro']:.4f}")
    print(f"Leakage inflation: {comp['leakage_inflation']:+.4f}")
    print(f"Wilcoxon p:        {comp.get('wilcoxon_p')}")
