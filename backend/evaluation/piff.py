"""
Project-Invariant Feature Filtering (PIFF)
==========================================
Novel contribution #1.

Core insight: Features with high variance *across* projects are likely encoding
project-specific style, not generalizable bug signals. Dropping them before
training improves cross-project (LOPO) generalization.

Algorithm
---------
1. Group training data by repository.
2. For each feature f, compute the per-project mean: mu_{f,p}.
3. Measure instability = CV(mu_{f,*}) = std(mu_{f,*}) / |mean(mu_{f,*})| + eps.
4. Sort features by CV ascending; select the k most stable.
5. Retrain XGB on stable features only.
6. Compare LOPO AUC: full features vs PIFF-filtered features.

Reference
---------
Hassan & Holt (2009): "The Impact of Independent Changes on Code Ownership"
Zimmermann et al. (2009): "Cross-Project Defect Prediction"
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
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False

STATIC_FEATURE_NAMES = [
    "cyclomatic_complexity", "cognitive_complexity",
    "max_function_complexity", "avg_function_complexity",
    "sloc", "comments", "blank_lines",
    "halstead_volume", "halstead_difficulty", "halstead_effort", "halstead_bugs",
    "n_long_functions", "n_complex_functions",
    "max_line_length", "avg_line_length", "n_lines_over_80",
    "n_functions",
]

JIT_FEATURE_NAMES = [
    "code_churn", "author_count", "file_age_days", "n_past_bugs", "commit_freq",
    "n_subsystems", "n_directories", "n_files", "entropy",
    "lines_added", "lines_deleted", "lines_touched", "developer_exp",
]

ALL_FEATURE_NAMES = STATIC_FEATURE_NAMES + JIT_FEATURE_NAMES  # 30-dim


def _load_bug_records(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _record_to_xy(rec: dict) -> tuple[list[float], int]:
    sf = rec["static_features"]  # 17-dim
    gf = rec.get("git_features", {})
    git_vec = [float(gf.get(k, 0) or 0) for k in JIT_FEATURE_NAMES]
    x = [float(v) for v in sf] + git_vec  # 30-dim
    return x, int(rec["label"])


def compute_feature_stability(
    records: list[dict],
    feature_names: list[str] = ALL_FEATURE_NAMES,
) -> dict[str, float]:
    """
    Returns a dict {feature_name: CV} where CV = coefficient of variation
    of the per-project feature mean.  Lower CV = more stable across projects.
    """
    projects: dict[str, list[list[float]]] = {}
    for rec in records:
        repo = rec.get("repo", "unknown").split("/")[-1]
        x, _ = _record_to_xy(rec)
        projects.setdefault(repo, []).append(x)

    project_means = {}
    for repo, xlist in projects.items():
        arr = np.array(xlist)
        project_means[repo] = arr.mean(axis=0)

    means_matrix = np.array(list(project_means.values()))  # (n_projects, n_features)
    global_mean = means_matrix.mean(axis=0)
    std_of_means = means_matrix.std(axis=0)
    cv = std_of_means / (np.abs(global_mean) + 1e-9)

    return {name: float(cv[i]) for i, name in enumerate(feature_names)}


def select_stable_features(
    stability: dict[str, float],
    top_k: Optional[int] = None,
    cv_threshold: float = 1.0,
) -> list[str]:
    """
    Select features with CV <= cv_threshold (or top-k most stable).
    """
    sorted_feats = sorted(stability.items(), key=lambda x: x[1])
    if top_k is not None:
        return [name for name, _ in sorted_feats[:top_k]]
    return [name for name, cv in sorted_feats if cv <= cv_threshold]


def _run_lopo(
    records: list[dict],
    feature_indices: list[int],
    seed: int = 42,
) -> list[dict]:
    """Run LOPO with a specific subset of feature indices. Returns per-project results."""
    if not _DEPS_OK:
        return []

    repos = sorted({rec.get("repo", "unknown").split("/")[-1] for rec in records})
    results = []

    for held_out in repos:
        train_recs = [r for r in records if r.get("repo", "").split("/")[-1] != held_out]
        test_recs  = [r for r in records if r.get("repo", "").split("/")[-1] == held_out]

        if len(test_recs) < 10:
            continue

        X_train, y_train, X_test, y_test = [], [], [], []
        for rec in train_recs:
            x, y = _record_to_xy(rec)
            X_train.append([x[i] for i in feature_indices])
            y_train.append(y)
        for rec in test_recs:
            x, y = _record_to_xy(rec)
            X_test.append([x[i] for i in feature_indices])
            y_test.append(y)

        if len(set(y_train)) < 2 or len(set(y_test)) < 2:
            continue

        X_tr = np.array(X_train, dtype=np.float32)
        y_tr = np.array(y_train)
        X_te = np.array(X_test, dtype=np.float32)
        y_te = np.array(y_test)

        # Replace inf/nan
        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=1e6, neginf=0.0)
        X_te = np.nan_to_num(X_te, nan=0.0, posinf=1e6, neginf=0.0)

        model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=seed,
            n_jobs=1, verbosity=0,
        )
        scale = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        model.set_params(scale_pos_weight=scale)
        model.fit(X_tr, y_tr)

        proba = model.predict_proba(X_te)[:, 1]
        auc = float(roc_auc_score(y_te, proba))
        ap  = float(average_precision_score(y_te, proba))
        results.append({"held_out": held_out, "auc": auc, "ap": ap,
                         "n_train": len(y_train), "n_test": len(y_test)})

    return results


def run_piff(
    data_path: str = "data/bug_dataset_v2.jsonl",
    cv_threshold: float = 1.0,
    top_k_stable: int = 15,
    seed: int = 42,
) -> dict:
    """
    Full PIFF evaluation: compare LOPO AUC with all 30 features vs stable-only subset.
    """
    if not _DEPS_OK:
        return {"error": "xgboost/sklearn not available"}

    logger.info("Loading bug dataset...")
    records = _load_bug_records(data_path)
    logger.info("  %d records, computing feature stability...", len(records))

    stability = compute_feature_stability(records, ALL_FEATURE_NAMES)
    sorted_stability = sorted(stability.items(), key=lambda x: x[1])

    # Select stable feature indices
    stable_names = select_stable_features(stability, top_k=top_k_stable)
    stable_indices = [ALL_FEATURE_NAMES.index(n) for n in stable_names]
    all_indices    = list(range(len(ALL_FEATURE_NAMES)))

    logger.info("  Stable features (%d): %s", len(stable_indices), stable_names)
    logger.info("  Running LOPO with all %d features...", len(all_indices))
    full_results   = _run_lopo(records, all_indices, seed=seed)

    logger.info("  Running LOPO with %d stable features (PIFF)...", len(stable_indices))
    piff_results   = _run_lopo(records, stable_indices, seed=seed)

    def _summarize(res):
        if not res:
            return {}
        aucs = [r["auc"] for r in res]
        aps  = [r["ap"]  for r in res]
        return {
            "mean_auc": float(np.mean(aucs)),
            "std_auc":  float(np.std(aucs)),
            "mean_ap":  float(np.mean(aps)),
            "std_ap":   float(np.std(aps)),
            "per_project": res,
        }

    report = {
        "method": "PIFF",
        "description": (
            "Project-Invariant Feature Filtering: features with cross-project CV > threshold "
            "are dropped before training to improve LOPO generalization."
        ),
        "n_features_total": len(ALL_FEATURE_NAMES),
        "n_features_stable": len(stable_indices),
        "stable_feature_names": stable_names,
        "cv_threshold": cv_threshold,
        "feature_stability": dict(sorted_stability),
        "full_features": _summarize(full_results),
        "piff_features": _summarize(piff_results),
        "delta_auc": (
            float(np.mean([r["auc"] for r in piff_results])) -
            float(np.mean([r["auc"] for r in full_results]))
            if full_results and piff_results else None
        ),
    }

    # Identify most/least stable features
    most_stable  = sorted_stability[:5]
    least_stable = sorted_stability[-5:]
    report["most_stable_features"]  = most_stable
    report["least_stable_features"] = least_stable

    full_auc  = report["full_features"].get("mean_auc", 0)
    piff_auc  = report["piff_features"].get("mean_auc", 0)
    delta     = report["delta_auc"] or 0

    logger.info("  Full features  LOPO AUC: %.4f", full_auc)
    logger.info("  PIFF features  LOPO AUC: %.4f  (delta=%+.4f)", piff_auc, delta)

    return report


if __name__ == "__main__":
    import io as _io
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    report = run_piff()
    out = Path("evaluation/results/piff_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {out}")
    print(f"Full AUC:  {report['full_features']['mean_auc']:.4f} +/- {report['full_features']['std_auc']:.4f}")
    print(f"PIFF AUC:  {report['piff_features']['mean_auc']:.4f} +/- {report['piff_features']['std_auc']:.4f}")
    print(f"Delta:     {report['delta_auc']:+.4f}")
