"""
Shift-Aware Evaluation Framework (SAEF)
========================================
Novel contribution #2.

Core insight: Instead of just *reporting* that LOPO AUC is low, we *quantify
why* — measuring the feature-distribution shift between projects and showing
that performance degrades predictably as shift increases.

This turns "our model fails" into "our model fails *proportionally* to
measurable distribution shift" — a publishable empirical finding.

Metrics
-------
- Jensen-Shannon divergence (symmetric KL) per feature
- Wasserstein-1 distance per feature
- Aggregate shift score = mean(JSD across features)
- Pearson correlation: shift_score vs LOPO AUC drop

Reference
---------
Briand et al. (2002): "Investigating the application of design metrics for fault prediction"
Turhan et al. (2009): "On the relative value of cross-company and within-company data for defect prediction"
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
    from scipy.stats import wasserstein_distance, pearsonr, spearmanr
    from scipy.special import rel_entr
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False


# ── helpers ──────────────────────────────────────────────────────────────────

def _js_divergence(p: np.ndarray, q: np.ndarray, n_bins: int = 20) -> float:
    """Jensen-Shannon divergence between two continuous distributions (via histogram)."""
    combined = np.concatenate([p, q])
    lo, hi = combined.min(), combined.max()
    if hi - lo < 1e-9:
        return 0.0
    bins = np.linspace(lo, hi, n_bins + 1)
    ph, _ = np.histogram(p, bins=bins, density=True)
    qh, _ = np.histogram(q, bins=bins, density=True)
    ph = ph + 1e-10
    qh = qh + 1e-10
    ph /= ph.sum()
    qh /= qh.sum()
    m = 0.5 * (ph + qh)
    jsd = 0.5 * np.sum(rel_entr(ph, m)) + 0.5 * np.sum(rel_entr(qh, m))
    return float(np.clip(jsd, 0.0, 1.0))


def _load_bug_records(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


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

ALL_NAMES = STATIC_NAMES + JIT_NAMES


def _record_to_vec(rec: dict) -> list[float]:
    sf = rec["static_features"]
    gf = rec.get("git_features", {})
    git_vec = [float(gf.get(k, 0) or 0) for k in JIT_NAMES]
    return [float(v) for v in sf] + git_vec


# ── main SAEF function ────────────────────────────────────────────────────────

def compute_pairwise_shift(
    records: list[dict],
    feature_names: list[str] = ALL_NAMES,
    use_wasserstein: bool = True,
) -> dict:
    """
    Compute pairwise distribution shift between all project pairs.
    Returns per-pair JSD and Wasserstein scores plus per-feature breakdown.
    """
    # Group by repo
    project_data: dict[str, np.ndarray] = {}
    for rec in records:
        repo = rec.get("repo", "unknown").split("/")[-1]
        vec = _record_to_vec(rec)
        if repo not in project_data:
            project_data[repo] = []
        project_data[repo].append(vec)

    for repo in project_data:
        project_data[repo] = np.array(project_data[repo], dtype=np.float64)

    repos = sorted(project_data.keys())
    n_feats = len(feature_names)
    pairs = []

    for i in range(len(repos)):
        for j in range(i + 1, len(repos)):
            r1, r2 = repos[i], repos[j]
            X1, X2 = project_data[r1], project_data[r2]

            per_feature_jsd = []
            per_feature_w1  = []

            for fi in range(n_feats):
                col1 = X1[:, fi]
                col2 = X2[:, fi]
                # Log-transform heavy-tailed features (Halstead, churn, etc.)
                if feature_names[fi] in {
                    "halstead_volume", "halstead_effort", "halstead_difficulty",
                    "code_churn", "lines_added", "lines_deleted", "lines_touched",
                }:
                    col1 = np.log1p(col1)
                    col2 = np.log1p(col2)

                jsd = _js_divergence(col1, col2)
                per_feature_jsd.append(jsd)

                if use_wasserstein:
                    try:
                        w1 = float(wasserstein_distance(col1, col2))
                        # Normalise by range
                        rng = max(col1.max(), col2.max()) - min(col1.min(), col2.min()) + 1e-9
                        per_feature_w1.append(w1 / rng)
                    except Exception:
                        per_feature_w1.append(0.0)

            agg_jsd = float(np.mean(per_feature_jsd))
            agg_w1  = float(np.mean(per_feature_w1)) if per_feature_w1 else None

            pairs.append({
                "project_a": r1,
                "project_b": r2,
                "aggregate_jsd": agg_jsd,
                "aggregate_w1": agg_w1,
                "per_feature_jsd": {
                    n: float(v) for n, v in zip(feature_names, per_feature_jsd)
                },
                "per_feature_w1": {
                    n: float(v) for n, v in zip(feature_names, per_feature_w1)
                } if per_feature_w1 else {},
            })

    # Overall shift per project (average shift vs all others)
    project_shift: dict[str, float] = {}
    for repo in repos:
        related = [p for p in pairs if repo in (p["project_a"], p["project_b"])]
        if related:
            project_shift[repo] = float(np.mean([p["aggregate_jsd"] for p in related]))

    return {
        "repos": repos,
        "n_features": n_feats,
        "pairwise": pairs,
        "project_avg_shift": project_shift,
    }


def correlate_shift_with_performance(
    shift_data: dict,
    lopo_results: list[dict],
) -> dict:
    """
    Correlate per-project shift score with LOPO AUC.
    Higher shift from training set -> lower AUC expected.
    """
    project_shift = shift_data["project_avg_shift"]
    shifts, aucs = [], []

    for res in lopo_results:
        repo = res["held_out_repo"].split("/")[-1]
        if repo in project_shift and res.get("auc") is not None:
            shifts.append(project_shift[repo])
            aucs.append(res["auc"])

    if len(shifts) < 3:
        return {"pearson_r": None, "spearman_r": None, "n_points": len(shifts),
                "note": "insufficient data for correlation"}

    shifts = np.array(shifts)
    aucs   = np.array(aucs)

    r_p, p_p = pearsonr(shifts, aucs)
    r_s, p_s = spearmanr(shifts, aucs)

    return {
        "n_points": len(shifts),
        "pearson_r":  float(r_p),
        "pearson_p":  float(p_p),
        "spearman_r": float(r_s),
        "spearman_p": float(p_s),
        "shift_auc_pairs": [
            {"repo": res["held_out_repo"].split("/")[-1],
             "shift": project_shift.get(res["held_out_repo"].split("/")[-1]),
             "auc":   res.get("auc")}
            for res in lopo_results
            if res["held_out_repo"].split("/")[-1] in project_shift
        ],
        "interpretation": (
            "Negative Pearson r confirms: higher distribution shift -> lower LOPO AUC. "
            "This quantifies *why* cross-project generalization fails."
        ),
    }


def run_saef(
    data_path:  str = "data/bug_dataset_v2.jsonl",
    lopo_path:  str = "evaluation/results/lopo_bug.json",
) -> dict:
    if not _DEPS_OK:
        return {"error": "scipy not available"}

    logger.info("SAEF: loading data...")
    records = _load_bug_records(data_path)

    logger.info("  Computing pairwise distribution shift...")
    shift_data = compute_pairwise_shift(records)

    # Load LOPO results
    lopo_results = []
    try:
        with open(lopo_path, encoding="utf-8") as f:
            lopo = json.load(f)
        lopo_results = lopo.get("project_results", [])
    except Exception as e:
        logger.warning("Could not load LOPO results: %s", e)

    logger.info("  Correlating shift with LOPO performance...")
    correlation = correlate_shift_with_performance(shift_data, lopo_results)

    # Rank features by how much they contribute to shift
    all_pairs = shift_data["pairwise"]
    if all_pairs:
        feat_shift_contrib: dict[str, list[float]] = {}
        for pair in all_pairs:
            for fname, jsd in pair["per_feature_jsd"].items():
                feat_shift_contrib.setdefault(fname, []).append(jsd)
        feat_avg_jsd = {
            k: float(np.mean(v)) for k, v in feat_shift_contrib.items()
        }
        top_shifting = sorted(feat_avg_jsd.items(), key=lambda x: -x[1])[:8]
        low_shifting = sorted(feat_avg_jsd.items(), key=lambda x: x[1])[:8]
    else:
        top_shifting = []
        low_shifting = []
        feat_avg_jsd = {}

    report = {
        "method": "SAEF",
        "description": (
            "Shift-Aware Evaluation Framework: quantifies feature-distribution "
            "shift between projects using Jensen-Shannon divergence and Wasserstein-1 "
            "distance, then correlates shift magnitude with LOPO performance degradation."
        ),
        "n_records": len(records),
        "n_projects": len(shift_data["repos"]),
        "repos": shift_data["repos"],
        "pairwise_shift": shift_data["pairwise"],
        "project_avg_shift": shift_data["project_avg_shift"],
        "feature_avg_jsd": feat_avg_jsd,
        "top_shifting_features":  top_shifting,
        "low_shifting_features":  low_shifting,
        "shift_performance_correlation": correlation,
    }

    if correlation.get("pearson_r") is not None:
        logger.info(
            "  Pearson r(shift, AUC) = %.3f (p=%.3f) -- %s",
            correlation["pearson_r"], correlation["pearson_p"],
            "negative = shift hurts performance" if correlation["pearson_r"] < 0 else "positive",
        )

    return report


if __name__ == "__main__":
    import io as _io
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    report = run_saef()
    out = Path("evaluation/results/saef_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {out}")
    corr = report["shift_performance_correlation"]
    print(f"Pearson r(shift, AUC): {corr.get('pearson_r')}")
    print(f"Spearman r:            {corr.get('spearman_r')}")
