"""
Actionability Analysis: False Positive Rate and Precision@k
=============================================================
Addresses the review: "No evaluation of false positives, developer trust,
fix adoption rate."

A tool is only actionable if its false positive rate is low enough that
developers can trust its warnings.  We evaluate:

  1. FPR at operating thresholds (what % of clean code gets flagged?)
  2. Precision@k (of the top-k highest-risk files, how many are truly buggy?)
  3. False Discovery Rate (FDR) = FP / (FP + TP)
  4. Workload reduction: what % of codebase must be reviewed to catch X% of bugs?
  5. Lift@k: how much better than random is the tool?
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
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        precision_recall_curve,
    )
    from xgboost import XGBClassifier
    _DEPS_OK = True
except ImportError:
    _DEPS_OK = False

RESULTS_DIR = Path("evaluation/results")

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


def _load_records(path: str) -> list[dict]:
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line.strip()))
    return recs


def _records_to_xy(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for rec in records:
        sf = rec["static_features"]
        gf = rec.get("git_features", {})
        git_vec = [float(gf.get(k, 0) or 0) for k in JIT_NAMES]
        X.append([float(v) for v in sf] + git_vec)
        y.append(int(rec["label"]))
    X_arr = np.nan_to_num(np.array(X, dtype=np.float32), nan=0, posinf=1e6, neginf=0)
    return X_arr, np.array(y)


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Precision in the top-k ranked samples."""
    top_k = np.argsort(scores)[::-1][:k]
    return float(y_true[top_k].mean())


def lift_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Lift = precision@k / overall positive rate."""
    base_rate = float(y_true.mean())
    if base_rate < 1e-9:
        return 1.0
    return precision_at_k(y_true, scores, k) / base_rate


def pofb_at_pct(
    y_true: np.ndarray,
    scores: np.ndarray,
    pct: float = 0.20,
) -> float:
    """
    Proportion of bugs found when inspecting the top-pct fraction of code
    (ranked by predicted risk). Effort-aware metric from Kamei et al.
    """
    k = max(1, int(len(y_true) * pct))
    top_k = np.argsort(scores)[::-1][:k]
    total_bugs = y_true.sum()
    if total_bugs == 0:
        return 0.0
    return float(y_true[top_k].sum() / total_bugs)


def fdr_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """
    False Discovery Rate = FP / (FP + TP) at a given operating threshold.
    Also returns FPR, recall, and the alert rate (how much of codebase is flagged).
    """
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())

    fdr      = fp / (fp + tp + 1e-9)
    fpr      = fp / (fp + tn + 1e-9)
    recall   = tp / (tp + fn + 1e-9)
    alert_rate = (tp + fp) / len(y_true)

    return {
        "threshold": threshold,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "FDR": round(fdr, 4),
        "FPR": round(fpr, 4),
        "recall": round(recall, 4),
        "alert_rate": round(alert_rate, 4),
        "alerts_per_true_bug": round((tp + fp) / max(tp, 1), 2),
    }


def run_actionability_analysis(
    data_path: str = "data/bug_dataset_v2.jsonl",
    seed: int = 42,
) -> dict:
    if not _DEPS_OK:
        return {"error": "sklearn/xgboost not available"}

    logger.info("Actionability: loading data...")
    records = _load_records(data_path)
    repos = sorted({r["repo"].split("/")[-1] for r in records})

    all_results = {}

    for held_out in repos:
        train_recs = [r for r in records if r["repo"].split("/")[-1] != held_out]
        test_recs  = [r for r in records if r["repo"].split("/")[-1] == held_out]
        if len(test_recs) < 20:
            continue

        X_tr, y_tr = _records_to_xy(train_recs)
        X_te, y_te = _records_to_xy(test_recs)
        if len(set(y_te)) < 2:
            continue

        model = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            scale_pos_weight=(y_tr == 0).sum() / max((y_tr == 1).sum(), 1),
            eval_metric="logloss", random_state=seed,
            n_jobs=1, verbosity=0,
        )
        model.fit(X_tr, y_tr)
        scores = model.predict_proba(X_te)[:, 1]

        auc = float(roc_auc_score(y_te, scores))
        ap  = float(average_precision_score(y_te, scores))

        # Precision@k and Lift@k
        ks = [5, 10, 20, int(len(y_te) * 0.10), int(len(y_te) * 0.20)]
        ks = sorted(set(max(1, k) for k in ks))

        precision_k = {f"P@{k}": round(precision_at_k(y_te, scores, k), 4) for k in ks}
        lift_k      = {f"L@{k}": round(lift_at_k(y_te, scores, k), 4) for k in ks}

        # PofB@20% (inspect top 20% of code, how many bugs found)
        pofb20 = pofb_at_pct(y_te, scores, pct=0.20)
        pofb10 = pofb_at_pct(y_te, scores, pct=0.10)

        # FDR at multiple thresholds
        fdr_analysis = {}
        for t in [0.30, 0.40, 0.50, 0.60]:
            fdr_analysis[f"t={t}"] = fdr_at_threshold(y_te, scores, threshold=t)

        # Find threshold for FDR <= 0.50 (at most 50% of alerts are false alarms)
        # and threshold for recall >= 0.60
        prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_te, scores)
        fdr_arr = 1.0 - prec_arr

        acceptable_thresh = None
        for pr, re, th in zip(prec_arr, rec_arr, thresh_arr):
            if (1 - pr) <= 0.50 and re >= 0.40:
                acceptable_thresh = {"threshold": round(float(th), 3),
                                     "precision": round(float(pr), 4),
                                     "recall": round(float(re), 4),
                                     "fdr": round(float(1 - pr), 4)}
                break

        all_results[held_out] = {
            "n_test": int(len(y_te)),
            "positive_rate": round(float(y_te.mean()), 4),
            "auc": round(auc, 4),
            "ap":  round(ap, 4),
            "pofb20": round(pofb20, 4),
            "pofb10": round(pofb10, 4),
            "precision_at_k": precision_k,
            "lift_at_k": lift_k,
            "fdr_at_thresholds": fdr_analysis,
            "acceptable_operating_point": acceptable_thresh,
        }
        logger.info(
            "  %s: AUC=%.3f PofB@20=%.3f P@10=%.3f FDR@0.50=%.3f",
            held_out, auc, pofb20,
            precision_at_k(y_te, scores, 10),
            fdr_at_threshold(y_te, scores, 0.50)["FDR"],
        )

    # ── Summary across projects ───────────────────────────────────────────────
    def _mean(key):
        vals = [all_results[r][key] for r in all_results if key in all_results[r]]
        return float(np.mean(vals)) if vals else None

    mean_pofb20  = _mean("pofb20")
    mean_fdr_050 = float(np.mean([
        all_results[r]["fdr_at_thresholds"].get("t=0.5", {}).get("FDR", 1.0)
        for r in all_results
    ]))
    mean_auc = _mean("auc")

    # Workload reduction: to catch 60% of bugs, inspect what fraction of code?
    workload_reduction = {}
    for held_out, res in all_results.items():
        workload_reduction[held_out] = {
            "inspect_20pct_catches": f"{res['pofb20']*100:.1f}% of bugs",
            "inspect_10pct_catches": f"{res['pofb10']*100:.1f}% of bugs",
            "fdr_at_default_0.5": res["fdr_at_thresholds"].get("t=0.5", {}).get("FDR"),
        }

    report = {
        "method": "Actionability_Analysis",
        "description": (
            "Evaluates whether the bug predictor is actionable in practice: "
            "precision@k, false discovery rate, PofB@20% (effort-aware), "
            "lift, and per-project operating point analysis."
        ),
        "per_project": all_results,
        "summary": {
            "mean_auc": mean_auc,
            "mean_pofb20": mean_pofb20,
            "mean_fdr_at_threshold_0.50": mean_fdr_050,
        },
        "workload_reduction": workload_reduction,
        "finding": (
            f"Mean PofB@20%={mean_pofb20:.3f}: inspecting only 20% of commits "
            f"(highest-risk) catches {mean_pofb20*100:.1f}% of bugs on average. "
            f"Mean FDR at threshold=0.50: {mean_fdr_050:.3f} "
            f"({mean_fdr_050*100:.1f}% of flagged files are false alarms). "
            + (
                "The tool provides meaningful workload reduction even under LOPO."
                if mean_pofb20 > 0.50 else
                "Limited workload reduction — cross-project generalisation is poor."
            )
        ),
    }

    return report


if __name__ == "__main__":
    import io as _io
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    report = run_actionability_analysis()
    out = Path("evaluation/results/actionability_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {out}")
    s = report["summary"]
    print(f"Mean AUC:     {s['mean_auc']:.4f}")
    print(f"Mean PofB@20: {s['mean_pofb20']:.4f}")
    print(f"Mean FDR@0.5: {s['mean_fdr_at_threshold_0.50']:.4f}")
    print(f"\nFinding: {report['finding']}")
