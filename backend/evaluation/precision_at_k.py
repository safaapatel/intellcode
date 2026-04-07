"""
Precision@K and Effort-Aware Evaluation Metrics
=================================================
Implements evaluation metrics that reflect how users actually interact with
code analysis tools: they triage findings in order, not by overall AUC.

The right metrics for code quality tools:
    - Security:   Precision@K  -- "if I look at the top K findings, how many are real?"
    - Bug pred:   PofB20        -- "how many bugs are in the top-20% of ranked files?"
    - All tasks:  Effort@N      -- "at what inspection effort do I find N% of issues?"

These are already partially computed in conformal_coverage.py. This module
provides a standalone, reusable implementation suitable for any ranked-output
model comparison.

AUC comparison (for reference):
    - AUC=0.494 (security LOPO) is near chance -- but AUC hides triage performance.
    - A model with AUC=0.60 that ranks the 3 real CVEs first out of 1000 files
      has Precision@3=1.00, which is what matters for security analysts.

References:
    Yang et al. 2016 -- "Effort-Aware Just-In-Time Defect Prediction"
    Kamei et al. 2013 -- "A Large-Scale Empirical Study of JIT Quality Assurance"
    Manning et al. 2008 -- "Introduction to Information Retrieval" (Precision@K)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Core ranking metrics
# ---------------------------------------------------------------------------

def precision_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int,
) -> float:
    """
    Precision@K: fraction of the top-K ranked items that are positive.

    P@K = |{relevant in top-K}| / K

    Args:
        y_true:  (N,) binary ground truth labels {0, 1}.
        y_score: (N,) predicted scores / probabilities (higher = more positive).
        k:       Number of top items to consider.

    Returns:
        Precision@K in [0, 1].
    """
    if k <= 0 or len(y_true) == 0:
        return 0.0
    k = min(k, len(y_true))
    order = np.argsort(y_score)[::-1]
    top_k_labels = y_true[order[:k]]
    return float(top_k_labels.sum() / k)


def recall_at_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k: int,
) -> float:
    """
    Recall@K: fraction of all positives that appear in the top-K.

    R@K = |{relevant in top-K}| / |{all relevant}|
    """
    n_pos = int(y_true.sum())
    if n_pos == 0 or k <= 0:
        return 0.0
    k = min(k, len(y_true))
    order = np.argsort(y_score)[::-1]
    top_k_labels = y_true[order[:k]]
    return float(top_k_labels.sum() / n_pos)


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Average Precision (area under Precision-Recall curve).

    AP = sum_k [P@k * (R@k - R@(k-1))]
    """
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]
    precisions = []
    n_found = 0
    for i, label in enumerate(y_sorted):
        if label == 1:
            n_found += 1
            precisions.append(n_found / (i + 1))
    return float(np.mean(precisions)) if precisions else 0.0


def pofb_at_effort(
    y_true: np.ndarray,
    y_score: np.ndarray,
    effort: float = 0.20,
) -> float:
    """
    Proportion of Bugs Found at given inspection Effort (PofB@effort).

    Ranks files by predicted bug probability. Returns the fraction of all
    bugs that are found when inspecting the top (effort * 100)% of files.

    PofB@20 >= 0.60 is the deployment threshold for IntelliCode bug prediction.

    Args:
        y_true:  (N,) binary labels (1=buggy).
        y_score: (N,) predicted bug probabilities.
        effort:  Fraction of files to inspect (default: 0.20 = top 20%).

    Returns:
        Fraction of bugs found in [0, 1].
    """
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 0.0
    n_inspect = max(1, int(len(y_true) * effort))
    order = np.argsort(y_score)[::-1]
    found = int(y_true[order[:n_inspect]].sum())
    return float(found / n_pos)


def effort_at_recall(
    y_true: np.ndarray,
    y_score: np.ndarray,
    recall_target: float = 0.80,
) -> float:
    """
    Effort required to achieve a target recall level.

    Returns the fraction of the ranked list that must be inspected to find
    (recall_target * 100)% of all bugs.

    Effort@80 = 0.30 means: inspecting 30% of files finds 80% of bugs.

    Args:
        y_true:         (N,) binary labels.
        y_score:        (N,) predicted probabilities.
        recall_target:  Target recall level (default: 0.80).

    Returns:
        Effort fraction in [0, 1], or 1.0 if target is never reached.
    """
    n_total = len(y_true)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return 1.0

    order = np.argsort(y_score)[::-1]
    y_sorted = y_true[order]
    target_found = math.ceil(recall_target * n_pos)
    found = 0
    for i, label in enumerate(y_sorted):
        found += int(label)
        if found >= target_found:
            return float((i + 1) / n_total)
    return 1.0


def optimal_effort_curve(y_true: np.ndarray) -> np.ndarray:
    """
    Compute the optimal (oracle) effort curve for PofB20 comparison.

    The optimal model always ranks all bugs before all clean files.
    Returns the fraction of bugs found at each inspection effort level
    for the optimal ranker.
    """
    n = len(y_true)
    n_pos = int(y_true.sum())
    efforts = np.linspace(0, 1, 101)
    pofbs = []
    for eff in efforts:
        n_inspect = max(1, int(n * eff))
        found = min(n_inspect, n_pos)
        pofbs.append(found / max(1, n_pos))
    return np.array(pofbs)


# ---------------------------------------------------------------------------
# Multi-K evaluation
# ---------------------------------------------------------------------------

def evaluate_at_multiple_k(
    y_true: np.ndarray,
    y_score: np.ndarray,
    ks: Optional[list[int]] = None,
) -> dict:
    """
    Compute Precision@K and Recall@K for multiple values of K.

    Args:
        y_true:  (N,) binary labels.
        y_score: (N,) predicted scores.
        ks:      List of K values. Defaults to [1, 3, 5, 10, 20, 50].

    Returns:
        Dict with keys like "p@1", "r@1", "p@3", ..., "ap", "pofb20".
    """
    if ks is None:
        ks = [1, 3, 5, 10, 20, 50]

    results = {}
    for k in ks:
        results[f"p@{k}"] = round(precision_at_k(y_true, y_score, k), 4)
        results[f"r@{k}"] = round(recall_at_k(y_true, y_score, k), 4)

    results["ap"]      = round(average_precision(y_true, y_score), 4)
    results["pofb20"]  = round(pofb_at_effort(y_true, y_score, 0.20), 4)
    results["pofb50"]  = round(pofb_at_effort(y_true, y_score, 0.50), 4)
    results["effort80"] = round(effort_at_recall(y_true, y_score, 0.80), 4)
    results["n_positive"] = int(y_true.sum())
    results["n_total"]    = len(y_true)

    return results


# ---------------------------------------------------------------------------
# Model comparison table
# ---------------------------------------------------------------------------

def compare_models(
    y_true: np.ndarray,
    model_scores: dict[str, np.ndarray],
    ks: Optional[list[int]] = None,
) -> dict[str, dict]:
    """
    Compare multiple models on Precision@K and effort-aware metrics.

    Args:
        y_true:        (N,) ground truth binary labels.
        model_scores:  {model_name: (N,) score array} for each model.
        ks:            K values for Precision@K.

    Returns:
        {model_name: metrics_dict} for all models.
    """
    results = {}
    for name, scores in model_scores.items():
        results[name] = evaluate_at_multiple_k(y_true, scores, ks)
    return results


def print_comparison_table(comparison: dict[str, dict]) -> str:
    """
    Format model comparison as a readable ASCII table.

    Returns the table as a string (also prints it).
    """
    if not comparison:
        return ""

    # Collect all metric keys
    sample = next(iter(comparison.values()))
    metric_keys = [k for k in sample.keys()
                   if k not in ("n_positive", "n_total")]

    # Header
    col_width = 12
    header = f"{'Model':<25}" + "".join(f"{m:>{col_width}}" for m in metric_keys)
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for model_name, metrics in comparison.items():
        row = f"{model_name:<25}" + "".join(
            f"{metrics.get(m, 0):>{col_width}.4f}" for m in metric_keys
        )
        lines.append(row)

    lines.append(sep)
    table = "\n".join(lines)
    print(table)
    return table


def latex_comparison_table(
    comparison: dict[str, dict],
    caption: str = "Model Comparison",
    label: str = "tab:comparison",
    ks: Optional[list[int]] = None,
) -> str:
    """
    Generate a LaTeX table from model comparison results.

    Returns:
        LaTeX table string for inclusion in a paper.
    """
    if ks is None:
        ks = [5, 10]
    metric_keys = [f"p@{k}" for k in ks] + ["ap", "pofb20", "effort80"]

    col_spec = "l" + "r" * len(metric_keys)
    header_row = " & ".join(
        ["Model"] + [m.replace("@", r"\texttt{@}").replace("_", r"\_") for m in metric_keys]
    )

    rows = []
    for model_name, metrics in comparison.items():
        vals = [f"{metrics.get(m, 0):.3f}" for m in metric_keys]
        rows.append(f"    {model_name} & " + " & ".join(vals) + r" \\")

    latex = (
        r"\begin{table}[h]" + "\n"
        r"\centering" + "\n"
        r"\caption{" + caption + r"}" + "\n"
        r"\label{" + label + r"}" + "\n"
        r"\begin{tabular}{" + col_spec + r"}" + "\n"
        r"\toprule" + "\n"
        f"    {header_row} " + r"\\" + "\n"
        r"\midrule" + "\n"
        + "\n".join(rows) + "\n"
        r"\bottomrule" + "\n"
        r"\end{tabular}" + "\n"
        r"\end{table}"
    )
    return latex


# ---------------------------------------------------------------------------
# Deployment threshold checking
# ---------------------------------------------------------------------------

def check_deployment_thresholds(metrics: dict, task: str) -> tuple[bool, list[str]]:
    """
    Check whether a model meets the deployment thresholds for its task.

    Thresholds (from research roadmap):
        Security:   P@10 >= 0.50, AP >= 0.40
        Bug:        PofB20 >= 0.60, Effort@80 <= 0.50
        Complexity: Coverage >= 0.90 (conformal)

    Args:
        metrics: Output of evaluate_at_multiple_k().
        task:    "security" | "bug" | "complexity"

    Returns:
        (passes: bool, reasons: list[str])
    """
    reasons = []
    passes = True

    if task == "security":
        if metrics.get("p@10", 0) < 0.50:
            passes = False
            reasons.append(f"P@10={metrics.get('p@10', 0):.3f} < 0.50 (security threshold)")
        if metrics.get("ap", 0) < 0.40:
            passes = False
            reasons.append(f"AP={metrics.get('ap', 0):.3f} < 0.40 (security threshold)")

    elif task == "bug":
        if metrics.get("pofb20", 0) < 0.60:
            passes = False
            reasons.append(f"PofB20={metrics.get('pofb20', 0):.3f} < 0.60 (bug threshold)")
        if metrics.get("effort80", 1.0) > 0.50:
            passes = False
            reasons.append(
                f"Effort@80={metrics.get('effort80', 1.0):.3f} > 0.50 (bug threshold)"
            )

    if not reasons:
        reasons.append("All thresholds met.")

    return passes, reasons
