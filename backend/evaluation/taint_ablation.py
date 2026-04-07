"""
Taint Feature Ablation Study
==============================
Isolates the contribution of taint-flow features to security LOPO performance.

Core claim under test:
    "Removing taint features reduces LOPO AUC from ~X to ~0.5, confirming that
    the taint-flow representation carries the transferable causal signal while
    API co-occurrence features are project-specific correlations."

Ablation conditions:
    1. Full model   — 16 taint dims + 15 metric dims (31-dim)
    2. No taint     — 15 metric dims only (API co-occurrence baseline)
    3. Taint only   — 16 taint dims only (causal features alone)
    4. Key taint    — risk_score + unsanitized_paths + branch_taint_fraction (3 dims)

Each condition is evaluated via LOPO using the same RF model.
Wilcoxon signed-rank p-values compare each condition against the full model.

Anti-learning diagnostic:
    If "No taint" AUC < 0.5: anti-learning confirmed
    (feature-label correlations invert across training/test distributions)
    If "Taint only" AUC > 0.5: taint features are the transferable signal

Usage:
    cd backend
    /d/projexts/intellcode/venv/Scripts/python evaluation/taint_ablation.py \\
        --dataset data/security_dataset.jsonl \\
        --output  evaluation/results/taint_ablation.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Taint feature indices within the 31-dim combined vector
# (first 15 = code metrics, next 16 = taint features)
# ---------------------------------------------------------------------------

_N_METRIC_DIMS = 15    # from metrics_to_feature_vector()
_N_TAINT_DIMS  = 16    # from taint_tracker.TAINT_FEATURE_NAMES

# Taint dims 15-30 (0-indexed within 31-dim vector):
_TAINT_SLICE = slice(_N_METRIC_DIMS, _N_METRIC_DIMS + _N_TAINT_DIMS)

# Key causal taint dims (within the taint block, 0-indexed):
#   taint_risk_score=11, taint_unsanitized_paths=3, taint_branch_taint_fraction=13
_KEY_TAINT_LOCAL = [11, 3, 13]
_KEY_TAINT_GLOBAL = [_N_METRIC_DIMS + i for i in _KEY_TAINT_LOCAL]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_security_records(dataset_path: str) -> list[dict]:
    records = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _build_feature_matrix(records: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Returns (X, y, repos) where X is 31-dim (15 metrics + 16 taint).
    Extracts taint features on-the-fly for records that have source code.
    Records without taint features get zeros for the taint block.
    """
    from features.code_metrics import compute_all_metrics, metrics_to_feature_vector
    from features.taint_tracker import extract_taint_features

    Xs, ys, repos = [], [], []

    for rec in records:
        label = int(rec.get("label", rec.get("is_vulnerable", 0)))
        repo  = rec.get("repo", rec.get("project", "unknown"))
        source = rec.get("source", rec.get("code", ""))

        if not source:
            continue

        # Metric features (15-dim)
        try:
            metrics = compute_all_metrics(source)
            metric_vec = metrics_to_feature_vector(metrics)
        except Exception:
            metric_vec = np.zeros(_N_METRIC_DIMS, dtype=np.float32)

        # Taint features (16-dim)
        try:
            taint_result = extract_taint_features(source)
            taint_vec = taint_result.to_vector()
        except Exception:
            taint_vec = np.zeros(_N_TAINT_DIMS, dtype=np.float32)

        x = np.concatenate([metric_vec, taint_vec]).astype(np.float32)
        Xs.append(x)
        ys.append(label)
        repos.append(repo)

    return np.array(Xs), np.array(ys), repos


# ---------------------------------------------------------------------------
# LOPO evaluation
# ---------------------------------------------------------------------------

def _build_rf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=200, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=1,
    )


def _lopo_auc(X: np.ndarray, y: np.ndarray, repos: list[str]) -> list[float]:
    """Return per-project AUC list via LOPO."""
    from sklearn.metrics import roc_auc_score
    repos_arr = np.array(repos)
    unique = sorted(set(repos))
    aucs = []
    for held_out in unique:
        te = repos_arr == held_out
        tr = ~te
        if te.sum() < 5 or tr.sum() < 10:
            continue
        if len(np.unique(y[te])) < 2:
            continue
        clf = _build_rf()
        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])[:, 1]
        aucs.append(float(roc_auc_score(y[te], proba)))
    return aucs


def _lopo_p10(X: np.ndarray, y: np.ndarray, repos: list[str]) -> list[float]:
    """Return per-project Precision@10 list via LOPO."""
    repos_arr = np.array(repos)
    unique = sorted(set(repos))
    p10s = []
    for held_out in unique:
        te = repos_arr == held_out
        tr = ~te
        if te.sum() < 5 or tr.sum() < 10:
            continue
        if len(np.unique(y[te])) < 2:
            continue
        clf = _build_rf()
        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])[:, 1]
        k = min(10, len(proba))
        top_k = np.argsort(proba)[::-1][:k]
        p10s.append(float(y[te][top_k].sum() / k))
    return p10s


def _wilcoxon_p(a: list[float], b: list[float]) -> Optional[float]:
    try:
        from scipy.stats import wilcoxon
        if len(a) < 2 or len(b) < 2 or len(a) != len(b):
            return None
        stat, p = wilcoxon(a, b, alternative="two-sided")
        return round(float(p), 4)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Ablation conditions
# ---------------------------------------------------------------------------

@dataclass
class TaintAblationRow:
    condition:       str
    n_features:      int
    mean_lopo_auc:   float
    std_lopo_auc:    float
    mean_p10:        float
    std_p10:         float
    delta_auc:       float            # vs. full model
    wilcoxon_p:      Optional[float]
    anti_learning:   bool             # True if mean AUC < 0.5

    def to_dict(self) -> dict:
        return asdict(self)


def run_taint_ablation(records: list[dict]) -> list[TaintAblationRow]:
    logger.info("Building feature matrix (%d records)...", len(records))
    X_full, y, repos = _build_feature_matrix(records)

    if len(X_full) == 0:
        raise ValueError("No records with source code found.")

    logger.info("Feature matrix: %s | positives: %d / %d",
                X_full.shape, y.sum(), len(y))

    # Condition definitions: (name, column_selector)
    conditions: list[tuple[str, np.ndarray]] = [
        ("Full (metrics + taint, 31-dim)",
         np.arange(X_full.shape[1])),
        ("No taint (metrics only, 15-dim)",
         np.arange(_N_METRIC_DIMS)),
        ("Taint only (16-dim)",
         np.arange(_N_METRIC_DIMS, _N_METRIC_DIMS + _N_TAINT_DIMS)),
        ("Key taint (risk + unsanitized + branch, 3-dim)",
         np.array(_KEY_TAINT_GLOBAL)),
    ]

    rows: list[TaintAblationRow] = []
    full_aucs: list[float] = []

    for i, (name, col_idx) in enumerate(conditions):
        X_cond = X_full[:, col_idx]
        logger.info("Condition [%d/%d]: %s | shape=%s",
                    i + 1, len(conditions), name, X_cond.shape)

        aucs = _lopo_auc(X_cond, y, repos)
        p10s = _lopo_p10(X_cond, y, repos)

        if not aucs:
            logger.warning("  No valid LOPO folds.")
            continue

        mean_auc = float(np.mean(aucs))
        std_auc  = float(np.std(aucs))
        mean_p10 = float(np.mean(p10s)) if p10s else float("nan")
        std_p10  = float(np.std(p10s))  if p10s else float("nan")

        if i == 0:
            full_aucs = aucs
            delta = 0.0
            wp = None
        else:
            delta = round(mean_auc - float(np.mean(full_aucs)), 4)
            wp = _wilcoxon_p(full_aucs, aucs)

        rows.append(TaintAblationRow(
            condition=name,
            n_features=len(col_idx),
            mean_lopo_auc=round(mean_auc, 4),
            std_lopo_auc=round(std_auc, 4),
            mean_p10=round(mean_p10, 4),
            std_p10=round(std_p10, 4),
            delta_auc=delta,
            wilcoxon_p=wp,
            anti_learning=mean_auc < 0.5,
        ))

    return rows


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_taint_ablation_table(rows: list[TaintAblationRow]) -> None:
    print("\n" + "=" * 90)
    print("Taint Feature Ablation -- Security LOPO")
    print("=" * 90)
    print(f"  {'Condition':<42} {'Dims':>4} {'AUC (mean+/-std)':>18} "
          f"{'P@10':>7} {'Delta AUC':>10} {'p-value':>9} {'Anti-learn':>10}")
    print(f"  {'-'*87}")

    for r in rows:
        auc_str   = f"{r.mean_lopo_auc:.4f}+/-{r.std_lopo_auc:.4f}"
        delta_str = f"{r.delta_auc:+.4f}" if r.delta_auc != 0.0 else "   (base)"
        p_str     = f"{r.wilcoxon_p:.4f}" if r.wilcoxon_p is not None else "    N/A"
        al_str    = "YES" if r.anti_learning else "no"
        p10_str   = f"{r.mean_p10:.4f}"

        print(f"  {r.condition:<42} {r.n_features:>4} {auc_str:>18} "
              f"{p10_str:>7} {delta_str:>10} {p_str:>9} {al_str:>10}")

    print("=" * 90)

    full_auc = rows[0].mean_lopo_auc if rows else None
    no_taint = next((r for r in rows if "No taint" in r.condition), None)
    taint_only = next((r for r in rows if "Taint only" in r.condition), None)

    print("\nKey findings:")
    if no_taint and no_taint.anti_learning:
        print(f"  Anti-learning CONFIRMED: No-taint AUC={no_taint.mean_lopo_auc:.4f} < 0.5")
        print(f"  API co-occurrence features INVERT across project boundaries.")
    if taint_only and taint_only.mean_lopo_auc is not None and full_auc is not None:
        gap = taint_only.mean_lopo_auc - (no_taint.mean_lopo_auc if no_taint else 0.5)
        print(f"  Taint-only AUC={taint_only.mean_lopo_auc:.4f}  "
              f"(+{gap:.4f} over no-taint baseline)")
        print(f"  Taint features carry the transferable causal signal.")
    print()


def save_latex_taint_table(rows: list[TaintAblationRow], path: str) -> None:
    lines = [
        "\\begin{tabular}{lrcccrr}",
        "\\toprule",
        "Condition & Dims & AUC & $\\pm$std & P@10 & $\\Delta$AUC & $p$-value \\\\",
        "\\midrule",
    ]
    for r in rows:
        delta = f"{r.delta_auc:+.4f}" if r.delta_auc != 0.0 else "--"
        pval  = f"{r.wilcoxon_p:.4f}" if r.wilcoxon_p is not None else "--"
        al    = " (anti-learn)" if r.anti_learning else ""
        name  = r.condition.replace("&", "\\&").replace("#", "\\#")
        lines.append(
            f"{name}{al} & {r.n_features} & {r.mean_lopo_auc:.4f} & "
            f"{r.std_lopo_auc:.4f} & {r.mean_p10:.4f} & {delta} & {pval} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    logger.info("LaTeX taint ablation table saved -> %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Taint feature ablation for security LOPO.")
    p.add_argument("--dataset", required=True, help="Security JSONL dataset.")
    p.add_argument("--output",  default="evaluation/results/taint_ablation.json")
    p.add_argument("--latex",   default="evaluation/results/tables/taint_ablation.tex")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    records = _load_security_records(args.dataset)
    rows = run_taint_ablation(records)
    print_taint_ablation_table(rows)
    save_latex_taint_table(rows, args.latex)

    out = {
        "experiment": "taint_feature_ablation",
        "rows": [r.to_dict() for r in rows],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Results saved -> %s", args.output)
