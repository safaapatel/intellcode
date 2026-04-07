"""
Abstention Coverage Curve + Hard Negative Mining Impact
=========================================================

Two experiments in one script:

Experiment A — Abstention Coverage vs Accuracy Curve
------------------------------------------------------
Shows that entropy-based abstention trades coverage for reliability
in a controlled, monotonic way.

Output table:
    Coverage    Accuracy    Abstain threshold
    100%        X           0.00 (no abstention)
    80%         X           ...
    60%         X           ...
    40%         X           ...

Expected result: accuracy increases monotonically as coverage decreases,
confirming that the model abstains on its worst predictions first.

Experiment B — Hard Negative Mining Impact
-------------------------------------------
Compares false positive rate and Precision@10 on a held-out clean set:
    - Baseline:  model trained without hard negatives
    - After HN:  model trained with hard negatives (sample_weight=3.0)

Output table:
    Setting             FP rate     Precision@10    AUC
    No hard negatives   X           X               X
    With hard negatives X           X               X

Even a 5-10% improvement in FP rate closes the loop on the mining claim.

Usage:
    cd backend
    /d/projexts/intellcode/venv/Scripts/python evaluation/abstention_and_hn_impact.py \\
        --dataset      data/security_dataset.jsonl \\
        --hn-dataset   data/security_dataset_hn.jsonl \\
        --output       evaluation/results/abstention_hn.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

_N_METRIC_DIMS = 15
_N_TAINT_DIMS  = 16


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _binary_entropy(p: float) -> float:
    p = max(1e-9, min(1.0 - 1e-9, p))
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _entropy_confidence(p: float) -> float:
    return max(0.0, 1.0 - _binary_entropy(p))


def _load_and_featurise(dataset_path: str):
    from features.code_metrics import compute_all_metrics, metrics_to_feature_vector
    from features.taint_tracker import extract_taint_features

    Xs, ys, repos, weights = [], [], [], []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            label  = int(rec.get("label", rec.get("is_vulnerable", 0)))
            repo   = rec.get("repo", rec.get("project", "unknown"))
            source = rec.get("source", rec.get("code", ""))
            w      = float(rec.get("sample_weight", 1.0))
            if not source:
                continue
            try:
                m = metrics_to_feature_vector(compute_all_metrics(source))
            except Exception:
                m = np.zeros(_N_METRIC_DIMS, dtype=np.float32)
            try:
                t = extract_taint_features(source).to_vector()
            except Exception:
                t = np.zeros(_N_TAINT_DIMS, dtype=np.float32)
            Xs.append(np.concatenate([m, t]))
            ys.append(label)
            repos.append(repo)
            weights.append(w)

    return (np.array(Xs, dtype=np.float32),
            np.array(ys),
            repos,
            np.array(weights))


def _build_rf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=42, n_jobs=1,
    )


# ---------------------------------------------------------------------------
# Experiment A: Abstention coverage curve
# ---------------------------------------------------------------------------

@dataclass
class CoverageRow:
    coverage_pct: float      # percentage of predictions kept (descending confidence order)
    accuracy:     float      # accuracy on kept predictions
    threshold:    float      # confidence threshold used
    n_kept:       int

    def to_dict(self) -> dict:
        return asdict(self)


def run_abstention_curve(dataset_path: str) -> list[CoverageRow]:
    """
    LOPO evaluation: for each project, get probabilities + labels.
    Pool all results, sort by confidence (descending).
    Report accuracy at each coverage decile.
    """
    X, y, repos, _ = _load_and_featurise(dataset_path)
    repos_arr = np.array(repos)
    unique = sorted(set(repos))

    all_probs, all_labels = [], []

    for held_out in unique:
        te = repos_arr == held_out
        tr = ~te
        if te.sum() < 5 or tr.sum() < 10 or len(np.unique(y[te])) < 2:
            continue
        clf = _build_rf()
        clf.fit(X[tr], y[tr])
        raw = clf.predict_proba(X[te])[:, 1]
        all_probs.extend(raw.tolist())
        all_labels.extend(y[te].tolist())

    if not all_probs:
        logger.warning("No LOPO folds produced results.")
        return []

    probs  = np.array(all_probs)
    labels = np.array(all_labels)
    confs  = np.array([_entropy_confidence(float(p)) for p in probs])

    # Sort by confidence descending (most confident first)
    order = np.argsort(confs)[::-1]
    sorted_probs  = probs[order]
    sorted_labels = labels[order]
    sorted_confs  = confs[order]

    rows: list[CoverageRow] = []
    # Evaluate at each 10% coverage step from 100% down to 10%
    for pct in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]:
        n_keep = max(1, int(pct * len(probs)))
        kept_probs  = sorted_probs[:n_keep]
        kept_labels = sorted_labels[:n_keep]
        threshold   = float(sorted_confs[n_keep - 1])

        preds   = (kept_probs >= 0.5).astype(int)
        acc     = float((preds == kept_labels).mean())

        rows.append(CoverageRow(
            coverage_pct=round(pct * 100, 1),
            accuracy=round(acc, 4),
            threshold=round(threshold, 4),
            n_kept=n_keep,
        ))

    return rows


def print_abstention_curve(rows: list[CoverageRow]) -> None:
    print("\n" + "=" * 60)
    print("Abstention Coverage vs Accuracy Curve")
    print("=" * 60)
    print(f"  {'Coverage':>10} {'Accuracy':>10} {'Conf thresh':>12} {'N kept':>8}")
    print(f"  {'-'*48}")
    for r in rows:
        print(f"  {r.coverage_pct:>9.1f}% {r.accuracy:>10.4f} "
              f"{r.threshold:>12.4f} {r.n_kept:>8}")
    print("=" * 60)

    full  = next((r for r in rows if r.coverage_pct == 100.0), None)
    half  = next((r for r in rows if r.coverage_pct == 50.0), None)
    if full and half:
        gain = half.accuracy - full.accuracy
        print(f"\n  Abstention trades 50% coverage for "
              f"+{gain:.4f} accuracy gain.")
        print(f"  Monotonic? "
              + ("YES" if all(rows[i].accuracy <= rows[i+1].accuracy
                              for i in range(len(rows)-1)) else "NO"))
    print()


# ---------------------------------------------------------------------------
# Experiment B: Hard negative mining impact
# ---------------------------------------------------------------------------

@dataclass
class HNImpactRow:
    setting:       str
    fp_rate:       float      # FP rate on clean held-out set
    p_at_10:       float      # Precision@10 on all test items
    auc:           float

    def to_dict(self) -> dict:
        return asdict(self)


def run_hn_impact(
    baseline_path: str,
    hn_path: Optional[str],
) -> list[HNImpactRow]:
    """
    Compare FP rate, P@10, AUC before and after hard negative augmentation.
    LOPO protocol on the clean (label=0) subset of the test set as held-out.
    """
    from sklearn.metrics import roc_auc_score

    def _evaluate(dataset_path: str, setting: str) -> Optional[HNImpactRow]:
        X, y, repos, weights = _load_and_featurise(dataset_path)
        if len(X) == 0:
            return None
        repos_arr = np.array(repos)
        unique = sorted(set(repos))

        fp_rates, p10s, aucs = [], [], []

        for held_out in unique:
            te = repos_arr == held_out
            tr = ~te
            if te.sum() < 5 or tr.sum() < 10 or len(np.unique(y[te])) < 2:
                continue

            clf = _build_rf()
            try:
                clf.fit(X[tr], y[tr], sample_weight=weights[tr])
            except Exception:
                clf.fit(X[tr], y[tr])

            proba = clf.predict_proba(X[te])[:, 1]
            preds = (proba >= 0.5).astype(int)

            # FP rate: on clean (label=0) samples in test set
            clean_mask = y[te] == 0
            if clean_mask.sum() > 0:
                fp_rate = float(preds[clean_mask].mean())
                fp_rates.append(fp_rate)

            # P@10
            k = min(10, len(proba))
            top_k = np.argsort(proba)[::-1][:k]
            p10s.append(float(y[te][top_k].sum() / k))

            # AUC
            try:
                aucs.append(float(roc_auc_score(y[te], proba)))
            except Exception:
                pass

        if not fp_rates:
            return None
        return HNImpactRow(
            setting=setting,
            fp_rate=round(float(np.mean(fp_rates)), 4),
            p_at_10=round(float(np.mean(p10s)) if p10s else float("nan"), 4),
            auc=round(float(np.mean(aucs)) if aucs else float("nan"), 4),
        )

    rows: list[HNImpactRow] = []

    logger.info("Evaluating baseline (no hard negatives)...")
    base_row = _evaluate(baseline_path, "No hard negatives")
    if base_row:
        rows.append(base_row)

    if hn_path and Path(hn_path).exists():
        logger.info("Evaluating after hard negative mining...")
        hn_row = _evaluate(hn_path, "With hard negatives")
        if hn_row:
            rows.append(hn_row)
    else:
        logger.warning("HN dataset not found at %s — skipping after-mining comparison.", hn_path)

    return rows


def print_hn_impact_table(rows: list[HNImpactRow]) -> None:
    print("\n" + "=" * 65)
    print("Hard Negative Mining Impact")
    print("=" * 65)
    print(f"  {'Setting':<28} {'FP rate':>9} {'P@10':>9} {'AUC':>9}")
    print(f"  {'-'*55}")
    for r in rows:
        print(f"  {r.setting:<28} {r.fp_rate:>9.4f} {r.p_at_10:>9.4f} {r.auc:>9.4f}")
    print("=" * 65)

    if len(rows) == 2:
        fp_delta  = rows[1].fp_rate - rows[0].fp_rate
        p10_delta = rows[1].p_at_10 - rows[0].p_at_10
        direction = "reduced" if fp_delta < 0 else "increased"
        print(f"\n  FP rate: {direction} by {abs(fp_delta):.4f} "
              f"({100*abs(fp_delta/max(rows[0].fp_rate,1e-9)):.1f}% relative)")
        print(f"  P@10: delta={p10_delta:+.4f}")
        if fp_delta < 0:
            print(f"  Hard negative mining REDUCES false positives on clean code.")
        else:
            print(f"  No improvement from hard negative mining — check mining threshold.")
    print()


# ---------------------------------------------------------------------------
# Combined report + LaTeX
# ---------------------------------------------------------------------------

def save_latex_abstention(rows: list[CoverageRow], path: str) -> None:
    lines = [
        "\\begin{tabular}{rcc}",
        "\\toprule",
        "Coverage (\\%) & Accuracy & Conf. threshold \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(f"{r.coverage_pct:.0f}\\% & {r.accuracy:.4f} & {r.threshold:.4f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    logger.info("LaTeX abstention curve saved -> %s", path)


def save_latex_hn(rows: list[HNImpactRow], path: str) -> None:
    lines = [
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Setting & FP Rate $\\downarrow$ & P@10 $\\uparrow$ & AUC $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(f"{r.setting} & {r.fp_rate:.4f} & {r.p_at_10:.4f} & {r.auc:.4f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    logger.info("LaTeX hard negative impact table saved -> %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Abstention curve + hard negative mining impact evaluation."
    )
    p.add_argument("--dataset",    required=True, help="Security JSONL dataset (baseline).")
    p.add_argument("--hn-dataset", default=None,
                   help="Dataset with hard negatives appended (output of hard_negative_miner.py).")
    p.add_argument("--output",     default="evaluation/results/abstention_hn.json")
    p.add_argument("--latex-dir",  default="evaluation/results/tables")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    logger.info("=== Experiment A: Abstention Coverage Curve ===")
    cov_rows = run_abstention_curve(args.dataset)
    print_abstention_curve(cov_rows)
    save_latex_abstention(cov_rows, str(Path(args.latex_dir) / "abstention_curve.tex"))

    logger.info("=== Experiment B: Hard Negative Mining Impact ===")
    hn_rows = run_hn_impact(args.dataset, args.hn_dataset)
    print_hn_impact_table(hn_rows)
    save_latex_hn(hn_rows, str(Path(args.latex_dir) / "hn_impact.tex"))

    out = {
        "experiment":       "abstention_and_hn_impact",
        "abstention_curve": [r.to_dict() for r in cov_rows],
        "hn_impact":        [r.to_dict() for r in hn_rows],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Results saved -> %s", args.output)
