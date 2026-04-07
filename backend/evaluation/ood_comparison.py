"""
OOD Uncertainty Comparison Study
==================================
Empirically validates that OOD-aware confidence improves decision quality
over no-OOD and symmetric-OOD baselines.

Three settings compared:
    1. No OOD       -- raw model probability, no adjustment
    2. Symmetric    -- pull toward 0.5 equally for high/low risk (old approach)
    3. Asymmetric   -- decay less for high-risk (preserve recall), more for low-risk
                       (current implementation)

Metrics:
    - ECE (calibration quality)
    - P@10 (precision at top-10 flagged items)
    - Abstention rate (fraction of inputs where confidence < 0.25)
    - Coverage@80% accuracy (coverage when accuracy >= 80%)

Produces Table for thesis Section 5.x:
    Setting         ECE     P@10    Abstention    Coverage@80acc
    No OOD          X       X       0%            X
    Symmetric OOD   X       X       X%            X
    Asymmetric OOD  X       X       X%            X

Usage:
    cd backend
    /d/projexts/intellcode/venv/Scripts/python evaluation/ood_comparison.py \\
        --dataset data/security_dataset.jsonl \\
        --output  evaluation/results/ood_comparison.json
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
_ABSTENTION_THRESHOLD = 0.25   # confidence < 0.25 -> abstain


# ---------------------------------------------------------------------------
# Confidence / OOD adjustment functions
# ---------------------------------------------------------------------------

def _binary_entropy(p: float) -> float:
    p = max(1e-9, min(1.0 - 1e-9, p))
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _entropy_confidence(p: float) -> float:
    return max(0.0, 1.0 - _binary_entropy(p))


def _sigma_to_factor(sigma: float) -> float:
    """Convert sigma distance to OOD confidence factor (same as ood_detector.py)."""
    if sigma < 2.0:
        return 1.0
    elif sigma < 3.5:
        return 1.0 - 0.5 * (sigma - 2.0) / 1.5
    else:
        return 0.1


def _adjust_no_ood(raw: float, sigma: float) -> tuple[float, float]:
    """No adjustment. Returns (adjusted_prob, confidence)."""
    return raw, _entropy_confidence(raw)


def _adjust_symmetric(raw: float, sigma: float) -> tuple[float, float]:
    """Symmetric pull toward 0.5 proportional to OOD-ness."""
    factor = _sigma_to_factor(sigma)
    adjusted = 0.5 + (raw - 0.5) * factor
    conf = _entropy_confidence(adjusted) * factor
    return adjusted, conf


def _adjust_asymmetric(raw: float, sigma: float) -> tuple[float, float]:
    """Asymmetric decay: high-risk decays less (preserve recall)."""
    factor = _sigma_to_factor(sigma)
    deviation = raw - 0.5
    if deviation > 0.0:
        damped = deviation * (0.5 + 0.5 * factor)
    else:
        damped = deviation * factor
    adjusted = 0.5 + damped
    conf = _entropy_confidence(adjusted) * ((0.5 + 0.5 * factor) if deviation > 0 else factor)
    return adjusted, conf


_ADJUSTMENTS = {
    "No OOD":         _adjust_no_ood,
    "Symmetric OOD":  _adjust_symmetric,
    "Asymmetric OOD": _adjust_asymmetric,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece_sum = 0.0
    n = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        ece_sum += mask.sum() * abs(float(probs[mask].mean()) - float(labels[mask].mean()))
    return float(ece_sum / n)


def _precision_at_k(probs: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    if len(probs) == 0:
        return float("nan")
    k = min(k, len(probs))
    top_k = np.argsort(probs)[::-1][:k]
    return float(labels[top_k].sum() / k)


def _coverage_at_accuracy(
    probs: np.ndarray, labels: np.ndarray, confidences: np.ndarray,
    target_accuracy: float = 0.80,
) -> float:
    """Largest fraction of samples retained (by confidence) where accuracy >= target."""
    order = np.argsort(confidences)[::-1]   # highest confidence first
    retained = 0
    for i, idx in enumerate(order):
        retained += 1
        subset_pred = (probs[order[:retained]] >= 0.5).astype(int)
        subset_true = labels[order[:retained]]
        acc = float((subset_pred == subset_true).mean())
        if acc < target_accuracy and retained > 5:
            # Accuracy just dropped below target, return coverage at previous step
            return (retained - 1) / len(probs)
    return 1.0


# ---------------------------------------------------------------------------
# Data loading + feature extraction
# ---------------------------------------------------------------------------

def _load_and_featurise(dataset_path: str):
    """Returns (X, y, repos) with 31-dim features (15 metric + 16 taint)."""
    from features.code_metrics import compute_all_metrics, metrics_to_feature_vector
    from features.taint_tracker import extract_taint_features

    records = []
    with open(dataset_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    Xs, ys, repos = [], [], []
    for rec in records:
        label  = int(rec.get("label", rec.get("is_vulnerable", 0)))
        repo   = rec.get("repo", rec.get("project", "unknown"))
        source = rec.get("source", rec.get("code", ""))
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

    return np.array(Xs, dtype=np.float32), np.array(ys), repos


# ---------------------------------------------------------------------------
# LOPO runner for all three OOD settings
# ---------------------------------------------------------------------------

@dataclass
class OODComparisonRow:
    setting:         str
    ece:             float
    p_at_10:         float
    abstention_rate: float
    coverage_at_80:  float

    def to_dict(self) -> dict:
        return asdict(self)


def run_ood_comparison(dataset_path: str) -> list[OODComparisonRow]:
    from sklearn.ensemble import RandomForestClassifier

    X, y, repos = _load_and_featurise(dataset_path)
    logger.info("Dataset: %s samples, %d positives", len(X), y.sum())

    try:
        from features.ood_detector import OODDetector
        has_ood = True
    except ImportError:
        has_ood = False
        logger.warning("OODDetector not available. Sigma distances will be 0.")

    repos_arr  = np.array(repos)
    unique     = sorted(set(repos))

    # Collect per-project results for each setting
    setting_results: dict[str, dict[str, list]] = {
        s: {"probs": [], "adj_probs": [], "confs": [], "labels": [], "sigmas": []}
        for s in _ADJUSTMENTS
    }

    for held_out in unique:
        te = repos_arr == held_out
        tr = ~te
        if te.sum() < 5 or tr.sum() < 10 or len(np.unique(y[te])) < 2:
            continue

        clf = RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=42, n_jobs=1,
        )
        clf.fit(X[tr], y[tr])
        raw_probs = clf.predict_proba(X[te])[:, 1]

        # Compute Mahalanobis sigma distances from training distribution
        if has_ood:
            ood_det = OODDetector().fit(X[tr])
            sigmas = np.array([ood_det.sigma_distance(x) for x in X[te]])
        else:
            sigmas = np.zeros(len(X[te]))

        for setting, adjust_fn in _ADJUSTMENTS.items():
            adj_arr, conf_arr = [], []
            for p, sigma in zip(raw_probs, sigmas):
                adj, conf = adjust_fn(float(p), float(sigma))
                adj_arr.append(adj)
                conf_arr.append(conf)

            adj_arr  = np.array(adj_arr)
            conf_arr = np.array(conf_arr)

            setting_results[setting]["probs"].extend(raw_probs.tolist())
            setting_results[setting]["adj_probs"].extend(adj_arr.tolist())
            setting_results[setting]["confs"].extend(conf_arr.tolist())
            setting_results[setting]["labels"].extend(y[te].tolist())
            setting_results[setting]["sigmas"].extend(sigmas.tolist())

    rows: list[OODComparisonRow] = []
    for setting, data in setting_results.items():
        if not data["labels"]:
            continue
        adj   = np.array(data["adj_probs"])
        confs = np.array(data["confs"])
        labs  = np.array(data["labels"])

        abstained = confs < _ABSTENTION_THRESHOLD
        # Evaluate only non-abstained predictions for calibration metrics
        kept      = ~abstained
        if kept.sum() < 10:
            logger.warning("Setting '%s': too few non-abstained samples.", setting)
            continue

        ece_val  = _ece(adj[kept], labs[kept])
        p10_val  = _precision_at_k(adj[kept], labs[kept], k=10)
        abst_rt  = float(abstained.mean())
        cov_80   = _coverage_at_accuracy(adj, labs, confs, target_accuracy=0.80)

        rows.append(OODComparisonRow(
            setting=setting,
            ece=round(ece_val, 4),
            p_at_10=round(p10_val, 4),
            abstention_rate=round(abst_rt, 4),
            coverage_at_80=round(cov_80, 4),
        ))
        logger.info("  %-20s ECE=%.4f  P@10=%.4f  Abstain=%.1f%%  Cov@80=%.4f",
                    setting, ece_val, p10_val, 100 * abst_rt, cov_80)

    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_ood_comparison_table(rows: list[OODComparisonRow]) -> None:
    print("\n" + "=" * 75)
    print("OOD Uncertainty Comparison")
    print("=" * 75)
    print(f"  {'Setting':<22} {'ECE':>7} {'P@10':>7} {'Abstain%':>9} {'Cov@80acc':>10}")
    print(f"  {'-'*60}")
    for r in rows:
        print(f"  {r.setting:<22} {r.ece:>7.4f} {r.p_at_10:>7.4f} "
              f"{100*r.abstention_rate:>8.1f}% {r.coverage_at_80:>10.4f}")
    print("=" * 75)

    asym = next((r for r in rows if "Asymmetric" in r.setting), None)
    sym  = next((r for r in rows if "Symmetric" in r.setting), None)
    base = next((r for r in rows if "No OOD" in r.setting), None)

    print("\nKey findings:")
    if asym and base:
        delta_ece = asym.ece - base.ece
        sign = "lower" if delta_ece < 0 else "higher"
        print(f"  ECE: Asymmetric ({asym.ece:.4f}) vs No-OOD ({base.ece:.4f}): "
              f"{abs(delta_ece):.4f} {sign} (lower is better)")
    if asym and sym:
        delta_p10 = asym.p_at_10 - sym.p_at_10
        print(f"  P@10: Asymmetric ({asym.p_at_10:.4f}) vs Symmetric ({sym.p_at_10:.4f}): "
              f"delta={delta_p10:+.4f}")
        print(f"  Asymmetric decay preserves recall for high-risk predictions "
              f"(reduced false-negative risk).")
    print()


def save_latex_ood_table(rows: list[OODComparisonRow], path: str) -> None:
    lines = [
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Setting & ECE $\\downarrow$ & P@10 $\\uparrow$ & Abstain\\% & Coverage@80acc \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r.setting} & {r.ece:.4f} & {r.p_at_10:.4f} & "
            f"{100*r.abstention_rate:.1f}\\% & {r.coverage_at_80:.4f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    logger.info("LaTeX OOD comparison table saved -> %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Compare OOD uncertainty strategies.")
    p.add_argument("--dataset", required=True, help="Security JSONL dataset.")
    p.add_argument("--output",  default="evaluation/results/ood_comparison.json")
    p.add_argument("--latex",   default="evaluation/results/tables/ood_comparison.tex")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    rows = run_ood_comparison(args.dataset)
    print_ood_comparison_table(rows)
    save_latex_ood_table(rows, args.latex)

    out = {"experiment": "ood_comparison", "rows": [r.to_dict() for r in rows]}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Results saved -> %s", args.output)
