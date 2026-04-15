"""
Run all novel contribution evaluations.
Saves results to evaluation/results/ and prints a summary table.

Usage:
    python run_novel_contributions.py [--skip <name>] [--only <name>]

Novel contributions implemented
--------------------------------
1. PIFF  -- Project-Invariant Feature Filtering
2. SAEF  -- Shift-Aware Evaluation Framework
3. LAMV  -- Leakage-Aware Model Validation
4. TDCP  -- Time-Decayed Code Prediction
5. CACM  -- Calibration Analysis of Code ML Models
6. SIMP  -- Simplicity Baselines Comparison
7. FTAX  -- Failure Mode Taxonomy
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path

# Fix Windows cp1252 console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("evaluation/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _save(name: str, report: dict):
    path = RESULTS_DIR / f"{name}_results.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("  Saved: %s", path)


def _run(label: str, fn, **kwargs) -> dict:
    logger.info("=" * 60)
    logger.info("Running %s ...", label)
    t0 = time.time()
    try:
        result = fn(**kwargs)
    except Exception as e:
        logger.error("  FAILED: %s", e, exc_info=True)
        result = {"error": str(e)}
    elapsed = time.time() - t0
    logger.info("  Done in %.1fs", elapsed)
    return result


# ── main ──────────────────────────────────────────────────────────────────────

def main(skip: set[str] = frozenset(), only: set[str] = frozenset()):
    summary_rows = []

    def should_run(name: str) -> bool:
        if only:
            return name in only
        return name not in skip

    # 1. PIFF
    if should_run("PIFF"):
        from evaluation.piff import run_piff
        report = _run("PIFF", run_piff,
                      data_path="data/bug_dataset_v2.jsonl",
                      top_k_stable=15, seed=42)
        _save("piff", report)
        if "error" not in report:
            full_auc = report["full_features"].get("mean_auc", 0)
            piff_auc = report["piff_features"].get("mean_auc", 0)
            delta    = report.get("delta_auc", 0) or 0
            summary_rows.append({
                "contribution": "PIFF",
                "key_metric":  f"LOPO AUC: {full_auc:.4f} -> {piff_auc:.4f} (delta={delta:+.4f})",
                "stable_features": len(report.get("stable_feature_names", [])),
            })

    # 2. SAEF
    if should_run("SAEF"):
        from evaluation.saef import run_saef
        report = _run("SAEF", run_saef,
                      data_path="data/bug_dataset_v2.jsonl",
                      lopo_path="evaluation/results/lopo_bug.json")
        _save("saef", report)
        if "error" not in report:
            corr = report.get("shift_performance_correlation", {})
            summary_rows.append({
                "contribution": "SAEF",
                "key_metric": (
                    f"Pearson r(shift,AUC)={corr.get('pearson_r', 'N/A')}, "
                    f"p={corr.get('pearson_p', 'N/A')}"
                ),
            })

    # 3. LAMV
    if should_run("LAMV"):
        from evaluation.lamv import run_lamv
        report = _run("LAMV", run_lamv,
                      data_path="data/bug_dataset_v2.jsonl")
        _save("lamv", report)
        if "error" not in report:
            summary_rows.append({
                "contribution": "LAMV",
                "key_metric": (
                    f"Leakage detected: {report['summary']['leakage_detected']}  |  "
                    f"is_fix flagged: {report['known_leakage_verified']['is_fix_flagged']}  |  "
                    f"Suspicious: {report['summary']['suspicious_features']}"
                ),
            })

    # 4. TDCP
    if should_run("TDCP"):
        from evaluation.tdcp import run_tdcp
        report = _run("TDCP", run_tdcp,
                      data_path="data/bug_dataset_v2.jsonl",
                      lambdas=[0.0, 0.5, 1.0, 2.0])
        _save("tdcp", report)
        if "error" not in report:
            summary_rows.append({
                "contribution": "TDCP",
                "key_metric": (
                    f"Temporal AUC: {report['baseline_temporal_auc']:.4f} "
                    f"-> {report['best_temporal_auc']:.4f} "
                    f"(best lambda={report['best_lambda']}, "
                    f"delta={report['delta_vs_baseline']:+.4f})"
                ),
            })

    # 5. CACM (Calibration)
    if should_run("CACM"):
        from evaluation.calibration_analysis import run_calibration_analysis
        report = _run("CACM", run_calibration_analysis,
                      data_path="data/bug_dataset_v2.jsonl")
        _save("calibration", report)
        if "error" not in report:
            deg = report.get("calibration_degradation", {})
            summary_rows.append({
                "contribution": "CACM",
                "key_metric": (
                    f"ECE: {deg.get('ece_in_distribution', 0):.4f} (in-dist) "
                    f"-> {deg.get('ece_lopo', 0):.4f} (LOPO) "
                    f"(+{deg.get('ece_increase', 0):.4f})"
                ),
            })

    # 6. Simplicity Baselines
    if should_run("SIMP"):
        from evaluation.simplicity_baselines import run_simplicity_baselines
        report = _run("SIMP", run_simplicity_baselines,
                      data_path="data/bug_dataset_v2.jsonl")
        _save("simplicity_baselines", report)
        if "error" not in report:
            ranking = report.get("ranking", [])
            top3 = ", ".join(f"{n}={auc:.4f}" for n, auc in ranking[:3])
            summary_rows.append({
                "contribution": "Simplicity Baselines",
                "key_metric": f"LOPO ranking: {top3}",
                "finding": report.get("finding", "")[:100],
            })

    # 7. Failure Taxonomy
    if should_run("FTAX"):
        from evaluation.failure_taxonomy import run_failure_taxonomy
        report = _run("FTAX", run_failure_taxonomy,
                      lopo_bug_path="evaluation/results/lopo_bug.json",
                      lopo_complexity_path="evaluation/results/lopo_complexity.json",
                      lopo_security_path="evaluation/results/lopo_security.json",
                      bug_data_path="data/bug_dataset_v2.jsonl")
        _save("failure_taxonomy", report)
        if "error" not in report:
            modes = report.get("overall_mode_frequency", {})
            top_mode = max(modes, key=modes.get) if modes else "N/A"
            summary_rows.append({
                "contribution": "Failure Taxonomy",
                "key_metric": (
                    f"Top mode: {top_mode} ({modes.get(top_mode, 0)} cases)  |  "
                    f"Severe: {report['overall_severity'].get('severe', 0)}"
                ),
            })

    # 8. Pattern Leakage Fix
    if should_run("PLF"):
        from evaluation.pattern_leakage_fix import run_pattern_leakage_fix
        report = _run("PLF", run_pattern_leakage_fix,
                      data_path="data/pattern_dataset.jsonl")
        _save("pattern_leakage_fix", report)
        if "error" not in report:
            comp = report["model_comparison"]
            summary_rows.append({
                "contribution": "Pattern Leakage Fix",
                "key_metric": (
                    f"Full F1={comp['full_model']['mean_f1_macro']:.4f} vs "
                    f"Debiased F1={comp['debiased_model']['mean_f1_macro']:.4f} "
                    f"(inflation={comp['leakage_inflation']:+.4f}, "
                    f"Wilcoxon p={comp.get('wilcoxon_p', 'N/A')})"
                ),
            })

    # 9. Statistical Significance Tests
    if should_run("SIG"):
        from evaluation.significance_tests import run_significance_tests
        report = _run("SIG", run_significance_tests)
        _save("significance_tests", report)
        if "error" not in report:
            n_sig = sum(1 for r in report["summary_table"] if r["significant"])
            summary_rows.append({
                "contribution": "Significance Tests",
                "key_metric": (
                    f"{n_sig}/{len(report['summary_table'])} tests significant at alpha=0.05  |  "
                    f"Note: n=4 projects limits power (min p=0.0625 for Wilcoxon)  |  "
                    f"PIFF Cohen's d={report['tests'].get('PIFF_vs_Full', {}).get('cohens_d', '?')} (medium effect)"
                ),
            })

    # 10. Actionability Analysis
    if should_run("ACT"):
        from evaluation.actionability_analysis import run_actionability_analysis
        report = _run("ACT", run_actionability_analysis,
                      data_path="data/bug_dataset_v2.jsonl")
        _save("actionability", report)
        if "error" not in report:
            s = report["summary"]
            summary_rows.append({
                "contribution": "Actionability (FDR/PofB)",
                "key_metric": (
                    f"PofB@20={s['mean_pofb20']:.4f} (inspect 20% -> find {s['mean_pofb20']*100:.1f}% bugs)  |  "
                    f"FDR@0.5={s['mean_fdr_at_threshold_0.50']:.4f} ({s['mean_fdr_at_threshold_0.50']*100:.1f}% false alarms)"
                ),
                "finding": report.get("finding", "")[:120],
            })

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("NOVEL CONTRIBUTIONS SUMMARY")
    print("=" * 70)
    for row in summary_rows:
        print(f"\n[{row['contribution']}]")
        print(f"  {row['key_metric']}")
        if "finding" in row:
            print(f"  Finding: {row['finding']}")

    print("\n" + "=" * 70)
    print(f"Results saved to: {RESULTS_DIR.resolve()}")
    print("=" * 70)

    return summary_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Skip these contributions (e.g. --skip PIFF SAEF)")
    parser.add_argument("--only", nargs="*", default=[],
                        help="Run only these contributions")
    args = parser.parse_args()
    main(skip=set(args.skip), only=set(args.only))
