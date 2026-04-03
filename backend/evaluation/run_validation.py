"""
Master Validation Runner
=========================
Orchestrates all IntelliCode validation experiments and produces a unified
thesis-ready report covering all six validation dimensions.

Validation dimensions run:
  1. Cognitive Complexity — SonarQube calibration (builtin benchmark + optional file)
  2. Cross-Project (LOPO) — leave-one-project-out for bug, security, complexity, pattern
  3. Baseline Comparison  — IntelliCode vs Bandit / radon / pylint on security data
  4. Ablation Study       — feature group + data fraction ablation with Cohen's d
  5. Conformal Coverage   — empirical MAPIE interval coverage on complexity model
  6. Effort-Aware Metrics — Popt and PofB20 for bug prediction

Outputs:
  evaluation/results/
    validation_summary.json       — machine-readable combined report
    validation_report.txt         — human-readable console dump
    tables/
      lopo_{task}.tex             — LaTeX LOPO table per task
      ablation_{task}.tex         — LaTeX ablation table per task
      baseline_comparison.tex     — LaTeX baseline comparison table

Usage:
    cd backend
    python evaluation/run_validation.py [options]

    # Minimal run using only built-in benchmarks (no data files required):
    python evaluation/run_validation.py --builtin-only

    # Full run with real datasets:
    python evaluation/run_validation.py \
        --security-data  data/security_dataset.jsonl \
        --bug-data        data/bug_dataset.jsonl \
        --complexity-data data/complexity_dataset.jsonl \
        --pattern-data    data/pattern_dataset.jsonl

    # Include IntelliCode model predictions in baseline comparison:
    python evaluation/run_validation.py \
        --security-data data/security_dataset.jsonl \
        --with-intellicode
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("evaluation/results")
TABLES_DIR  = RESULTS_DIR / "tables"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _section(title: str, width: int = 72) -> str:
    bar = "=" * width
    return f"\n{bar}\n  {title}\n{bar}"


def _save_text(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# 1 — Cognitive Complexity Validation
# ---------------------------------------------------------------------------

def run_cc_validation(benchmark_path: Optional[str]) -> dict:
    from evaluation.cognitive_complexity_validator import (
        BUILTIN_BENCHMARK, CognitiveComplexityValidator,
    )

    validator = CognitiveComplexityValidator()

    # Always run against builtin benchmark
    result_builtin = validator.validate(BUILTIN_BENCHMARK)
    print(_section("1. Cognitive Complexity — Built-in Benchmark"))
    result_builtin.print_summary()

    out = {
        "builtin": result_builtin.to_dict(),
        "pass": result_builtin.passes_threshold(),
    }

    # Optional external benchmark file
    if benchmark_path and Path(benchmark_path).exists():
        with open(benchmark_path) as f:
            ext_bench = json.load(f)
        result_ext = validator.validate(ext_bench)
        print("\n  External benchmark:")
        result_ext.print_summary()
        out["external"] = result_ext.to_dict()

    return out


# ---------------------------------------------------------------------------
# 2 — Cross-Project (LOPO) Benchmark
# ---------------------------------------------------------------------------

def run_lopo(records_by_task: dict[str, list[dict]]) -> dict:
    from evaluation.cross_project_benchmark import CrossProjectBenchmark

    results = {}
    for task, records in records_by_task.items():
        if not records:
            logger.info("  [SKIP] %s — no records provided", task)
            continue
        print(_section(f"2. LOPO Benchmark — Task: {task}  ({len(records)} records)"))
        bench = CrossProjectBenchmark(task=task)

        try:
            # LOPO evaluation
            report = bench.run(records)
            bench.print_report(report)

            # Random-split baseline for degradation measurement
            rs = bench.run_random_split_baseline(records, n_trials=5)
            if rs.get("random_split_auc_mean") is not None and report.mean_auc is not None:
                degradation = round(rs["random_split_auc_mean"] - report.mean_auc, 4)
                print(f"  Random-split AUC: {rs['random_split_auc_mean']:.4f} +/- "
                      f"{rs['random_split_auc_std']:.4f}")
                print(f"  Cross-project degradation: {degradation:+.4f} AUC points")
                report.random_split_auc  = rs["random_split_auc_mean"]
                report.degradation_auc   = degradation

            results[task] = report.to_dict()

            # Save per-task LaTeX table
            tex_path = TABLES_DIR / f"lopo_{task}.tex"
            bench.save_latex_table(report, str(tex_path))
            print(f"  LaTeX -> {tex_path}")

        except ValueError as e:
            # Datasets without per-repo labels — fall back to 5-fold CV
            print(f"  [INFO] LOPO unavailable ({e}). Running 5-fold CV instead.")
            try:
                rs = bench.run_random_split_baseline(records, n_trials=5)
                auc_m = rs.get("random_split_auc_mean")
                auc_s = rs.get("random_split_auc_std", 0)
                if auc_m is not None:
                    print(f"  5-fold CV AUC: {auc_m:.4f} +/- {auc_s:.4f}")
                results[task] = {
                    "protocol": "5fold_cv_fallback",
                    "note": "Repo labels absent; LOPO requires >= 2 distinct repos in data",
                    **rs,
                }
            except Exception as e2:
                results[task] = {"error": str(e2)}

        except Exception as e:
            logger.warning("  LOPO failed for %s: %s", task, e)
            results[task] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------
# 3 — Baseline Comparison
# ---------------------------------------------------------------------------

def run_baselines(
    security_records: list[dict],
    complexity_records: list[dict],
    bug_records: list[dict],
    with_intellicode: bool,
) -> dict:
    from evaluation.baseline_comparison import (
        evaluate_security_baselines,
        evaluate_complexity_baselines,
        evaluate_bug_baselines,
        _print_latex_table,
    )

    results: dict = {}
    print(_section("3. Baseline Comparison"))
    print("  Baselines: keyword-scan (naive), Bandit, Semgrep, LOC-threshold, Halstead LR")

    # ── Security: keyword scan vs Bandit vs Semgrep vs IntelliCode ───────────
    if security_records:
        print(f"\n  Security ({len(security_records)} records) ...")
        ic_system = None
        if with_intellicode:
            try:
                from models.security_detection import EnsembleSecurityModel
                ic_system = EnsembleSecurityModel(checkpoint_dir="checkpoints/security")
                print("  [OK] Loaded IntelliCode security model")
            except Exception as e:
                print(f"  [WARN] Security model unavailable: {e}")
        try:
            sec_res = evaluate_security_baselines(security_records, ic_system)
            results["security"] = sec_res
            for name in ("keyword_scan", "bandit", "semgrep", "intellicode"):
                m = sec_res.get(name, {})
                if isinstance(m, dict) and "f1" in m:
                    print(f"  {name:<18} — F1: {m['f1']:.4f}  AUC: {m['auc']:.4f}  AP: {m['ap']:.4f}")
        except Exception as e:
            print(f"  [ERROR] {e}")
            results["security"] = {"error": str(e)}
    else:
        print("  [SKIP] Security baselines — no data")

    # ── Complexity: LOC naive vs radon vs IntelliCode ─────────────────────────
    if complexity_records:
        print(f"\n  Complexity ({len(complexity_records)} records) ...")
        ic_cpx = None
        if with_intellicode:
            try:
                from models.complexity_prediction import ComplexityPredictionModel
                ic_cpx = ComplexityPredictionModel(
                    checkpoint_path="checkpoints/complexity/model.pkl"
                )
                print("  [OK] Loaded IntelliCode complexity model")
            except Exception as e:
                print(f"  [WARN] Complexity model unavailable: {e}")
        try:
            cpx_res = evaluate_complexity_baselines(complexity_records, ic_cpx)
            results["complexity"] = cpx_res
            for name, m in cpx_res.items():
                if isinstance(m, dict) and "spearman" in m:
                    print(f"  {name:<18} — rho: {m['spearman']:.4f}  "
                          f"RMSE: {m.get('rmse', 0):.3f}  R2: {m.get('r2', 0):.4f}")
        except Exception as e:
            print(f"  [ERROR] {e}")
            results["complexity"] = {"error": str(e)}
    else:
        print("  [SKIP] Complexity baselines — no data")

    # ── Bugs: majority-class vs LOC-threshold vs Halstead LR vs IntelliCode ──
    if bug_records:
        print(f"\n  Bugs ({len(bug_records)} records) ...")
        ic_bug = None
        if with_intellicode:
            try:
                from models.bug_predictor import BugPredictionModel
                ic_bug = BugPredictionModel()
                print("  [OK] Loaded IntelliCode bug model")
            except Exception as e:
                print(f"  [WARN] Bug model unavailable: {e}")
        try:
            bug_res = evaluate_bug_baselines(bug_records, ic_bug)
            results["bugs"] = bug_res
        except Exception as e:
            print(f"  [ERROR] {e}")
            results["bugs"] = {"error": str(e)}
    else:
        print("  [SKIP] Bug baselines — no data")

    # ── Save LaTeX table ──────────────────────────────────────────────────────
    try:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _print_latex_table(results)
        latex_out = buf.getvalue()
        # Extract just the tabular block for the .tex file
        _save_text(latex_out, TABLES_DIR / "baseline_comparison.tex")
        print(f"\n  LaTeX table -> {TABLES_DIR / 'baseline_comparison.tex'}")
    except Exception as e:
        logger.debug("LaTeX baseline table failed: %s", e)
        _save_baseline_latex(results, TABLES_DIR / "baseline_comparison.tex")

    return results


def _save_baseline_latex(results: dict, path: Path) -> None:
    """Write a LaTeX table comparing IntelliCode vs baselines."""
    lines = [
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Task & System & Precision & Recall & F1 & AUC \\\\",
        "\\midrule",
    ]

    for task, task_res in results.items():
        for system, metrics in task_res.items():
            if system.startswith("significance") or not isinstance(metrics, dict):
                continue
            p   = metrics.get("precision", metrics.get("spearman"))
            r   = metrics.get("recall",    metrics.get("mae"))
            f1  = metrics.get("f1")
            auc = metrics.get("auc")
            p_s   = f"{p:.4f}"   if p   is not None else "--"
            r_s   = f"{r:.4f}"   if r   is not None else "--"
            f1_s  = f"{f1:.4f}"  if f1  is not None else "--"
            auc_s = f"{auc:.4f}" if auc is not None else "--"
            task_s = task.capitalize().replace("_", " ")
            sys_s  = system.replace("_", " ").capitalize()
            lines.append(f"{task_s} & {sys_s} & {p_s} & {r_s} & {f1_s} & {auc_s} \\\\")

    lines += ["\\bottomrule", "\\end{tabular}"]
    _save_text("\n".join(lines), path)
    print(f"  LaTeX -> {path}")


# ---------------------------------------------------------------------------
# 4 — Ablation Study
# ---------------------------------------------------------------------------

def run_ablation(records_by_task: dict[str, list[dict]]) -> dict:
    from evaluation.ablation_study import (
        AblationStudy, STATIC_FEATURE_GROUPS, JIT_FEATURE_GROUPS,
    )
    from evaluation.cross_project_benchmark import _PREPARERS

    results = {}

    for task, records in records_by_task.items():
        if not records:
            continue
        print(_section(f"4. Ablation Study — Task: {task}  ({len(records)} records)"))

        try:
            X, y, repos = _PREPARERS[task](records)
            if len(X) < 50:
                print(f"  [SKIP] Insufficient data ({len(X)} samples)")
                continue

            study = AblationStudy(task=task)

            # Feature group ablation
            groups = STATIC_FEATURE_GROUPS.copy()
            if task == "bug":
                groups.update(JIT_FEATURE_GROUPS)

            feat_report = study.run_feature_ablation(X, y, list(repos), groups, n_trials=5)
            study.print_table(feat_report)
            study.save_latex_table(feat_report, str(TABLES_DIR / f"ablation_{task}.tex"))

            # Data fraction ablation (learning curve)
            data_report = study.run_data_ablation(X, y, list(repos))
            results[task] = {
                "feature_ablation": feat_report.to_dict(),
                "data_ablation":    data_report.to_dict(),
            }

        except Exception as e:
            logger.warning("  Ablation failed for %s: %s", task, e)
            results[task] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------
# 5 — Conformal Coverage
# ---------------------------------------------------------------------------

def run_conformal(complexity_records: list[dict]) -> dict:
    from evaluation.conformal_coverage import run_coverage_from_mapie_model, print_coverage_report

    print(_section("5. Conformal Prediction Coverage"))

    if not complexity_records:
        print("  [SKIP] No complexity data provided")
        return {}

    results = run_coverage_from_mapie_model(
        complexity_records,
        checkpoint_path="checkpoints/complexity/model.pkl",
    )
    if results:
        print_coverage_report(results)
        return {"coverage": [r.to_dict() for r in results]}

    print("  [WARN] Coverage validation unavailable — no MAPIE checkpoint or insufficient data")
    return {}


# ---------------------------------------------------------------------------
# 6 — Effort-Aware Metrics (Popt, PofB20)
# ---------------------------------------------------------------------------

def run_effort(bug_records: list[dict], with_intellicode: bool) -> dict:
    from evaluation.conformal_coverage import compute_effort_metrics, print_effort_report

    print(_section("6. Effort-Aware Bug Prediction (Popt, PofB20)"))

    if not bug_records:
        print("  [SKIP] No bug data provided")
        return {}

    bug_model = None
    if with_intellicode:
        try:
            from models.bug_predictor import BugPredictionModel
            bug_model = BugPredictionModel(checkpoint_dir="checkpoints/bug_predictor")
            print("  [OK] Bug predictor loaded")
        except Exception as e:
            print(f"  [WARN] Bug model unavailable: {e}")

    metrics = compute_effort_metrics(bug_records, bug_model)
    print_effort_report(metrics)
    return metrics.to_dict()


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(report: dict) -> None:
    print(_section("VALIDATION SUMMARY", width=80))

    # CC
    cc = report.get("cognitive_complexity", {})
    if cc.get("builtin"):
        b = cc["builtin"]
        status = "PASS" if cc.get("pass") else "FAIL"
        print(f"  {'CC Validation':<35} Pearson r={b['pearson_r']:.4f}  "
              f"MAE={b['mae']:.3f}  [{status}]")

    # LOPO
    lopo = report.get("lopo", {})
    for task, res in lopo.items():
        if isinstance(res, dict) and res.get("mean_auc") is not None:
            m = res["mean_auc"]
            s = res.get("std_auc", 0)
            d = res.get("degradation_auc")
            d_s = f"  (vs random: {d:+.4f})" if d is not None else ""
            print(f"  {'LOPO ' + task:<35} AUC={m:.4f} ± {s:.4f}{d_s}")
        elif isinstance(res, dict) and res.get("mean_rmse") is not None:
            m = res["mean_rmse"]
            s = res.get("std_rmse", 0)
            spr = res.get("mean_spearman")
            spr_s = f"  Spearman={spr:.4f}" if spr else ""
            print(f"  {'LOPO ' + task:<35} RMSE={m:.4f} ± {s:.4f}{spr_s}")

    # Baselines
    bl = report.get("baselines", {})
    for task, task_res in bl.items():
        for system, metrics in task_res.items():
            if system.startswith("significance") or not isinstance(metrics, dict):
                continue
            auc = metrics.get("auc")
            f1  = metrics.get("f1")
            if auc or f1:
                label = f"Baseline {task}/{system}"
                line = f"  {label:<35}"
                if f1:  line += f" F1={f1:.4f}"
                if auc: line += f"  AUC={auc:.4f}"
                print(line)

    # Conformal
    cov = report.get("conformal", {}).get("coverage", [])
    for c in cov:
        status = "PASS" if c.get("valid") else "FAIL"
        print(f"  {'Conformal ' + str(int(c['claimed_coverage']*100)) + '%':<35}"
              f" empirical={c['empirical_coverage']*100:.1f}%  [{status}]")

    # Effort
    eff = report.get("effort", {})
    if eff.get("popt") is not None:
        print(f"  {'Effort (Popt / PofB20)':<35} "
              f"Popt={eff['popt']:.4f}  PofB20={eff['pofb20']:.4f}")

    print("=" * 80)


def save_latex_summary(report: dict, path: Path) -> None:
    """Write a combined thesis validation chapter table."""
    lines = [
        "\\begin{tabular}{lll}",
        "\\toprule",
        "Dimension & Metric & Result \\\\",
        "\\midrule",
    ]
    # CC
    cc = report.get("cognitive_complexity", {}).get("builtin", {})
    if cc:
        lines.append(f"CC Calibration & Pearson $r$ & {cc.get('pearson_r', '--'):.4f} \\\\")
        lines.append(f"CC Calibration & MAE & {cc.get('mae', '--'):.3f} \\\\")

    # LOPO
    for task, res in report.get("lopo", {}).items():
        if not isinstance(res, dict):
            continue
        if res.get("mean_auc") is not None:
            lines.append(f"LOPO {task.capitalize()} & AUC & "
                         f"{res['mean_auc']:.4f} $\\pm$ {res.get('std_auc',0):.4f} \\\\")
        if res.get("mean_rmse") is not None:
            lines.append(f"LOPO {task.capitalize()} & RMSE & "
                         f"{res['mean_rmse']:.4f} $\\pm$ {res.get('std_rmse',0):.4f} \\\\")

    # Effort
    eff = report.get("effort", {})
    if eff.get("popt") is not None:
        lines.append(f"Effort-aware & Popt & {eff['popt']:.4f} \\\\")
        lines.append(f"Effort-aware & PofB20 & {eff['pofb20']:.4f} \\\\")

    lines += ["\\bottomrule", "\\end{tabular}"]
    _save_text("\n".join(lines), path)
    print(f"\nValidation summary LaTeX -> {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(
        description="Run all IntelliCode thesis validation experiments."
    )
    parser.add_argument("--security-data",   default="data/security_dataset.jsonl")
    parser.add_argument("--bug-data",        default="data/bug_dataset.jsonl")
    parser.add_argument("--complexity-data", default="data/complexity_dataset.jsonl")
    parser.add_argument("--pattern-data",    default="data/pattern_dataset.jsonl")
    parser.add_argument("--cc-benchmark",    default=None,
                        help="Optional external CC benchmark JSON file")
    parser.add_argument("--output",          default="evaluation/results/validation_summary.json")
    parser.add_argument("--builtin-only",    action="store_true",
                        help="Skip all dataset-dependent experiments (CC benchmark only)")
    parser.add_argument("--with-intellicode", action="store_true",
                        help="Include IntelliCode model predictions in baseline comparison")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Load datasets (silently skip missing files)
    def _try_load(path: str) -> list[dict]:
        if not path or not Path(path).exists():
            return []
        try:
            recs = _load_jsonl(path)
            print(f"  Loaded {len(recs):>6} records from {path}")
            return recs
        except Exception as e:
            logger.warning("Could not load %s: %s", path, e)
            return []

    # Load datasets BEFORE setting up the Tee so loading messages print immediately
    print("Loading datasets...")
    sec_recs  = [] if args.builtin_only else _try_load(args.security_data)
    bug_recs  = [] if args.builtin_only else _try_load(args.bug_data)
    cpx_recs  = [] if args.builtin_only else _try_load(args.complexity_data)
    pat_recs  = [] if args.builtin_only else _try_load(args.pattern_data)
    print()

    t0 = time.perf_counter()
    full_report: dict = {}
    console_buf = io.StringIO()

    # Capture all output to both console and buffer.
    # Use errors="replace" on the real stdout to survive non-ASCII on Windows cp1252.
    import io as _io
    safe_stdout = _io.TextIOWrapper(
        sys.stdout.buffer,
        encoding=sys.stdout.encoding or "utf-8",
        errors="replace",
        line_buffering=True,
    ) if hasattr(sys.stdout, "buffer") else sys.stdout

    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                except Exception:
                    pass
        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except Exception:
                    pass

    tee = Tee(safe_stdout, console_buf)
    old_stdout = sys.stdout
    sys.stdout = tee  # type: ignore[assignment]

    try:
        # 1. CC Validation (always runs — uses builtin benchmark)
        full_report["cognitive_complexity"] = run_cc_validation(args.cc_benchmark)

        records_by_task = {
            "bug":        bug_recs,
            "security":   sec_recs,
            "complexity": cpx_recs,
            "pattern":    pat_recs,
        }

        # 2. LOPO
        full_report["lopo"] = run_lopo(
            {t: r for t, r in records_by_task.items() if r}
        )

        # 3. Baselines
        full_report["baselines"] = run_baselines(
            sec_recs, cpx_recs, bug_recs, args.with_intellicode
        )

        # 4. Ablation
        full_report["ablation"] = run_ablation(
            {t: r for t, r in records_by_task.items() if r}
        )

        # 5. Conformal coverage
        full_report["conformal"] = run_conformal(cpx_recs)

        # 6. Effort-aware metrics
        full_report["effort"] = run_effort(bug_recs, args.with_intellicode)

        # Summary table
        print_summary_table(full_report)

    finally:
        sys.stdout = old_stdout

    # Save text report
    report_text = console_buf.getvalue()
    txt_path = RESULTS_DIR / "validation_report.txt"
    _save_text(report_text, txt_path)
    print(f"\nText report  -> {txt_path}")

    # Save combined JSON
    full_report["_meta"] = {
        "duration_seconds": round(time.perf_counter() - t0, 2),
        "records_loaded": {
            "security":   len(sec_recs),
            "bug":        len(bug_recs),
            "complexity": len(cpx_recs),
            "pattern":    len(pat_recs),
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"JSON report  -> {out_path}")

    # Save thesis summary LaTeX table
    save_latex_summary(full_report, TABLES_DIR / "validation_summary.tex")

    elapsed = round(time.perf_counter() - t0, 1)
    print(f"\nAll validation completed in {elapsed}s")


if __name__ == "__main__":
    main()
