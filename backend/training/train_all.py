"""
Master Training Pipeline — Train All IntelliCode ML Models

Runs the full pipeline with a single command:
  1. Fetch REAL datasets from public sources (Bandit, GitHub, PyDriller)
  2. Train Complexity model    (XGBoost)
  3. Train Security model      (Random Forest + 1D CNN)
  4. Train Bug Predictor       (Logistic Regression + MLP)
  5. Optionally train Pattern  (CodeBERT fine-tune — slow, GPU recommended)

Real data sources used (no GitHub token required):
  Security  — Bandit's own labelled examples + security-fix commits (PyDriller)
  Complexity — Real Python files from psf/requests, pallets/flask, etc.
  Bug        — Real bug-fix commits from Django, Flask, requests, pip, etc.
  Pattern    — Real Python functions labelled by code-quality heuristics

Usage:
    cd backend

    # Full pipeline with real data (recommended)
    python training/train_all.py

    # Everything including CodeBERT (requires GPU for reasonable speed)
    python training/train_all.py --with-pattern

    # Skip data fetching if you already have datasets
    python training/train_all.py --skip-datagen

    # Smaller dataset for a quick smoke test
    python training/train_all.py --n 300 --no-mlp

Outputs:
    data/                          — JSONL datasets (real data)
    checkpoints/complexity/        — XGBoost model + metrics
    checkpoints/security/          — RF + CNN models + metrics
    checkpoints/bug_predictor/     — LR + MLP models + metrics
    checkpoints/pattern/           — CodeBERT checkpoint (optional)
    checkpoints/training_report.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _h(text: str):
    print(f"\n{_BOLD}{'=' * 60}{_RESET}")
    print(f"{_BOLD}{_GREEN}  {text}{_RESET}")
    print(f"{_BOLD}{'=' * 60}{_RESET}\n")


def _ok(text: str):
    print(f"{_GREEN}[OK] {text}{_RESET}")


def _warn(text: str):
    print(f"{_YELLOW}[!!] {text}{_RESET}")


def _fail(text: str):
    print(f"{_RED}[FAIL] {text}{_RESET}")


def run_step(label: str, fn, *args, **kwargs) -> tuple[bool, dict, float]:
    """Run a training step, catching errors and measuring time."""
    _h(label)
    t0 = time.time()
    try:
        result = fn(*args, **kwargs)
        elapsed = time.time() - t0
        _ok(f"{label} completed in {elapsed:.1f}s")
        return True, result or {}, elapsed
    except Exception as exc:
        elapsed = time.time() - t0
        _fail(f"{label} FAILED after {elapsed:.1f}s: {exc}")
        import traceback
        traceback.print_exc()
        return False, {"error": str(exc)}, elapsed


# ---------------------------------------------------------------------------
# Step wrappers
# ---------------------------------------------------------------------------

def step_datagen(n: int, out_dir: str, tasks: list[str]):
    """Fetch REAL datasets from public sources (Bandit, GitHub, PyDriller)."""
    import importlib.util, sys
    from pathlib import Path as P

    backend_dir = P(__file__).resolve().parent.parent
    sys.path.insert(0, str(backend_dir))

    spec = importlib.util.spec_from_file_location(
        "fetch_real_datasets",
        backend_dir / "training" / "fetch_real_datasets.py",
    )
    fetcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetcher)

    counts = {}
    for task in tasks:
        out_path = f"{out_dir}/{task}_dataset.jsonl"
        print(f"\nFetching real {task} dataset (target {n} samples)…")
        if task == "complexity":
            samples = fetcher.fetch_complexity_dataset(n=n, out=out_path)
        elif task == "security":
            samples = fetcher.fetch_security_dataset(n=n, out=out_path)
        elif task == "pattern":
            samples = fetcher.fetch_pattern_dataset(n=n, out=out_path)
        elif task == "bug":
            samples = fetcher.fetch_bug_dataset(n=n, out=out_path)
        else:
            samples = []
        counts[task] = len(samples)
    return counts


def step_complexity(data_path: str, out_dir: str, cv_folds: int):
    import importlib.util, sys
    from pathlib import Path as P

    backend_dir = P(__file__).resolve().parent.parent
    sys.path.insert(0, str(backend_dir))

    spec = importlib.util.spec_from_file_location(
        "train_complexity",
        backend_dir / "training" / "train_complexity.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.train(data_path=data_path, output_dir=out_dir, cv_folds=cv_folds)


def step_security(data_path: str, out_dir: str):
    import importlib.util, sys
    from pathlib import Path as P

    backend_dir = P(__file__).resolve().parent.parent
    sys.path.insert(0, str(backend_dir))

    spec = importlib.util.spec_from_file_location(
        "train_security",
        backend_dir / "training" / "train_security.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.train(data_path=data_path, output_dir=out_dir)


def step_bugs(data_path: str, out_dir: str, cv_folds: int, use_mlp: bool, epochs: int):
    import importlib.util, sys
    from pathlib import Path as P

    backend_dir = P(__file__).resolve().parent.parent
    sys.path.insert(0, str(backend_dir))

    spec = importlib.util.spec_from_file_location(
        "train_bugs",
        backend_dir / "training" / "train_bugs.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.train(
        data_path=data_path,
        output_dir=out_dir,
        cv_folds=cv_folds,
        use_mlp=use_mlp,
        epochs=epochs,
    )


def step_pattern(data_path: str, out_dir: str):
    import importlib.util, sys
    from pathlib import Path as P

    backend_dir = P(__file__).resolve().parent.parent
    sys.path.insert(0, str(backend_dir))

    spec = importlib.util.spec_from_file_location(
        "train_pattern",
        backend_dir / "training" / "train_pattern.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.train(data_path=data_path, output_dir=out_dir)


def step_cgam(data_path: str, out_dir: str):
    import importlib.util, sys
    from pathlib import Path as P
    backend_dir = P(__file__).resolve().parent.parent
    sys.path.insert(0, str(backend_dir))
    spec = importlib.util.spec_from_file_location(
        "train_cgam", backend_dir / "training" / "train_cgam.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.train(sources=mod.load_sources_from_dataset(data_path), output_dir=out_dir)


def step_apcr(data_path: str, out_dir: str):
    import importlib.util, sys
    from pathlib import Path as P
    backend_dir = P(__file__).resolve().parent.parent
    sys.path.insert(0, str(backend_dir))
    spec = importlib.util.spec_from_file_location(
        "train_apcr", backend_dir / "training" / "train_apcr.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.train(data_path=data_path, output_dir=out_dir)


def step_dre(data_path: str, out_dir: str):
    import importlib.util, sys
    from pathlib import Path as P
    backend_dir = P(__file__).resolve().parent.parent
    sys.path.insert(0, str(backend_dir))
    spec = importlib.util.spec_from_file_location(
        "train_dre", backend_dir / "training" / "train_dre.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.train(data_path=data_path, output_dir=out_dir)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(report: dict):
    _h("Training Report")
    total_time = sum(s.get("elapsed_seconds", 0) for s in report["steps"].values())
    print(f"{'Step':<25} {'Status':<10} {'Time':>8}")
    print("-" * 47)
    for name, info in report["steps"].items():
        status = _GREEN + "PASS" + _RESET if info["success"] else _RED + "FAIL" + _RESET
        print(f"  {name:<23} {status}   {info['elapsed_seconds']:>6.1f}s")
    print("-" * 47)
    print(f"  {'Total':<23}        {total_time:>6.1f}s")
    print()

    checkpoints = list(Path("checkpoints").glob("**/*.pkl")) + \
                  list(Path("checkpoints").glob("**/*.pt"))
    if checkpoints:
        print(f"Saved checkpoints:")
        for cp in sorted(checkpoints):
            size_kb = cp.stat().st_size / 1024
            print(f"  {cp}  ({size_kb:.0f} KB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="IntelliCode — train all ML models in one command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=1500,
                        help="Samples per dataset (default 1500)")
    parser.add_argument("--data-dir", default="data",
                        help="Directory for generated datasets")
    parser.add_argument("--checkpoint-dir", default="checkpoints",
                        help="Root directory for model checkpoints")
    parser.add_argument("--cv", type=int, default=5,
                        help="Cross-validation folds (default 5)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="MLP training epochs (default 50)")
    parser.add_argument("--no-mlp", action="store_true",
                        help="Skip MLP for bug predictor (faster)")
    parser.add_argument("--with-pattern", action="store_true",
                        help="Also fine-tune CodeBERT for pattern recognition")
    parser.add_argument("--skip-datagen", action="store_true",
                        help="Skip dataset generation (use existing files)")
    parser.add_argument("--only", choices=["datagen", "complexity", "security", "bugs", "pattern",
                                            "cgam", "apcr", "dre"],
                        help="Run only one step")
    args = parser.parse_args()

    data_dir = args.data_dir
    ckpt_dir = args.checkpoint_dir
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    tasks_for_datagen = ["complexity", "security", "bug", "pattern"]

    report = {"steps": {}, "args": vars(args)}
    t_start = time.time()

    # ── 1. Data fetching ─────────────────────────────────────────────────────
    if not args.skip_datagen and args.only in (None, "datagen"):
        ok, res, elapsed = run_step(
            "Fetch Real Datasets",
            step_datagen,
            n=args.n,
            out_dir=data_dir,
            tasks=tasks_for_datagen,
        )
        report["steps"]["datagen"] = {"success": ok, "result": res, "elapsed_seconds": elapsed}
        if not ok and args.only == "datagen":
            sys.exit(1)
    elif args.skip_datagen:
        _warn("Skipping data generation (--skip-datagen)")

    # ── 2. Complexity ────────────────────────────────────────────────────────
    if args.only in (None, "complexity"):
        data_path = f"{data_dir}/complexity_dataset.jsonl"
        if Path(data_path).exists():
            ok, res, elapsed = run_step(
                "Train Complexity (XGBoost)",
                step_complexity,
                data_path=data_path,
                out_dir=f"{ckpt_dir}/complexity",
                cv_folds=args.cv,
            )
        else:
            _warn(f"Skipping complexity: {data_path} not found")
            ok, res, elapsed = False, {"error": "dataset missing"}, 0.0
        report["steps"]["complexity"] = {"success": ok, "result": res, "elapsed_seconds": elapsed}

    # ── 3. Security ──────────────────────────────────────────────────────────
    if args.only in (None, "security"):
        data_path = f"{data_dir}/security_dataset.jsonl"
        if Path(data_path).exists():
            ok, res, elapsed = run_step(
                "Train Security (RF + CNN)",
                step_security,
                data_path=data_path,
                out_dir=f"{ckpt_dir}/security",
            )
        else:
            _warn(f"Skipping security: {data_path} not found")
            ok, res, elapsed = False, {"error": "dataset missing"}, 0.0
        report["steps"]["security"] = {"success": ok, "result": res, "elapsed_seconds": elapsed}

    # ── 4. Bug Predictor ─────────────────────────────────────────────────────
    if args.only in (None, "bugs"):
        data_path = f"{data_dir}/bug_dataset.jsonl"
        if Path(data_path).exists():
            ok, res, elapsed = run_step(
                "Train Bug Predictor (LR + MLP)",
                step_bugs,
                data_path=data_path,
                out_dir=f"{ckpt_dir}/bug_predictor",
                cv_folds=args.cv,
                use_mlp=not args.no_mlp,
                epochs=args.epochs,
            )
        else:
            _warn(f"Skipping bug predictor: {data_path} not found")
            ok, res, elapsed = False, {"error": "dataset missing"}, 0.0
        report["steps"]["bugs"] = {"success": ok, "result": res, "elapsed_seconds": elapsed}

    # ── 5. Pattern (RF classifier, always runs) ──────────────────────────────
    if args.only in (None, "pattern"):
        data_path = f"{data_dir}/pattern_dataset.jsonl"
        if Path(data_path).exists():
            ok, res, elapsed = run_step(
                "Train Pattern Classifier (RF)",
                step_pattern,
                data_path=data_path,
                out_dir=f"{ckpt_dir}/pattern",
            )
        else:
            _warn(f"Skipping pattern: {data_path} not found")
            ok, res, elapsed = False, {"error": "dataset missing"}, 0.0
        report["steps"]["pattern"] = {"success": ok, "result": res, "elapsed_seconds": elapsed}

    # ── 6. CGAM — trains on complexity dataset (clean code sources) ──────────
    if args.only in (None, "cgam"):
        data_path = f"{data_dir}/complexity_dataset.jsonl"
        if Path(data_path).exists():
            ok, res, elapsed = run_step("Train CGAM (grammar anomaly)",
                step_cgam, data_path=data_path, out_dir=f"{ckpt_dir}/cgam")
        else:
            _warn("Skipping CGAM: complexity dataset not found")
            ok, res, elapsed = False, {"error": "dataset missing"}, 0.0
        report["steps"]["cgam"] = {"success": ok, "result": res, "elapsed_seconds": elapsed}

    # ── 7. APCR — trains on complexity dataset ────────────────────────────────
    if args.only in (None, "apcr"):
        data_path = f"{data_dir}/complexity_dataset.jsonl"
        if Path(data_path).exists():
            ok, res, elapsed = run_step("Train APCR (asymmetric regressor)",
                step_apcr, data_path=data_path, out_dir=f"{ckpt_dir}/apcr")
        else:
            _warn("Skipping APCR: complexity dataset not found")
            ok, res, elapsed = False, {"error": "dataset missing"}, 0.0
        report["steps"]["apcr"] = {"success": ok, "result": res, "elapsed_seconds": elapsed}

    # ── 8. DRE — trains on bug dataset ───────────────────────────────────────
    if args.only in (None, "dre"):
        data_path = f"{data_dir}/bug_dataset.jsonl"
        if Path(data_path).exists():
            ok, res, elapsed = run_step("Train DRE (differential risk encoder)",
                step_dre, data_path=data_path, out_dir=f"{ckpt_dir}/dre")
        else:
            _warn("Skipping DRE: bug dataset not found")
            ok, res, elapsed = False, {"error": "dataset missing"}, 0.0
        report["steps"]["dre"] = {"success": ok, "result": res, "elapsed_seconds": elapsed}

    # ── Summary ──────────────────────────────────────────────────────────────
    report["total_seconds"] = time.time() - t_start

    report_path = Path(ckpt_dir) / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print_report(report)
    print(f"Full report -> {report_path}")

    n_failed = sum(1 for s in report["steps"].values() if not s["success"])
    if n_failed:
        _fail(f"{n_failed} step(s) failed. Check output above.")
        sys.exit(1)
    else:
        _ok("All steps completed successfully.")


if __name__ == "__main__":
    main()
