"""
MTL Comparison Script
======================
Compares single-task baselines vs. MTL with Kendall loss vs. MTL with MGDA loss
across all four code quality tasks.

For each task the canonical model family is used:
  - complexity : XGBoost regression
  - security   : Random Forest classification
  - bugs        : XGBoost classification
  - pattern    : Random Forest classification

Results are saved to evaluation/results/mtl_comparison.json and
evaluation/results/tables/mtl_comparison.tex before any printing so that
encoding errors in the console cannot cause data loss.

Usage (from backend/):
    python evaluation/mtl_comparison.py \\
        --data-dir  ../data \\
        --out-dir   evaluation/results

Constraints:
  - n_jobs=1 (Windows, no multiprocessing)
  - ASCII + cp1252 only in all print() calls (no Greek letters, no box-drawing)
  - try/except around every variant; fallback to {"error": str(e)} or
    {"skipped": True, "reason": "..."} when a model is unavailable
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature dimension constants (must match training)
# ---------------------------------------------------------------------------
# metrics_to_feature_vector() returns 15 dims (cognitive_complexity excluded)
STATIC_DIM   = 15
# Security RF uses 31-dim extended feature vector (updated Apr 2026)
SECURITY_DIM = 31
# Bug predictor: 15 static + 14 JIT = 29 dims
BUG_DIM      = 29


# ===========================================================================
# Data loaders
# ===========================================================================

def _load_jsonl(path: str) -> list:
    records = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _load_complexity(data_dir: Path):
    """Load complexity dataset -> (X, y) using 15-dim feature vector."""
    path = data_dir / "complexity_dataset.jsonl"
    if not path.exists():
        return None, None
    records = _load_jsonl(str(path))
    X, y = [], []
    COG_IDX = 1
    for r in records:
        feat = r.get("features", [])
        if not feat:
            continue
        # Dataset stores 17-dim raw features; training target = feat[COG_IDX=1]
        # Model input = [feat[i] for i in range(16) if i != COG_IDX] (15-dim)
        if len(feat) >= 16:
            target = float(feat[COG_IDX])
            x_vec  = [feat[i] for i in range(16) if i != COG_IDX]  # 15-dim
        elif len(feat) == 15:
            # Already stripped; use record["target"] as y
            t = r.get("target")
            if t is None:
                continue
            target = float(t)
            x_vec  = feat
        else:
            continue
        if len(x_vec) != STATIC_DIM:
            continue
        X.append([float(v) for v in x_vec])
        y.append(target)
    if not X:
        return None, None
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _load_security(data_dir: Path):
    """Load security dataset -> (X, y) using 16-dim RF feature vector."""
    path = data_dir / "security_dataset.jsonl"
    if not path.exists():
        return None, None
    records = _load_jsonl(str(path))
    X, y = [], []
    for r in records:
        try:
            if "features" in r:
                feat = list(r["features"])
            elif "source" in r:
                from models.security_detection import _build_rf_feature_vector
                feat = list(_build_rf_feature_vector(r["source"]))
            else:
                continue
            # Pad or truncate to SECURITY_DIM
            if len(feat) < SECURITY_DIM:
                feat = feat + [0.0] * (SECURITY_DIM - len(feat))
            feat = feat[:SECURITY_DIM]
            X.append([float(v) for v in feat])
            y.append(int(r["label"]))
        except Exception:
            continue
    if not X or len(set(y)) < 2:
        return None, None
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def _load_bugs(data_dir: Path):
    """Load bug dataset -> (X, y) using static + JIT features (29-dim)."""
    path = data_dir / "bug_dataset.jsonl"
    if not path.exists():
        return None, None
    records = _load_jsonl(str(path))
    X, y = [], []
    JIT_KEYS = [
        "NS", "ND", "NF", "Entropy", "LA", "LD", "LT", "FIX",
        "NDEV", "AGE", "NUC", "EXP", "REXP", "SEXP",
    ]
    # git_features proxy keys (present when full JIT features unavailable)
    GIT_PROXY = {"LA": "code_churn", "NDEV": "author_count"}
    for r in records:
        static = list(r.get("static_features", []))
        # jit_features or git_features (proxy)
        jit = r.get("jit_features") or {}
        if not jit:
            gf = r.get("git_features") or {}
            jit = {JIT_KEYS[0]: 0}  # will use proxy below
            for jk, gk in GIT_PROXY.items():
                jit[jk] = gf.get(gk, 0)
        if not static:
            continue
        # Handle 17-dim static (COG_IDX=1, same as complexity)
        if len(static) >= 16:
            static = [static[i] for i in range(16) if i != 1]  # 15-dim
        if len(static) != STATIC_DIM:
            continue
        jit_vec = [float(jit.get(k, 0)) for k in JIT_KEYS]
        feat = static + jit_vec  # 29-dim
        X.append([float(v) for v in feat])
        y.append(int(r["label"]))
    if not X or len(set(y)) < 2:
        return None, None
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def _load_patterns(data_dir: Path):
    """Load pattern dataset -> (X, y) using 15-dim metrics_to_feature_vector."""
    path = data_dir / "pattern_dataset.jsonl"
    if not path.exists():
        return None, None
    from models.multi_task_model import PATTERN_CLASS_TO_IDX
    from features.code_metrics import compute_all_metrics, metrics_to_feature_vector
    records = _load_jsonl(str(path))
    X, y = [], []
    for r in records:
        code  = r.get("code", "")
        label = r.get("label", "")
        if not code or label not in PATTERN_CLASS_TO_IDX:
            continue
        try:
            m    = compute_all_metrics(code)
            feat = metrics_to_feature_vector(m)
            if len(feat) != STATIC_DIM:
                continue
            X.append([float(v) for v in feat])
            y.append(PATTERN_CLASS_TO_IDX[label])
        except Exception:
            continue
    if not X:
        return None, None
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ===========================================================================
# Single-task canonical models
# ===========================================================================

def _train_single_complexity(X_tr, y_tr, X_te, y_te) -> dict:
    """XGBoost regression for cognitive complexity."""
    try:
        from xgboost import XGBRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import mean_squared_error, r2_score
        from scipy.stats import spearmanr

        mdl = Pipeline([
            ("scaler", StandardScaler()),
            ("xgb",    XGBRegressor(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=1, verbosity=0,
            )),
        ])
        mdl.fit(X_tr, y_tr)
        y_pred = mdl.predict(X_te)
        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        r2   = float(r2_score(y_te, y_pred))
        rho, _ = spearmanr(y_te, y_pred)
        return {"rmse": round(rmse, 4), "r2": round(r2, 4), "spearman": round(float(rho), 4)}
    except Exception as exc:
        return {"error": str(exc)}


def _train_single_security(X_tr, y_tr, X_te, y_te) -> dict:
    """Random Forest classification for security vulnerabilities."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

        mdl = Pipeline([
            ("scaler", StandardScaler()),
            ("rf",     RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                random_state=42, n_jobs=1,
            )),
        ])
        mdl.fit(X_tr, y_tr)
        y_prob = mdl.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        auc = float(roc_auc_score(y_te, y_prob))
        f1  = float(f1_score(y_te, y_pred, zero_division=0))
        ap  = float(average_precision_score(y_te, y_prob))
        return {"auc": round(auc, 4), "f1": round(f1, 4), "ap": round(ap, 4)}
    except Exception as exc:
        return {"error": str(exc)}


def _train_single_bugs(X_tr, y_tr, X_te, y_te) -> dict:
    """XGBoost classification for bug prediction."""
    try:
        from xgboost import XGBClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

        scale_pos = float(np.sum(y_tr == 0)) / max(float(np.sum(y_tr == 1)), 1.0)
        mdl = Pipeline([
            ("scaler", StandardScaler()),
            ("xgb",    XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pos,
                random_state=42, n_jobs=1, verbosity=0,
                eval_metric="logloss",
            )),
        ])
        mdl.fit(X_tr, y_tr)
        y_prob = mdl.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        auc = float(roc_auc_score(y_te, y_prob))
        f1  = float(f1_score(y_te, y_pred, zero_division=0))
        ap  = float(average_precision_score(y_te, y_prob))
        return {"auc": round(auc, 4), "f1": round(f1, 4), "ap": round(ap, 4)}
    except Exception as exc:
        return {"error": str(exc)}


def _train_single_patterns(X_tr, y_tr, X_te, y_te) -> dict:
    """Random Forest classification for code patterns."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import f1_score, accuracy_score

        mdl = Pipeline([
            ("scaler", StandardScaler()),
            ("rf",     RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                random_state=42, n_jobs=1,
            )),
        ])
        mdl.fit(X_tr, y_tr)
        y_pred = mdl.predict(X_te)
        f1  = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
        acc = float(accuracy_score(y_te, y_pred))
        return {"f1_macro": round(f1, 4), "accuracy": round(acc, 4)}
    except Exception as exc:
        return {"error": str(exc)}


# ===========================================================================
# MTL shallow model (runs MultiTaskTrainer.train_shallow in-process)
# ===========================================================================

def _train_mtl_variant(
    data_dir: Path,
    out_dir:  Path,
    strategy: str,
    X_splits: dict,
    y_splits: dict,
) -> dict:
    """
    Train a shallow MTL model using `strategy` ("kendall" | "mgda" | "equal")
    and return per-task test-set metrics.

    The MTL shallow model trains independent sklearn heads; the loss strategy
    affects only the MGDA weight computation (logged but not used for fitting).

    Returns dict mapping task -> metrics_dict.
    """
    try:
        from training.multi_task_trainer import MultiTaskTrainer

        ckpt_dir = str(out_dir / f"mtl_{strategy}_ckpt")
        trainer  = MultiTaskTrainer(
            loss_strategy=strategy,
            output_dir=ckpt_dir,
            test_split=0.20,
        )

        # We pass None for datasets that are unavailable, trainer handles gracefully
        def _p(task: str, fname: str) -> Optional[str]:
            p = data_dir / fname
            return str(p) if p.exists() else None

        result = trainer.train_shallow(
            complexity_data=_p("complexity", "complexity_dataset.jsonl"),
            security_data=_p("security",    "security_dataset.jsonl"),
            bug_data=_p("bug",              "bug_dataset.jsonl"),
            pattern_data=_p("pattern",      "pattern_dataset.jsonl"),
        )
        # result["tasks"] has per-task metrics from the MTL trainer
        return result.get("tasks", {})

    except Exception as exc:
        logger.warning("MTL %s variant failed: %s", strategy, exc)
        return {"error": str(exc)}


# ===========================================================================
# Per-task evaluation extractor
# ===========================================================================

def _task_metrics_from_mtl(mtl_tasks: dict, task: str) -> dict:
    """Extract canonical metrics for a task from MTL result dict."""
    if "error" in mtl_tasks:
        return {"error": mtl_tasks["error"]}
    task_m = mtl_tasks.get(task)
    if task_m is None:
        return {"skipped": True, "reason": "task not trained in MTL model"}
    return task_m


# ===========================================================================
# Main comparison runner
# ===========================================================================

def run_comparison(data_dir: Path, out_dir: Path) -> dict:
    """
    Run the full single-task vs MTL-kendall vs MTL-mgda comparison matrix.

    Returns a nested dict:
        {task: {strategy: metrics_dict}}
    """
    from sklearn.model_selection import train_test_split

    results: dict = {}

    # ------------------------------------------------------------------
    # Load all datasets
    # ------------------------------------------------------------------
    logger.info("Loading datasets from %s ...", data_dir)

    datasets: dict = {}
    loaders = {
        "complexity": _load_complexity,
        "security":   _load_security,
        "bugs":       _load_bugs,
        "patterns":   _load_patterns,
    }
    for task, loader in loaders.items():
        try:
            X, y = loader(data_dir)
            if X is not None:
                datasets[task] = (X, y)
                logger.info("  %s: %d samples, %d features", task, len(X), X.shape[1])
            else:
                logger.warning("  %s: dataset not found or empty", task)
        except Exception as exc:
            logger.warning("  %s: load error — %s", task, exc)

    # ------------------------------------------------------------------
    # Build train/test splits (80/20 stratified)
    # ------------------------------------------------------------------
    splits: dict = {}
    for task, (X, y) in datasets.items():
        try:
            is_regression = (task == "complexity")
            if is_regression:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=0.20, random_state=42
                )
            else:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=0.20, random_state=42, stratify=y
                )
            splits[task] = (X_tr, X_te, y_tr, y_te)
        except Exception as exc:
            logger.warning("  %s split failed: %s", task, exc)

    # ------------------------------------------------------------------
    # Single-task baselines
    # ------------------------------------------------------------------
    logger.info("Training single-task baselines ...")

    single_fns = {
        "complexity": _train_single_complexity,
        "security":   _train_single_security,
        "bugs":       _train_single_bugs,
        "patterns":   _train_single_patterns,
    }

    for task, fn in single_fns.items():
        results.setdefault(task, {})
        if task not in splits:
            results[task]["single_task"] = {
                "skipped": True, "reason": "dataset not available"
            }
            continue
        X_tr, X_te, y_tr, y_te = splits[task]
        logger.info("  %s single_task ...", task)
        try:
            results[task]["single_task"] = fn(X_tr, y_tr, X_te, y_te)
        except Exception as exc:
            results[task]["single_task"] = {"error": str(exc)}

    # ------------------------------------------------------------------
    # MTL variants (kendall, mgda, equal)
    # ------------------------------------------------------------------
    # Map internal task names to the MTL trainer's task names
    TASK_MAP = {
        "complexity": "complexity",
        "security":   "security",
        "bugs":       "bug",
        "patterns":   "pattern",
    }
    # Column names differ between tasks
    MTL_METRIC_MAP = {
        "complexity": {"rmse": "rmse", "spearman": "spearman"},   # ridge trainer
        "security":   {"auc": "auc",   "ap": "ap"},
        "bugs":       {"auc": "auc",   "ap": "ap"},
        "patterns":   {"accuracy": "accuracy", "f1_macro": "f1_macro"},
    }

    for strategy in ("kendall", "mgda", "equal"):
        logger.info("Training MTL strategy=%s ...", strategy)
        try:
            mtl_tasks = _train_mtl_variant(
                data_dir=data_dir,
                out_dir=out_dir,
                strategy=strategy,
                X_splits=splits,
                y_splits={},
            )
        except Exception as exc:
            logger.warning("MTL %s outer failure: %s", strategy, exc)
            mtl_tasks = {"error": str(exc)}

        strategy_key = f"mtl_{strategy}"
        for task in ("complexity", "security", "bugs", "patterns"):
            results.setdefault(task, {})
            mtl_name = TASK_MAP[task]
            try:
                task_m = _task_metrics_from_mtl(mtl_tasks, mtl_name)
                results[task][strategy_key] = task_m
            except Exception as exc:
                results[task][strategy_key] = {"error": str(exc)}

    return results


# ===========================================================================
# Output: JSON + table printing + LaTeX
# ===========================================================================

def _fmt(val, digits: int = 4) -> str:
    """Format a numeric value or return '-' for missing/error."""
    if val is None:
        return "-"
    try:
        return f"{float(val):.{digits}f}"
    except (TypeError, ValueError):
        return str(val)[:8]


# Task-specific column definitions
TASK_COLUMNS = {
    "complexity": [
        ("RMSE",     "rmse",     4),
        ("R2",       "r2",       4),
        ("Spearman", "spearman", 4),
    ],
    "security": [
        ("AUC", "auc", 4),
        ("F1",  "f1",  4),
        ("AP",  "ap",  4),
    ],
    "bugs": [
        ("AUC", "auc", 4),
        ("F1",  "f1",  4),
        ("AP",  "ap",  4),
    ],
    "patterns": [
        ("F1-macro", "f1_macro", 4),
        ("Accuracy", "accuracy", 4),
    ],
}

STRATEGIES = ["single_task", "mtl_kendall", "mtl_mgda", "mtl_equal"]


def _print_table(results: dict) -> None:
    """Print a plain-text ASCII comparison table (cp1252 safe)."""
    col_w = [12, 14, 10, 10, 10]
    sep   = "-" * 60

    print("\n" + sep)
    print(f"  MTL Comparison Results")
    print(sep)

    for task in ("complexity", "security", "bugs", "patterns"):
        cols = TASK_COLUMNS.get(task, [])
        col_names = [c[0] for c in cols]
        header = f"{'Task':<12} {'Strategy':<14} " + " ".join(f"{n:>10}" for n in col_names)
        print("\n" + header)
        print("-" * len(header))

        task_res = results.get(task, {})
        for strat in STRATEGIES:
            m = task_res.get(strat, {})
            if not m:
                continue
            if m.get("skipped"):
                row_vals = ["(skipped)"] + ["-"] * (len(cols) - 1)
            elif "error" in m:
                row_vals = [f"ERR:{m['error'][:8]}"] + ["-"] * (len(cols) - 1)
            else:
                row_vals = [_fmt(m.get(key), digits) for _, key, digits in cols]
            vals_str = " ".join(f"{v:>10}" for v in row_vals)
            print(f"  {task:<10} {strat:<14} {vals_str}")

    print("\n" + sep + "\n")


def _build_latex_table(results: dict) -> str:
    """Build a LaTeX table fragment comparing all strategies across all tasks."""
    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Single-task vs. MTL (Kendall / MGDA / Equal) comparison}")
    lines.append(r"\label{tab:mtl_comparison}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Task & Strategy & Metric 1 & Metric 2 & Metric 3 \\")
    lines.append(r"\midrule")

    LATEX_TASK_NAMES = {
        "complexity": "Complexity",
        "security":   "Security",
        "bugs":       "Bugs",
        "patterns":   "Patterns",
    }

    for task in ("complexity", "security", "bugs", "patterns"):
        cols = TASK_COLUMNS.get(task, [])
        task_res = results.get(task, {})
        for idx, strat in enumerate(STRATEGIES):
            m = task_res.get(strat, {})
            if not m:
                continue
            task_label = LATEX_TASK_NAMES.get(task, task) if idx == 0 else ""
            strat_label = strat.replace("_", r"\_")
            if m.get("skipped"):
                vals = ["---"] * len(cols)
            elif "error" in m:
                vals = [r"\textit{error}"] * len(cols)
            else:
                vals = [_fmt(m.get(key), digits) for _, key, digits in cols]
            # Pad to 3 columns
            while len(vals) < 3:
                vals.append("")
            row = f"  {task_label} & {strat_label} & " + " & ".join(vals[:3]) + r" \\"
            lines.append(row)
        lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ===========================================================================
# CLI entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare single-task vs. MTL-kendall vs. MTL-mgda",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", default=str(_BACKEND / "data"),
        help="Directory containing *_dataset.jsonl files",
    )
    parser.add_argument(
        "--out-dir", default=str(_BACKEND / "evaluation" / "results"),
        help="Output directory for JSON and LaTeX results",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Data dir : %s", data_dir)
    logger.info("Output   : %s", out_dir)

    # Run comparison
    results = run_comparison(data_dir=data_dir, out_dir=out_dir)

    # ------------------------------------------------------------------
    # Save JSON FIRST (before any print that could crash on encoding)
    # ------------------------------------------------------------------
    json_path = out_dir / "mtl_comparison.json"
    try:
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        logger.info("Results saved -> %s", json_path)
    except Exception as exc:
        logger.error("Failed to save JSON: %s", exc)

    # Save LaTeX
    latex_path = tables_dir / "mtl_comparison.tex"
    try:
        latex = _build_latex_table(results)
        with open(latex_path, "w", encoding="utf-8") as fh:
            fh.write(latex + "\n")
        logger.info("LaTeX  saved -> %s", latex_path)
    except Exception as exc:
        logger.error("Failed to save LaTeX: %s", exc)

    # ------------------------------------------------------------------
    # Print table (ASCII + cp1252 safe only)
    # ------------------------------------------------------------------
    try:
        _print_table(results)
    except Exception as exc:
        logger.error("Table print failed: %s", exc)

    return results


if __name__ == "__main__":
    main()
