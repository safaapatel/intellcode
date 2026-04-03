"""
Ablation study runner for all IntelliCode tasks.
Run from backend/ directory with the project venv.
"""
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("run_ablation")

from evaluation.cross_project_benchmark import (
    _prepare_complexity, _prepare_bug, _prepare_security,
    _prepare_pattern, _repo_from_record,
)
from evaluation.ablation_study import AblationStudy

DATA_DIR = Path("data")
OUT_DIR  = Path("evaluation/results")
TEX_DIR  = OUT_DIR / "tables"
TEX_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(path: Path) -> list:
    records = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    logger.info("Loaded %d records from %s", len(records), path)
    return records


# ---------------------------------------------------------------------------
# Feature groups per task (indices into X after task-specific preparation)
# ---------------------------------------------------------------------------

# Complexity X is 15-dim (cog_complexity dropped from index 1 of 16-dim vector)
# Remaining order: [CC(0), maxCC(1), avgCC(2), sloc(3), comments(4), blank(5),
#                   vol(6), diff(7), effort(8), bugs(9),
#                   n_long(10), n_complex(11), max_line(12), avg_line(13), over80(14)]
COMPLEXITY_GROUPS = {
    "cyclomatic":  [0, 1, 2],       # CC, maxCC, avgCC
    "size":        [3, 4, 5],       # sloc, comments, blank
    "halstead":    [6, 7, 8, 9],    # vol, diff, effort, bugs
    "functions":   [10, 11],        # n_long, n_complex
    "line_style":  [12, 13, 14],    # max_line, avg_line, over80
}

# Bug X is 30-dim (16 static + 14 JIT Kamei features)
# JIT order: NS,ND,NF,Entropy,LA,LD,LT,FIX,NDEV,AGE,NUC,EXP,REXP,SEXP
BUG_GROUPS = {
    "static_features": list(range(16)),
    "jit_diffusion":   [16, 17, 18, 19],    # NS, ND, NF, Entropy
    "jit_size_change": [20, 21, 22],         # LA, LD, LT
    "jit_purpose":     [23],                 # FIX
    "jit_history":     [24, 25, 26],         # NDEV, AGE, NUC
    "jit_experience":  [27, 28, 29],         # EXP, REXP, SEXP
}

# Security X is 16-dim (ast-derived features)
SECURITY_GROUPS = {
    "call_import":   [0, 1],         # n_calls, n_imports
    "control_flow":  [6, 7, 8, 9],   # n_try, n_returns, n_loops, n_conditionals
    "assignments":   [2, 3, 4, 5],   # n_assign, n_compare, n_augassign, n_exceptions
    "strings_bytes": [11, 12, 13],   # n_string_literals, n_bytes, n_lines
    "danger_ops":    [14, 15],        # has_eval, has_exec
}

# Pattern X is 15-dim (same as metrics_to_feature_vector output):
# [CC(0), maxCC(1), avgCC(2), sloc(3), comments(4), blank(5),
#  vol(6), diff(7), effort(8), bugs(9),
#  n_long(10), n_complex(11), max_line(12), avg_line(13), over80(14)]
PATTERN_GROUPS = {
    "cyclomatic":  [0, 1, 2],        # CC, maxCC, avgCC
    "size":        [3, 4, 5],        # sloc, comments, blank
    "halstead":    [6, 7, 8, 9],     # vol, diff, effort, bugs
    "functions":   [10, 11],         # n_long, n_complex
    "line_style":  [12, 13, 14],     # max_line, avg_line, over80
}


def run_task(task: str, records: list, groups: dict):
    logger.info("=" * 60)
    logger.info("ABLATION: %s  (%d records)", task.upper(), len(records))

    if task == "complexity":
        X, y, repos = _prepare_complexity(records)
    elif task == "bug":
        X, y, repos = _prepare_bug(records)
    elif task == "security":
        X, y, repos = _prepare_security(records)
    elif task == "pattern":
        X, y, repos = _prepare_pattern(records)
    else:
        raise ValueError(task)

    logger.info("X shape: %s  y shape: %s  unique repos: %d",
                X.shape, y.shape, len(set(repos)))

    if X.shape[0] < 30:
        logger.warning("Too few samples for %s ablation — skipping", task)
        return

    study = AblationStudy(task=task, n_jobs=1, random_state=42)
    report = study.run_feature_ablation(X, y, repos, groups, n_trials=5)

    # Also run data ablation (learning curve)
    data_report = study.run_data_ablation(X, y, repos,
                                           fractions=[0.1, 0.25, 0.5, 0.75, 1.0])

    # Save JSON first (so data is preserved even if print crashes)
    study.save_json(report,      str(OUT_DIR / f"ablation_{task}.json"))
    study.save_json(data_report, str(OUT_DIR / f"ablation_{task}_data.json"))
    study.save_latex_table(report,      str(TEX_DIR / f"ablation_{task}.tex"))
    study.save_latex_table(data_report, str(TEX_DIR / f"ablation_{task}_data.tex"))
    study.print_table(report)


def main():
    tasks = {
        "complexity": (DATA_DIR / "complexity_dataset.jsonl", COMPLEXITY_GROUPS),
        "bug":        (DATA_DIR / "bug_dataset.jsonl",        BUG_GROUPS),
        "security":   (DATA_DIR / "security_dataset.jsonl",   SECURITY_GROUPS),
        "pattern":    (DATA_DIR / "pattern_dataset.jsonl",    PATTERN_GROUPS),
    }

    for task, (data_path, groups) in tasks.items():
        if not data_path.exists():
            logger.warning("Dataset not found: %s — skipping %s", data_path, task)
            continue
        records = load_jsonl(data_path)
        try:
            run_task(task, records, groups)
        except Exception as e:
            logger.error("Ablation failed for %s: %s", task, e, exc_info=True)

    logger.info("Ablation study complete.")


if __name__ == "__main__":
    main()
