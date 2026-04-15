"""
Train Code Grammar Anomaly Model (CGAM)
========================================
Trains the Variable-Order Markov Model over DFS-order AST node sequences
on UNLABELED clean Python code.

Usage:
    cd backend
    python training/train_cgam.py --data data/complexity_dataset.jsonl
    python training/train_cgam.py --repos data/cgam_sources.jsonl
    python training/train_cgam.py --n 3000   # fetch from clean repos directly

Data sources (priority order):
    1. --data   : existing complexity dataset (re-uses source fields if stored)
    2. --repos  : JSONL of {content: "..."} records
    3. --n      : fetch from SECURITY_NEGATIVE_REPOS using GitHub API

Outputs:
    checkpoints/cgam/vomm.pkl
    checkpoints/cgam/calibration.json
    checkpoints/cgam/metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Clean repos to train on if no dataset provided
CLEAN_REPOS = [
    "django/django",
    "pallets/flask",
    "psf/requests",
    "sqlalchemy/sqlalchemy",
    "encode/httpx",
    "pyca/cryptography",
    "pytest-dev/pytest",
    "psf/black",
    "python-attrs/attrs",
    "pydantic/pydantic",
    "tiangolo/fastapi",
    "aio-libs/aiohttp",
    "pallets/werkzeug",
]


def load_sources_from_dataset(data_path: str) -> list[str]:
    """Load Python source strings from a JSONL dataset file.

    Accepts complexity, pattern, or any JSONL with a 'code' or 'content' field.
    """
    sources = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            src = rec.get("code") or rec.get("content") or rec.get("source", "")
            if src and len(src) > 30:
                sources.append(src)
    logger.info("Loaded %d source strings from %s", len(sources), data_path)
    return sources


def load_sources_from_repos_jsonl(repos_path: str) -> list[str]:
    """Load source strings from a JSONL of {content: ...} records."""
    sources = []
    with open(repos_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            src = rec.get("content") or rec.get("code", "")
            if src and len(src) > 30:
                sources.append(src)
    logger.info("Loaded %d source strings from %s", len(sources), repos_path)
    return sources


def fetch_sources_from_github(n: int, token: str) -> list[str]:
    """Fetch Python files from curated clean repos."""
    from training.dataset_builder import iter_python_files

    sources = []
    per_repo = max(n // len(CLEAN_REPOS), 20)
    for file_info in iter_python_files(CLEAN_REPOS, token, per_repo):
        sources.append(file_info["content"])
        if len(sources) >= n:
            break

    logger.info("Fetched %d files from GitHub", len(sources))
    return sources


def train(
    sources: list[str],
    output_dir: str = "checkpoints/cgam",
) -> dict:
    """Train CGAM on source strings and save checkpoint."""
    from models.code_grammar_anomaly import CodeGrammarAnomalyModel

    if len(sources) < 10:
        raise ValueError(f"Need at least 10 source files; got {len(sources)}")

    model = CodeGrammarAnomalyModel()
    metrics = model.fit(sources, output_dir=output_dir)

    # Quick evaluation: score 20% held-out as anomaly check
    n_eval = max(1, len(sources) // 5)
    eval_sources = sources[-n_eval:]
    scores = [model.predict(s).anomaly_score for s in eval_sources]

    import numpy as np
    metrics["eval_anomaly_score_mean"] = round(float(np.mean(scores)), 4)
    metrics["eval_anomaly_score_p95"]  = round(float(np.percentile(scores, 95)), 4)
    metrics["n_eval_files"] = n_eval

    # Overwrite metrics with eval results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(
        "CGAM training complete: vocab=%d  trigrams=%d  eval_mean_score=%.4f",
        metrics["vocab_size"], metrics["n_trigrams"],
        metrics["eval_anomaly_score_mean"],
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train the Code Grammar Anomaly Model (CGAM)"
    )
    parser.add_argument("--data", help="Path to existing JSONL dataset (any task)")
    parser.add_argument("--repos", help="Path to JSONL of {content: ...} records")
    parser.add_argument("--n", type=int, default=3000,
                        help="Max files to fetch from GitHub if no --data (default 3000)")
    parser.add_argument("--out", default="checkpoints/cgam",
                        help="Checkpoint output directory")
    args = parser.parse_args()

    sources: list[str] = []

    if args.data and Path(args.data).exists():
        sources = load_sources_from_dataset(args.data)
    elif args.repos and Path(args.repos).exists():
        sources = load_sources_from_repos_jsonl(args.repos)
    else:
        token = os.environ.get("GITHUB_TOKEN", "")
        if not token:
            # Try to use existing complexity or pattern dataset
            for fallback in ["data/complexity_dataset.jsonl", "data/pattern_dataset.jsonl"]:
                if Path(fallback).exists():
                    logger.info("No GitHub token — using fallback dataset: %s", fallback)
                    sources = load_sources_from_dataset(fallback)
                    break
            if not sources:
                logger.error(
                    "No data source available. Set GITHUB_TOKEN or provide --data path."
                )
                sys.exit(1)
        else:
            sources = fetch_sources_from_github(args.n, token)

    metrics = train(sources, output_dir=args.out)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
