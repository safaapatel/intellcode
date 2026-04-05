"""
Dataset Expansion Script
========================
Expands all four training datasets using locally cached repositories.
No GitHub API calls required — uses data/repos_cache/*.

Expansion targets:
  complexity : 875  -> 3,500+ (mine all .py files in repos_cache)
  security   : 1286 -> 2,500+ (add prod-code negative samples)
  bug        : 1200 -> 2,000+ (additional pydriller commit mining)
  pattern    : 1153 -> 2,000+ (more function-level heuristic labels)

Usage:
    cd backend
    python expand_datasets.py [--tasks all|complexity|security|bug|pattern]
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import os
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = DATA / "repos_cache"

sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _iter_py_files(base: Path):
    for fpath in base.rglob("*.py"):
        try:
            src = fpath.read_text(encoding="utf-8", errors="replace")
            repo = fpath.relative_to(CACHE).parts[0]
            yield fpath, src, repo
        except Exception:
            continue


def _load_existing(path: Path) -> set[str]:
    """Return set of origin strings already in the dataset."""
    seen = set()
    if not path.exists():
        return seen
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
                orig = d.get("origin", "")
                if orig:
                    seen.add(orig)
            except Exception:
                continue
    return seen


def _append_records(path: Path, records: list[dict]) -> int:
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")
    return len(records)


# ---------------------------------------------------------------------------
# Complexity expansion
# ---------------------------------------------------------------------------

def expand_complexity(max_new: int = 4000) -> int:
    from features.code_metrics import compute_all_metrics, metrics_to_feature_vector

    out_path = DATA / "complexity_dataset.jsonl"
    existing = _load_existing(out_path)
    logger.info("Complexity: %d existing records", len(existing))

    new_records = []
    for fpath, src, repo in _iter_py_files(CACHE):
        if len(new_records) >= max_new:
            break
        rel = str(fpath.relative_to(CACHE))
        if rel in existing:
            continue

        try:
            metrics = compute_all_metrics(src)
            feat = metrics_to_feature_vector(metrics)

            # metrics_to_feature_vector returns 15-dim (cog excluded)
            # We need to add cognitive_complexity as the target (not in feat)
            cog = float(getattr(metrics, "cognitive_complexity", 0))
            if not math.isfinite(cog):
                continue
            if not all(math.isfinite(v) for v in feat):
                continue

            # Old format: 17-dim [cc, cog, maxCC, avgCC, sloc, comments, blank,
            #   vol, diff, effort, bugs, nlong, ncomplex, maxline, avgline, over80, MI]
            # train_complexity.py: target=feats[1]=cog, input=feats[0]+feats[2:16]
            mi = float(getattr(metrics, "maintainability_index", 0.0))
            feat17 = [float(feat[0]), cog] + [float(v) for v in feat[1:]] + [mi]
            if len(feat17) != 17:
                continue
            new_records.append({
                "features": feat17,
                "target":   cog,
                "origin":   rel,
            })
        except Exception:
            continue

    added = _append_records(out_path, new_records)
    logger.info("Complexity: added %d records (total now ~%d)", added, len(existing) + added)
    return added


# ---------------------------------------------------------------------------
# Security expansion (negative samples from production code)
# ---------------------------------------------------------------------------

_DANGER_PATTERNS = [
    r"\beval\s*\(",
    r"\bexec\s*\(",
    r"pickle\.loads?\s*\(",
    r"yaml\.load\s*\(",
    r"subprocess\.(call|Popen|run)\s*\(",
    r"os\.(system|popen)\s*\(",
    r"__import__\s*\(",
    r"hashlib\.(md5|sha1)\s*\(",
    r"random\.random\s*\(",
]
_DANGER_RE = re.compile("|".join(_DANGER_PATTERNS))


def _extract_functions(src: str) -> list[str]:
    """Extract function bodies from a Python source file."""
    funcs = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return funcs
    lines = src.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                start = node.lineno - 1
                end = node.end_lineno if hasattr(node, "end_lineno") else start + 30
                body = "\n".join(lines[start:end])
                if len(body.strip()) > 50:
                    funcs.append(body)
            except Exception:
                continue
    return funcs


def _security_features_from_source(src: str) -> list[float]:
    """Build minimal 16-dim security feature vector from source text."""
    try:
        from models.security_detection import _build_rf_feature_vector
        feat = _build_rf_feature_vector(src)
        if len(feat) < 16:
            feat = list(feat) + [0.0] * (16 - len(feat))
        return [float(v) for v in feat[:16]]
    except Exception:
        # Fallback: manual feature extraction
        tokens = src.split()
        return [
            float(len(tokens)),
            float(src.count("def ")),
            float(src.count("import ")),
            float(src.count("(")),
            float(src.count(".")),
            float(len(src.splitlines())),
            float(src.count("try:")),
            float(src.count("except")),
            float(src.count("return")),
            float(src.count("raise")),
            float(src.count("os.")),
            float(src.count("subprocess")),
            float(src.count("eval")),
            float(src.count("exec")),
            float(src.count("open(")),
            float(len(set(tokens))),
        ]


def _tokenize(src: str, max_len: int = 512) -> list[str]:
    return re.findall(r"[a-zA-Z_]\w*|[0-9]+|[^\s\w]", src)[:max_len]


def expand_security(max_new_neg: int = 1500) -> int:
    out_path = DATA / "security_dataset_filtered.jsonl"
    existing = _load_existing(out_path)
    logger.info("Security: %d existing records", len(existing))

    new_records = []
    # Add clean production-code functions as negatives
    for fpath, src, repo in _iter_py_files(CACHE):
        if len(new_records) >= max_new_neg:
            break

        funcs = _extract_functions(src)
        for i, func_src in enumerate(funcs):
            if len(new_records) >= max_new_neg:
                break

            origin = f"{fpath.relative_to(CACHE)}::func{i}"
            if origin in existing:
                continue

            # Skip if function contains dangerous patterns (may be false negative)
            if _DANGER_RE.search(func_src):
                continue

            tokens = _tokenize(func_src)
            if len(tokens) < 10:
                continue

            try:
                feats = _security_features_from_source(func_src)
                new_records.append({
                    "label":    0,
                    "source":   func_src[:2000],
                    "tokens":   tokens[:512],
                    "n_calls":  func_src.count("("),
                    "n_imports": func_src.count("import "),
                    "origin":   origin,
                    "data_source": "expansion_prod",
                })
            except Exception:
                continue

    added = _append_records(out_path, new_records)
    logger.info("Security: added %d negative records (total ~%d)", added, len(existing) + added)
    return added


# ---------------------------------------------------------------------------
# Bug expansion via pydriller
# ---------------------------------------------------------------------------

def expand_bug(max_new: int = 1000) -> int:
    try:
        import pydriller
    except ImportError:
        logger.warning("pydriller not installed — skipping bug expansion")
        return 0

    from features.code_metrics import compute_all_metrics, metrics_to_feature_vector

    out_path = DATA / "bug_dataset.jsonl"
    existing_count = sum(1 for _ in open(out_path, encoding="utf-8", errors="replace"))
    logger.info("Bug: %d existing records", existing_count)

    new_records = []

    # Mine commit history from cached repos that have .git directories
    BUG_FIX_KEYWORDS = re.compile(
        r"\b(fix|bug|error|crash|fault|defect|patch|correct|repair|resolve)\b",
        re.IGNORECASE,
    )

    for repo_dir in sorted(CACHE.iterdir()):
        if not (repo_dir / ".git").exists():
            continue
        if len(new_records) >= max_new:
            break

        repo_name = repo_dir.name
        logger.info("  Mining %s ...", repo_name)

        try:
            from pydriller import Repository
            commits = list(Repository(
                str(repo_dir),
                only_modifications_with_file_types=[".py"],
                order="reverse",
            ).traverse_commits())
        except Exception as e:
            logger.warning("  pydriller failed on %s: %s", repo_name, e)
            continue

        for commit in commits[:300]:  # limit per repo
            if len(new_records) >= max_new:
                break
            try:
                is_fix = bool(BUG_FIX_KEYWORDS.search(commit.msg))
                for mod in commit.modified_files:
                    if not mod.filename.endswith(".py"):
                        continue
                    source = mod.source_code or mod.source_code_before or ""
                    if not source or len(source) < 50:
                        continue
                    try:
                        metrics = compute_all_metrics(source)
                        static_feat = list(metrics_to_feature_vector(metrics))
                    except Exception:
                        continue

                    # Minimal git features (partial — no full REXP etc.)
                    git_feat = [
                        float(mod.added_lines),
                        float(mod.deleted_lines),
                        float(commit.insertions),
                        float(commit.deletions),
                        float(len(list(commit.modified_files))),
                        float(len(commit.msg)),
                        float(is_fix),
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # pad to 14
                    ]
                    git_feat = git_feat[:14]

                    new_records.append({
                        "label":          int(is_fix),
                        "static_features": static_feat,
                        "git_features":    git_feat,
                        "origin":         f"{repo_name}/{mod.filename}/{commit.hash[:8]}",
                    })
            except Exception:
                continue

    added = _append_records(out_path, new_records)
    logger.info("Bug: added %d records (total ~%d)", added, existing_count + added)
    return added


# ---------------------------------------------------------------------------
# Pattern expansion
# ---------------------------------------------------------------------------

_SMELL_SLOC_THRESHOLD  = 30    # long method
_SMELL_PARAMS_THRESHOLD = 5    # long parameter list
_SMELL_CALLS_THRESHOLD  = 8    # feature envy proxy

def _detect_smells(src: str) -> str:
    """Return smell label: 'clean', 'long_method', 'feature_envy', 'long_params'."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return "clean"

    lines = [l for l in src.splitlines() if l.strip()]
    sloc = len(lines)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Long method
            if sloc > _SMELL_SLOC_THRESHOLD:
                return "long_method"
            # Long parameter list
            n_params = len(node.args.args) + len(node.args.posonlyargs)
            if n_params > _SMELL_PARAMS_THRESHOLD:
                return "long_params"
    # Feature envy: high density of attribute accesses to external objects
    calls = src.count(".")
    if calls > _SMELL_CALLS_THRESHOLD:
        return "feature_envy"
    return "clean"


_LABEL_MAP = {"clean": 0, "long_method": 1, "feature_envy": 2, "long_params": 3}


def expand_pattern(max_new: int = 1500) -> int:
    out_path = DATA / "pattern_dataset.jsonl"
    existing = _load_existing(out_path)
    logger.info("Pattern: %d existing records", len(existing))

    new_records = []
    label_counts = {k: 0 for k in _LABEL_MAP}

    for fpath, src, repo in _iter_py_files(CACHE):
        if len(new_records) >= max_new:
            break

        funcs = _extract_functions(src)
        for i, func_src in enumerate(funcs):
            if len(new_records) >= max_new:
                break

            origin = f"{fpath.relative_to(CACHE)}::func{i}"
            if origin in existing:
                continue
            if len(func_src.strip()) < 30:
                continue

            smell = _detect_smells(func_src)
            label_int = _LABEL_MAP[smell]

            # Balance: don't let clean dominate
            if smell == "clean" and label_counts["clean"] > label_counts.get("long_method", 0) * 3:
                continue

            new_records.append({
                "code":   func_src[:3000],
                "label":  label_int,
                "origin": origin,
            })
            label_counts[smell] = label_counts.get(smell, 0) + 1

    added = _append_records(out_path, new_records)
    logger.info(
        "Pattern: added %d records (total ~%d). Labels: %s",
        added, len(existing) + added, label_counts,
    )
    return added


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+",
                    default=["complexity", "security", "bug", "pattern"],
                    choices=["all", "complexity", "security", "bug", "pattern"])
    ap.add_argument("--max-new", type=int, default=3000,
                    help="Max new records per dataset")
    args = ap.parse_args()

    tasks = args.tasks
    if "all" in tasks:
        tasks = ["complexity", "security", "bug", "pattern"]

    totals = {}

    if "complexity" in tasks:
        totals["complexity"] = expand_complexity(max_new=args.max_new)

    if "security" in tasks:
        totals["security"] = expand_security(max_new_neg=args.max_new)

    if "bug" in tasks:
        totals["bug"] = expand_bug(max_new=args.max_new)

    if "pattern" in tasks:
        totals["pattern"] = expand_pattern(max_new=args.max_new)

    print("\n=== EXPANSION SUMMARY ===")
    for task, n in totals.items():
        print(f"  {task:12s}: +{n} new records")

    # Print final counts
    print("\n=== FINAL DATASET SIZES ===")
    for fname in ["complexity_dataset.jsonl", "security_dataset_filtered.jsonl",
                  "bug_dataset.jsonl", "pattern_dataset.jsonl"]:
        fpath = DATA / fname
        if fpath.exists():
            n = sum(1 for _ in open(fpath, encoding="utf-8", errors="replace"))
            print(f"  {fname:40s}: {n}")


if __name__ == "__main__":
    main()
