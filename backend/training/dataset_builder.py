"""
Dataset Builder
Utilities for mining GitHub repositories to build labeled training datasets
for all three ML models.

Requirements:
    pip install PyGithub pydriller pandas tqdm

Usage:
    python dataset_builder.py --task complexity --out data/complexity_dataset.jsonl
    python dataset_builder.py --task security  --out data/security_dataset.jsonl
    python dataset_builder.py --task pattern   --out data/pattern_dataset.jsonl
    python dataset_builder.py --task bug       --out data/bug_dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterator

# Add parent to path so features/ imports work when run from training/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# GitHub mining helpers
# ---------------------------------------------------------------------------

def iter_python_files(
    repo_names: list[str],
    github_token: str,
    max_files_per_repo: int = 500,
) -> Iterator[dict]:
    """
    Yield dicts with keys: repo, path, content, stars, language.
    Requires: pip install PyGithub
    """
    try:
        from github import Github
    except ImportError:
        raise RuntimeError("PyGithub not installed. pip install PyGithub")

    g = Github(github_token)
    for repo_name in repo_names:
        try:
            repo = g.get_repo(repo_name)
            contents = repo.get_contents("")
            count = 0
            queue = list(contents)
            while queue and count < max_files_per_repo:
                item = queue.pop(0)
                if item.type == "dir":
                    queue.extend(repo.get_contents(item.path))
                elif item.name.endswith(".py") and item.size < 100_000:
                    try:
                        source = item.decoded_content.decode("utf-8", errors="replace")
                        yield {
                            "repo": repo_name,
                            "path": item.path,
                            "content": source,
                            "stars": repo.stargazers_count,
                        }
                        count += 1
                    except Exception:
                        continue
        except Exception as e:
            print(f"[WARN] Skipping {repo_name}: {e}")
        time.sleep(0.5)  # respect rate limits


# ---------------------------------------------------------------------------
# Complexity dataset builder
# ---------------------------------------------------------------------------

def build_complexity_dataset(
    repo_names: list[str],
    github_token: str,
    output_path: str,
    max_per_repo: int = 500,
):
    """
    For each Python file, compute metrics and use maintainability_index
    (computed from Halstead + cyclomatic) as the training target.
    """
    from features.code_metrics import compute_all_metrics, metrics_to_feature_vector

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w") as out:
        for file_info in iter_python_files(repo_names, github_token, max_per_repo):
            source = file_info["content"]
            try:
                metrics = compute_all_metrics(source)
                feat_vec = metrics_to_feature_vector(metrics)
                target_score = metrics.maintainability_index

                record = {
                    "repo": file_info["repo"],
                    "path": file_info["path"],
                    "features": feat_vec,
                    "target": target_score,
                }
                out.write(json.dumps(record) + "\n")
                count += 1
                if count % 100 == 0:
                    print(f"  Processed {count} files...")
            except Exception:
                continue

    print(f"Complexity dataset: {count} samples → {output_path}")


# ---------------------------------------------------------------------------
# Security dataset builder
# ---------------------------------------------------------------------------

# High-quality repos known to have security vulnerabilities (or fixes)
SECURITY_POSITIVE_REPOS = [
    "juice-shop/juice-shop",          # intentionally vulnerable app
    "OWASP/WebGoat",
    "appsecco/dvna",                   # Damn Vulnerable Node App
]

SECURITY_NEGATIVE_REPOS = [
    "django/django",                   # well-audited codebase
    "pallets/flask",
    "psf/requests",
]


def build_security_dataset(
    positive_repos: list[str],
    negative_repos: list[str],
    github_token: str,
    output_path: str,
    max_per_repo: int = 300,
):
    """
    Label 1 (vulnerable) for intentionally-vulnerable repos,
    Label 0 (clean) for well-audited repos.

    Also uses the deterministic security scanner to auto-label borderline files.
    """
    from features.security_patterns import scan_security_patterns
    from features.ast_extractor import ASTExtractor, tokenize_code, tokens_to_ids

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w") as out:
        for base_label, repo_list in [(1, positive_repos), (0, negative_repos)]:
            for file_info in iter_python_files(repo_list, github_token, max_per_repo):
                source = file_info["content"]
                try:
                    ast_feats = ASTExtractor().extract(source)
                    tokens = tokenize_code(source)[:512]

                    # Override label with scanner for conservative repos
                    actual_label = base_label
                    if base_label == 0:
                        scan = scan_security_patterns(source)
                        if scan.critical:
                            actual_label = 1

                    record = {
                        "repo": file_info["repo"],
                        "path": file_info["path"],
                        "label": actual_label,
                        "tokens": tokens[:512],
                        "n_calls": ast_feats.get("n_calls", 0),
                        "n_imports": ast_feats.get("n_imports", 0),
                    }
                    out.write(json.dumps(record) + "\n")
                    count += 1
                    if count % 100 == 0:
                        print(f"  Processed {count} files...")
                except Exception:
                    continue

    print(f"Security dataset: {count} samples → {output_path}")


# ---------------------------------------------------------------------------
# Pattern recognition dataset builder
# ---------------------------------------------------------------------------

PATTERN_LABELS = {
    "code_smell": [
        "getsentry/sentry",       # large codebase, some smells
    ],
    "anti_pattern": [
        "minimaxir/textgenrnn",
    ],
    "clean": [
        "django/django",
        "pallets/flask",
        "psf/requests",
        "numpy/numpy",
    ],
}


def build_pattern_dataset(
    github_token: str,
    output_path: str,
    max_per_label: int = 500,
):
    """
    Build a JSONL dataset for CodeBERT fine-tuning.
    Format: {"code": "...", "label": "clean|code_smell|anti_pattern|style_violation"}

    Note: Precise labeling requires human annotation or static analysis tools (pylint).
    This builder uses heuristics + pylint scores to approximate labels.
    """
    from features.code_metrics import compute_all_metrics

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w") as out:
        for label, repos in PATTERN_LABELS.items():
            per_repo = max_per_label // len(repos)
            for file_info in iter_python_files(repos, github_token, per_repo):
                source = file_info["content"]
                try:
                    # Use function-level snippets for CodeBERT (short context)
                    import ast
                    tree = ast.parse(source)
                    lines = source.splitlines()

                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            start = node.lineno - 1
                            end = node.end_lineno or (start + 30)
                            snippet = "\n".join(lines[start:end])

                            # Refine label with metrics
                            actual_label = label
                            if label == "clean":
                                m = compute_all_metrics(snippet)
                                if m.cyclomatic_complexity > 10:
                                    actual_label = "code_smell"
                                elif m.n_lines_over_80 > 3:
                                    actual_label = "style_violation"

                            record = {
                                "code": snippet[:2000],
                                "label": actual_label,
                                "repo": file_info["repo"],
                                "path": file_info["path"],
                            }
                            out.write(json.dumps(record) + "\n")
                            count += 1
                except Exception:
                    continue

    print(f"Pattern dataset: {count} samples → {output_path}")


# ---------------------------------------------------------------------------
# Bug prediction dataset builder (uses PyDriller)
# ---------------------------------------------------------------------------

def build_bug_dataset(
    repo_urls: list[str],
    output_path: str,
    max_commits: int = 1000,
):
    """
    Mine commit history to find bug-fix commits.
    Files touched in bug-fix commits are labeled 1; other files labeled 0.

    Bug-fix detection: commit message contains keywords like 'fix', 'bug', 'issue'.
    """
    try:
        from pydriller import Repository
    except ImportError:
        raise RuntimeError("pydriller not installed. pip install pydriller")

    from features.code_metrics import compute_all_metrics, metrics_to_feature_vector
    from features.ast_extractor import ASTExtractor

    BUG_KEYWORDS = {"fix", "bug", "defect", "issue", "error", "crash", "patch", "repair"}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w") as out:
        for repo_url in repo_urls:
            print(f"Mining {repo_url}...")
            try:
                commits_processed = 0
                for commit in Repository(repo_url).traverse_commits():
                    if commits_processed >= max_commits:
                        break
                    commits_processed += 1

                    msg = commit.msg.lower()
                    is_bug_fix = any(kw in msg for kw in BUG_KEYWORDS)

                    for modified_file in commit.modified_files:
                        if not (modified_file.filename.endswith(".py")
                                and modified_file.source_code):
                            continue

                        source = modified_file.source_code
                        try:
                            metrics = compute_all_metrics(source)
                            ast_feats = ASTExtractor().extract(source)
                            feat_vec = metrics_to_feature_vector(metrics)

                            record = {
                                "repo": repo_url,
                                "file": modified_file.filename,
                                "commit": commit.hash[:8],
                                "label": 1 if is_bug_fix else 0,
                                "static_features": feat_vec,
                                "git_features": {
                                    "code_churn": (modified_file.added_lines or 0)
                                    + (modified_file.deleted_lines or 0),
                                    "author_count": 1,
                                    "file_age_days": 0,
                                    "n_past_bugs": 0,
                                    "commit_freq": 0.0,
                                },
                            }
                            out.write(json.dumps(record) + "\n")
                            count += 1
                        except Exception:
                            continue
            except Exception as e:
                print(f"[WARN] Error mining {repo_url}: {e}")

    print(f"Bug dataset: {count} samples → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build ML training datasets from GitHub")
    parser.add_argument("--task", choices=["complexity", "security", "pattern", "bug"],
                        required=True)
    parser.add_argument("--out", default="data/dataset.jsonl")
    parser.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""),
                        help="GitHub personal access token")
    parser.add_argument("--max", type=int, default=300,
                        help="Max files per repository")
    args = parser.parse_args()

    if args.task in ("complexity", "security", "pattern") and not args.token:
        print("ERROR: --token or GITHUB_TOKEN env var required for GitHub mining")
        sys.exit(1)

    if args.task == "complexity":
        repos = [
            "django/django", "pallets/flask", "psf/requests",
            "numpy/numpy", "pandas-dev/pandas", "scikit-learn/scikit-learn",
        ]
        build_complexity_dataset(repos, args.token, args.out, args.max)

    elif args.task == "security":
        build_security_dataset(
            positive_repos=SECURITY_POSITIVE_REPOS,
            negative_repos=SECURITY_NEGATIVE_REPOS,
            github_token=args.token,
            output_path=args.out,
            max_per_repo=args.max,
        )

    elif args.task == "pattern":
        build_pattern_dataset(args.token, args.out, max_per_label=args.max)

    elif args.task == "bug":
        bug_repos = [
            "https://github.com/django/django",
            "https://github.com/pallets/flask",
            "https://github.com/psf/requests",
        ]
        build_bug_dataset(bug_repos, args.out, max_commits=args.max)


if __name__ == "__main__":
    main()
