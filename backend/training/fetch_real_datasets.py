"""
Real Dataset Fetcher for IntelliCode ML Models

Strategy (Windows-friendly, no cloning needed for most tasks):

Security  → Bandit test-case files (explicitly labelled vulnerable Python)
            + clean code from well-audited repos via direct GitHub raw API.
            Auto-labels clean files with pattern scanner.

Complexity → Real Python files downloaded via GitHub raw API from popular
             open-source projects. Target = maintainability_index.

Bug        → Real bug-fix commits via PyDriller (commit history required).
             Uses a persistent cache dir to avoid Windows cleanup issues.

Pattern    → Real Python functions from GitHub raw API; labelled by
             code-quality heuristics (complexity, depth, line length).

Usage (cd backend first):
    python training/fetch_real_datasets.py --all --out data/
    python training/fetch_real_datasets.py --task security   --out data/
    python training/fetch_real_datasets.py --task complexity --out data/
    python training/fetch_real_datasets.py --task bug        --out data/
    python training/fetch_real_datasets.py --task pattern    --out data/
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterator

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_jsonl(samples: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"  Saved {len(samples)} records -> {path}")


def _raw_github(owner: str, repo: str, path: str, ref: str = "main") -> str | None:
    """Download a single raw file from GitHub (no token, no cloning)."""
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200 and r.text.strip():
            return r.text
    except Exception:
        pass
    return None


def _github_tree(
    owner: str, repo: str, ref: str = "main", extension: str = ".py",
) -> list[str]:
    """Return all file paths with given extension in a GitHub repo tree (no auth)."""
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{ref}?recursive=1"
    try:
        r = requests.get(url, timeout=20,
                         headers={"Accept": "application/vnd.github.v3+json"})
        if r.status_code == 200:
            return [
                item["path"] for item in r.json().get("tree", [])
                if item["type"] == "blob"
                and item["path"].endswith(extension)
                and item.get("size", 0) < 80_000
            ]
    except Exception:
        pass
    return []


def _iter_repo_files(
    repo_list: list[tuple[str, str, str]],   # (owner, repo, ref)
    max_per_repo: int = 150,
) -> Iterator[tuple[str, str]]:
    """Yield (source_code, origin_tag) for Python files from GitHub API."""
    for owner, repo, ref in repo_list:
        print(f"  Downloading {owner}/{repo}@{ref} …")
        paths = _github_tree(owner, repo, ref)
        random.shuffle(paths)
        count = 0
        for path in paths:
            if count >= max_per_repo:
                break
            source = _raw_github(owner, repo, path, ref)
            if not source or len(source) < 50 or len(source) > 150_000:
                continue
            yield source, f"{owner}/{repo}/{path}"
            count += 1
            time.sleep(0.05)   # respect GitHub rate limits


# ─────────────────────────────────────────────────────────────────────────────
# Bandit labelled examples
# ─────────────────────────────────────────────────────────────────────────────

# Vulnerable example files from PyCQA/bandit — verified real paths on main branch
BANDIT_VULN_FILES = [
    "examples/binding.py",
    "examples/cipher-modes.py",
    "examples/ciphers.py",
    "examples/crypto-md5.py",
    "examples/dill.py",
    "examples/django_sql_injection_extra.py",
    "examples/django_sql_injection_raw.py",
    "examples/eval.py",
    "examples/exec.py",
    "examples/flask_debug.py",
    "examples/ftplib.py",
    "examples/hardcoded-passwords.py",
    "examples/hardcoded-tmp.py",
    "examples/hashlib_new_insecure_functions.py",
    "examples/huggingface_unsafe_download.py",
    "examples/jinja2_templating.py",
    "examples/jsonpickle.py",
    "examples/logging_config_insecure_listen.py",
    "examples/mako_templating.py",
    "examples/mark_safe_insecure.py",
    "examples/markupsafe_markup_xss.py",
    "examples/markupsafe_markup_xss_extend_markup_names.py",
    "examples/marshal_deserialize.py",
    "examples/mktemp.py",
    "examples/no_host_key_verification.py",
    "examples/os-chmod.py",
    "examples/os-exec.py",
    "examples/os-popen.py",
    "examples/os-spawn.py",
    "examples/os-startfile.py",
    "examples/os_system.py",
    "examples/pandas_read_pickle.py",
    "examples/paramiko_injection.py",
    "examples/partial_path_process.py",
    "examples/pickle_deserialize.py",
    "examples/popen_wrappers.py",
    "examples/pycrypto.py",
    "examples/pycryptodome.py",
    "examples/pytorch_load.py",
    "examples/random_module.py",
    "examples/requests-missing-timeout.py",
    "examples/requests-ssl-verify-disabled.py",
    "examples/shelve_open.py",
    "examples/snmp.py",
    "examples/sql_multiline_statements.py",
    "examples/sql_statements.py",
    "examples/ssl-insecure-version.py",
    "examples/subprocess_shell.py",
    "examples/tarfile_extractall.py",
    "examples/telnetlib.py",
    "examples/try_except_pass.py",
    "examples/unverified_context.py",
    "examples/urlopen.py",
    "examples/weak_cryptographic_key_sizes.py",
    "examples/wildcard-injection.py",
    "examples/xml_etree_celementtree.py",
    "examples/xml_etree_elementtree.py",
    "examples/xml_expatbuilder.py",
    "examples/xml_expatreader.py",
    "examples/xml_minidom.py",
    "examples/xml_pulldom.py",
    "examples/xml_sax.py",
    "examples/xml_xmlrpc.py",
    "examples/yaml_load.py",
]

# Explicitly safe files from Bandit
BANDIT_CLEAN_FILES = [
    "examples/okay.py",
    "examples/mark_safe_secure.py",
    "examples/markupsafe_markup_xss_allowed_calls.py",
    "examples/new_candidates-none.py",
    "examples/init-py-test/subdirectory-okay.py",
]


def _fetch_bandit_examples() -> list[dict]:
    """Download Bandit's labelled security examples via raw GitHub API."""
    from features.ast_extractor import ASTExtractor, tokenize_code

    samples = []

    print("  Downloading Bandit vulnerable examples …")
    for path in tqdm(BANDIT_VULN_FILES, desc="  vuln"):
        source = _raw_github("PyCQA", "bandit", path)
        if not source or len(source) < 20:
            continue
        try:
            ast_feats = ASTExtractor().extract(source)
            tokens = tokenize_code(source)[:512]
            samples.append({
                "label": 1,
                "source": source,
                "tokens": tokens,
                "n_calls": ast_feats.get("n_calls", 0),
                "n_imports": ast_feats.get("n_imports", 0),
                "origin": f"bandit/{path}",
            })
        except Exception:
            continue

    print("  Downloading Bandit clean examples …")
    for path in tqdm(BANDIT_CLEAN_FILES, desc="  clean"):
        source = _raw_github("PyCQA", "bandit", path)
        if not source or len(source) < 20:
            continue
        try:
            ast_feats = ASTExtractor().extract(source)
            tokens = tokenize_code(source)[:512]
            samples.append({
                "label": 0,
                "source": source,
                "tokens": tokens,
                "n_calls": ast_feats.get("n_calls", 0),
                "n_imports": ast_feats.get("n_imports", 0),
                "origin": f"bandit/{path}",
            })
        except Exception:
            continue

    n_v = sum(1 for s in samples if s["label"] == 1)
    n_c = sum(1 for s in samples if s["label"] == 0)
    print(f"  Bandit: {n_v} vulnerable, {n_c} clean")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# CVE / security-fix commit mining
# ─────────────────────────────────────────────────────────────────────────────

# Repos with well-documented security patches in commit history
SECURITY_CVE_REPOS = [
    # Original web frameworks
    "https://github.com/django/django",
    "https://github.com/psf/requests",
    "https://github.com/paramiko/paramiko",
    "https://github.com/pypa/pip",
    "https://github.com/pallets/flask",
    "https://github.com/aio-libs/aiohttp",
    "https://github.com/encode/httpx",
    "https://github.com/tornadoweb/tornado",
    # High-CVE-count Python projects
    "https://github.com/urllib3/urllib3",
    "https://github.com/pallets/jinja",
    "https://github.com/python-pillow/Pillow",
    "https://github.com/lxml/lxml",
    "https://github.com/sqlalchemy/sqlalchemy",
    "https://github.com/celery/celery",
    "https://github.com/scrapy/scrapy",
    "https://github.com/encode/starlette",
    "https://github.com/yaml/pyyaml",
    "https://github.com/marshmallow-code/marshmallow",
    # Infrastructure / DevOps (broader domain coverage)
    "https://github.com/ansible/ansible",
    "https://github.com/saltstack/salt",
    "https://github.com/apache/airflow",
    "https://github.com/buildbot/buildbot",
    # Scientific / data engineering
    "https://github.com/sympy/sympy",
    "https://github.com/networkx/networkx",
    "https://github.com/pydata/xarray",
    # CLI / packaging tools
    "https://github.com/pypa/pipenv",
    "https://github.com/python-poetry/poetry",
]

SECURITY_CVE_KEYWORDS = {
    "cve", "security", "vulnerability", "injection", "xss", "csrf",
    "auth", "overflow", "bypass", "privilege", "exploit", "rce",
    "sqli", "ssrf", "directory traversal", "path traversal", "exec",
}


def _fetch_cve_security_samples(max_per_repo: int = 80) -> list[dict]:
    """
    Mine security-fix commits from Python repos via PyDriller.
    - source_code_before a security-fix commit  → label 1 (vulnerable)
    - source_code from non-security commits      → label 0 (clean)
    """
    try:
        from pydriller import Repository
    except ImportError:
        print("  [SKIP] pydriller not installed — skipping CVE mining")
        return []

    from features.ast_extractor import ASTExtractor, tokenize_code

    cache_dir = Path(__file__).resolve().parent.parent / "data" / "repos_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    seen: set[int] = set()

    for repo_url in SECURITY_CVE_REPOS:
        repo_name = repo_url.rstrip("/").split("/")[-1]
        clone_path = cache_dir / repo_name
        print(f"  CVE mining {repo_url} …")

        try:
            if clone_path.exists():
                repo_iter = Repository(str(clone_path), order="date")
            else:
                repo_iter = Repository(repo_url, clone_repo_to=str(cache_dir), order="date")

            vuln_count = 0
            clean_count = 0
            processed = 0

            for commit in repo_iter.traverse_commits():
                if processed >= 500 or (vuln_count >= max_per_repo and clean_count >= max_per_repo):
                    break

                msg = commit.msg.lower()
                is_security = any(kw in msg for kw in SECURITY_CVE_KEYWORDS)

                for mf in commit.modified_files:
                    if not (mf.filename or "").endswith(".py"):
                        continue

                    if is_security:
                        # Pre-fix code = vulnerable
                        src = mf.source_code_before
                        label = 1
                        if vuln_count >= max_per_repo:
                            continue
                    else:
                        # Post-commit code = clean
                        src = mf.source_code
                        label = 0
                        if clean_count >= max_per_repo:
                            continue

                    if not src or len(src) < 50 or len(src) > 150_000:
                        continue
                    h = hash(src)
                    if h in seen:
                        continue
                    seen.add(h)

                    try:
                        ast_feats = ASTExtractor().extract(src)
                        tokens = tokenize_code(src)[:512]
                        samples.append({
                            "label": label,
                            "source": src[:8000],
                            "tokens": tokens,
                            "n_calls": ast_feats.get("n_calls", 0),
                            "n_imports": ast_feats.get("n_imports", 0),
                            "origin": f"cve_mining/{repo_name}/{mf.filename}",
                        })
                        if label == 1:
                            vuln_count += 1
                        else:
                            clean_count += 1
                    except Exception:
                        continue

                processed += 1

            print(f"  CVE {repo_name}: {vuln_count} vuln, {clean_count} clean")

        except Exception as e:
            print(f"  [WARN] CVE mining {repo_url}: {e}")

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 1. SECURITY DATASET
# ─────────────────────────────────────────────────────────────────────────────

# Well-audited repos for clean samples (no cloning — direct GitHub API)
SECURITY_CLEAN_REPOS = [
    # Web frameworks (original)
    ("psf", "requests", "main"),
    ("pallets", "flask", "main"),
    ("pallets", "click", "main"),
    ("bottlepy", "bottle", "master"),
    ("encode", "httpx", "master"),
    ("tqdm", "tqdm", "master"),
    ("pypa", "pip", "main"),
    # Well-audited security-conscious libraries
    ("urllib3", "urllib3", "main"),
    ("certifi", "python-certifi", "master"),
    ("pyca", "cryptography", "main"),
    ("paramiko", "paramiko", "main"),
    # Async / networking
    ("aio-libs", "aiohttp", "master"),
    ("encode", "starlette", "master"),
    ("tiangolo", "fastapi", "master"),
    # Data handling
    ("pydantic", "pydantic", "main"),
    ("pallets", "jinja", "main"),
    ("marshmallow-code", "marshmallow", "dev"),
    # Scientific computing (expands beyond web-framework domain)
    ("sympy", "sympy", "master"),
    ("networkx", "networkx", "main"),
    ("scikit-learn", "scikit-learn", "main"),
    # Infrastructure / DevOps (highest domain-shift diversity)
    ("pypa", "pipenv", "main"),
    ("pytest-dev", "pytest", "main"),
    ("PyCQA", "pylint", "main"),
]


def fetch_security_dataset(n: int = 3000, out: str = "data/security_dataset.jsonl"):
    """
    Real security dataset:
      - Bandit's own labelled examples (real vulnerable Python, ~63 files)
      - Clean Python files from well-audited repos (auto-labelled with scanner)
    """
    print("\n=== Security Dataset (Real Sources) ===")
    from features.ast_extractor import ASTExtractor, tokenize_code
    from features.security_patterns import scan_security_patterns

    samples = _fetch_bandit_examples()
    print("  Mining CVE-fix commits for additional vulnerable samples …")
    cve_samples = _fetch_cve_security_samples(max_per_repo=80)
    samples.extend(cve_samples)
    print(f"  After CVE mining: {len(samples)} total samples")

    # Download clean Python code from well-audited repos
    n_clean_needed = max(n - len(samples), len([s for s in samples if s["label"] == 1]) * 2)
    max_per_repo = max(30, n_clean_needed // len(SECURITY_CLEAN_REPOS))

    print(f"  Downloading clean Python files (target {n_clean_needed}) …")
    for source, origin in _iter_repo_files(SECURITY_CLEAN_REPOS, max_per_repo):
        try:
            # Re-label with scanner — clean file might still have issues
            scan = scan_security_patterns(source)
            label = 1 if scan.critical else 0
            ast_feats = ASTExtractor().extract(source)
            tokens = tokenize_code(source)[:512]
            samples.append({
                "label": label,
                "source": source[:8000],
                "tokens": tokens,
                "n_calls": ast_feats.get("n_calls", 0),
                "n_imports": ast_feats.get("n_imports", 0),
                "origin": origin,
            })
        except Exception:
            continue

    random.shuffle(samples)
    _write_jsonl(samples, out)
    n_v = sum(1 for s in samples if s["label"] == 1)
    n_c = sum(1 for s in samples if s["label"] == 0)
    print(f"  Security: {len(samples)} total ({n_v} vulnerable, {n_c} clean)")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 2. COMPLEXITY DATASET
# ─────────────────────────────────────────────────────────────────────────────

COMPLEXITY_REPOS = [
    # Web frameworks (original)
    ("psf", "requests", "main"),
    ("pallets", "flask", "main"),
    ("pallets", "click", "main"),
    ("bottlepy", "bottle", "master"),
    ("encode", "httpx", "master"),
    ("pypa", "pip", "main"),
    ("tqdm", "tqdm", "master"),
    ("tiangolo", "fastapi", "master"),
    ("pydantic", "pydantic", "main"),
    ("sqlalchemy", "sqlalchemy", "main"),
    # Data science / scientific computing
    ("numpy", "numpy", "main"),
    ("pandas-dev", "pandas", "main"),
    ("scikit-learn", "scikit-learn", "main"),
    ("matplotlib", "matplotlib", "main"),
    ("scipy", "scipy", "main"),
    # CLI / utilities
    ("Textualize", "rich", "master"),
    ("tiangolo", "typer", "master"),
    ("httpie", "httpie", "master"),
    ("psutil", "psutil", "master"),
    ("giampaolo", "pyftpdlib", "master"),
    # Infrastructure / DevOps
    ("fabric", "fabric", "main"),
    ("pypa", "virtualenv", "main"),
    ("PyCQA", "flake8", "main"),
    ("PyCQA", "pylint", "main"),
    ("pytest-dev", "pytest", "main"),
    # Databases / async
    ("encode", "databases", "master"),
    ("MagicStack", "asyncpg", "master"),
    ("redis", "redis-py", "master"),
    ("pymongo", "mongo-python-driver", "master"),
    # Parsing / templating
    ("pallets", "jinja", "main"),
    ("yaml", "pyyaml", "master"),
    ("Legrandin", "pycryptodome", "master"),
    # Networking / security
    ("paramiko", "paramiko", "main"),
    ("urllib3", "urllib3", "main"),
    ("certifi", "python-certifi", "master"),
]


def fetch_complexity_dataset(n: int = 6000, out: str = "data/complexity_dataset.jsonl"):
    """
    Real complexity dataset from popular Python packages.
    Downloads Python files via GitHub raw API, computes maintainability index.
    """
    print("\n=== Complexity Dataset (Real Sources) ===")
    from features.code_metrics import compute_all_metrics, metrics_to_feature_vector

    samples = []
    seen: set[int] = set()
    max_per_repo = max(60, n // len(COMPLEXITY_REPOS) + 20)

    for source, origin in tqdm(
        _iter_repo_files(COMPLEXITY_REPOS, max_per_repo),
        desc="  complexity",
        total=n,
    ):
        if len(samples) >= n:
            break
        h = hash(source)
        if h in seen:
            continue
        seen.add(h)
        try:
            metrics = compute_all_metrics(source)
            feat_vec = metrics_to_feature_vector(metrics)
            cog = float(metrics.cognitive_complexity)
            if cog < 0:
                continue
            # Build 16-dim vector: insert cognitive_complexity at index 1
            # (COG_IDX=1 in train_complexity.py; it is extracted as target there)
            full_vec = [feat_vec[0], cog] + feat_vec[1:]
            samples.append({
                "features": [float(x) for x in full_vec],
                "target": cog,
                "origin": origin,
                "code": source,
            })
        except Exception:
            continue

    print(f"  Complexity: {len(samples)} files")
    _write_jsonl(samples, out)
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 3. BUG PREDICTION DATASET  (uses PyDriller — needs git history)
# ─────────────────────────────────────────────────────────────────────────────

BUG_REPOS = [
    # Original web framework set
    "https://github.com/django/django",
    "https://github.com/pallets/flask",
    "https://github.com/psf/requests",
    "https://github.com/pypa/pip",
    "https://github.com/tqdm/tqdm",
    "https://github.com/bottlepy/bottle",
    # Major Python projects (added previously)
    "https://github.com/scrapy/scrapy",
    "https://github.com/sqlalchemy/sqlalchemy",
    "https://github.com/celery/celery",
    "https://github.com/encode/httpx",
    "https://github.com/pytest-dev/pytest",
    "https://github.com/aio-libs/aiohttp",
    "https://github.com/paramiko/paramiko",
    "https://github.com/tornadoweb/tornado",
    "https://github.com/twisted/twisted",
    "https://github.com/pydantic/pydantic",
    "https://github.com/tiangolo/fastapi",
    "https://github.com/fabric/fabric",
    "https://github.com/gitpython-developers/GitPython",
    "https://github.com/pypa/setuptools",
    # Data science / scientific (diverse domain)
    "https://github.com/pandas-dev/pandas",
    "https://github.com/scikit-learn/scikit-learn",
    "https://github.com/matplotlib/matplotlib",
    "https://github.com/sympy/sympy",
    # CLI / developer tools
    "https://github.com/Textualize/rich",
    "https://github.com/httpie/httpie",
    "https://github.com/PyCQA/pylint",
    "https://github.com/PyCQA/flake8",
    "https://github.com/HypothesisWorks/hypothesis",
    # Infrastructure / devops
    "https://github.com/pypa/virtualenv",
    "https://github.com/psutil/psutil",
    "https://github.com/encode/starlette",
    "https://github.com/urllib3/urllib3",
    "https://github.com/marshmallow-code/marshmallow",
]

BUG_KEYWORDS = {
    "fix", "bug", "defect", "issue", "error", "crash",
    "patch", "repair", "hotfix", "regression", "fault",
}


def fetch_bug_dataset(n: int = 15000, out: str = "data/bug_dataset.jsonl"):
    """
    Real bug prediction dataset mined from commit history via PyDriller.
    Uses a persistent cache dir to avoid Windows temp cleanup issues.
    """
    print("\n=== Bug Prediction Dataset (Real Sources) ===")
    try:
        from pydriller import Repository
    except ImportError:
        raise RuntimeError("pydriller not installed. pip install pydriller")

    from features.code_metrics import compute_all_metrics, metrics_to_feature_vector

    # Use a persistent cache dir — avoids Windows TemporaryDirectory cleanup crash
    cache_dir = Path(__file__).resolve().parent.parent / "data" / "repos_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    seen: set[int] = set()
    n_buggy = 0
    n_clean_count = 0

    for repo_url in BUG_REPOS:
        if len(samples) >= n:
            break
        repo_name = repo_url.rstrip("/").split("/")[-1]
        clone_path = cache_dir / repo_name
        print(f"  Mining {repo_url} …")

        try:
            # Re-use existing clone if present, otherwise clone fresh
            if clone_path.exists():
                repo_iter = Repository(str(clone_path), order="date")
            else:
                repo_iter = Repository(repo_url, clone_repo_to=str(cache_dir), order="date")

            processed = 0
            for commit in repo_iter.traverse_commits():
                if processed >= 600 or len(samples) >= n:
                    break

                msg = commit.msg.lower()
                is_bug = any(kw in msg for kw in BUG_KEYWORDS)

                for mf in commit.modified_files:
                    if not (mf.filename or "").endswith(".py"):
                        continue
                    src = mf.source_code
                    if not src or len(src) < 50 or len(src) > 200_000:
                        continue

                    h = hash(src)
                    if h in seen:
                        continue
                    seen.add(h)

                    label = 1 if is_bug else 0
                    # Keep roughly balanced
                    if label == 1 and n_buggy > n_clean_count * 2:
                        continue
                    if label == 0 and n_clean_count > n_buggy * 2:
                        continue

                    try:
                        metrics = compute_all_metrics(src)
                        feat_vec = metrics_to_feature_vector(metrics)
                        churn = (mf.added_lines or 0) + (mf.deleted_lines or 0)
                        samples.append({
                            "label": label,
                            "static_features": [float(x) for x in feat_vec],
                            "git_features": {
                                "code_churn": churn,
                                "author_count": 1,
                                "file_age_days": 0,
                                # n_past_bugs is 0 — we don't have per-file history
                                # per commit; model learns from static code features.
                                "n_past_bugs": 0,
                                "commit_freq": float(churn > 50),
                            },
                            "origin": f"{repo_name}/{mf.filename}",
                        })
                        if label == 1:
                            n_buggy += 1
                        else:
                            n_clean_count += 1
                    except Exception:
                        continue

                processed += 1

        except Exception as e:
            print(f"  [WARN] {repo_url}: {e}")

    print(f"  Bug: {len(samples)} files ({n_buggy} buggy, {n_clean_count} clean)")
    _write_jsonl(samples, out)
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# 4. PATTERN RECOGNITION DATASET
# ─────────────────────────────────────────────────────────────────────────────

PATTERN_REPOS = [
    # Web frameworks (original)
    ("pallets", "flask", "main"),
    ("psf", "requests", "main"),
    ("pallets", "click", "main"),
    ("tqdm", "tqdm", "master"),
    ("bottlepy", "bottle", "master"),
    ("encode", "httpx", "master"),
    ("pypa", "pip", "main"),
    ("tiangolo", "fastapi", "master"),
    # Data science
    ("scikit-learn", "scikit-learn", "main"),
    ("pandas-dev", "pandas", "main"),
    ("matplotlib", "matplotlib", "main"),
    # CLI / utilities
    ("Textualize", "rich", "main"),
    ("tiangolo", "typer", "master"),
    ("httpie", "httpie", "master"),
    # Testing / quality
    ("pytest-dev", "pytest", "main"),
    ("HypothesisWorks", "hypothesis", "master"),
    # Infrastructure
    ("fabric", "fabric", "main"),
    ("PyCQA", "pylint", "main"),
    ("PyCQA", "flake8", "main"),
    # Async / networking
    ("aio-libs", "aiohttp", "master"),
    ("encode", "starlette", "master"),
    ("MagicStack", "uvloop", "master"),
    # Parsing / data handling
    ("pallets", "jinja", "main"),
    ("marshmallow-code", "marshmallow", "dev"),
    ("pydantic", "pydantic", "main"),
    ("sqlalchemy", "sqlalchemy", "main"),
]


def _label_function(node: ast.FunctionDef, source_lines: list[str]) -> str | None:
    """
    Label a single function with a stricter 2-of-N consensus oracle.

    Returns None for borderline cases (< 2 signals agree) so they are dropped
    rather than labelled with a single-author guess — this reduces the circular
    label supervision that inflated the old model by +0.082 F1.

    For binary mode the caller maps "anti_pattern"/"code_smell" -> "pattern",
    "style_violation" -> "pattern", "clean" -> "clean". None cases are skipped.
    """
    start = node.lineno - 1
    end = getattr(node, "end_lineno", start + 30)
    snippet_lines = source_lines[start:end]

    n_lines = end - start
    n_args = len(node.args.args) + len(node.args.posonlyargs) + len(node.args.kwonlyargs)

    max_depth = 0
    for line in snippet_lines:
        stripped = line.lstrip()
        if stripped:
            depth = (len(line) - len(stripped)) // 4
            max_depth = max(max_depth, depth)

    n_long = sum(1 for line in snippet_lines if len(line) > 100)

    branch_count = sum(
        1 for line in snippet_lines
        if any(line.strip().startswith(kw) for kw in
               ("if ", "elif ", "for ", "while ", "try:"))
    )

    has_docstring = (
        node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
        and isinstance(node.body[0].value.value, str)
    )

    # --- Anti-pattern: requires 2+ strong signals ---
    anti_signals = [
        n_args > 7,
        n_lines > 80 and branch_count > 8,
        max_depth >= 5,
        n_lines > 100,
    ]
    if sum(anti_signals) >= 2:
        return "anti_pattern"

    # --- Code smell: requires 2+ signals ---
    smell_signals = [
        n_lines > 50,
        branch_count > 6,
        not has_docstring and n_lines > 40,
        max_depth >= 4,
    ]
    if sum(smell_signals) >= 2:
        return "code_smell"

    # --- Style violation: requires 2+ mild signals ---
    style_signals = [
        n_long > 2,
        not has_docstring and n_lines > 25,
        branch_count > 4,
    ]
    if sum(style_signals) >= 2:
        return "style_violation"

    # --- Clean: requires 2+ clean signals (no ambiguous cases labelled clean) ---
    clean_signals = [
        n_lines <= 20,
        branch_count <= 2,
        has_docstring,
        max_depth <= 2,
    ]
    if sum(clean_signals) >= 2:
        return "clean"

    # Borderline — insufficient consensus, drop this sample
    return None


def fetch_pattern_dataset(n: int = 4000, out: str = "data/pattern_dataset.jsonl"):
    """
    Real pattern dataset: Python functions from public repos,
    labelled with code-quality heuristics.
    """
    print("\n=== Pattern Dataset (Real Sources) ===")
    samples = []
    label_counts: dict[str, int] = {
        "clean": 0, "code_smell": 0, "anti_pattern": 0, "style_violation": 0
    }
    seen: set[int] = set()
    max_per_label = n // 4 + 50   # slight buffer

    for source, origin in tqdm(
        _iter_repo_files(PATTERN_REPOS, max_per_repo=200),
        desc="  pattern",
    ):
        if all(v >= max_per_label for v in label_counts.values()):
            break

        h = hash(source)
        if h in seen:
            continue
        seen.add(h)

        try:
            tree = ast.parse(source)
            source_lines = source.splitlines()
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                start = node.lineno - 1
                end = getattr(node, "end_lineno", start + 30)
                snippet = "\n".join(source_lines[start:end])
                if len(snippet) < 30 or len(snippet) > 3000:
                    continue
                label = _label_function(node, source_lines)
                # None = borderline consensus — drop rather than mislabel
                if label is None:
                    continue
                if label_counts.get(label, 0) >= max_per_label:
                    continue
                samples.append({
                    "code": snippet,
                    "label": label,
                    "origin": origin,
                })
                label_counts[label] = label_counts.get(label, 0) + 1
        except Exception:
            continue

    n_dropped = 0  # tracked implicitly via None returns above
    random.shuffle(samples)
    print(f"  Pattern: {len(samples)} functions | {label_counts} (borderline samples dropped)")
    _write_jsonl(samples, out)
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fetch REAL public datasets for IntelliCode ML models"
    )
    parser.add_argument(
        "--task",
        choices=["security", "complexity", "bug", "pattern"],
        help="Which dataset to build (omit with --all)",
    )
    parser.add_argument("--all", action="store_true", help="Build all datasets")
    parser.add_argument("--n", type=int, default=4000, help="Target samples per dataset")
    parser.add_argument("--out", default="data", help="Output directory")
    args = parser.parse_args()

    if not args.all and not args.task:
        parser.error("Specify --task or --all")

    tasks = (
        ["security", "complexity", "bug", "pattern"] if args.all else [args.task]
    )

    for task in tasks:
        out_path = f"{args.out}/{task}_dataset.jsonl"
        if task == "security":
            fetch_security_dataset(n=args.n, out=out_path)
        elif task == "complexity":
            fetch_complexity_dataset(n=args.n, out=out_path)
        elif task == "bug":
            fetch_bug_dataset(n=args.n, out=out_path)
        elif task == "pattern":
            fetch_pattern_dataset(n=args.n, out=out_path)

    print("\nAll real datasets fetched successfully.")


if __name__ == "__main__":
    main()
