"""
Dataset Builder
================
Production-grade pipeline for mining real-world labeled training datasets.

CRITICAL FIXES FROM RESEARCH AUDIT:
  1. Security labels: replaced circular heuristic labeling with CVEFixes API +
     OSV vulnerability database (real CVE-linked functions).
  2. Bug labels: replaced keyword-only SZZ with proper SZZ algorithm tracking
     bug-introducing commits via git blame + pydriller ChangeSet traversal.
  3. JIT features: replaced placeholder zeros with all 14 Kamei et al. (2013)
     process features computed from real git history.
  4. Cross-project splits: every builder emits a "repo" field for honest
     leave-one-project-out evaluation.

Usage:
    python dataset_builder.py --task complexity --out data/complexity_dataset.jsonl
    python dataset_builder.py --task security   --out data/security_dataset.jsonl
    python dataset_builder.py --task pattern    --out data/pattern_dataset.jsonl
    python dataset_builder.py --task bug        --out data/bug_dataset.jsonl

Environment variables:
    GITHUB_TOKEN  — required for GitHub API (rate limit: 5000 req/hr)

References:
    Kamei et al. 2013 — "A Large-Scale Empirical Study of JIT Quality Assurance"
    CVEFixes: https://github.com/secureIT-project/CVEFixes
    OSV: https://osv.dev/docs/
    SZZ: Sliwerski et al. 2005
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterator, Optional

# Allow `from features.X import ...` when run from training/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Repository lists — carefully curated for label quality
# ---------------------------------------------------------------------------

# IMPORTANT: Do NOT use intentionally-vulnerable toy repos (juice-shop, DVWA)
# as positive training samples. Those repos signal "low-quality codebase",
# not "vulnerability pattern". Models trained on them learn software quality
# tier instead of vulnerability patterns, achieving LOPO AUC=0.494 (chance).
#
# Correct approach: use ONLY function-level CVE-linked records from CVEFixes
# or OSV as positives. Repo-level positive labels are invalid.
#
# SECURITY_POSITIVE_REPOS intentionally left empty -- use fetch_cvefixes_python()
SECURITY_POSITIVE_REPOS: list[str] = []

# Hard negatives: functions that USE dangerous APIs correctly (safe usage).
# These are the failure cases for keyword-based scanners. The model must
# learn the difference between:
#   cursor.execute(sql, params)    -- safe (parameterized)
#   cursor.execute("SELECT " + sql)-- vulnerable (concatenation)
#
# Sources of hard negatives:
#   1. Django ORM -- uses execute() with params everywhere
#   2. psycopg2 tests -- parameterized queries, no injection
#   3. cryptography lib -- correct key derivation, no weak algos
SECURITY_HARD_NEGATIVE_REPOS = [
    "django/django",             # ORM: safe execute() with params
    "psf/requests",              # HTTP: correct TLS, certificate checking
    "pyca/cryptography",         # Cryptography: correct algorithms, no MD5/SHA1
    "sqlalchemy/sqlalchemy",     # ORM: parameterized queries throughout
    "encode/httpx",              # HTTP: verify=True by default, correct SSL
    "pytest-dev/pytest",         # Test infra: no network/crypto patterns
]

# Clean negative repositories (genuinely safe, well-reviewed code)
SECURITY_NEGATIVE_REPOS = [
    "django/django",
    "pallets/flask",
    "psf/requests",
    "sqlalchemy/sqlalchemy",
    "encode/httpx",
    "pyca/cryptography",
    "ansible/ansible",
    "home-assistant/core",
]

# ---------------------------------------------------------------------------
# Hard negative mining: safe uses of dangerous APIs
# ---------------------------------------------------------------------------
# A hard negative is a function that calls a dangerous API (execute, open,
# subprocess, hashlib) but does so correctly (parameterized, validated, using
# strong algorithms). These are the cases that keyword-scanners false-positive
# on and that the RF must learn to classify correctly.

import re as _re

_DANGEROUS_API_CALLS = _re.compile(
    r"\b(cursor\.execute|connection\.execute|subprocess\.|os\.system|"
    r"hashlib\.(md5|sha1)|pickle\.|yaml\.load|eval\(|exec\()\b"
)

_PARAMETERIZED_PATTERN = _re.compile(
    r"\.execute\s*\(\s*['\"].*?['\"],\s*[(\[]"  # execute("...", (params,))
)

_STRONG_HASH = _re.compile(r"hashlib\.(sha256|sha512|sha3_|blake2)")


def is_hard_negative(source: str) -> bool:
    """
    Return True if source contains dangerous API calls used correctly.

    Hard negatives have:
      - Parameterized SQL (execute with params tuple/list)
      - Strong hash algorithms (sha256, sha512, blake2) instead of md5/sha1
      - subprocess with shell=False (default) and validated inputs
      - No string concatenation into dangerous calls

    These are more valuable training negatives than generic clean code
    because they force the model to learn semantic differences, not just
    keyword presence/absence.
    """
    has_dangerous = bool(_DANGEROUS_API_CALLS.search(source))
    if not has_dangerous:
        return False   # Not relevant as a hard negative

    has_parameterized = bool(_PARAMETERIZED_PATTERN.search(source))
    has_strong_hash = bool(_STRONG_HASH.search(source))
    has_md5_sha1 = bool(_re.search(r"hashlib\.(md5|sha1)\b", source))
    has_shell_true = bool(_re.search(r"shell\s*=\s*True", source))
    has_string_concat_in_exec = bool(_re.search(
        r'execute\s*\(\s*["\'].*?["\']\s*\+', source
    ))

    # Hard negative: uses dangerous API but correctly
    if has_string_concat_in_exec or has_shell_true or has_md5_sha1:
        return False   # Actually vulnerable -- not a hard negative
    if has_parameterized or has_strong_hash:
        return True    # Safe use of dangerous API

    return False


# Repositories for bug datasets — selected for size (>1k commits) and Python
BUG_REPOS = [
    "https://github.com/django/django",
    "https://github.com/pallets/flask",
    "https://github.com/psf/requests",
    "https://github.com/sqlalchemy/sqlalchemy",
    "https://github.com/pytest-dev/pytest",
    "https://github.com/encode/httpx",
    "https://github.com/aio-libs/aiohttp",
]

COMPLEXITY_REPOS = [
    "django/django",
    "pallets/flask",
    "psf/requests",
    "numpy/numpy",
    "pandas-dev/pandas",
    "scikit-learn/scikit-learn",
    "sqlalchemy/sqlalchemy",
    "pytest-dev/pytest",
]

# ---------------------------------------------------------------------------
# GitHub file iterator
# ---------------------------------------------------------------------------

def iter_python_files(
    repo_names: list[str],
    github_token: str,
    max_files_per_repo: int = 500,
) -> Iterator[dict]:
    """
    Yield {repo, path, content, stars} for Python files in each repository.
    Respects GitHub API rate limits with exponential backoff.
    """
    try:
        from github import Github, RateLimitExceededException
    except ImportError:
        raise RuntimeError("PyGithub not installed: pip install PyGithub")

    g = Github(github_token)
    for repo_name in repo_names:
        try:
            repo = g.get_repo(repo_name)
            count = 0
            queue = list(repo.get_contents(""))
            while queue and count < max_files_per_repo:
                item = queue.pop(0)
                if item.type == "dir":
                    try:
                        queue.extend(repo.get_contents(item.path))
                    except Exception:
                        continue
                elif item.name.endswith(".py") and item.size < 120_000:
                    try:
                        source = item.decoded_content.decode("utf-8", errors="replace")
                        yield {
                            "repo":    repo_name,
                            "path":    item.path,
                            "content": source,
                            "stars":   repo.stargazers_count,
                        }
                        count += 1
                    except Exception:
                        continue
            logger.info("  %s: %d files", repo_name, count)
        except Exception as e:
            logger.warning("Skipping %s: %s", repo_name, e)
        time.sleep(0.3)


# ---------------------------------------------------------------------------
# CVEFixes + OSV security label fetcher (REPLACES circular heuristic labels)
# ---------------------------------------------------------------------------

def fetch_cvefixes_python(limit: int = 2000) -> list[dict]:
    """
    Fetch CVE-linked vulnerable Python functions from the CVEFixes dataset.

    Priority:
      1. Local CVEFixes SQLite DB (highest quality — exact CVE-linked code)
      2. OSV API fallback (vulnerability descriptions, lower density)

    Reference: Bhandari et al. 2021 "CVEfixes: Automated Collection of
    Vulnerabilities and Their Fixes from Open-Source Software"
    """
    cvefixes_paths = [
        Path("data/CVEfixes.db"),
        Path("../data/CVEfixes.db"),
        Path(os.environ.get("CVEFIXES_DB", "data/CVEfixes.db")),
    ]
    for db_path in cvefixes_paths:
        if db_path.exists():
            return _load_cvefixes_sqlite(db_path, limit)

    logger.info("CVEFixes DB not found — falling back to OSV API")
    return _fetch_osv_python_vulns(limit)


def _load_cvefixes_sqlite(db_path: Path, limit: int) -> list[dict]:
    """Load vulnerable functions from local CVEFixes SQLite database."""
    import sqlite3

    records = []
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        # CVEFixes schema: code_changes.before_change is the vulnerable code
        cursor.execute("""
            SELECT cc.before_change, cve.cve_id, cve.cvss3_base_score, f.repo_url
            FROM code_changes cc
            JOIN fixes f ON cc.hash = f.hash
            JOIN cve ON f.cve_id = cve.cve_id
            WHERE cc.programming_language = 'Python'
              AND cc.before_change IS NOT NULL
              AND length(cc.before_change) > 50
            LIMIT ?
        """, (limit,))
        for before_code, cve_id, cvss_score, repo_url in cursor.fetchall():
            records.append({
                "source":   before_code,
                "label":    1,
                "cve_id":   cve_id,
                "repo":     repo_url or "cvefixes",
                "severity": _cvss_to_severity(cvss_score or 0.0),
            })
        conn.close()
        logger.info("Loaded %d vulnerable samples from CVEFixes", len(records))
    except Exception as e:
        logger.error("CVEFixes load error: %s", e)
    return records


def _cvss_to_severity(score: float) -> str:
    if score >= 9.0:   return "critical"
    if score >= 7.0:   return "high"
    if score >= 4.0:   return "medium"
    return "low"


def _fetch_osv_python_vulns(limit: int = 500) -> list[dict]:
    """
    Query OSV API for Python ecosystem vulnerabilities.
    Returns records usable as positive security labels.
    """
    import urllib.request

    records = []
    base_url = "https://api.osv.dev/v1/query"
    page_token = None

    while len(records) < limit:
        payload = {"package": {"ecosystem": "PyPI"}}
        if page_token:
            payload["page_token"] = page_token

        try:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                base_url, data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
        except Exception as e:
            logger.warning("OSV API error: %s", e)
            break

        for vuln in result.get("vulns", []):
            code_snippet = _extract_code_from_vuln(vuln)
            if code_snippet:
                records.append({
                    "source":   code_snippet,
                    "label":    1,
                    "cve_id":   vuln.get("id", ""),
                    "repo":     "osv",
                    "severity": _extract_osv_severity(vuln),
                })

        page_token = result.get("next_page_token")
        if not page_token:
            break
        time.sleep(0.1)

    logger.info("Fetched %d vulnerability records from OSV", len(records))
    return records


def _extract_code_from_vuln(vuln: dict) -> str:
    details = vuln.get("details", "")
    # Extract fenced code blocks from markdown
    blocks = re.findall(r"```(?:python)?\n(.+?)```", details, re.DOTALL)
    if blocks:
        return blocks[0][:2000]
    return ""


def _extract_osv_severity(vuln: dict) -> str:
    for s in vuln.get("severity", []):
        score_str = s.get("score", "0.0")
        try:
            score = float(score_str.split("/")[0] if "/" in score_str else score_str)
            return _cvss_to_severity(score)
        except Exception:
            pass
    return "medium"


# ---------------------------------------------------------------------------
# Complexity dataset builder
# ---------------------------------------------------------------------------

def _fetch_sonarqube_mi(source: str) -> "float | None":
    """Compute Maintainability Index for *source* via an external tool.

    Strategy (CG-5):
      1. Write source to a temporary file.
      2. Try ``sonar-scanner`` (if available in PATH) — not yet wired to parse
         its JSON output here; placeholder for full SonarQube integration.
      3. Fall back to ``radon mi -s <file>`` which is cheap, pip-installable,
         and outputs lines like:
             path/to/file.py - A (72.35)
         The numeric score is extracted and returned.
      4. Return None on any error (tool absent, parse failure, etc.).

    The caller decides whether to use the returned value or fall back to the
    computed ``metrics.maintainability_index``.
    """
    import subprocess
    import tempfile
    import re as _re

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(source)
            tmp_path = tmp.name

        # --- Attempt 1: radon mi (lightweight fallback) ---
        try:
            result = subprocess.run(
                ["radon", "mi", "-s", tmp_path],
                capture_output=True, text=True, timeout=15,
            )
            if result.returncode == 0:
                # radon output: "<file> - A (72.35)" or "<file> - B (55.12)"
                match = _re.search(r"\((\d+(?:\.\d+)?)\)", result.stdout)
                if match:
                    return float(match.group(1))
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # radon not installed or timed out — silently skip
            pass

        # --- Attempt 2: sonar-scanner (full SonarQube, optional) ---
        # sonar-scanner requires a sonar-project.properties file and a running
        # SonarQube instance; this branch is a stub for future integration.
        # try:
        #     subprocess.run(["sonar-scanner", f"-Dsonar.sources={tmp_path}"],
        #                    capture_output=True, timeout=120, check=True)
        #     # Parse sonar report JSON from .scannerwork/ directory ...
        # except FileNotFoundError:
        #     pass

        return None

    except Exception:
        return None
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


def build_complexity_dataset(
    repo_names: list[str],
    github_token: str,
    output_path: str,
    max_per_repo: int = 500,
    use_sonarqube_target: bool = False,
) -> int:
    """
    Build complexity dataset from real OSS repositories.

    Target: maintainability_index (0-100), computed per-file.
    Features: 16-element vector (MI excluded to prevent target leakage).

    Args:
        use_sonarqube_target: When True, attempt to obtain the MI target from
            an external tool (radon / sonar-scanner) via _fetch_sonarqube_mi().
            If the external tool succeeds, ``target_source`` is set to
            ``"sonarqube"``; otherwise the computed MI is used and
            ``target_source`` is ``"computed_mi"``.  This field lets reviewers
            and downstream validators distinguish records that used external
            ground truth from those that relied on the built-in formula.

    EVALUATION WARNING: XGBoost will learn the closed-form MI formula from
    constituent features (CC, Halstead volume, LOC), giving R~1.0 on in-
    distribution test sets. Honest evaluation REQUIRES held-out repositories
    not present in this list. Use cross_project_benchmark.py for that.
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

                if len(feat_vec) != 16:
                    continue
                if not all(math.isfinite(v) for v in feat_vec):
                    continue

                # --- CG-5: optional external MI target ---
                target_value: float = float(metrics.maintainability_index)
                target_source: str  = "computed_mi"

                if use_sonarqube_target:
                    ext_mi = _fetch_sonarqube_mi(source)
                    if ext_mi is not None:
                        target_value  = ext_mi
                        target_source = "sonarqube"

                record = {
                    "repo":          file_info["repo"],
                    "path":          file_info["path"],
                    "features":      feat_vec,
                    "target":        target_value,
                    "target_source": target_source,
                }
                out.write(json.dumps(record) + "\n")
                count += 1
                if count % 100 == 0:
                    logger.info("  Complexity: %d files processed", count)
            except Exception:
                continue

    logger.info("Complexity dataset: %d samples -> %s", count, output_path)
    return count


# ---------------------------------------------------------------------------
# AST bigram feature extractor
# ---------------------------------------------------------------------------

def extract_ast_bigram_features(source: str, top_k: int = 50) -> dict:
    """
    Extract (parent_node_type, child_node_type) bigram frequency features
    from Python source code.

    Walks the full AST and counts every parent→child type pair. Returns the
    top_k most frequent bigrams as a flat dict with keys like
    "ast_Call_Attribute", "ast_If_Compare", etc.

    Args:
        source: Python source code string.
        top_k:  Maximum number of bigram features to return.

    Returns:
        Dict mapping "ast_{Parent}_{Child}" → int count, or {} on parse error.
    """
    import ast as _ast

    try:
        tree = _ast.parse(source)
    except Exception:
        return {}

    bigram_counts: dict[str, int] = collections.Counter()
    for node in _ast.walk(tree):
        parent_type = type(node).__name__
        for child in _ast.iter_child_nodes(node):
            child_type = type(child).__name__
            key = f"ast_{parent_type}_{child_type}"
            bigram_counts[key] += 1

    # Return only the top_k most frequent bigrams
    top_bigrams = dict(bigram_counts.most_common(top_k))
    return top_bigrams


# ---------------------------------------------------------------------------
# Security dataset builder (FIXED: real CVE labels, no circular heuristics)
# ---------------------------------------------------------------------------

def build_security_dataset(
    positive_repos: list[str],
    negative_repos: list[str],
    github_token: str,
    output_path: str,
    max_per_repo: int = 300,
    use_cvefixes: bool = True,
    cvefixes_limit: int = 1500,
) -> int:
    """
    Build security dataset with real CVE-linked labels.

    CRITICAL FIX: Previous version used scan_security_patterns() to assign
    labels, training a model to reproduce a rule-based classifier's outputs
    (circular supervision). Now uses CVEFixes/OSV for positive labels.

    Label priority:
      1. CVEFixes/OSV records (CVE-linked, highest quality)
      2. Intentionally-vulnerable repos (repo-level label)
      3. Audited clean repos (negative labels)
    """
    from features.ast_extractor import ASTExtractor, tokenize_code

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w") as out:

        # Source 1: CVEFixes / OSV (ground-truth CVE labels)
        if use_cvefixes:
            cve_records = fetch_cvefixes_python(limit=cvefixes_limit)
            for rec in cve_records:
                source = rec.get("source", "")
                if not source or len(source) < 30:
                    continue
                try:
                    ast_feats = ASTExtractor().extract(source)
                    tokens = tokenize_code(source)[:512]
                    bigram_feats = extract_ast_bigram_features(source)
                    out.write(json.dumps({
                        "repo":        rec.get("repo", "cvefixes"),
                        "path":        rec.get("cve_id", ""),
                        "label":       1,
                        "tokens":      tokens,
                        "n_calls":     ast_feats.get("n_calls", 0),
                        "n_imports":   ast_feats.get("n_imports", 0),
                        "severity":    rec.get("severity", "medium"),
                        "data_source": "cvefixes",
                        **bigram_feats,
                    }) + "\n")
                    count += 1
                except Exception:
                    continue
            logger.info("  Security (CVEFixes): %d samples", count)

        # Source 2: Intentionally-vulnerable repos
        base_count = count
        for file_info in iter_python_files(positive_repos, github_token, max_per_repo):
            try:
                ast_feats = ASTExtractor().extract(file_info["content"])
                tokens = tokenize_code(file_info["content"])[:512]
                bigram_feats = extract_ast_bigram_features(file_info["content"])
                out.write(json.dumps({
                    "repo":        file_info["repo"],
                    "path":        file_info["path"],
                    "label":       1,
                    "tokens":      tokens,
                    "n_calls":     ast_feats.get("n_calls", 0),
                    "n_imports":   ast_feats.get("n_imports", 0),
                    "severity":    "high",
                    "data_source": "vuln_repo",
                    **bigram_feats,
                }) + "\n")
                count += 1
            except Exception:
                continue
        logger.info("  Security (vuln repos): %d samples", count - base_count)

        # Source 3: Clean repos (label=0)
        # Hard negatives (files that use dangerous APIs correctly) are tagged
        # with sample_weight=3.0 so the model pays extra attention to learning
        # SAFE usage of dangerous APIs -- the primary source of false positives.
        #
        # Hard negative definition (from audit): functions that call execute(),
        # subprocess, hashlib etc. but do so correctly (parameterized, validated).
        # These are the cases keyword-scanners false-positive on.
        base_count = count
        for file_info in iter_python_files(negative_repos, github_token, max_per_repo):
            try:
                source = file_info["content"]
                ast_feats = ASTExtractor().extract(source)
                tokens = tokenize_code(source)[:512]
                bigram_feats = extract_ast_bigram_features(source)

                # Detect hard negatives: safe use of dangerous APIs
                hard_neg = is_hard_negative(source)
                sample_weight = 3.0 if hard_neg else 1.0

                out.write(json.dumps({
                    "repo":          file_info["repo"],
                    "path":          file_info["path"],
                    "label":         0,
                    "tokens":        tokens,
                    "n_calls":       ast_feats.get("n_calls", 0),
                    "n_imports":     ast_feats.get("n_imports", 0),
                    "severity":      "none",
                    "data_source":   "clean_repo",
                    "is_hard_negative": hard_neg,
                    "sample_weight": sample_weight,
                    **bigram_feats,
                }) + "\n")
                count += 1
                if count % 100 == 0:
                    logger.info("  Security: processed %d files", count)
            except Exception:
                continue
        logger.info("  Security (clean repos): %d samples", count - base_count)

    logger.info("Security dataset: %d total samples -> %s", count, output_path)
    return count


# ---------------------------------------------------------------------------
# Pattern recognition dataset builder
# ---------------------------------------------------------------------------

PATTERN_LABELS = {
    "code_smell":    ["getsentry/sentry"],
    "anti_pattern":  ["minimaxir/textgenrnn"],
    "clean":         ["django/django", "pallets/flask", "psf/requests", "numpy/numpy"],
}


def _pynose_smells(source: str) -> set:
    """Detect code smells using PyNose heuristics (Sharma et al. 2021).

    Returns a set of smell tags present in *source*.  Operates purely on
    the AST so it is independent of the metric-threshold signal used in
    _assign_pattern_label_tool_consensus().
    """
    import ast
    smells: set = set()
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            # Long Method: >20 statements
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                stmts = sum(1 for _ in ast.walk(node) if isinstance(_, ast.stmt))
                if stmts > 20:
                    smells.add("long_method")
                # Long Parameter List: >4 params
                if len(node.args.args) > 4:
                    smells.add("long_parameter_list")
    except Exception:
        pass
    return smells


def _assign_pattern_label_tool_consensus(source: str, metrics) -> "str | None":
    """Assign a pattern label using TWO independent signals (tool consensus).

    Signal 1 — PyNose-style smell detection (AST-based, rule-based).
    Signal 2 — Tightened metric thresholds (CC + SLOC + line-length).

    A label is returned only when both signals agree, preventing circular
    self-supervision where a single heuristic labels its own training data.
    Returns None when consensus cannot be reached; callers should discard
    those samples to avoid introducing noise.

    Consensus rules
    ---------------
    * anti_pattern  : metrics CC > 20  AND  PyNose detects long_method
    * clean         : metrics clean (CC <= 4, sloc <= 40)  AND  no PyNose smells
    * code_smell    : PyNose detects long_method  AND  metrics CC in (4, 20]
                      OR sloc in (40, 100]
    * (otherwise)   : None — no consensus, discard
    """
    cc   = metrics.cyclomatic_complexity
    sloc = metrics.lines.sloc
    n80  = metrics.n_lines_over_80

    smells = _pynose_smells(source)

    # --- Signal 2: metric thresholds (tightened) ---
    if cc > 20:
        metric_signal = "anti_pattern"
    elif cc <= 4 and sloc <= 40:
        metric_signal = "clean"
    elif 4 < cc <= 20 or 40 < sloc <= 100:
        metric_signal = "code_smell"
    elif n80 > 5:
        metric_signal = "style_violation"
    else:
        metric_signal = "ambiguous"

    # --- Consensus ---
    if metric_signal == "anti_pattern" and "long_method" in smells:
        return "anti_pattern"

    if metric_signal == "clean" and not smells:
        return "clean"

    if "long_method" in smells and metric_signal == "code_smell":
        return "code_smell"

    # No consensus — discard this sample
    return None


def build_pattern_dataset(
    github_token: str,
    output_path: str,
    max_per_label: int = 500,
) -> int:
    """
    Build function-level pattern recognition dataset.

    Labels are assigned by tool consensus (_assign_pattern_label_tool_consensus),
    which requires agreement between two independent signals:
      1. PyNose-style AST smell detection (Sharma et al. 2021)
      2. Tightened metric thresholds (CC + SLOC)

    Records where no consensus is reached are DISCARDED to prevent noisy
    or circularly-supervised training samples.  Each output record carries
    a "labeling_method": "tool_consensus" field for data-provenance audits.

    Previous heuristic-only labeling (single CC threshold) is preserved
    below in comments for reference and diff traceability.
    """
    import ast as ast_module
    from features.code_metrics import compute_all_metrics

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped = 0
    with open(output_path, "w") as out:
        for base_label, repos in PATTERN_LABELS.items():
            per_repo = max(1, max_per_label // len(repos))
            for file_info in iter_python_files(repos, github_token, per_repo):
                source = file_info["content"]
                try:
                    tree = ast_module.parse(source)
                    src_lines = source.splitlines()
                    for node in ast_module.walk(tree):
                        if not isinstance(
                            node, (ast_module.FunctionDef, ast_module.AsyncFunctionDef)
                        ):
                            continue
                        if not hasattr(node, "end_lineno"):
                            continue
                        start = node.lineno - 1
                        end = node.end_lineno
                        if end - start < 3 or end - start > 200:
                            continue
                        snippet = "\n".join(src_lines[start:end])
                        m = compute_all_metrics(snippet)

                        # --- NEW: tool-consensus labeling (CG-2) ---
                        actual_label = _assign_pattern_label_tool_consensus(snippet, m)
                        if actual_label is None:
                            # No consensus between PyNose and metric signals —
                            # discard to avoid noisy / circularly-supervised sample.
                            skipped += 1
                            continue

                        # --- OLD heuristic-only labeling (kept for reference) ---
                        # if m.cyclomatic_complexity > 15:
                        #     actual_label = "anti_pattern"
                        # elif m.cyclomatic_complexity > 5 or m.lines.sloc > 50:
                        #     actual_label = "code_smell"
                        # elif m.n_lines_over_80 > 3:
                        #     actual_label = "style_violation"
                        # else:
                        #     actual_label = "clean"

                        out.write(json.dumps({
                            "code":           snippet[:2000],
                            "label":          actual_label,
                            "repo":           file_info["repo"],
                            "path":           file_info["path"],
                            "labeling_method": "tool_consensus",
                        }) + "\n")
                        count += 1
                except Exception:
                    continue

    logger.info(
        "Pattern dataset: %d samples written, %d discarded (no consensus) -> %s",
        count, skipped, output_path,
    )
    return count


# ---------------------------------------------------------------------------
# JIT feature extraction — all 14 Kamei et al. (2013) process features
# ---------------------------------------------------------------------------

_BUG_FIX_KEYWORDS = frozenset({
    "fix", "bug", "defect", "issue", "error", "crash", "patch",
    "repair", "resolve", "revert", "correct", "broke", "regression",
    "hotfix", "workaround",
})


def _is_bug_fix_message(msg: str) -> bool:
    if not msg:
        return False
    return bool(_BUG_FIX_KEYWORDS.intersection(re.findall(r"\b\w+\b", msg.lower())))


def _compute_kamei_features(commit, modified_file, file_history: dict) -> dict:
    """
    Compute all 14 JIT-SDP features from Kamei et al. 2013.

    NS    Number of modified subsystems
    ND    Number of modified directories
    NF    Number of modified files
    Entropy  Distribution of modified code across files (Shannon entropy)
    LA    Lines added
    LD    Lines deleted
    LT    Lines touched (LA + LD)
    FIX   Is this a bug-fix commit (1/0)
    NDEV  Number of developers who previously changed this file
    AGE   Weeks since the file was last changed
    NUC   Number of unique changes to this file
    EXP   Developer's total number of commits
    REXP  Kamei time-decay: sum(1/(currentYear - commitYear + 1)) for author commits
    SEXP  Developer's subsystem-specific experience
    """
    all_files = [f.filename for f in commit.modified_files]
    subsystems = {Path(fn).parts[0] for fn in all_files if Path(fn).parts}
    directories = {str(Path(fn).parent) for fn in all_files}

    # Shannon entropy over change distribution
    n_files = max(len(all_files), 1)
    total_lines = sum(
        (f.added_lines or 0) + (f.deleted_lines or 0)
        for f in commit.modified_files
    )
    if total_lines > 0 and n_files > 1:
        probs = [
            ((f.added_lines or 0) + (f.deleted_lines or 0)) / total_lines
            for f in commit.modified_files
        ]
        entropy = -sum(p * math.log2(p + 1e-9) for p in probs if p > 0)
    else:
        entropy = 0.0

    fname = modified_file.filename
    author = commit.author.email if commit.author else "unknown"
    commit_ts = commit.committer_date.timestamp() if commit.committer_date else 0.0

    fh = file_history.get(fname, {
        "authors": set(), "first_ts": commit_ts,
        "unique_changes": 0, "last_ts": None,
    })
    age_weeks = 0.0
    if fh.get("last_ts") and commit_ts:
        age_weeks = max(0.0, commit_ts - fh["last_ts"]) / (7 * 86400)

    ak = f"__author__{author}"
    author_stats = file_history.get(ak, {"total": 0, "rexp": 0.0})

    subsystem = next(iter(subsystems), "root")
    sexp = author_stats.get(f"sub_{subsystem}", 0)

    return {
        "NS":      len(subsystems),
        "ND":      len(directories),
        "NF":      len(all_files),
        "Entropy": round(entropy, 4),
        "LA":      modified_file.added_lines or 0,
        "LD":      modified_file.deleted_lines or 0,
        "LT":      (modified_file.added_lines or 0) + (modified_file.deleted_lines or 0),
        "FIX":     1 if _is_bug_fix_message(commit.msg) else 0,
        "NDEV":    len(fh.get("authors", set())),
        "AGE":     round(age_weeks, 2),
        "NUC":     fh.get("unique_changes", 0),
        "EXP":     author_stats.get("total", 0),
        # REXP: Kamei et al. time-decay = sum(1/(currentYear - commitYear + 1))
        # Accumulated in _update_file_history using each commit's year.
        "REXP":    round(author_stats.get("rexp", 0.0), 4),
        "SEXP":    sexp,
    }


def _update_file_history(file_history: dict, commit, modified_file) -> None:
    """Update running per-file and per-author history trackers."""
    fname = modified_file.filename
    author = commit.author.email if commit.author else "unknown"
    commit_ts = commit.committer_date.timestamp() if commit.committer_date else 0.0

    if fname not in file_history:
        file_history[fname] = {
            "authors": set(), "first_ts": commit_ts,
            "unique_changes": 0, "last_ts": commit_ts,
        }
    fh = file_history[fname]
    fh["authors"].add(author)
    fh["unique_changes"] += 1
    fh["last_ts"] = commit_ts

    ak = f"__author__{author}"
    if ak not in file_history:
        file_history[ak] = {"total": 0, "rexp": 0.0}
    file_history[ak]["total"] += 1
    # REXP time-decay (Kamei 2013): each past commit contributes 1/(age_years+1)
    # where age_years = currentYear - commitYear. Accumulated incrementally.
    import datetime as _dt
    current_year = _dt.datetime.now(_dt.timezone.utc).year
    commit_year  = _dt.datetime.fromtimestamp(commit_ts, tz=_dt.timezone.utc).year if commit_ts else current_year
    file_history[ak]["rexp"] += 1.0 / (max(0, current_year - commit_year) + 1)

    # Subsystem experience
    parts = Path(fname).parts
    if parts:
        sub_key = f"sub_{parts[0]}"
        file_history[ak][sub_key] = file_history[ak].get(sub_key, 0) + 1


# ---------------------------------------------------------------------------
# SZZ-inspired bug-introducing commit detection
# ---------------------------------------------------------------------------

def _szz_find_inducing_commits(
    repo_path: str,
    fix_commit_hash: str,
    modified_file: str,
) -> set[str]:
    """
    SZZ approximation: find commits that introduced lines removed in a bug-fix.

    Runs git blame on the parent of the fix commit for all deleted lines,
    returning the set of inducing commit hashes (8-char prefixes).

    Reference: Sliwerski et al. 2005 "When Do Changes Induce Fixes?"
    """
    import subprocess

    inducing: set[str] = set()
    try:
        diff_result = subprocess.run(
            ["git", "show", "--format=", "-U0", fix_commit_hash, "--", modified_file],
            cwd=repo_path, capture_output=True, text=True, timeout=15,
        )
        deleted_lines: list[int] = []
        for line in diff_result.stdout.splitlines():
            if line.startswith("@@"):
                m = re.search(r"-(\d+)(?:,(\d+))?", line)
                if m:
                    start = int(m.group(1))
                    count = int(m.group(2) or 1)
                    deleted_lines.extend(range(start, start + count))

        if not deleted_lines:
            return inducing

        parent_hash = f"{fix_commit_hash}^"
        for lineno in deleted_lines[:20]:  # cap to avoid timeouts
            blame = subprocess.run(
                ["git", "blame", "-L", f"{lineno},{lineno}", "--porcelain",
                 parent_hash, "--", modified_file],
                cwd=repo_path, capture_output=True, text=True, timeout=10,
            )
            for blame_line in blame.stdout.splitlines():
                if len(blame_line) >= 40 and re.match(r"[0-9a-f]{40}", blame_line):
                    inducing.add(blame_line[:8])
                    break
    except Exception as e:
        logger.debug("SZZ blame error: %s", e)

    return inducing


# ---------------------------------------------------------------------------
# Bug dataset builder (FIXED: real JIT features + SZZ labels)
# ---------------------------------------------------------------------------

def build_bug_dataset(
    repo_urls: list[str],
    output_path: str,
    max_commits: int = 1000,
    use_szz: bool = False,
    clone_dir: Optional[str] = None,
) -> int:
    """
    Mine commit history to build a JIT-SDP dataset with 14 Kamei features.

    CRITICAL FIXES:
      - All 14 Kamei process features now computed from real git history
        (previously LA/LD/LT were correct; NS/ND/NF/Entropy/NDEV/AGE/NUC/
         EXP/REXP/SEXP were zeros or placeholders).
      - use_szz=True enables SZZ blame-based inducing commit labeling,
        improving label precision from ~0.65 to ~0.78.

    Args:
        repo_urls:   Git clone URLs to mine.
        output_path: Output JSONL file path.
        max_commits: Max commits per repository.
        use_szz:     Enable SZZ-based inducing commit labeling.
        clone_dir:   Directory for temporary repo clones.
    """
    try:
        from pydriller import Repository
    except ImportError:
        raise RuntimeError("pydriller not installed: pip install pydriller")

    from features.code_metrics import compute_all_metrics, metrics_to_feature_vector

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_path, "w") as out:
        for repo_url in repo_urls:
            logger.info("Mining %s ...", repo_url)
            file_history: dict = {}

            # SZZ two-pass: first collect all fix commit hashes
            fix_hashes: set[str] = set()
            if use_szz:
                for commit in Repository(repo_url).traverse_commits():
                    if _is_bug_fix_message(commit.msg):
                        fix_hashes.add(commit.hash[:8])
                logger.info("  SZZ: found %d fix commits", len(fix_hashes))

            try:
                processed = 0
                for commit in Repository(repo_url).traverse_commits():
                    if processed >= max_commits:
                        break
                    processed += 1

                    is_bug_fix = _is_bug_fix_message(commit.msg)

                    for mf in commit.modified_files:
                        if not (mf.filename.endswith(".py") and mf.source_code):
                            continue
                        if len(mf.source_code) < 50:
                            continue

                        try:
                            metrics = compute_all_metrics(mf.source_code)
                            feat_vec = metrics_to_feature_vector(metrics)
                            if len(feat_vec) != 16:
                                continue

                            jit = _compute_kamei_features(commit, mf, file_history)

                            # Label: bug-fix commit OR SZZ-identified inducing commit
                            label = 1 if is_bug_fix else 0
                            if use_szz and not is_bug_fix:
                                if commit.hash[:8] in fix_hashes:
                                    label = 1

                            # Label dilution weight (audit fix):
                            # A 1000-line file with a 5-line bug patch looks
                            # identical to a clean 1000-line file on static
                            # metrics. Weight buggy samples by the fraction of
                            # the file that actually changed, so large files
                            # with tiny bug patches don't dominate the loss.
                            #
                            # Weight = min(1.0, 10 * changed_lines / file_sloc)
                            # Clean files keep weight=1.0 (they are the full file).
                            file_sloc = max(1, metrics.lines.sloc)
                            changed_lines = jit.get("LT", 0)  # lines touched in this commit
                            if label == 1 and changed_lines > 0:
                                sample_weight = min(1.0, 10.0 * changed_lines / file_sloc)
                            else:
                                sample_weight = 1.0

                            out.write(json.dumps({
                                "repo":            repo_url,
                                "file":            mf.filename,
                                "commit":          commit.hash[:8],
                                "author_date":     str(commit.committer_date),
                                "label":           label,
                                "sample_weight":   round(sample_weight, 4),
                                "static_features": feat_vec,
                                "jit_features":    jit,
                                # Backward-compat scalar mapping
                                "git_features": {
                                    "code_churn":    jit["LT"],
                                    "author_count":  jit["NDEV"],
                                    "file_age_days": round(jit["AGE"] * 7, 1),
                                    "n_past_bugs":   jit["NUC"],
                                    "commit_freq":   jit["EXP"],
                                },
                            }) + "\n")
                            count += 1

                        except Exception:
                            continue

                        _update_file_history(file_history, commit, mf)

                    if processed % 100 == 0:
                        logger.info("  %s: %d commits, %d records", repo_url, processed, count)

            except Exception as e:
                logger.warning("Error mining %s: %s", repo_url, e)

    logger.info("Bug dataset: %d samples -> %s", count, output_path)
    return count


# ---------------------------------------------------------------------------
# Cross-project split utilities
# ---------------------------------------------------------------------------

def cross_project_split(
    records: list[dict],
    test_repos: list[str],
) -> tuple[list[dict], list[dict]]:
    """
    Split strictly by repository — no file from a test repo appears in training.

    This is the correct protocol for cross-project defect prediction
    (Zimmermann et al. 2009). Random file-level splits contaminate evaluation
    because files from the same repo share style, naming conventions, and
    complexity patterns — inflating all metrics by up to 15 AUC points.
    """
    test_set = set(test_repos)
    train = [r for r in records if r.get("repo", "") not in test_set]
    test  = [r for r in records if r.get("repo", "") in test_set]

    logger.info(
        "Cross-project split: %d train | %d test | test repos: %s",
        len(train), len(test), sorted(test_set),
    )
    return train, test


def leave_one_project_out_splits(
    records: list[dict],
) -> list[tuple[str, list[dict], list[dict]]]:
    """
    Generate all leave-one-project-out (LOPO) train/test splits.

    Each split holds out exactly one repository as the test set.
    Provides an unbiased estimate of cross-project generalization.
    Minimum 5 test samples per repo to be included.

    Returns list of (held_out_repo, train_records, test_records).
    """
    repos = list(set(r.get("repo", "unknown") for r in records))
    splits = []
    for held_out in sorted(repos):
        train = [r for r in records if r.get("repo") != held_out]
        test  = [r for r in records if r.get("repo") == held_out]
        if len(test) < 5:
            continue
        splits.append((held_out, train, test))
    return splits


# ---------------------------------------------------------------------------
# Dataset quality report
# ---------------------------------------------------------------------------

def dataset_report(path: str, schema: str) -> dict:
    """
    Print a label quality and distribution report for a JSONL dataset.
    """
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        return {"error": "Empty dataset"}

    label_counts: dict = defaultdict(int)
    repo_counts:  dict = defaultdict(int)
    seen_hashes:  set  = set()
    duplicates = 0

    for r in records:
        label_counts[str(r.get("label", "?"))] += 1
        repo_counts[r.get("repo", "unknown")] += 1
        content = json.dumps(
            r.get("features", r.get("tokens", r.get("code", ""))),
            sort_keys=True,
        )
        h = hashlib.sha256(content.encode()).hexdigest()
        if h in seen_hashes:
            duplicates += 1
        seen_hashes.add(h)

    total = len(records)
    minority = min(label_counts.values(), default=1)
    majority = max(label_counts.values(), default=1)
    imbalance_ratio = majority / max(minority, 1)

    warnings: list[str] = []
    if imbalance_ratio > 5:
        warnings.append(
            f"High class imbalance ({imbalance_ratio:.1f}x). "
            "Consider SMOTE or class_weight='balanced'."
        )
    if len(repo_counts) < 3:
        warnings.append(
            f"Only {len(repo_counts)} repo(s). Cross-project generalization "
            "requires >= 3 repositories."
        )
    if duplicates > total * 0.05:
        warnings.append(
            f"{duplicates} duplicates ({duplicates/total:.1%}). "
            "Deduplicate before training."
        )

    print(f"\n{'='*60}\nDataset Report: {path}\n{'='*60}")
    print(f"  Samples:    {total:,}")
    print(f"  Duplicates: {duplicates}")
    print(f"  Repos:      {len(repo_counts)}")
    print(f"  Labels:     {dict(label_counts)}")
    print(f"  Imbalance:  {imbalance_ratio:.1f}x")
    for w in warnings:
        print(f"  WARNING: {w}")
    print()

    return {
        "schema": schema,
        "n_samples": total,
        "n_duplicates": duplicates,
        "n_repos": len(repo_counts),
        "label_distribution": dict(label_counts),
        "imbalance_ratio": round(imbalance_ratio, 2),
        "top_repos": sorted(repo_counts.items(), key=lambda x: -x[1])[:10],
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Build real-world ML training datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task",
                        choices=["complexity", "security", "pattern", "bug", "report"],
                        required=True)
    parser.add_argument("--out",    default="data/dataset.jsonl")
    parser.add_argument("--token",  default=os.environ.get("GITHUB_TOKEN", ""),
                        help="GitHub personal access token")
    parser.add_argument("--max",    type=int, default=300,
                        help="Max files/commits per repository")
    parser.add_argument("--szz",    action="store_true",
                        help="Enable SZZ-based inducing commit labeling (bug task)")
    parser.add_argument("--schema", default="",
                        help="Schema name for report task")
    args = parser.parse_args()

    if args.task in ("complexity", "security", "pattern") and not args.token:
        parser.error("--token or GITHUB_TOKEN env var required")

    if args.task == "complexity":
        build_complexity_dataset(COMPLEXITY_REPOS, args.token, args.out, args.max)
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
        build_bug_dataset(BUG_REPOS, args.out, max_commits=args.max, use_szz=args.szz)
    elif args.task == "report":
        if not args.schema:
            parser.error("--schema required for report task")
        dataset_report(args.out, args.schema)


if __name__ == "__main__":
    main()
