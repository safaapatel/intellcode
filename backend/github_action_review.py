"""
IntelliCode GitHub Actions Reviewer
=====================================
Self-contained script that:
  1. Reads the PR diff to find changed .py files
  2. Loads per-repo config from .intellicode.yml (optional)
  3. Runs IntelliCode analysis (no FastAPI server — imports models directly)
  4. Filters findings via inline `# intellicode: ignore` suppression comments
  5. Filters known false-positive paths (checkpoint loaders, test fixtures, etc.)
  6. Compares PR quality score against the base-branch score (delta)
  7. Posts findings as inline review comments via the GitHub REST API
  8. Optionally fails with exit code 1 if overall quality is below threshold

Environment variables (set by GitHub Actions):
  GITHUB_TOKEN          — automatically provided by Actions
  GITHUB_REPOSITORY     — "owner/repo"
  GITHUB_EVENT_PATH     — path to the webhook event JSON
  GITHUB_WORKSPACE      — root of the checked-out repo
  INTELLICODE_THRESHOLD — quality score threshold (0-100, default 60)
  INTELLICODE_FAIL_ON_ISSUES — "true" to fail CI when below threshold (default "false")

Per-repo configuration (.intellicode.yml at repo root):
  threshold: 65               # override INTELLICODE_THRESHOLD
  fail_on_issues: true        # override INTELLICODE_FAIL_ON_ISSUES
  ignore_paths:               # glob patterns to skip entirely
    - "tests/**"
    - "migrations/**"
  suppress_rules:             # rule IDs to silence globally
    - "insecure_deserialization"
    - "too_long"
  max_inline_comments: 20     # cap per review (default 30)

Inline suppression (in source code):
  pickle.loads(data)  # intellicode: ignore
  # intellicode: ignore[insecure_deserialization]

Usage (local trial):
  cd /path/to/repo
  GITHUB_TOKEN=... GITHUB_REPOSITORY=owner/repo GITHUB_EVENT_PATH=event.json \\
    python backend/github_action_review.py
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Bootstrap — add backend to path so we can import models directly
# ---------------------------------------------------------------------------
BACKEND = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND))

# ---------------------------------------------------------------------------
# Per-repo config (.intellicode.yml)
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: dict[str, Any] = {
    "threshold": 60.0,
    "fail_on_issues": False,
    "ignore_paths": [],
    "suppress_rules": [],
    "max_inline_comments": 30,
    # Confidence-gated fail rules. Each entry: {severity, confidence_threshold, count_threshold}
    # Example: fail if any critical finding has confidence >= 0.75
    "fail_on": [],
    # Use GitHub Checks API for annotations (requires checks:write permission)
    "use_checks_api": True,
    # Delta gating: fail if PR drops overall quality score by more than this many points.
    # Set to None (or omit) to disable delta gating entirely.
    # Example: max_score_drop: 5  -> blocks any PR that regresses quality by more than 5 pts
    "max_score_drop": None,
}

# Paths that commonly trigger false positives (checkpoint loaders, fixtures)
_AUTO_SUPPRESS_PATH_PATTERNS = [
    "*/checkpoints/*",
    "*/fixtures/*",
    "*/test_data/*",
    "*/conftest.py",
    "*/__pycache__/*",
]

# Rule IDs that are auto-suppressed when file matches checkpoint-like paths
_CHECKPOINT_RULES = {"insecure_deserialization"}


def load_config(workspace: Path) -> dict[str, Any]:
    """Load .intellicode.yml from the workspace root. Falls back to defaults."""
    config = dict(_DEFAULT_CONFIG)
    config_path = workspace / ".intellicode.yml"
    if not config_path.exists():
        return config
    try:
        # Use basic YAML-like parser to avoid pyyaml dependency
        raw = config_path.read_text(encoding="utf-8")
        parsed = _parse_simple_yaml(raw)
        config["threshold"] = float(parsed.get("threshold", config["threshold"]))
        config["fail_on_issues"] = str(parsed.get("fail_on_issues", "false")).lower() == "true"
        config["ignore_paths"] = list(parsed.get("ignore_paths", []))
        config["suppress_rules"] = [str(r) for r in parsed.get("suppress_rules", [])]
        config["max_inline_comments"] = int(parsed.get("max_inline_comments", 30))
        config["fail_on"] = list(parsed.get("fail_on", []))
        config["use_checks_api"] = parsed.get("use_checks_api", True)
        print(f"[info] Loaded .intellicode.yml: threshold={config['threshold']}, "
              f"ignore_paths={config['ignore_paths']}, suppress_rules={config['suppress_rules']}")
    except Exception as e:
        print(f"[warn] Failed to parse .intellicode.yml: {e} — using defaults")
    return config


def _parse_simple_yaml(text: str) -> dict:
    """
    Minimal YAML parser for .intellicode.yml — handles scalar keys and
    simple string lists. Avoids pyyaml dependency in CI.
    """
    result: dict[str, Any] = {}
    current_list_key: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- "):
            if current_list_key:
                value = stripped[2:].strip().strip('"').strip("'")
                result[current_list_key].append(value)
            continue
        if ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if val == "" or val == "[]":
                result[key] = []
                current_list_key = key if val == "" else None
            else:
                current_list_key = None
                if val.lower() == "true":
                    result[key] = True
                elif val.lower() == "false":
                    result[key] = False
                else:
                    try:
                        result[key] = int(val)
                    except ValueError:
                        try:
                            result[key] = float(val)
                        except ValueError:
                            result[key] = val
    return result


def _should_ignore_path(path: str, ignore_patterns: list[str]) -> bool:
    """Return True if path matches any ignore glob pattern."""
    for pattern in ignore_patterns + _AUTO_SUPPRESS_PATH_PATTERNS[:0]:  # user patterns only
        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(Path(path).name, pattern):
            return True
    return False


def _is_checkpoint_path(path: str) -> bool:
    """Return True if the file is likely a model checkpoint loader (not user data)."""
    for pattern in _AUTO_SUPPRESS_PATH_PATTERNS:
        if fnmatch.fnmatch(path, pattern):
            return True
    return False


# ---------------------------------------------------------------------------
# Inline suppression — # intellicode: ignore[rule] comments
# ---------------------------------------------------------------------------

def _build_suppression_map(source: str) -> dict[int, set[str]]:
    """
    Parse `# intellicode: ignore` and `# intellicode: ignore[rule1,rule2]`
    comments in source. Returns {lineno: set_of_suppressed_rule_ids}.
    An empty set means suppress ALL rules on that line.
    """
    suppressed: dict[int, set[str]] = {}
    pattern = re.compile(r"#\s*intellicode\s*:\s*ignore(?:\[([^\]]+)\])?", re.IGNORECASE)
    for lineno, line in enumerate(source.splitlines(), start=1):
        m = pattern.search(line)
        if m:
            rules_str = m.group(1)
            if rules_str:
                rules = {r.strip().lower() for r in rules_str.split(",")}
            else:
                rules = set()  # suppress all
            suppressed[lineno] = rules
    return suppressed


def _is_suppressed(
    rule_id: str,
    lineno: int | None,
    suppression_map: dict[int, set[str]],
    global_suppressed: set[str],
) -> bool:
    """Return True if this finding should be suppressed."""
    if rule_id.lower() in global_suppressed:
        return True
    if lineno and lineno in suppression_map:
        rules = suppression_map[lineno]
        if not rules or rule_id.lower() in rules:
            return True
    return False


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

GITHUB_API = "https://api.github.com"


def _github_request(
    method: str,
    path: str,
    body: dict | None = None,
    token: str | None = None,
) -> dict | list | None:
    url = f"{GITHUB_API}{path}"
    data = json.dumps(body).encode() if body else None
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "intellicode-review-bot/1.0",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if data:
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        body_text = e.read().decode(errors="replace")
        print(f"[warn] GitHub API {method} {path} -> {e.code}: {body_text[:300]}")
        return None
    except Exception as exc:
        print(f"[warn] GitHub API request failed: {exc}")
        return None


def get_pr_files(token: str, owner: str, repo: str, pr_number: int) -> list[dict]:
    result = _github_request("GET", f"/repos/{owner}/{repo}/pulls/{pr_number}/files", token=token)
    return result if isinstance(result, list) else []


def get_pr_commits(token: str, owner: str, repo: str, pr_number: int) -> list[str]:
    result = _github_request("GET", f"/repos/{owner}/{repo}/pulls/{pr_number}/commits", token=token)
    if not isinstance(result, list):
        return []
    return [c["sha"] for c in result]


def get_file_at_ref(token: str, owner: str, repo: str, path: str, ref: str) -> str | None:
    """Fetch raw file content at a specific git ref via GitHub API."""
    result = _github_request(
        "GET", f"/repos/{owner}/{repo}/contents/{path}?ref={ref}", token=token
    )
    if not isinstance(result, dict):
        return None
    import base64
    content = result.get("content", "")
    encoding = result.get("encoding", "")
    if encoding == "base64":
        try:
            return base64.b64decode(content.replace("\n", "")).decode("utf-8", errors="replace")
        except Exception:
            return None
    return None


def post_pr_review(
    token: str,
    owner: str,
    repo: str,
    pr_number: int,
    commit_sha: str,
    comments: list[dict],
    summary: str,
    event: str = "COMMENT",
) -> dict | None:
    body: dict[str, Any] = {
        "commit_id": commit_sha,
        "body": summary,
        "event": event,
        "comments": comments,
    }
    return _github_request(
        "POST", f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews",
        body=body, token=token,
    )


def post_issue_comment(
    token: str, owner: str, repo: str, pr_number: int, body: str
) -> dict | None:
    return _github_request(
        "POST", f"/repos/{owner}/{repo}/issues/{pr_number}/comments",
        body={"body": body}, token=token,
    )


def post_check_run(
    token: str,
    owner: str,
    repo: str,
    commit_sha: str,
    conclusion: str,
    summary: str,
    annotations: list[dict],
    title: str = "IntelliCode Review",
) -> dict | None:
    """
    Create a GitHub Check Run with line-level annotations.
    Annotations appear directly on diff lines in the PR — no manual mapping needed.

    conclusion: "success" | "failure" | "neutral" | "action_required"
    annotation keys: path, start_line, end_line, annotation_level, message, title
    annotation_level: "failure" | "warning" | "notice"

    Note: GitHub limits annotations to 50 per check run API call.
    We batch automatically for larger sets.
    """
    BATCH_SIZE = 50
    first_batch = annotations[:BATCH_SIZE]
    body: dict[str, Any] = {
        "name": "IntelliCode Review",
        "head_sha": commit_sha,
        "status": "completed",
        "conclusion": conclusion,
        "output": {
            "title": title,
            "summary": summary[:65535],  # GitHub limit
            "annotations": first_batch,
        },
    }
    result = _github_request(
        "POST", f"/repos/{owner}/{repo}/check-runs", body=body, token=token,
    )

    # Patch remaining annotation batches onto the same check run
    if result and isinstance(result, dict) and len(annotations) > BATCH_SIZE:
        check_id = result.get("id")
        if check_id:
            for i in range(BATCH_SIZE, len(annotations), BATCH_SIZE):
                batch = annotations[i:i + BATCH_SIZE]
                _github_request(
                    "PATCH",
                    f"/repos/{owner}/{repo}/check-runs/{check_id}",
                    body={"output": {"title": title, "summary": summary[:65535], "annotations": batch}},
                    token=token,
                )
    return result


def build_check_annotations(all_results: list[dict]) -> list[dict]:
    """
    Convert analysis results into GitHub Check Run annotation objects.
    Unlike PR review comments, annotations are not constrained to diff lines —
    they appear on ANY line in the file.
    """
    annotations = []
    LEVEL = {"critical": "failure", "high": "failure", "medium": "warning", "low": "notice"}

    for r in all_results:
        path = r.get("filename", "unknown")

        # Security annotations
        sec = r.get("security") or {}
        for vuln in sec.get("vulnerabilities", []):
            lineno = max(1, vuln.get("lineno") or 1)
            conf = vuln.get("confidence", 0.5)
            cwe = f" ({vuln['cwe']})" if vuln.get("cwe") else ""
            decision = vuln.get("decision", {})
            decision_label = decision.get("label", "")
            decision_note = (
                f"\n**Action:** {decision_label}" if decision_label else ""
            )
            annotations.append({
                "path": path,
                "start_line": lineno,
                "end_line": lineno,
                "annotation_level": LEVEL.get(vuln.get("severity", "low"), "notice"),
                "title": f"Security: {vuln.get('vuln_type', 'issue')}{cwe}",
                "message": (
                    f"{vuln.get('title', 'Security finding')}{decision_note}\n"
                    f"Confidence: {conf:.0%} | Severity: {vuln.get('severity', '?').upper()}\n"
                    f"{vuln.get('description', '')}"
                )[:1000],
            })

        # Complexity function issues
        comp = r.get("complexity") or {}
        for fi in comp.get("function_issues", []):
            lineno = max(1, fi.get("lineno") or 1)
            annotations.append({
                "path": path,
                "start_line": lineno,
                "end_line": lineno,
                "annotation_level": "warning",
                "title": f"Complexity: {fi.get('issue', 'issue')} in {fi.get('name', '?')}",
                "message": (
                    f"Function `{fi.get('name', '?')}` has {fi.get('issue', 'an issue')}. "
                    f"Consider splitting into smaller, focused functions."
                ),
            })

    return annotations


def _should_fail_on_confidence(all_results: list[dict], fail_on_rules: list[dict]) -> tuple[bool, str]:
    """
    Apply confidence-gated quality gate rules from .intellicode.yml `fail_on` list.

    Each rule: {severity: str, confidence_threshold: float, count_threshold: int}

    Returns (should_fail, reason_string).
    """
    if not fail_on_rules:
        return False, ""

    for rule in fail_on_rules:
        sev = str(rule.get("severity", "critical")).lower()
        conf_threshold = float(rule.get("confidence_threshold", 0.75))
        count_threshold = int(rule.get("count_threshold", 1))

        matching = []
        for r in all_results:
            sec = r.get("security") or {}
            for vuln in sec.get("vulnerabilities", []):
                if (vuln.get("severity", "").lower() == sev
                        and vuln.get("confidence", 0.0) >= conf_threshold):
                    matching.append(vuln)

        if len(matching) >= count_threshold:
            return (
                True,
                f"{len(matching)} {sev} finding(s) with confidence >= {conf_threshold:.0%} "
                f"(threshold: {count_threshold})",
            )

    return False, ""


# ---------------------------------------------------------------------------
# Analysis — import models directly (no server)
# ---------------------------------------------------------------------------

def _score_from_probability(prob: float) -> float:
    return round((1.0 - max(0.0, min(1.0, prob))) * 100, 1)


def _grade(score: float) -> str:
    if score >= 85: return "A"
    if score >= 70: return "B"
    if score >= 55: return "C"
    if score >= 40: return "D"
    return "F"


def _load_analyzer():
    """
    Lazy-import the IntelliCode analysis stack.
    Returns analyze_fn(source, filename) -> dict.
    """
    from models.complexity_prediction import ComplexityPredictionModel
    from models.security_detection import EnsembleSecurityModel
    from models.bug_predictor import BugPredictionModel
    from models.pattern_recognition import PatternRecognitionModel

    complexity_pred = ComplexityPredictionModel()
    security_det = EnsembleSecurityModel()
    bug_pred = BugPredictionModel()
    pattern_rec = PatternRecognitionModel()

    def analyze(source: str, filename: str) -> dict:
        results: dict[str, Any] = {"filename": filename}
        errors: list[str] = []

        try:
            c = complexity_pred.predict(source)
            results["complexity"] = {
                "score": c.score,
                "grade": c.grade,
                "cognitive": c.cognitive,
                "cyclomatic": c.cyclomatic,
                "function_issues": [
                    {"name": fi.name, "lineno": fi.lineno, "cyclomatic": fi.cyclomatic,
                     "body_lines": fi.body_lines, "n_params": fi.n_params, "issue": fi.issue}
                    for fi in (c.function_issues or [])
                ],
            }
        except Exception as e:
            errors.append(f"complexity: {e}")
            results["complexity"] = None

        try:
            vulns: list = security_det.predict(source)
            sev_penalty = {"critical": 40, "high": 25, "medium": 12, "low": 5}
            sec_score = 100.0
            for v in vulns:
                sec_score -= sev_penalty.get(getattr(v, "severity", "").lower(), 5)
            sec_score = max(0.0, sec_score)
            results["security"] = {
                "score": round(sec_score, 1),
                "grade": _grade(sec_score),
                "vulnerabilities": [
                    {"vuln_type": v.vuln_type, "severity": v.severity,
                     "confidence": v.confidence, "lineno": v.lineno,
                     "title": v.title, "description": v.description,
                     "snippet": v.snippet, "cwe": getattr(v, "cwe", "")}
                    for v in vulns
                ],
            }
        except Exception as e:
            errors.append(f"security: {e}")
            results["security"] = None

        try:
            b = bug_pred.predict(source)
            bug_score = _score_from_probability(b.bug_probability)
            results["bug"] = {
                "score": bug_score,
                "grade": _grade(bug_score),
                "risk_level": b.risk_level,
                "bug_probability": round(b.bug_probability, 3),
                "risk_factors": list(b.risk_factors or []),
            }
        except Exception as e:
            errors.append(f"bug: {e}")
            results["bug"] = None

        try:
            p = pattern_rec.predict(source)
            label_scores = {"clean": 100.0, "style_violation": 70.0,
                            "code_smell": 45.0, "anti_pattern": 25.0}
            pat_score = label_scores.get(p.label, 60.0)
            smells = []
            if p.label != "clean":
                smells.append({
                    "name": p.label,
                    "severity": "high" if p.label == "anti_pattern" else "medium",
                    "lineno": None,
                    "description": f"Detected pattern: {p.label} (confidence {p.confidence:.2f})",
                })
            results["pattern"] = {
                "score": pat_score,
                "grade": _grade(pat_score),
                "label": p.label,
                "confidence": round(p.confidence, 3),
                "smells": smells,
            }
        except Exception as e:
            errors.append(f"pattern: {e}")
            results["pattern"] = None

        scores = [
            float(results[k]["score"])
            for k in ("complexity", "security", "bug", "pattern")
            if results.get(k) and isinstance(results[k].get("score"), (int, float))
        ]
        results["overall_score"] = round(sum(scores) / len(scores), 1) if scores else 0.0
        results["errors"] = errors
        return results

    return analyze


# ---------------------------------------------------------------------------
# Base-branch delta — analyse the same files on the base branch
# ---------------------------------------------------------------------------

def compute_base_scores(
    token: str,
    owner: str,
    repo: str,
    base_ref: str,
    py_files: list[dict],
    analyze,
    workspace: Path,
) -> dict[str, float]:
    """
    For each changed file, fetch the base-branch version and analyse it.
    Returns {filename: overall_score}. Skips files that are new (no base version).

    Tries git locally first (fast), falls back to GitHub API.
    """
    base_scores: dict[str, float] = {}
    for file_info in py_files:
        path = file_info["filename"]
        if file_info.get("status") == "added":
            continue  # new file — no base to compare

        # Try local git first
        source: str | None = None
        try:
            result = subprocess.run(
                ["git", "show", f"{base_ref}:{path}"],
                cwd=str(workspace),
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                source = result.stdout
        except Exception:
            pass

        # Fall back to GitHub API
        if source is None and token:
            source = get_file_at_ref(token, owner, repo, path, base_ref)

        if source is None:
            continue

        try:
            r = analyze(source, path)
            base_scores[path] = r["overall_score"]
        except Exception:
            pass

    return base_scores


# ---------------------------------------------------------------------------
# Filtering — apply suppression rules and path ignores to findings
# ---------------------------------------------------------------------------

def filter_findings(
    file_result: dict,
    path: str,
    suppression_map: dict[int, set[str]],
    global_suppressed: set[str],
    is_checkpoint: bool,
) -> dict:
    """Return a copy of file_result with suppressed findings removed."""
    import copy
    r = copy.deepcopy(file_result)

    effective_suppressed = set(global_suppressed)
    if is_checkpoint:
        effective_suppressed |= _CHECKPOINT_RULES

    # Filter security vulnerabilities
    if r.get("security") and r["security"].get("vulnerabilities"):
        original = r["security"]["vulnerabilities"]
        filtered = [
            v for v in original
            if not _is_suppressed(
                v.get("vuln_type", ""),
                v.get("lineno"),
                suppression_map,
                effective_suppressed,
            )
        ]
        removed = len(original) - len(filtered)
        if removed:
            print(f"[info]     Suppressed {removed} security finding(s) in {path}")
        r["security"]["vulnerabilities"] = filtered
        # Recalculate score
        sev_penalty = {"critical": 40, "high": 25, "medium": 12, "low": 5}
        sec_score = 100.0
        for v in filtered:
            sec_score -= sev_penalty.get(v.get("severity", "").lower(), 5)
        r["security"]["score"] = round(max(0.0, sec_score), 1)
        r["security"]["grade"] = _grade(r["security"]["score"])

    # Filter complexity issues
    if r.get("complexity") and r["complexity"].get("function_issues"):
        original = r["complexity"]["function_issues"]
        filtered = [
            fi for fi in original
            if not _is_suppressed(
                fi.get("issue", ""),
                fi.get("lineno"),
                suppression_map,
                effective_suppressed,
            )
        ]
        r["complexity"]["function_issues"] = filtered

    # Filter pattern smells
    if r.get("pattern") and r["pattern"].get("smells"):
        original = r["pattern"]["smells"]
        filtered = [
            s for s in original
            if not _is_suppressed(
                s.get("name", ""),
                s.get("lineno"),
                suppression_map,
                effective_suppressed,
            )
        ]
        r["pattern"]["smells"] = filtered

    # Recalculate overall score
    scores = [
        float(r[k]["score"])
        for k in ("complexity", "security", "bug", "pattern")
        if r.get(k) and isinstance(r[k].get("score"), (int, float))
    ]
    r["overall_score"] = round(sum(scores) / len(scores), 1) if scores else 0.0
    return r


# ---------------------------------------------------------------------------
# Comment generation
# ---------------------------------------------------------------------------

SEVERITY_LABEL = {
    "critical": "!! CRITICAL",
    "high": "! HIGH",
    "medium": "MEDIUM",
    "low": "LOW",
    "info": "INFO",
}


def _severity_sort_key(severity: str) -> int:
    return {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}.get(severity.lower(), 5)


# ---------------------------------------------------------------------------
# Fix suggestion generation — produces GitHub "```suggestion" blocks
# ---------------------------------------------------------------------------

def _generate_fix_suggestion(
    vuln_type: str,
    source_lines: list[str],
    lineno: int | None,
) -> str | None:
    """
    Attempt to generate a single-line GitHub suggestion block for common vuln types.
    Returns a markdown ```suggestion block string, or None when no auto-fix applies.

    GitHub renders suggestion blocks as "Apply suggestion" buttons on PR diff lines,
    replacing the anchored line with the fixed version on click.
    """
    if not lineno or not source_lines or lineno > len(source_lines):
        return None

    original = source_lines[lineno - 1]

    if vuln_type == "sql_injection":
        # f-string in execute: cursor.execute(f"... {var} ...")
        m = re.search(
            r'((?:cursor|conn|db|c)\s*\.\s*execute\s*\()\s*f(["\'])(.+?)\2',
            original,
        )
        if m:
            prefix_call = m.group(1)
            query_body = m.group(3)
            params = re.findall(r'\{(\w+(?:\.\w+)*)\}', query_body)
            if params:
                safe_query = re.sub(r'\{[^}]+\}', '?', query_body)
                params_str = (
                    f"({params[0]},)" if len(params) == 1
                    else f"({', '.join(params)})"
                )
                indent = original[: len(original) - len(original.lstrip())]
                fixed = f'{indent}{prefix_call}"{safe_query}", {params_str})'
                # Preserve trailing comment if present
                after = original[m.end():]
                trailing = re.match(r'\s*#.*', after)
                if trailing:
                    fixed += trailing.group(0)
                return f"```suggestion\n{fixed}\n```"

        # %-formatting: execute("... %s ..." % var)
        m2 = re.search(
            r'((?:cursor|conn|db|c)\s*\.\s*execute\s*\()\s*(["\'].+?["\'])\s*%\s*(.+?)\)',
            original,
        )
        if m2:
            indent = original[: len(original) - len(original.lstrip())]
            query_raw = m2.group(2).strip('"\'')
            safe_query = re.sub(r'%s', '?', query_raw)
            param_expr = m2.group(3).strip()
            if not (param_expr.startswith('(') and param_expr.endswith(')')):
                param_expr = f"({param_expr},)"
            fixed = f'{indent}{m2.group(1)}"{safe_query}", {param_expr})'
            return f"```suggestion\n{fixed}\n```"
        return None

    elif vuln_type == "hardcoded_secret":
        # VAR = "literal" or VAR = 'literal'
        m = re.match(r'^(\s*)(\w+)\s*=\s*(["\'])(.+?)\3', original)
        if m:
            indent = m.group(1)
            var_name = m.group(2)
            # Try to infer a sensible env var name
            env_key = re.sub(r'[^A-Z0-9_]', '_', var_name.upper())
            fixed = f'{indent}{var_name} = os.environ.get("{env_key}")'
            return f"```suggestion\n{fixed}\n```"
        return None

    elif vuln_type == "weak_cryptography":
        # hashlib.md5 / hashlib.sha1 -> hashlib.sha256
        fixed = re.sub(r'\bhashlib\.(md5|sha1|sha)\b', 'hashlib.sha256', original)
        if fixed != original:
            return f"```suggestion\n{fixed.rstrip()}\n```"
        # MD5() / SHA1() bare calls (e.g. from Crypto)
        fixed2 = re.sub(r'\bMD5\s*\(', 'SHA256(', re.sub(r'\bSHA1\s*\(', 'SHA256(', original))
        if fixed2 != original:
            return f"```suggestion\n{fixed2.rstrip()}\n```"
        return None

    elif vuln_type == "code_injection":
        # eval(...) -> ast.literal_eval(...)
        fixed = re.sub(r'\beval\s*\(', 'ast.literal_eval(', original)
        if fixed != original:
            return f"```suggestion\n{fixed.rstrip()}\n```"
        # exec(...) -> no clean one-liner replacement, but flag it
        return None

    elif vuln_type == "command_injection":
        # os.system(cmd) -> subprocess.run([cmd], shell=False)
        m = re.match(r'^(\s*)(.*)os\.system\s*\((.+?)\)(.*)', original)
        if m:
            indent, pre, cmd_arg, post = m.group(1), m.group(2), m.group(3), m.group(4)
            fixed = f'{indent}{pre}subprocess.run({cmd_arg}, shell=False){post}'
            return f"```suggestion\n{fixed.rstrip()}\n```"
        # shell=True -> shell=False
        if 'shell=True' in original:
            fixed = original.replace('shell=True', 'shell=False', 1)
            return f"```suggestion\n{fixed.rstrip()}\n```"
        # subprocess.call(cmd) without shell arg — add shell=False
        m2 = re.search(r'subprocess\.(call|run|Popen)\s*\(', original)
        if m2 and 'shell=' not in original:
            insert_at = m2.end()
            fixed = original[:insert_at] + 'shell=False, ' + original[insert_at:]
            return f"```suggestion\n{fixed.rstrip()}\n```"
        return None

    elif vuln_type == "path_traversal":
        # open(user_path, ...) -> open(os.path.basename(user_path), ...)
        m = re.search(r'\bopen\s*\(\s*([^,\)]+)', original)
        if m:
            path_arg = m.group(1).strip()
            # Skip if already safe
            if 'basename' not in path_arg and 'resolve' not in path_arg:
                fixed = original.replace(
                    m.group(0),
                    m.group(0).replace(path_arg, f'os.path.basename({path_arg})'),
                    1,
                )
                return f"```suggestion\n{fixed.rstrip()}\n```"
        return None

    elif vuln_type == "insecure_deserialization":
        # pickle.loads / yaml.load -> suggest safe alternatives
        if 'pickle.loads' in original or 'pickle.load(' in original:
            indent = original[: len(original) - len(original.lstrip())]
            return (
                "```suggestion\n"
                f"{indent}# FIXME: Replace pickle with a safe format (json, msgpack, etc.)\n"
                "```"
            )
        if re.search(r'\byaml\.load\s*\(', original) and 'Loader=' not in original:
            fixed = re.sub(
                r'\byaml\.load\s*\((.+?)\)',
                r'yaml.safe_load(\1)',
                original,
            )
            if fixed != original:
                return f"```suggestion\n{fixed.rstrip()}\n```"
        return None

    elif vuln_type == "xxe":
        # xml.etree.ElementTree.parse / lxml without defusedxml
        if 'etree' in original and ('parse' in original or 'fromstring' in original):
            indent = original[: len(original) - len(original.lstrip())]
            return (
                "```suggestion\n"
                f"{indent}# FIXME: Use defusedxml.ElementTree instead of xml.etree.ElementTree\n"
                "```"
            )
        return None

    return None


def _generate_fix_suggestion_js(
    vuln_type: str,
    source_lines: list[str],
    lineno: int | None,
) -> str | None:
    """
    Generate GitHub suggestion blocks for JavaScript / TypeScript vulnerabilities.
    Same contract as _generate_fix_suggestion but for JS/TS source.
    """
    if not lineno or not source_lines or lineno > len(source_lines):
        return None

    original = source_lines[lineno - 1]

    if vuln_type == "sql_injection":
        # Template literal in query: db.query(`SELECT ... ${var}`)
        m = re.search(
            r'((?:db|pool|client|conn|connection|knex)\s*\.\s*(?:query|execute|run)\s*\()\s*`(.+?)`',
            original,
        )
        if m:
            call_prefix = m.group(1)
            query_body = m.group(2)
            params = re.findall(r'\$\{(\w+(?:\.\w+)*)\}', query_body)
            if params:
                safe_query = re.sub(r'\$\{[^}]+\}', '?', query_body)
                params_arr = f"[{', '.join(params)}]"
                indent = original[: len(original) - len(original.lstrip())]
                fixed = f'{indent}{call_prefix}"{safe_query}", {params_arr})'
                return f"```suggestion\n{fixed}\n```"
        # String concatenation: "SELECT ... " + var
        m2 = re.search(
            r'((?:db|pool|client|conn)\s*\.\s*(?:query|execute)\s*\()\s*(["\'].+?["\'])\s*\+',
            original,
        )
        if m2:
            indent = original[: len(original) - len(original.lstrip())]
            return (
                "```suggestion\n"
                f"{indent}// FIXME: Use parameterised query instead of string concatenation\n"
                "```"
            )
        return None

    elif vuln_type == "hardcoded_secret":
        # const/let/var TOKEN = "literal"
        m = re.match(
            r'^(\s*)(?:const|let|var)\s+(\w+)\s*=\s*(["\'])(.+?)\3',
            original,
        )
        if m:
            indent = m.group(1)
            decl = "const"
            var_name = m.group(2)
            env_key = re.sub(r'[^A-Z0-9_]', '_', var_name.upper())
            fixed = f'{indent}{decl} {var_name} = process.env.{env_key}'
            return f"```suggestion\n{fixed}\n```"
        return None

    elif vuln_type == "code_injection":
        # eval(...)
        fixed = re.sub(r'\beval\s*\(', 'JSON.parse(', original)
        if fixed != original:
            return f"```suggestion\n{fixed.rstrip()}\n```"
        # new Function(...)
        if 'new Function(' in original:
            indent = original[: len(original) - len(original.lstrip())]
            return (
                "```suggestion\n"
                f"{indent}// FIXME: Avoid new Function() — refactor to a static function\n"
                "```"
            )
        return None

    elif vuln_type == "command_injection":
        # child_process.exec(cmd) -> child_process.execFile([...], {shell: false})
        m = re.search(
            r'(?:exec|execSync)\s*\(\s*`([^`]+)`',
            original,
        )
        if m:
            indent = original[: len(original) - len(original.lstrip())]
            return (
                "```suggestion\n"
                f"{indent}// FIXME: Use execFile with an array of args to avoid shell injection\n"
                "```"
            )
        if re.search(r'shell\s*:\s*true', original):
            fixed = re.sub(r'shell\s*:\s*true', 'shell: false', original, count=1)
            return f"```suggestion\n{fixed.rstrip()}\n```"
        return None

    elif vuln_type == "xss":
        # innerHTML = ... -> textContent = ...
        if 'innerHTML' in original and '=' in original:
            fixed = original.replace('innerHTML', 'textContent', 1)
            return f"```suggestion\n{fixed.rstrip()}\n```"
        # document.write(...)
        if 'document.write(' in original:
            indent = original[: len(original) - len(original.lstrip())]
            return (
                "```suggestion\n"
                f"{indent}// FIXME: Replace document.write() with safe DOM manipulation\n"
                "```"
            )
        return None

    elif vuln_type == "path_traversal":
        # fs.readFile(userPath, ...) -> add path.basename()
        m = re.search(r'(?:fs\.readFile|fs\.readFileSync|fs\.writeFile)\s*\(\s*([^,\)]+)', original)
        if m:
            path_arg = m.group(1).strip()
            if 'basename' not in path_arg and 'resolve' not in path_arg:
                fixed = original.replace(
                    m.group(0),
                    m.group(0).replace(path_arg, f'path.basename({path_arg})'),
                    1,
                )
                return f"```suggestion\n{fixed.rstrip()}\n```"
        return None

    elif vuln_type == "weak_cryptography":
        # crypto.createHash('md5') -> crypto.createHash('sha256')
        fixed = re.sub(r"createHash\s*\(\s*['\"](?:md5|sha1)['\"]\s*\)",
                       "createHash('sha256')", original)
        if fixed != original:
            return f"```suggestion\n{fixed.rstrip()}\n```"
        return None

    elif vuln_type == "insecure_deserialization":
        # JSON.parse is safe; flag custom deserializers
        if re.search(r'deserializ|fromJSON|decode\s*\(', original, re.IGNORECASE):
            indent = original[: len(original) - len(original.lstrip())]
            return (
                "```suggestion\n"
                f"{indent}// FIXME: Validate schema after deserialisation (use Zod, Joi, etc.)\n"
                "```"
            )
        return None

    return None


def _generate_fix_suggestion_java(
    vuln_type: str,
    source_lines: list[str],
    lineno: int | None,
) -> str | None:
    """
    Generate GitHub suggestion blocks for Java vulnerabilities.
    """
    if not lineno or not source_lines or lineno > len(source_lines):
        return None

    original = source_lines[lineno - 1]

    if vuln_type == "sql_injection":
        # Statement.execute("SELECT ... " + var)  ->  PreparedStatement
        if re.search(r'(?:execute|executeQuery|executeUpdate)\s*\(["\']', original):
            indent = original[: len(original) - len(original.lstrip())]
            return (
                "```suggestion\n"
                f"{indent}// FIXME: Use PreparedStatement with ? placeholders instead of string concat\n"
                "```"
            )
        return None

    elif vuln_type == "hardcoded_secret":
        # String TOKEN = "literal"
        m = re.match(
            r'^(\s*)(?:(?:private|public|protected|static|final)\s+)*String\s+(\w+)\s*=\s*"(.+?)"',
            original,
        )
        if m:
            indent = m.group(1)
            var_name = m.group(2)
            env_key = re.sub(r'[^A-Z0-9_]', '_', var_name.upper())
            fixed = f'{indent}String {var_name} = System.getenv("{env_key}");'
            return f"```suggestion\n{fixed}\n```"
        return None

    elif vuln_type == "command_injection":
        # Runtime.exec("..." + var)
        if re.search(r'Runtime\.getRuntime\(\)\.exec\(', original):
            indent = original[: len(original) - len(original.lstrip())]
            return (
                "```suggestion\n"
                f"{indent}// FIXME: Use ProcessBuilder with a String[] args array to avoid shell injection\n"
                "```"
            )
        return None

    elif vuln_type == "weak_cryptography":
        # MessageDigest.getInstance("MD5")
        fixed = re.sub(
            r'MessageDigest\.getInstance\s*\(\s*"(?:MD5|SHA-1|SHA1)"\s*\)',
            'MessageDigest.getInstance("SHA-256")',
            original,
        )
        if fixed != original:
            return f"```suggestion\n{fixed.rstrip()}\n```"
        return None

    elif vuln_type == "insecure_deserialization":
        if re.search(r'ObjectInputStream|readObject\(\)', original):
            indent = original[: len(original) - len(original.lstrip())]
            return (
                "```suggestion\n"
                f"{indent}// FIXME: Replace Java deserialization with Jackson or Protocol Buffers\n"
                "```"
            )
        return None

    elif vuln_type == "xxe":
        if re.search(r'DocumentBuilderFactory|SAXParserFactory|XMLInputFactory', original):
            indent = original[: len(original) - len(original.lstrip())]
            return (
                "```suggestion\n"
                f"{indent}// FIXME: Disable external entity processing — set FEATURE_SECURE_PROCESSING\n"
                "```"
            )
        return None

    return None


def build_review_comments(
    file_result: dict,
    diff_lines: set[int],
    path: str,
    source_lines: list[str] | None = None,
) -> list[dict]:
    """
    Build inline GitHub review comment objects for one file.

    When source_lines is provided, security findings include a ```suggestion block
    that GitHub renders as a clickable "Apply suggestion" button, letting the reviewer
    commit the fix directly from the PR diff view.
    """
    comments: list[dict] = []

    def nearest_diff_line(lineno: int | None) -> int | None:
        if not diff_lines:
            return None
        if lineno and lineno in diff_lines:
            return lineno
        if lineno:
            for delta in range(1, 11):
                if lineno + delta in diff_lines:
                    return lineno + delta
                if lineno - delta in diff_lines and lineno - delta > 0:
                    return lineno - delta
        return min(diff_lines)

    # Security
    sec = file_result.get("security") or {}
    for vuln in sorted(sec.get("vulnerabilities", []),
                       key=lambda v: _severity_sort_key(v.get("severity", "info"))):
        lineno = vuln.get("lineno")
        line = nearest_diff_line(lineno)
        if line is None:
            continue

        sev_tag = SEVERITY_LABEL.get(vuln.get("severity", "").lower(), "ISSUE")
        cwe = f" ({vuln['cwe']})" if vuln.get("cwe") else ""
        suppress_hint = (
            f"\n\n> To suppress: add `# intellicode: ignore[{vuln['vuln_type']}]` "
            f"to the flagged line."
        )

        body = (
            f"**[IntelliCode Security - {sev_tag}]** {vuln['title']}{cwe}\n\n"
            f"{vuln['description']}{suppress_hint}\n"
        )

        # Attach a GitHub suggestion block when we can generate a safe one-liner fix
        lang = file_result.get("language", "python").lower()
        if lang in ("javascript", "typescript"):
            suggestion = _generate_fix_suggestion_js(
                vuln.get("vuln_type", ""),
                source_lines or [],
                lineno,
            )
        elif lang == "java":
            suggestion = _generate_fix_suggestion_java(
                vuln.get("vuln_type", ""),
                source_lines or [],
                lineno,
            )
        else:
            suggestion = _generate_fix_suggestion(
                vuln.get("vuln_type", ""),
                source_lines or [],
                lineno,
            )
        if suggestion:
            body += f"\n**Suggested fix:**\n{suggestion}\n"
        elif vuln.get("snippet"):
            # Fall back to showing the vulnerable snippet for context
            body += f"\n```python\n{vuln['snippet'][:300]}\n```\n"

        comments.append({"path": path, "line": line, "side": "RIGHT", "body": body})

    # Complexity
    comp = file_result.get("complexity") or {}
    for fi in comp.get("function_issues", []):
        line = nearest_diff_line(fi.get("lineno"))
        if line is None:
            continue
        label = {
            "high_complexity": "High Complexity",
            "too_long": "Function Too Long",
            "too_many_params": "Too Many Parameters",
        }.get(fi.get("issue", ""), fi.get("issue", ""))

        metrics = []
        if fi.get("cyclomatic"):
            metrics.append(f"cyclomatic={fi['cyclomatic']}")
        if fi.get("body_lines"):
            metrics.append(f"lines={fi['body_lines']}")
        if fi.get("n_params"):
            metrics.append(f"params={fi['n_params']}")
        metrics_str = f" ({', '.join(metrics)})" if metrics else ""

        body = (
            f"**[IntelliCode Complexity]** `{fi['name']}`: {label}{metrics_str}\n\n"
            f"Consider breaking this into smaller, focused functions. "
            f"Functions with cyclomatic complexity > 10 or > 50 lines are harder to test "
            f"and significantly more likely to contain defects.\n\n"
            f"> To suppress: add `# intellicode: ignore[{fi['issue']}]` to the function definition."
        )
        comments.append({"path": path, "line": line, "side": "RIGHT", "body": body})

    # Patterns
    pat = file_result.get("pattern") or {}
    for smell in pat.get("smells", []):
        line = nearest_diff_line(smell.get("lineno"))
        if line is None:
            continue
        body = (
            f"**[IntelliCode Pattern]** `{smell['name']}`"
            + (f": {smell['description']}" if smell.get("description") else "")
        )
        comments.append({"path": path, "line": line, "side": "RIGHT", "body": body})

    return comments


def _grade_bar(score: float) -> str:
    filled = int(score / 10)
    return "[" + "#" * filled + "-" * (10 - filled) + f"] {score:.0f}/100"


def _delta_str(current: float, base: float | None) -> str:
    if base is None:
        return ""
    delta = current - base
    sign = "+" if delta >= 0 else ""
    arrow = "^" if delta >= 0 else "v"
    return f"  ({arrow} {sign}{delta:.1f} vs base)"


def build_summary(
    all_results: list[dict],
    threshold: float,
    base_scores: dict[str, float] | None = None,
) -> tuple[str, str]:
    """Build the PR review summary. Returns (markdown, event_type)."""
    lines = ["## IntelliCode Review\n"]
    base_scores = base_scores or {}

    total_vulns = total_complex = total_smells = 0
    overall_scores: list[float] = []
    base_overall: list[float] = []

    for r in all_results:
        filename = r["filename"]
        score = r.get("overall_score", 0.0)
        overall_scores.append(score)
        base = base_scores.get(filename)
        if base is not None:
            base_overall.append(base)

        sec = r.get("security") or {}
        comp = r.get("complexity") or {}
        pat = r.get("pattern") or {}
        bug = r.get("bug") or {}

        vulns = sec.get("vulnerabilities", [])
        smells = pat.get("smells", [])
        fi = comp.get("function_issues", [])
        total_vulns += len(vulns)
        total_smells += len(smells)
        total_complex += len(fi)

        lines.append(
            f"### `{filename}`\n"
            f"Overall: {_grade_bar(score)}{_delta_str(score, base)}\n"
            f"Security:{sec.get('grade','?')}  "
            f"Complexity:{comp.get('grade','?')}  "
            f"Bug-risk:{bug.get('grade','?')}  "
            f"Patterns:{pat.get('grade','?')}\n"
        )
        if vulns:
            crit = [v for v in vulns if v.get("severity") == "critical"]
            high = [v for v in vulns if v.get("severity") == "high"]
            lines.append(
                f"- {len(vulns)} security finding(s)"
                + (f" ({len(crit)} critical, {len(high)} high)" if crit or high else "")
            )
        if fi:
            lines.append(f"- {len(fi)} complex function(s)")
        if smells:
            lines.append(f"- {len(smells)} code smell(s)")
        if bug.get("risk_factors"):
            top = bug["risk_factors"][:2]
            lines.append(f"- Bug risk factors: {'; '.join(top)}")
        lines.append("")

    mean_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
    mean_base = sum(base_overall) / len(base_overall) if base_overall else None

    lines.append("---")
    delta_part = _delta_str(mean_score, mean_base) if mean_base is not None else ""
    lines.append(
        f"**Files reviewed:** {len(all_results)}  |  "
        f"**Avg quality score:** {mean_score:.1f}/100{delta_part}"
    )
    lines.append(
        f"**Total findings:** {total_vulns} security, "
        f"{total_complex} complexity, {total_smells} pattern"
    )

    if mean_score < threshold:
        lines.append(
            f"\n> Quality score {mean_score:.1f} is below the configured threshold of {threshold:.0f}. "
            f"Add `# intellicode: ignore` to suppress individual findings, or update "
            f"`.intellicode.yml` to adjust the threshold."
        )
        event = "REQUEST_CHANGES"
    else:
        lines.append(
            f"\n> Quality score {mean_score:.1f} meets the configured threshold of {threshold:.0f}."
        )
        event = "COMMENT"

    lines.append(
        "\n*Powered by [IntelliCode](https://github.com/safaapatel/intellcode) -- "
        "ML-based code quality analysis*"
    )
    return "\n".join(lines), event


# ---------------------------------------------------------------------------
# SARIF output — GitHub Code Scanning (Security tab)
# ---------------------------------------------------------------------------

_SARIF_SEVERITY_MAP = {
    "critical": "error",
    "high":     "error",
    "medium":   "warning",
    "low":      "note",
    "info":     "note",
}

# CWE -> OWASP/SARIF rule definitions (id, name, full description)
_SARIF_RULES: dict[str, dict] = {
    "sql_injection":             {"id": "IC-SEC-001", "name": "SQLInjection",       "cwe": "CWE-89"},
    "command_injection":         {"id": "IC-SEC-002", "name": "CommandInjection",   "cwe": "CWE-78"},
    "path_traversal":            {"id": "IC-SEC-003", "name": "PathTraversal",      "cwe": "CWE-22"},
    "hardcoded_secret":          {"id": "IC-SEC-004", "name": "HardcodedSecret",    "cwe": "CWE-798"},
    "weak_cryptography":         {"id": "IC-SEC-005", "name": "WeakCryptography",   "cwe": "CWE-327"},
    "insecure_deserialization":  {"id": "IC-SEC-006", "name": "InsecureDeserialization", "cwe": "CWE-502"},
    "code_injection":            {"id": "IC-SEC-007", "name": "CodeInjection",      "cwe": "CWE-94"},
    "xxe":                       {"id": "IC-SEC-008", "name": "XXE",                "cwe": "CWE-611"},
    "xss":                       {"id": "IC-SEC-009", "name": "XSS",                "cwe": "CWE-79"},
    "high_complexity":           {"id": "IC-CC-001",  "name": "HighComplexity",     "cwe": None},
    "too_long":                  {"id": "IC-CC-002",  "name": "FunctionTooLong",    "cwe": None},
    "too_many_params":           {"id": "IC-CC-003",  "name": "TooManyParameters",  "cwe": None},
    "code_smell":                {"id": "IC-PAT-001", "name": "CodeSmell",          "cwe": None},
    "anti_pattern":              {"id": "IC-PAT-002", "name": "AntiPattern",        "cwe": None},
}


def build_sarif(all_results: list[dict]) -> dict:
    """
    Build a SARIF 2.1.0 document from IntelliCode analysis results.

    The output can be uploaded to GitHub Code Scanning via:
        gh api /repos/{owner}/{repo}/code-scanning/sarifs --input sarif.json
    or via the upload-sarif action.

    SARIF spec: https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html
    """
    # Collect unique rule IDs referenced in results
    referenced_rule_ids: set[str] = set()
    results_list: list[dict] = []

    for file_result in all_results:
        path = file_result.get("filename", "unknown")

        # Security vulnerabilities
        sec = file_result.get("security") or {}
        for vuln in sec.get("vulnerabilities", []):
            vtype = vuln.get("vuln_type", "unknown")
            rule = _SARIF_RULES.get(vtype, {"id": f"IC-SEC-{vtype}", "name": vtype, "cwe": None})
            rule_id = rule["id"]
            referenced_rule_ids.add(vtype)

            lineno = vuln.get("lineno") or 1
            sev = vuln.get("severity", "medium")

            sarif_result = {
                "ruleId": rule_id,
                "level": _SARIF_SEVERITY_MAP.get(sev, "warning"),
                "message": {"text": f"{vuln.get('title', vtype)}: {vuln.get('description', '')}"},
                "locations": [{
                    "physicalLocation": {
                        "artifactLocation": {"uri": path, "uriBaseId": "%SRCROOT%"},
                        "region": {"startLine": lineno},
                    }
                }],
            }
            if rule.get("cwe"):
                sarif_result["taxa"] = [{"id": rule["cwe"], "toolComponent": {"name": "CWE"}}]
            results_list.append(sarif_result)

        # Complexity issues
        comp = file_result.get("complexity") or {}
        for fi in comp.get("function_issues", []):
            issue = fi.get("issue", "high_complexity")
            rule = _SARIF_RULES.get(issue, {"id": f"IC-CC-{issue}", "name": issue, "cwe": None})
            rule_id = rule["id"]
            referenced_rule_ids.add(issue)

            lineno = fi.get("lineno") or 1
            cc = fi.get("cyclomatic", "")
            msg = f"Function `{fi.get('name', '?')}` has {issue.replace('_', ' ')}"
            if cc:
                msg += f" (cyclomatic={cc})"

            results_list.append({
                "ruleId": rule_id,
                "level": "warning",
                "message": {"text": msg},
                "locations": [{
                    "physicalLocation": {
                        "artifactLocation": {"uri": path, "uriBaseId": "%SRCROOT%"},
                        "region": {"startLine": lineno},
                    }
                }],
            })

        # Pattern smells
        pat = file_result.get("pattern") or {}
        for smell in pat.get("smells", []):
            issue = smell.get("type", "code_smell")
            rule = _SARIF_RULES.get(issue, {"id": f"IC-PAT-{issue}", "name": issue, "cwe": None})
            rule_id = rule["id"]
            referenced_rule_ids.add(issue)

            lineno = smell.get("lineno") or 1
            results_list.append({
                "ruleId": rule_id,
                "level": "note",
                "message": {"text": f"{smell.get('name', issue)}: {smell.get('description', '')}"},
                "locations": [{
                    "physicalLocation": {
                        "artifactLocation": {"uri": path, "uriBaseId": "%SRCROOT%"},
                        "region": {"startLine": lineno},
                    }
                }],
            })

    # Build rule descriptors for all referenced rules
    rules_list = []
    for vtype in referenced_rule_ids:
        r = _SARIF_RULES.get(vtype, {"id": f"IC-{vtype}", "name": vtype, "cwe": None})
        rule_entry: dict = {
            "id": r["id"],
            "name": r["name"],
            "shortDescription": {"text": r["name"]},
            "fullDescription": {"text": f"IntelliCode detected: {r['name']}"},
            "properties": {"tags": ["IntelliCode"]},
        }
        if r.get("cwe"):
            rule_entry["relationships"] = [{
                "target": {"id": r["cwe"], "toolComponent": {"name": "CWE"}},
                "kinds": ["superset"],
            }]
        rules_list.append(rule_entry)

    return {
        "version": "2.1.0",
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "runs": [{
            "tool": {
                "driver": {
                    "name": "IntelliCode",
                    "version": "1.0.0",
                    "informationUri": "https://github.com/safaapatel/intellcode",
                    "rules": rules_list,
                }
            },
            "results": results_list,
        }],
    }


def write_sarif(all_results: list[dict], output_path: str = "intellicode.sarif.json") -> str:
    """Write SARIF JSON to disk and return the path."""
    sarif_doc = build_sarif(all_results)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(sarif_doc, f, indent=2)
    return str(out)


# ---------------------------------------------------------------------------
# Diff parsing
# ---------------------------------------------------------------------------

def parse_pr_file_diff(patch: str) -> set[int]:
    """Return new-file line numbers of added/modified lines in a PR patch."""
    changed: set[int] = set()
    if not patch:
        return changed
    current_line = 0
    for raw_line in patch.splitlines():
        if raw_line.startswith("@@"):
            try:
                new_part = raw_line.split("+")[1].split("@@")[0].strip()
                current_line = int(new_part.split(",")[0]) - 1
            except (IndexError, ValueError):
                pass
        elif raw_line.startswith("+") and not raw_line.startswith("+++"):
            current_line += 1
            changed.add(current_line)
        elif not raw_line.startswith("-"):
            current_line += 1
    return changed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("[error] GITHUB_TOKEN not set")
        return 1

    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    try:
        with open(event_path, encoding="utf-8") as f:
            event = json.load(f)
    except Exception as e:
        print(f"[error] Cannot read GITHUB_EVENT_PATH={event_path!r}: {e}")
        return 1

    pr = event.get("pull_request", {})
    pr_number = pr.get("number")
    if not pr_number:
        print("[error] No pull_request.number in event payload")
        return 1

    repo_full = os.environ.get("GITHUB_REPOSITORY", "")
    if "/" not in repo_full:
        print(f"[error] GITHUB_REPOSITORY={repo_full!r} is not 'owner/repo'")
        return 1
    owner, repo = repo_full.split("/", 1)

    workspace = Path(os.environ.get("GITHUB_WORKSPACE", "."))
    config = load_config(workspace)

    # CLI env vars override config file
    threshold = float(os.environ.get("INTELLICODE_THRESHOLD", config["threshold"]))
    fail_on_issues = (
        os.environ.get("INTELLICODE_FAIL_ON_ISSUES", "").lower() == "true"
        or config["fail_on_issues"]
    )
    max_inline = config["max_inline_comments"]
    global_suppressed = set(config["suppress_rules"])

    base_ref = pr.get("base", {}).get("sha") or pr.get("base", {}).get("ref", "main")
    print(f"[info] Reviewing PR #{pr_number} in {repo_full} "
          f"(threshold={threshold}, base={base_ref})")

    # Get changed files — support Python, JavaScript, TypeScript, Java
    _SUPPORTED_EXTS = (".py", ".js", ".ts", ".jsx", ".tsx", ".java")

    def _file_lang(filename: str) -> str:
        ext = Path(filename).suffix.lower()
        return {
            ".py": "python",
            ".js": "javascript", ".jsx": "javascript",
            ".ts": "typescript", ".tsx": "typescript",
            ".java": "java",
        }.get(ext, "python")

    pr_files = get_pr_files(token, owner, repo, pr_number)
    py_files = [
        f for f in pr_files
        if Path(f.get("filename", "")).suffix.lower() in _SUPPORTED_EXTS
        and f.get("status") != "removed"
        and not _should_ignore_path(f["filename"], config["ignore_paths"])
    ]

    if not py_files:
        print("[info] No supported files to review (all excluded or none changed)")
        return 0

    ignored = [
        f["filename"] for f in pr_files
        if Path(f.get("filename", "")).suffix.lower() in _SUPPORTED_EXTS
        and _should_ignore_path(f["filename"], config["ignore_paths"])
    ]
    if ignored:
        print(f"[info] Ignored by config: {ignored}")

    print(f"[info] Analysing {len(py_files)} file(s): "
          + ", ".join(f["filename"] for f in py_files))

    commits = get_pr_commits(token, owner, repo, pr_number)
    commit_sha = commits[-1] if commits else pr.get("head", {}).get("sha", "")
    if not commit_sha:
        print("[error] Could not determine commit SHA")
        return 1

    print("[info] Loading IntelliCode analysis models...")
    t0 = time.time()
    try:
        analyze = _load_analyzer()
    except Exception as e:
        print(f"[error] Failed to load models: {e}")
        post_issue_comment(
            token, owner, repo, pr_number,
            f"## IntelliCode Review\n\n[warn] Model loading failed: `{e}`\n\nCheck workflow logs.",
        )
        return 1
    print(f"[info] Models loaded in {time.time()-t0:.1f}s")

    # Analyse PR HEAD files
    all_results: list[dict] = []
    all_comments: list[dict] = []

    for file_info in py_files:
        path = file_info["filename"]
        full_path = workspace / path
        patch = file_info.get("patch", "")
        diff_lines = parse_pr_file_diff(patch)

        try:
            source = full_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            print(f"[warn] Cannot read {path}: {e}")
            continue

        lang = _file_lang(path)
        print(f"[info]   {path} [{lang}] ({len(source)} chars, {len(diff_lines)} changed lines)...")
        raw_result = analyze(source, path)
        raw_result["language"] = lang   # propagate for language-aware suggestion dispatch

        # Build suppression map from inline comments
        suppression_map = _build_suppression_map(source)
        is_checkpoint = _is_checkpoint_path(path)

        result = filter_findings(
            raw_result, path, suppression_map, global_suppressed, is_checkpoint
        )
        all_results.append(result)

        if result.get("errors"):
            print(f"[warn]   Errors: {result['errors']}")

        if diff_lines:
            file_comments = build_review_comments(result, diff_lines, path, source.splitlines())
            all_comments.extend(file_comments)
            print(f"[info]   -> {len(file_comments)} comment(s) (score={result['overall_score']})")

    if not all_results:
        print("[info] No files could be analysed")
        return 0

    # Write SARIF for GitHub Code Scanning (uploaded as artifact by workflow)
    sarif_path = str(workspace / "backend" / "evaluation" / "results" / "intellicode.sarif.json")
    try:
        written = write_sarif(all_results, sarif_path)
        print(f"[info] SARIF written -> {written}")
    except Exception as e:
        print(f"[warn] SARIF write failed: {e}")

    # Base-branch delta
    print(f"[info] Computing base-branch ({base_ref}) quality scores for delta...")
    try:
        base_scores = compute_base_scores(
            token, owner, repo, base_ref, py_files, analyze, workspace
        )
        if base_scores:
            print(f"[info] Base scores: " +
                  ", ".join(f"{k}={v}" for k, v in base_scores.items()))
    except Exception as e:
        print(f"[warn] Base-branch delta failed: {e}")
        base_scores = {}

    # Build summary
    summary, event_type = build_summary(all_results, threshold, base_scores)

    # Cap and prioritise inline comments
    if len(all_comments) > max_inline:
        print(f"[info] Capping inline comments at {max_inline} (was {len(all_comments)})")

        def comment_priority(c: dict) -> int:
            body = c.get("body", "")
            if "Security" in body and "CRITICAL" in body: return 0
            if "Security" in body and "HIGH" in body:    return 1
            if "Security" in body:                        return 2
            if "Complexity" in body:                      return 3
            return 4

        all_comments.sort(key=comment_priority)
        all_comments = all_comments[:max_inline]

    # Check confidence-gated quality gates
    conf_fail, conf_reason = _should_fail_on_confidence(all_results, config.get("fail_on", []))
    if conf_fail:
        print(f"[info] Confidence gate triggered: {conf_reason}")

    # Determine overall CI conclusion
    mean_score = sum(r.get("overall_score", 0) for r in all_results) / len(all_results)
    score_fail = fail_on_issues and (event_type == "REQUEST_CHANGES")

    # Delta gating: fail if this PR drops quality by more than max_score_drop points
    max_score_drop = config.get("max_score_drop")
    delta_fail = False
    delta_reason = ""
    if max_score_drop is not None and base_scores:
        base_overall_scores = [
            v for k, v in base_scores.items()
            if any(r.get("filename") == k for r in all_results)
        ]
        if base_overall_scores:
            mean_base_score = sum(base_overall_scores) / len(base_overall_scores)
            drop = mean_base_score - mean_score
            if drop > float(max_score_drop):
                delta_fail = True
                delta_reason = (
                    f"Quality dropped {drop:.1f} pts (base={mean_base_score:.1f}, "
                    f"PR={mean_score:.1f}, max_allowed_drop={max_score_drop})"
                )
                print(f"[info] Delta gate triggered: {delta_reason}")

    should_fail = score_fail or conf_fail or delta_fail
    conclusion = "failure" if should_fail else "success"

    # Post GitHub Checks API run with annotations (line-level, appears on diff)
    use_checks = config.get("use_checks_api", True)
    if use_checks:
        annotations = build_check_annotations(all_results)
        extra_note = ""
        if conf_fail:
            extra_note = f" | {conf_reason}"
        elif delta_fail:
            extra_note = f" | {delta_reason}"
        check_title = (
            f"Quality: {mean_score:.0f}/100 — "
            + ("FAILED" if should_fail else "PASSED")
            + extra_note
        )
        print(f"[info] Posting Check Run with {len(annotations)} annotation(s)...")
        check_result = post_check_run(
            token, owner, repo,
            commit_sha=commit_sha,
            conclusion=conclusion,
            summary=summary,
            annotations=annotations,
            title=check_title,
        )
        if check_result:
            print(f"[info] Check Run posted: {check_result.get('html_url', 'ok')}")
        else:
            print("[warn] Check Run failed — falling back to PR review comments")
            use_checks = False  # fall through to legacy path

    # Also post PR review comments (contains the human-readable summary with file breakdown)
    print(f"[info] Posting review: {len(all_comments)} inline comment(s), event={event_type}")
    result_obj = post_pr_review(
        token, owner, repo, pr_number,
        commit_sha=commit_sha,
        comments=all_comments,
        summary=summary,
        event=event_type,
    )

    if result_obj is None:
        print("[warn] Review post failed — posting plain comment fallback")
        post_issue_comment(token, owner, repo, pr_number, summary)

    if should_fail:
        if conf_fail:
            print(f"[fail] Confidence gate: {conf_reason} — failing CI")
        elif delta_fail:
            print(f"[fail] Delta gate: {delta_reason} — failing CI")
        else:
            print(f"[fail] Quality score {mean_score:.1f} below threshold {threshold} — failing CI")
        return 1

    print("[info] IntelliCode review complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
