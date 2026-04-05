"""
IntelliCode GitHub Actions Reviewer
=====================================
Self-contained script that:
  1. Reads the PR diff to find changed .py files
  2. Runs IntelliCode analysis (no FastAPI server — imports models directly)
  3. Posts findings as inline review comments via the GitHub REST API
  4. Optionally fails with exit code 1 if overall quality is below threshold

Environment variables (set by GitHub Actions):
  GITHUB_TOKEN          — automatically provided by Actions
  GITHUB_REPOSITORY     — "owner/repo"
  GITHUB_EVENT_PATH     — path to the webhook event JSON
  INTELLICODE_THRESHOLD — quality score threshold (0-100, default 60)
  INTELLICODE_FAIL_ON_ISSUES — "true" to fail CI when issues found (default "false")

Usage:
  python backend/github_action_review.py
"""

from __future__ import annotations

import json
import os
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
    """Return list of files changed in the PR."""
    result = _github_request(
        "GET", f"/repos/{owner}/{repo}/pulls/{pr_number}/files", token=token
    )
    return result if isinstance(result, list) else []


def get_pr_commits(token: str, owner: str, repo: str, pr_number: int) -> list[str]:
    """Return list of commit SHAs for the PR (most recent last)."""
    result = _github_request(
        "GET", f"/repos/{owner}/{repo}/pulls/{pr_number}/commits", token=token
    )
    if not isinstance(result, list):
        return []
    return [c["sha"] for c in result]


def post_pr_review(
    token: str,
    owner: str,
    repo: str,
    pr_number: int,
    commit_sha: str,
    comments: list[dict],
    summary: str,
    event: str = "COMMENT",  # "APPROVE" | "REQUEST_CHANGES" | "COMMENT"
) -> dict | None:
    """Post a PR review with inline comments."""
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
    """Post a plain comment on the PR (fallback when inline comments fail)."""
    return _github_request(
        "POST", f"/repos/{owner}/{repo}/issues/{pr_number}/comments",
        body={"body": body}, token=token,
    )


# ---------------------------------------------------------------------------
# Analysis — import models directly (no server)
# ---------------------------------------------------------------------------

def _score_from_probability(prob: float) -> float:
    """Convert a bug probability (0–1) to a quality score (0–100, higher = safer)."""
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
    Returns analyze_fn where analyze_fn(source: str, filename: str) -> dict.

    Actual class names and predict() signatures:
      ComplexityPredictionModel.predict(source) -> ComplexityResult
        .score, .grade, .cyclomatic, .cognitive, .function_issues (list[FunctionIssue])
      EnsembleSecurityModel.predict(source) -> list[VulnerabilityPrediction]
        each: .vuln_type, .severity, .confidence, .lineno, .title, .description, .snippet, .cwe
      BugPredictionModel.predict(source, git_metadata=None) -> BugPrediction
        .bug_probability, .risk_level, .risk_factors, .confidence, .static_score, .git_score
      PatternRecognitionModel.predict(code_snippet) -> PatternPrediction
        .label (str), .confidence, .label_id, .all_scores (dict)
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

        # --- Complexity ---
        try:
            c = complexity_pred.predict(source)
            results["complexity"] = {
                "score": c.score,
                "grade": c.grade,
                "cognitive": c.cognitive,
                "cyclomatic": c.cyclomatic,
                "function_issues": [
                    {
                        "name": fi.name,
                        "lineno": fi.lineno,
                        "cyclomatic": fi.cyclomatic,
                        "body_lines": fi.body_lines,
                        "n_params": fi.n_params,
                        "issue": fi.issue,
                    }
                    for fi in (c.function_issues or [])
                ],
            }
        except Exception as e:
            errors.append(f"complexity: {e}")
            results["complexity"] = None

        # --- Security ---
        # EnsembleSecurityModel.predict() returns list[VulnerabilityPrediction]
        # We derive a score from the highest-severity finding.
        try:
            vulns: list = security_det.predict(source)
            # Score: start at 100, penalise by severity
            sev_penalty = {"critical": 40, "high": 25, "medium": 12, "low": 5}
            sec_score = 100.0
            for v in vulns:
                sec_score -= sev_penalty.get(getattr(v, "severity", "").lower(), 5)
            sec_score = max(0.0, sec_score)
            results["security"] = {
                "score": round(sec_score, 1),
                "grade": _grade(sec_score),
                "vulnerabilities": [
                    {
                        "vuln_type": v.vuln_type,
                        "severity": v.severity,
                        "confidence": v.confidence,
                        "lineno": v.lineno,
                        "title": v.title,
                        "description": v.description,
                        "snippet": v.snippet,
                        "cwe": getattr(v, "cwe", ""),
                    }
                    for v in vulns
                ],
            }
        except Exception as e:
            errors.append(f"security: {e}")
            results["security"] = None

        # --- Bug risk ---
        # BugPredictionModel.predict() returns BugPrediction
        # .bug_probability (0–1), .risk_level (str), .risk_factors (list[str])
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

        # --- Patterns / code smells ---
        # PatternRecognitionModel.predict(code_snippet) -> PatternPrediction
        # .label (str: 'clean'|'code_smell'|'anti_pattern'|'style_violation')
        # .confidence, .all_scores (dict label->score)
        try:
            p = pattern_rec.predict(source)
            # Score: clean=100, style_violation=70, code_smell=45, anti_pattern=25
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

        # --- Overall quality score (average of available scores) ---
        scores = []
        for k in ("complexity", "security", "bug", "pattern"):
            r = results.get(k)
            if r and isinstance(r.get("score"), (int, float)):
                scores.append(float(r["score"]))
        results["overall_score"] = round(sum(scores) / len(scores), 1) if scores else 0.0
        results["errors"] = errors
        return results

    return analyze


# ---------------------------------------------------------------------------
# Comment generation — convert analysis findings into GitHub review comments
# ---------------------------------------------------------------------------

SEVERITY_EMOJI = {
    "critical": "!! CRITICAL",
    "high": "! HIGH",
    "medium": "MEDIUM",
    "low": "LOW",
    "info": "INFO",
}

GRADE_EMOJI = {"A": "A", "B": "B", "C": "C", "D": "D", "F": "F"}


def _severity_sort_key(severity: str) -> int:
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    return order.get(severity.lower(), 5)


def build_review_comments(
    file_result: dict,
    diff_lines: set[int],
    path: str,
) -> list[dict]:
    """
    Build a list of GitHub pull-request review comment objects for one file.

    GitHub requires comments to be on lines that appear in the diff.
    We attach each finding to the nearest changed line, or skip it if
    no changed line is nearby (within 10 lines).
    """
    comments: list[dict] = []

    def nearest_diff_line(lineno: int | None) -> int | None:
        if not diff_lines:
            return None
        if lineno and lineno in diff_lines:
            return lineno
        if lineno:
            # Search within +/- 10 lines
            for delta in range(1, 11):
                if lineno + delta in diff_lines:
                    return lineno + delta
                if lineno - delta in diff_lines and lineno - delta > 0:
                    return lineno - delta
        # Fall back to first changed line in file
        return min(diff_lines)

    # Security vulnerabilities
    sec = file_result.get("security") or {}
    for vuln in sorted(
        sec.get("vulnerabilities", []),
        key=lambda v: _severity_sort_key(v.get("severity", "info")),
    ):
        line = nearest_diff_line(vuln.get("lineno"))
        if line is None:
            continue
        sev_tag = SEVERITY_EMOJI.get(vuln.get("severity", "").lower(), "ISSUE")
        cwe = f" ({vuln['cwe']})" if vuln.get("cwe") else ""
        body = (
            f"**[IntelliCode Security - {sev_tag}]** {vuln['title']}{cwe}\n\n"
            f"{vuln['description']}\n"
        )
        if vuln.get("snippet"):
            body += f"\n```python\n{vuln['snippet'][:300]}\n```\n"
        comments.append({"path": path, "line": line, "side": "RIGHT", "body": body})

    # Complexity hot functions
    comp = file_result.get("complexity") or {}
    for fi in comp.get("function_issues", []):
        lineno = fi.get("lineno") if isinstance(fi, dict) else getattr(fi, "lineno", None)
        issue = fi.get("issue") if isinstance(fi, dict) else getattr(fi, "issue", "")
        name = fi.get("name") if isinstance(fi, dict) else getattr(fi, "name", "")
        line = nearest_diff_line(lineno)
        if line is None:
            continue
        label = {
            "high_complexity": "High Complexity",
            "too_long": "Function Too Long",
            "too_many_params": "Too Many Parameters",
        }.get(issue, issue)
        body = f"**[IntelliCode Complexity]** `{name}`: {label}\n\nConsider breaking this function into smaller, focused units."
        comments.append({"path": path, "line": line, "side": "RIGHT", "body": body})

    # Code smells
    pat = file_result.get("pattern") or {}
    for smell in pat.get("smells", []):
        lineno = smell.get("lineno")
        line = nearest_diff_line(lineno)
        if line is None:
            continue
        body = (
            f"**[IntelliCode Pattern]** `{smell['name']}`"
            + (f": {smell['description']}" if smell.get("description") else "")
        )
        comments.append({"path": path, "line": line, "side": "RIGHT", "body": body})

    return comments


def _grade_bar(score: float) -> str:
    """ASCII progress bar for a 0-100 score."""
    filled = int(score / 10)
    return "[" + "#" * filled + "-" * (10 - filled) + f"] {score:.0f}/100"


def build_summary(all_results: list[dict], threshold: float) -> tuple[str, str]:
    """
    Build the PR review summary body and determine review event type.
    Returns (summary_markdown, event_type).
    """
    lines = ["## IntelliCode Review\n"]

    total_vulns = 0
    total_smells = 0
    total_complex = 0
    overall_scores: list[float] = []

    for r in all_results:
        filename = r["filename"]
        score = r.get("overall_score", 0.0)
        overall_scores.append(score)

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

        sec_grade = sec.get("grade", "?")
        comp_grade = comp.get("grade", "?")
        bug_grade = bug.get("grade", "?")
        pat_grade = pat.get("grade", "?")

        lines.append(
            f"### `{filename}`\n"
            f"Overall: {_grade_bar(score)}\n"
            f"Security:{sec_grade}  Complexity:{comp_grade}  Bug-risk:{bug_grade}  Patterns:{pat_grade}\n"
        )

        if vulns:
            crit = [v for v in vulns if v.get("severity") == "critical"]
            high = [v for v in vulns if v.get("severity") == "high"]
            lines.append(f"- {len(vulns)} security finding(s)"
                         + (f" ({len(crit)} critical, {len(high)} high)" if crit or high else ""))
        if fi:
            lines.append(f"- {len(fi)} complex function(s)")
        if smells:
            lines.append(f"- {len(smells)} code smell(s)")
        lines.append("")

    mean_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0

    lines.append("---")
    lines.append(f"**Files reviewed:** {len(all_results)}  |  "
                 f"**Avg quality score:** {mean_score:.1f}/100")
    lines.append(f"**Total findings:** {total_vulns} security, "
                 f"{total_complex} complexity, {total_smells} pattern")

    if mean_score < threshold:
        lines.append(f"\n> Quality score {mean_score:.1f} is below the configured threshold of {threshold}.")
        event = "REQUEST_CHANGES"
    else:
        lines.append(f"\n> Quality score {mean_score:.1f} meets the configured threshold of {threshold}.")
        event = "COMMENT"

    lines.append("\n*Powered by [IntelliCode](https://github.com/safaapatel/intellcode) -- ML-based code quality analysis*")
    return "\n".join(lines), event


# ---------------------------------------------------------------------------
# Diff parsing — extract line numbers of changed lines per file
# ---------------------------------------------------------------------------

def parse_pr_file_diff(patch: str) -> set[int]:
    """
    Parse a GitHub PR file patch string and return the set of new-file line
    numbers that were added or modified (lines starting with '+').
    """
    changed: set[int] = set()
    if not patch:
        return changed

    current_line = 0
    for raw_line in patch.splitlines():
        if raw_line.startswith("@@"):
            # @@ -old_start,old_count +new_start,new_count @@
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
# Main entry point
# ---------------------------------------------------------------------------

def main() -> int:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("[error] GITHUB_TOKEN not set")
        return 1

    # Parse event metadata
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
        print(f"[error] GITHUB_REPOSITORY={repo_full!r} is not in 'owner/repo' format")
        return 1
    owner, repo = repo_full.split("/", 1)

    threshold = float(os.environ.get("INTELLICODE_THRESHOLD", "60"))
    fail_on_issues = os.environ.get("INTELLICODE_FAIL_ON_ISSUES", "false").lower() == "true"

    print(f"[info] Reviewing PR #{pr_number} in {repo_full} (threshold={threshold})")

    # Get changed files
    pr_files = get_pr_files(token, owner, repo, pr_number)
    py_files = [
        f for f in pr_files
        if f.get("filename", "").endswith(".py")
        and f.get("status") != "removed"
    ]

    if not py_files:
        print("[info] No Python files changed in this PR — nothing to review")
        return 0

    print(f"[info] Analysing {len(py_files)} changed Python file(s): "
          + ", ".join(f["filename"] for f in py_files))

    # Get latest commit SHA
    commits = get_pr_commits(token, owner, repo, pr_number)
    commit_sha = commits[-1] if commits else pr.get("head", {}).get("sha", "")
    if not commit_sha:
        print("[error] Could not determine commit SHA")
        return 1

    # Load analysis engine (this is the slow part — model imports)
    print("[info] Loading IntelliCode analysis models...")
    t0 = time.time()
    try:
        analyze = _load_analyzer()
    except Exception as e:
        print(f"[error] Failed to load analysis models: {e}")
        # Post a fallback comment so reviewers know the check ran
        post_issue_comment(
            token, owner, repo, pr_number,
            f"## IntelliCode Review\n\n[warn] Model loading failed: {e}\n\nPlease check the workflow logs.",
        )
        return 1
    print(f"[info] Models loaded in {time.time()-t0:.1f}s")

    # Checkout files and analyse
    workspace = Path(os.environ.get("GITHUB_WORKSPACE", "."))
    all_results: list[dict] = []
    all_comments: list[dict] = []

    for file_info in py_files:
        path = file_info["filename"]
        full_path = workspace / path
        patch = file_info.get("patch", "")
        diff_lines = parse_pr_file_diff(patch)

        try:
            source = full_path.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            print(f"[warn] File not found (maybe deleted): {full_path}")
            continue
        except Exception as e:
            print(f"[warn] Cannot read {path}: {e}")
            continue

        print(f"[info]   Analysing {path} ({len(source)} chars, {len(diff_lines)} changed lines)...")
        result = analyze(source, path)
        all_results.append(result)

        if result.get("errors"):
            print(f"[warn]   Analysis errors: {result['errors']}")

        # Only create inline comments for lines in the diff
        if diff_lines:
            file_comments = build_review_comments(result, diff_lines, path)
            all_comments.extend(file_comments)
            print(f"[info]   -> {len(file_comments)} inline comment(s) queued")

    if not all_results:
        print("[info] No files could be analysed")
        return 0

    # Build summary and choose review event
    summary, event_type = build_summary(all_results, threshold)

    # GitHub caps inline comments per review at 30 to avoid noise
    MAX_INLINE = 30
    if len(all_comments) > MAX_INLINE:
        print(f"[info] Capping inline comments at {MAX_INLINE} (was {len(all_comments)})")
        # Prioritise security > complexity > pattern
        def comment_priority(c: dict) -> int:
            body = c.get("body", "")
            if "Security" in body and "CRITICAL" in body:
                return 0
            if "Security" in body and "HIGH" in body:
                return 1
            if "Security" in body:
                return 2
            if "Complexity" in body:
                return 3
            return 4
        all_comments.sort(key=comment_priority)
        all_comments = all_comments[:MAX_INLINE]

    print(f"[info] Posting review with {len(all_comments)} inline comment(s), event={event_type}")
    result_obj = post_pr_review(
        token, owner, repo, pr_number,
        commit_sha=commit_sha,
        comments=all_comments,
        summary=summary,
        event=event_type,
    )

    if result_obj is None:
        print("[warn] Review post may have failed — trying plain comment fallback")
        post_issue_comment(token, owner, repo, pr_number, summary)

    # Final exit code
    if fail_on_issues and event_type == "REQUEST_CHANGES":
        mean_score = sum(r.get("overall_score", 0) for r in all_results) / len(all_results)
        print(f"[fail] Quality score {mean_score:.1f} below threshold {threshold} — failing CI")
        return 1

    print("[info] IntelliCode review complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
