"""
Documentation Quality Analyzer
Scores the quality and completeness of docstrings in Python source code.

Checks per function/class:
    - Presence of a docstring
    - Minimum useful length (not just "TODO" or a single word)
    - Parameter documentation (Args / Parameters section)
    - Return value documentation (Returns section)
    - Raises documentation (Raises section) when exceptions are raised
    - Example / Usage section presence (bonus)

Overall file score:
    coverage    — % of public functions/classes that have docstrings
    quality     — average quality score (0–100) of existing docstrings
    grade       — A (≥90), B (≥75), C (≥60), D (≥40), F (<40)
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class DocIssue:
    issue_type: str          # "missing" | "too_short" | "missing_params" | ...
    severity: str            # "error" | "warning" | "info"
    symbol: str              # function or class name
    location: str
    start_line: int
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SymbolDocScore:
    name: str
    kind: str                # "function" | "class" | "method"
    start_line: int
    has_docstring: bool
    quality_score: float     # 0–100
    issues: list[str]        # brief issue labels

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DocQualityResult:
    coverage: float              # 0–1 fraction of symbols documented
    average_quality: float       # 0–100 average score of documented symbols
    grade: str                   # A–F
    total_symbols: int
    documented_symbols: int
    symbol_scores: list[SymbolDocScore]
    issues: list[DocIssue]
    summary: str

    def to_dict(self) -> dict:
        return {
            "coverage": round(self.coverage, 3),
            "average_quality": round(self.average_quality, 1),
            "grade": self.grade,
            "total_symbols": self.total_symbols,
            "documented_symbols": self.documented_symbols,
            "symbol_scores": [s.to_dict() for s in self.symbol_scores],
            "issues": [i.to_dict() for i in self.issues],
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Docstring parsing helpers
# ---------------------------------------------------------------------------

_PARAM_PATTERNS = [
    re.compile(r"^\s*(Args|Parameters|Params)\s*:", re.MULTILINE | re.IGNORECASE),
]
_RETURN_PATTERNS = [
    re.compile(r"^\s*(Returns?|Return value)\s*:", re.MULTILINE | re.IGNORECASE),
]
_RAISES_PATTERNS = [
    re.compile(r"^\s*(Raises?|Exceptions?)\s*:", re.MULTILINE | re.IGNORECASE),
]
_EXAMPLE_PATTERNS = [
    re.compile(r"^\s*(Examples?|Usage|Example usage)\s*:", re.MULTILINE | re.IGNORECASE),
    re.compile(r">>>\s+\w+", re.MULTILINE),  # doctest style
]


def _has_section(doc: str, patterns: list[re.Pattern]) -> bool:
    return any(p.search(doc) for p in patterns)


def _get_docstring(node: ast.AST) -> Optional[str]:
    if not (hasattr(node, "body") and node.body):
        return None
    first = node.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
        val = first.value.value
        if isinstance(val, str):
            return val
    return None


def _count_params(func: ast.FunctionDef) -> int:
    args = func.args
    params = [a for a in args.args if a.arg not in ("self", "cls")]
    params += args.kwonlyargs
    if args.vararg:
        params.append(args.vararg)
    if args.kwarg:
        params.append(args.kwarg)
    return len(params)


def _has_explicit_return(func: ast.FunctionDef) -> bool:
    for node in ast.walk(func):
        if isinstance(node, ast.Return) and node.value is not None:
            return True
    return False


def _has_raises(func: ast.FunctionDef) -> bool:
    for node in ast.walk(func):
        if isinstance(node, ast.Raise):
            return True
    return False


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_docstring(
    doc: str,
    n_params: int,
    has_return: bool,
    has_raises_in_code: bool,
) -> tuple[float, list[str]]:
    """
    Score a docstring 0–100 and return a list of issue labels.
    """
    score = 0.0
    issues = []
    words = doc.split()

    # Presence of meaningful content (up to 30 pts)
    if len(words) >= 3:
        score += 30
    elif len(words) > 0:
        score += 10
        issues.append("too_short")

    # First line is a summary sentence (up to 20 pts)
    first_line = doc.strip().splitlines()[0].strip()
    if len(first_line) >= 10 and first_line[0].isupper():
        score += 20
    elif len(first_line) >= 5:
        score += 10
        issues.append("weak_summary")

    # Parameter docs (up to 25 pts)
    if n_params > 0:
        if _has_section(doc, _PARAM_PATTERNS):
            score += 25
        else:
            issues.append("missing_params_section")

    # Return docs (up to 15 pts)
    if has_return:
        if _has_section(doc, _RETURN_PATTERNS):
            score += 15
        else:
            issues.append("missing_returns_section")

    # Raises docs (up to 5 pts)
    if has_raises_in_code:
        if _has_section(doc, _RAISES_PATTERNS):
            score += 5
        else:
            issues.append("missing_raises_section")

    # Example / doctest (bonus up to 5 pts)
    if _has_section(doc, _EXAMPLE_PATTERNS):
        score += 5

    return min(100.0, score), issues


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

def _grade(score: float) -> str:
    if score >= 90:
        return "A"
    elif score >= 75:
        return "B"
    elif score >= 60:
        return "C"
    elif score >= 40:
        return "D"
    else:
        return "F"


class DocQualityAnalyzer:
    """
    Analyzes documentation quality for all public functions, methods, and
    classes in a Python source file.

    Usage:
        analyzer = DocQualityAnalyzer()
        result = analyzer.analyze(source_code)
    """

    def analyze(self, source: str) -> DocQualityResult:
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return DocQualityResult(
                coverage=0.0,
                average_quality=0.0,
                grade="F",
                total_symbols=0,
                documented_symbols=0,
                symbol_scores=[],
                issues=[],
                summary=f"Syntax error: {e}",
            )

        symbol_scores: list[SymbolDocScore] = []
        doc_issues: list[DocIssue] = []

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "function"
                # Skip private/dunder unless it's __init__
                if node.name.startswith("_") and node.name != "__init__":
                    continue
                score, issues = self._score_function(node, doc_issues)
                symbol_scores.append(SymbolDocScore(
                    name=node.name,
                    kind=kind,
                    start_line=node.lineno,
                    has_docstring=_get_docstring(node) is not None,
                    quality_score=score,
                    issues=issues,
                ))

            elif isinstance(node, ast.ClassDef):
                if node.name.startswith("_"):
                    continue
                score, issues = self._score_class(node, doc_issues)
                symbol_scores.append(SymbolDocScore(
                    name=node.name,
                    kind="class",
                    start_line=node.lineno,
                    has_docstring=_get_docstring(node) is not None,
                    quality_score=score,
                    issues=issues,
                ))

        total = len(symbol_scores)
        documented = sum(1 for s in symbol_scores if s.has_docstring)
        coverage = documented / total if total > 0 else 1.0
        scored = [s.quality_score for s in symbol_scores if s.has_docstring]
        avg_quality = sum(scored) / len(scored) if scored else 0.0

        # Overall grade: combine coverage and quality
        overall = coverage * 50 + avg_quality * 0.5
        grade = _grade(overall)

        summary = (
            f"Documentation coverage: {documented}/{total} symbols "
            f"({coverage:.0%}). "
            f"Average docstring quality: {avg_quality:.0f}/100. "
            f"Grade: {grade}."
        )

        return DocQualityResult(
            coverage=coverage,
            average_quality=avg_quality,
            grade=grade,
            total_symbols=total,
            documented_symbols=documented,
            symbol_scores=sorted(symbol_scores, key=lambda s: s.quality_score),
            issues=doc_issues,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _score_function(
        self,
        func: ast.FunctionDef,
        issues_out: list[DocIssue],
    ) -> tuple[float, list[str]]:
        doc = _get_docstring(func)
        loc = f"line {func.lineno}"

        if doc is None:
            issues_out.append(DocIssue(
                issue_type="missing",
                severity="warning",
                symbol=func.name,
                location=loc,
                start_line=func.lineno,
                message=f"Function '{func.name}' has no docstring.",
            ))
            return 0.0, ["missing"]

        n_params = _count_params(func)
        has_ret = _has_explicit_return(func)
        has_raises = _has_raises(func)
        score, issue_labels = _score_docstring(doc, n_params, has_ret, has_raises)

        for label in issue_labels:
            issues_out.append(DocIssue(
                issue_type=label,
                severity="info",
                symbol=func.name,
                location=loc,
                start_line=func.lineno,
                message=f"Docstring for '{func.name}': {label.replace('_', ' ')}.",
            ))

        return score, issue_labels

    def _score_class(
        self,
        cls: ast.ClassDef,
        issues_out: list[DocIssue],
    ) -> tuple[float, list[str]]:
        doc = _get_docstring(cls)
        loc = f"line {cls.lineno}"

        if doc is None:
            issues_out.append(DocIssue(
                issue_type="missing",
                severity="warning",
                symbol=cls.name,
                location=loc,
                start_line=cls.lineno,
                message=f"Class '{cls.name}' has no docstring.",
            ))
            return 0.0, ["missing"]

        score, issue_labels = _score_docstring(doc, 0, False, False)
        return score, issue_labels
