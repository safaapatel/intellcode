"""
Code Readability Scorer
Scores how easy a Python file is to read and understand.

Dimensions scored (each 0–100, then combined):

    naming_quality      — identifier length, style (snake_case), meaningfulness
    comment_density     — ratio of comment/docstring lines to code lines
    structural_clarity  — average function length, nesting, line length
    consistency         — mixed naming styles, inconsistent spacing patterns
    cognitive_load      — number of concepts (branches + loops + functions) per SLOC

Final score: weighted average → letter grade A–F.

Individual findings flag specific identifiers or lines that hurt readability.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SNAKE_CASE = re.compile(r"^[a-z_][a-z0-9_]*$")
_CAMEL_CASE = re.compile(r"^[a-z][a-zA-Z0-9]+$")
_PASCAL_CASE = re.compile(r"^[A-Z][a-zA-Z0-9]+$")
_SCREAMING_SNAKE = re.compile(r"^[A-Z_][A-Z0-9_]+$")

# Single-letter variables that are acceptable (loop counters, math)
_ACCEPTABLE_SHORT = {"i", "j", "k", "n", "x", "y", "z", "f", "e", "t", "s", "v"}

# Meaningless names (score penalty)
_MEANINGLESS = {
    "tmp", "temp", "var", "val", "foo", "bar", "baz", "data2", "result2",
    "stuff", "thing", "obj", "obj2", "item", "items2", "a", "b", "c",
    "d", "p", "q", "r", "u", "w", "m",
}

# Weights for final score
_WEIGHTS = {
    "naming_quality": 0.30,
    "comment_density": 0.20,
    "structural_clarity": 0.30,
    "cognitive_load": 0.20,
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ReadabilityIssue:
    dimension: str       # which score dimension this affects
    title: str
    description: str
    location: str
    start_line: int
    penalty: float       # 0–10 points deducted from that dimension

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DimensionScore:
    name: str
    score: float         # 0–100
    weight: float
    issues: list[str]    # brief labels

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReadabilityResult:
    overall_score: float
    grade: str
    dimensions: list[DimensionScore]
    issues: list[ReadabilityIssue]
    top_improvements: list[str]   # up to 3 highest-impact recommendations
    summary: str

    def to_dict(self) -> dict:
        return {
            "overall_score": round(self.overall_score, 1),
            "grade": self.grade,
            "dimensions": [d.to_dict() for d in self.dimensions],
            "issues": [i.to_dict() for i in self.issues],
            "top_improvements": self.top_improvements,
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grade(score: float) -> str:
    if score >= 90: return "A"
    if score >= 75: return "B"
    if score >= 60: return "C"
    if score >= 40: return "D"
    return "F"


def _count_sloc(source: str) -> int:
    count = 0
    for line in source.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            count += 1
    return max(1, count)


def _count_comment_lines(source: str) -> int:
    count = 0
    in_docstring = False
    quote_char = ""
    for line in source.splitlines():
        s = line.strip()
        if in_docstring:
            count += 1
            if quote_char in s:
                in_docstring = False
                quote_char = ""
        elif s.startswith('"""') or s.startswith("'''"):
            count += 1
            q = '"""' if s.startswith('"""') else "'''"
            # Only enter docstring mode if the closing quote is not on the same line
            if q not in s[3:]:
                in_docstring = True
                quote_char = q
        elif s.startswith("#"):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Dimension scorers
# ---------------------------------------------------------------------------

def _score_naming(tree: ast.Module, issues_out: list[ReadabilityIssue]) -> float:
    """
    Score naming quality: penalize too-short, meaningless, or inconsistent names.
    """
    penalties = 0.0
    checked = 0

    for node in ast.walk(tree):
        # Function names
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
            checked += 1
            if name.startswith("__"):
                continue
            if not _SNAKE_CASE.match(name):
                penalties += 3
                issues_out.append(ReadabilityIssue(
                    dimension="naming_quality",
                    title=f"Non-snake_case function name: '{name}'",
                    description=f"Function '{name}' (line {node.lineno}) should use snake_case.",
                    location=f"line {node.lineno}",
                    start_line=node.lineno,
                    penalty=3,
                ))
            if len(name) <= 2 and name not in _ACCEPTABLE_SHORT:
                penalties += 5
                issues_out.append(ReadabilityIssue(
                    dimension="naming_quality",
                    title=f"Too-short function name: '{name}'",
                    description=f"Function name '{name}' (line {node.lineno}) is too short to be meaningful.",
                    location=f"line {node.lineno}",
                    start_line=node.lineno,
                    penalty=5,
                ))

        # Class names — should be PascalCase
        elif isinstance(node, ast.ClassDef):
            name = node.name
            checked += 1
            if not _PASCAL_CASE.match(name) and not name.startswith("_"):
                penalties += 3
                issues_out.append(ReadabilityIssue(
                    dimension="naming_quality",
                    title=f"Non-PascalCase class name: '{name}'",
                    description=f"Class '{name}' (line {node.lineno}) should use PascalCase.",
                    location=f"line {node.lineno}",
                    start_line=node.lineno,
                    penalty=3,
                ))

        # Variable names
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            name = node.id
            if name.startswith("_") or name in ("self", "cls"):
                continue
            checked += 1
            if name in _MEANINGLESS:
                penalties += 4
                issues_out.append(ReadabilityIssue(
                    dimension="naming_quality",
                    title=f"Meaningless variable name: '{name}'",
                    description=(
                        f"Variable '{name}' at line {node.lineno} is a generic name "
                        f"that conveys no domain meaning. Use a descriptive name."
                    ),
                    location=f"line {node.lineno}",
                    start_line=node.lineno,
                    penalty=4,
                ))
            elif len(name) == 1 and name.lower() not in _ACCEPTABLE_SHORT:
                penalties += 3
                issues_out.append(ReadabilityIssue(
                    dimension="naming_quality",
                    title=f"Single-letter variable: '{name}' (line {node.lineno})",
                    description=f"'{name}' is a single-letter variable. Use a descriptive name.",
                    location=f"line {node.lineno}",
                    start_line=node.lineno,
                    penalty=3,
                ))

    if checked == 0:
        return 100.0

    # Normalize: each penalty point reduces score by (100 / max_expected_penalty)
    max_expected = checked * 5  # worst case: every symbol has 5-pt penalty
    score = max(0.0, 100.0 - (penalties / max_expected * 100))
    return score


def _score_comment_density(source: str, issues_out: list[ReadabilityIssue]) -> float:
    sloc = _count_sloc(source)
    comments = _count_comment_lines(source)
    ratio = comments / sloc

    # Ideal: 0.15 – 0.40 (15–40 comment lines per 100 code lines)
    if ratio < 0.05:
        issues_out.append(ReadabilityIssue(
            dimension="comment_density",
            title=f"Very low comment density ({ratio:.0%})",
            description=(
                f"Only {ratio:.0%} of lines are comments or docstrings. "
                f"Aim for 15–40% to explain the 'why', not just the 'what'."
            ),
            location="module-level",
            start_line=1,
            penalty=40,
        ))
        return 30.0
    elif ratio < 0.10:
        return 55.0
    elif ratio <= 0.50:
        # Sweet spot
        return min(100.0, 60 + ratio * 100)
    else:
        # Over-commented: lots of noise
        return max(50.0, 100 - (ratio - 0.5) * 50)


def _score_structural_clarity(
    tree: ast.Module,
    source: str,
    issues_out: list[ReadabilityIssue],
) -> float:
    score = 100.0
    lines = source.splitlines()

    # Long lines — PEP 8: soft limit 79, hard limit 99
    long_lines_soft = [i + 1 for i, l in enumerate(lines) if len(l) > 79]
    long_lines_hard = [i + 1 for i, l in enumerate(lines) if len(l) > 99]
    if len(long_lines_hard) > 3:
        penalty = min(20, len(long_lines_hard) * 2)
        score -= penalty
        issues_out.append(ReadabilityIssue(
            dimension="structural_clarity",
            title=f"{len(long_lines_hard)} line(s) exceed 99 characters (PEP 8 hard limit)",
            description=(
                f"PEP 8 recommends a hard maximum of 99 characters per line. "
                f"{len(long_lines_soft)} line(s) exceed the soft 79-char limit. "
                f"First offender: line {long_lines_hard[0]}."
            ),
            location=f"line {long_lines_hard[0]}",
            start_line=long_lines_hard[0],
            penalty=penalty,
        ))
    elif len(long_lines_soft) > 10:
        penalty = min(10, len(long_lines_soft))
        score -= penalty
        issues_out.append(ReadabilityIssue(
            dimension="structural_clarity",
            title=f"{len(long_lines_soft)} line(s) exceed 79 characters (PEP 8 soft limit)",
            description=(
                f"PEP 8 recommends wrapping lines at 79 characters for readability. "
                f"Consider using implicit line continuation inside parentheses. "
                f"First offender: line {long_lines_soft[0]}."
            ),
            location=f"line {long_lines_soft[0]}",
            start_line=long_lines_soft[0],
            penalty=penalty,
        ))

    # Function length
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            n = getattr(node, "end_lineno", node.lineno) - node.lineno
            if n > 60:
                score -= 10
                issues_out.append(ReadabilityIssue(
                    dimension="structural_clarity",
                    title=f"'{node.name}' is {n} lines long",
                    description=(
                        f"Very long functions are hard to read. "
                        f"Extract sub-tasks into named helper functions."
                    ),
                    location=f"lines {node.lineno}–{node.end_lineno}",
                    start_line=node.lineno,
                    penalty=10,
                ))
            elif n > 40:
                score -= 5

    # Mixed indentation (tabs + spaces)
    has_tabs = any("\t" in l for l in lines)
    has_spaces = any(l.startswith("  ") for l in lines)
    if has_tabs and has_spaces:
        score -= 15
        issues_out.append(ReadabilityIssue(
            dimension="structural_clarity",
            title="Mixed tabs and spaces",
            description="File mixes tab and space indentation. Use spaces only (PEP 8).",
            location="module-level",
            start_line=1,
            penalty=15,
        ))

    return max(0.0, score)


def _score_cognitive_load(
    tree: ast.Module,
    source: str,
    issues_out: list[ReadabilityIssue],
) -> float:
    sloc = _count_sloc(source)

    branches = sum(1 for n in ast.walk(tree) if isinstance(n, ast.If))
    loops = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.For, ast.While)))
    functions = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    classes = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    exceptions = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Try))

    concepts = branches + loops + functions + classes + exceptions
    density = concepts / sloc  # concepts per source line

    # Target: < 0.15 is clear, 0.15–0.30 moderate, > 0.30 heavy
    if density < 0.10:
        score = 100.0
    elif density < 0.20:
        score = 80.0
    elif density < 0.30:
        score = 60.0
    elif density < 0.40:
        score = 40.0
        issues_out.append(ReadabilityIssue(
            dimension="cognitive_load",
            title=f"High concept density ({density:.2f} constructs/line)",
            description=(
                f"The file has {concepts} control-flow constructs across {sloc} lines "
                f"({density:.2f}/line). This is hard to follow. Split into smaller modules."
            ),
            location="module-level",
            start_line=1,
            penalty=20,
        ))
    else:
        score = max(0.0, 100 - density * 150)
        issues_out.append(ReadabilityIssue(
            dimension="cognitive_load",
            title=f"Very high concept density ({density:.2f} constructs/line)",
            description=(
                f"{concepts} control constructs in {sloc} SLOC is very dense. "
                f"Simplify logic and distribute across smaller functions/modules."
            ),
            location="module-level",
            start_line=1,
            penalty=35,
        ))

    return score


# ---------------------------------------------------------------------------
# Main scorer
# ---------------------------------------------------------------------------

class ReadabilityScorer:
    """
    Scores code readability across four dimensions and produces an A–F grade.

    Usage:
        scorer = ReadabilityScorer()
        result = scorer.score(source_code)
    """

    def score(self, source: str) -> ReadabilityResult:
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return ReadabilityResult(
                overall_score=0.0,
                grade="F",
                dimensions=[],
                issues=[],
                top_improvements=[],
                summary=f"Syntax error: {e}",
            )

        issues: list[ReadabilityIssue] = []

        naming = _score_naming(tree, issues)
        comments = _score_comment_density(source, issues)
        structure = _score_structural_clarity(tree, source, issues)
        cog_load = _score_cognitive_load(tree, source, issues)

        dimensions = [
            DimensionScore("naming_quality", naming, _WEIGHTS["naming_quality"],
                           [i.title for i in issues if i.dimension == "naming_quality"]),
            DimensionScore("comment_density", comments, _WEIGHTS["comment_density"],
                           [i.title for i in issues if i.dimension == "comment_density"]),
            DimensionScore("structural_clarity", structure, _WEIGHTS["structural_clarity"],
                           [i.title for i in issues if i.dimension == "structural_clarity"]),
            DimensionScore("cognitive_load", cog_load, _WEIGHTS["cognitive_load"],
                           [i.title for i in issues if i.dimension == "cognitive_load"]),
        ]

        overall = sum(d.score * d.weight for d in dimensions)
        grade = _grade(overall)

        # Top 3 improvements: lowest-scoring dimensions first
        sorted_dims = sorted(dimensions, key=lambda d: d.score)
        top_improvements = []
        improvement_hints = {
            "naming_quality": "Rename short/meaningless identifiers with descriptive names.",
            "comment_density": "Add docstrings to public functions and explain non-obvious logic.",
            "structural_clarity": "Shorten long functions and keep lines under 99 chars.",
            "cognitive_load": "Reduce branching by extracting helpers and simplifying logic.",
        }
        for d in sorted_dims[:3]:
            if d.score < 80:
                top_improvements.append(improvement_hints[d.name])

        issues.sort(key=lambda i: -i.penalty)

        summary = (
            f"Readability score: {overall:.0f}/100 (Grade {grade}). "
            f"Naming: {naming:.0f}, Comments: {comments:.0f}, "
            f"Structure: {structure:.0f}, Cognitive load: {cog_load:.0f}."
        )

        return ReadabilityResult(
            overall_score=overall,
            grade=grade,
            dimensions=dimensions,
            issues=issues,
            top_improvements=top_improvements,
            summary=summary,
        )
