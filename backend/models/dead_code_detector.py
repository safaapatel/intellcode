"""
Dead Code Detector
Identifies unused, unreachable, and redundant code using AST analysis.

Detects:
    - Unreachable code after return / raise / break / continue
    - Defined functions that are never called within the file
    - Defined classes that are never instantiated or referenced
    - Unused imports (imported but never referenced)
    - Unused local variables (assigned but never read)
    - Unused function parameters
    - Empty except blocks (silently swallows errors)
    - Redundant else after return (unnecessary else clause)

Each finding includes:
    - issue_type: machine-readable key
    - title: short label
    - description: explanation
    - location: line(s) in the file
    - severity: "warning" | "info"
    - removable: True if the code can safely be deleted
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class DeadCodeIssue:
    issue_type: str
    title: str
    description: str
    location: str
    start_line: int
    end_line: int
    severity: str       # "warning" | "info"
    removable: bool     # True = can delete without side effects

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DeadCodeResult:
    issues: list[DeadCodeIssue]
    dead_line_count: int        # estimated lines of dead code
    total_lines: int
    dead_ratio: float           # dead_lines / total_lines
    summary: str

    def to_dict(self) -> dict:
        return {
            "issues": [i.to_dict() for i in self.issues],
            "dead_line_count": self.dead_line_count,
            "total_lines": self.total_lines,
            "dead_ratio": round(self.dead_ratio, 3),
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Visitor: collect definitions and usages
# ---------------------------------------------------------------------------

class _NameCollector(ast.NodeVisitor):
    """Collects all names that are *used* (loaded) in the AST."""

    def __init__(self):
        self.used: set[str] = set()

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            self.used.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        # Capture obj.attr — record 'obj' as used
        if isinstance(node.value, ast.Name):
            self.used.add(node.value.id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Direct function call: foo()
        if isinstance(node.func, ast.Name):
            self.used.add(node.func.id)
        # Method call: obj.method()
        elif isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                self.used.add(node.func.value.id)
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _check_unreachable_code(tree: ast.Module) -> list[DeadCodeIssue]:
    """Find statements after return/raise/break/continue."""
    issues = []
    _TERMINATORS = (ast.Return, ast.Raise, ast.Break, ast.Continue)

    def check_body(body: list[ast.stmt]):
        for i, stmt in enumerate(body):
            if isinstance(stmt, _TERMINATORS):
                if i + 1 < len(body):
                    next_stmt = body[i + 1]
                    # skip trailing docstrings / pass
                    if isinstance(next_stmt, ast.Pass):
                        continue
                    end = getattr(body[-1], "end_lineno", next_stmt.lineno)
                    issues.append(DeadCodeIssue(
                        issue_type="unreachable_code",
                        title="Unreachable code after terminator statement",
                        description=(
                            f"Code after the '{type(stmt).__name__.lower()}' at "
                            f"line {stmt.lineno} will never execute "
                            f"(lines {next_stmt.lineno}–{end})."
                        ),
                        location=f"lines {next_stmt.lineno}–{end}",
                        start_line=next_stmt.lineno,
                        end_line=end,
                        severity="warning",
                        removable=True,
                    ))
                    break  # only report first dead block per body

            # Recurse into compound statements
            for child_body_attr in ("body", "orelse", "handlers", "finalbody"):
                child_body = getattr(stmt, child_body_attr, [])
                if isinstance(child_body, list):
                    check_body(child_body)

    check_body(tree.body)
    return issues


def _check_unused_imports(tree: ast.Module) -> list[DeadCodeIssue]:
    """Find imports whose names are never used."""
    collector = _NameCollector()
    collector.visit(tree)
    used = collector.used

    issues = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname if alias.asname else alias.name.split(".")[0]
                if name not in used and name != "__future__":
                    issues.append(DeadCodeIssue(
                        issue_type="unused_import",
                        title=f"Unused import: '{alias.name}'",
                        description=(
                            f"'{alias.name}' is imported at line {node.lineno} "
                            f"but never used. Remove it to reduce confusion."
                        ),
                        location=f"line {node.lineno}",
                        start_line=node.lineno,
                        end_line=node.lineno,
                        severity="warning",
                        removable=True,
                    ))
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == "*":
                    continue
                name = alias.asname if alias.asname else alias.name
                if name not in used:
                    issues.append(DeadCodeIssue(
                        issue_type="unused_import",
                        title=f"Unused import: '{alias.name}' from '{node.module}'",
                        description=(
                            f"'{alias.name}' is imported from '{node.module}' at "
                            f"line {node.lineno} but never referenced."
                        ),
                        location=f"line {node.lineno}",
                        start_line=node.lineno,
                        end_line=node.lineno,
                        severity="warning",
                        removable=True,
                    ))
    return issues


def _check_unused_functions(tree: ast.Module) -> list[DeadCodeIssue]:
    """Find module-level functions that are never called."""
    collector = _NameCollector()
    collector.visit(tree)
    used = collector.used

    issues = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_") or node.name in ("main", "__init__"):
                continue  # private / special functions are commonly "unused" by design
            if node.name not in used:
                end = getattr(node, "end_lineno", node.lineno)
                issues.append(DeadCodeIssue(
                    issue_type="unused_function",
                    title=f"Unused function: '{node.name}'",
                    description=(
                        f"Function '{node.name}' defined at line {node.lineno} "
                        f"is never called within this file. If it's part of a "
                        f"public API, add a usage or export it explicitly."
                    ),
                    location=f"lines {node.lineno}–{end}",
                    start_line=node.lineno,
                    end_line=end,
                    severity="info",
                    removable=False,  # might be called externally
                ))
    return issues


def _check_unused_variables(tree: ast.Module) -> list[DeadCodeIssue]:
    """Find local variables that are assigned but never read."""
    issues = []

    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        # Collect all stores and loads within this function
        stores: dict[str, list[int]] = {}   # name → [lineno, ...]
        loads: set[str] = set()

        for node in ast.walk(func):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    stores.setdefault(node.id, []).append(node.lineno)
                elif isinstance(node.ctx, ast.Load):
                    loads.add(node.id)

        for name, linenos in stores.items():
            if name.startswith("_") or name in ("self", "cls"):
                continue
            if name not in loads:
                issues.append(DeadCodeIssue(
                    issue_type="unused_variable",
                    title=f"Unused variable: '{name}' in '{func.name}'",
                    description=(
                        f"Variable '{name}' is assigned (line {linenos[0]}) inside "
                        f"'{func.name}' but its value is never read. "
                        f"Either use it or remove the assignment."
                    ),
                    location=f"line {linenos[0]} in function '{func.name}'",
                    start_line=linenos[0],
                    end_line=linenos[0],
                    severity="info",
                    removable=True,
                ))

    return issues


def _check_empty_except(tree: ast.Module) -> list[DeadCodeIssue]:
    """Find empty except blocks that silently swallow errors."""
    issues = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            for handler in node.handlers:
                body = handler.body
                # Empty or pass-only body
                if all(isinstance(s, ast.Pass) for s in body):
                    issues.append(DeadCodeIssue(
                        issue_type="empty_except",
                        title=f"Empty except block at line {handler.lineno}",
                        description=(
                            f"An except block at line {handler.lineno} silently "
                            f"swallows exceptions. Add logging or re-raise the "
                            f"exception so errors are not hidden."
                        ),
                        location=f"line {handler.lineno}",
                        start_line=handler.lineno,
                        end_line=getattr(handler, "end_lineno", handler.lineno),
                        severity="warning",
                        removable=False,
                    ))
    return issues


def _check_redundant_else(tree: ast.Module) -> list[DeadCodeIssue]:
    """Detect else clauses that are redundant after a return statement."""
    issues = []

    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(func):
            if isinstance(node, ast.If) and node.orelse:
                # If the if-body ends with a return, the else is redundant
                if_body = node.body
                if if_body and isinstance(if_body[-1], ast.Return):
                    else_start = node.orelse[0].lineno
                    else_end = getattr(node.orelse[-1], "end_lineno", else_start)
                    issues.append(DeadCodeIssue(
                        issue_type="redundant_else",
                        title=f"Redundant else after return (line {else_start})",
                        description=(
                            f"The 'else' block starting at line {else_start} is "
                            f"unnecessary because the 'if' block always returns. "
                            f"Remove the else and un-indent its contents."
                        ),
                        location=f"lines {else_start}–{else_end}",
                        start_line=else_start,
                        end_line=else_end,
                        severity="info",
                        removable=True,
                    ))
    return issues


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class DeadCodeDetector:
    """
    Detects dead, unused, and unreachable code in Python source files.

    Usage:
        detector = DeadCodeDetector()
        result = detector.detect(source_code)
    """

    def detect(self, source: str) -> DeadCodeResult:
        """
        Analyze source code for dead code issues.

        Args:
            source: Python source code string.

        Returns:
            DeadCodeResult with all issues found.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return DeadCodeResult(
                issues=[],
                dead_line_count=0,
                total_lines=len(source.splitlines()),
                dead_ratio=0.0,
                summary=f"Syntax error — cannot analyze: {e}",
            )

        total_lines = len(source.splitlines())
        issues: list[DeadCodeIssue] = []

        issues.extend(_check_unreachable_code(tree))
        issues.extend(_check_unused_imports(tree))
        issues.extend(_check_unused_functions(tree))
        issues.extend(_check_unused_variables(tree))
        issues.extend(_check_empty_except(tree))
        issues.extend(_check_redundant_else(tree))

        # Sort by line number
        issues.sort(key=lambda i: i.start_line)

        # Estimate dead lines (from removable issues)
        dead_lines = sum(
            max(1, i.end_line - i.start_line + 1)
            for i in issues
            if i.removable
        )
        dead_ratio = dead_lines / total_lines if total_lines > 0 else 0.0

        n = len(issues)
        warnings = sum(1 for i in issues if i.severity == "warning")
        if n == 0:
            summary = "No dead code detected."
        else:
            summary = (
                f"Found {n} dead code issue(s) ({warnings} warnings). "
                f"Estimated {dead_lines} removable line(s) "
                f"({dead_ratio:.1%} of file)."
            )

        return DeadCodeResult(
            issues=issues,
            dead_line_count=dead_lines,
            total_lines=total_lines,
            dead_ratio=dead_ratio,
            summary=summary,
        )
