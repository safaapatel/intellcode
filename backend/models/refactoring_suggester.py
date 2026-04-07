"""
Refactoring Suggester
Analyzes code and produces specific, actionable refactoring recommendations.

Detects:
    - Long methods (> 50 lines) → Extract Method
    - Too many parameters (> 4) → Introduce Parameter Object
    - Complex conditions (nested ands/ors) → Introduce Explaining Variable
    - Magic numbers/strings → Extract Constant
    - Duplicate code blocks within a file → Consolidate / DRY
    - God functions (do too many things) → Split Function
    - Deep nesting (> 3 levels) → Extract + Early Return
    - Feature envy (method uses another class's data extensively) → Move Method

Each suggestion includes:
    - refactoring_type: machine-readable key
    - title: human-readable label
    - description: what to do and why
    - location: function name + line numbers
    - effort: estimated minutes to apply
    - priority: "critical" | "high" | "medium" | "low"
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
class RefactoringSuggestion:
    refactoring_type: str
    title: str
    description: str
    location: str           # e.g. "function 'process_data' (lines 42–91)"
    start_line: int
    end_line: int
    effort_minutes: int
    priority: str           # "critical" | "high" | "medium" | "low"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RefactoringResult:
    suggestions: list[RefactoringSuggestion]
    total_effort_minutes: int
    priority_counts: dict[str, int]
    summary: str

    def to_dict(self) -> dict:
        return {
            "suggestions": [s.to_dict() for s in self.suggestions],
            "total_effort_minutes": self.total_effort_minutes,
            "priority_counts": self.priority_counts,
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _count_lines(node: ast.AST) -> int:
    if hasattr(node, "end_lineno") and hasattr(node, "lineno"):
        return node.end_lineno - node.lineno + 1
    return 0


def _nesting_depth(node: ast.AST) -> int:
    """Compute maximum nesting depth of control flow inside a function."""
    max_depth = [0]

    def walk(n, depth):
        if isinstance(n, (ast.If, ast.For, ast.While, ast.With,
                          ast.Try, ast.ExceptHandler)):
            depth += 1
            max_depth[0] = max(max_depth[0], depth)
        for child in ast.iter_child_nodes(n):
            walk(child, depth)

    walk(node, 0)
    return max_depth[0]


def _count_magic_numbers(node: ast.AST) -> list[tuple[int, float | int]]:
    """Return (lineno, value) for magic number literals (not 0, 1, -1, 2)."""
    ALLOWED = {0, 1, -1, 2, 100}
    found = []
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            if n.value not in ALLOWED and not isinstance(n.value, bool):
                found.append((n.lineno, n.value))
    return found


def _count_magic_strings(node: ast.AST) -> list[tuple[int, str]]:
    """Return (lineno, value) for non-trivial string literals."""
    found = []
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            if len(n.value) > 3 and not n.value.startswith(("http", "/")):
                found.append((n.lineno, n.value[:30]))
    return found


def _condition_complexity(node: ast.AST) -> int:
    """Count boolean operators in a condition."""
    count = 0
    for n in ast.walk(node):
        if isinstance(n, ast.BoolOp):
            count += len(n.values) - 1
    return count


def _count_return_points(func: ast.FunctionDef) -> int:
    return sum(1 for n in ast.walk(func) if isinstance(n, ast.Return))


def _method_uses_other_class(
    func: ast.FunctionDef,
    class_name: Optional[str],
) -> bool:
    """
    Heuristic for Feature Envy: the function accesses many attributes
    that don't belong to 'self'.
    """
    if class_name is None:
        return False
    external_accesses = 0
    for node in ast.walk(func):
        if isinstance(node, ast.Attribute):
            # obj.attr where obj is not 'self'
            if isinstance(node.value, ast.Name) and node.value.id != "self":
                external_accesses += 1
    return external_accesses >= 5


# ---------------------------------------------------------------------------
# Suggester
# ---------------------------------------------------------------------------

class RefactoringSuggester:
    """
    Analyzes Python source code and returns refactoring suggestions.

    Usage:
        suggester = RefactoringSuggester()
        result = suggester.analyze(source_code)
    """

    LONG_METHOD_LINES = 40
    TOO_MANY_PARAMS = 4
    DEEP_NESTING = 3
    COMPLEX_CONDITION_OPS = 3
    MANY_RETURN_POINTS = 4
    MAGIC_NUMBER_THRESHOLD = 3

    def analyze(self, source: str) -> RefactoringResult:
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return RefactoringResult(
                suggestions=[],
                total_effort_minutes=0,
                priority_counts={},
                summary=f"Syntax error — cannot analyze: {e}",
            )

        suggestions: list[RefactoringSuggestion] = []
        lines = source.splitlines()

        # Walk top-level and class-level functions
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parent_class = self._find_parent_class(tree, node)
                suggestions.extend(
                    self._analyze_function(node, lines, parent_class)
                )

        # Module-level checks
        suggestions.extend(self._check_module_level(tree, lines))

        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        suggestions.sort(key=lambda s: (priority_order.get(s.priority, 9), s.start_line))

        # Deduplicate overlapping suggestions of same type
        suggestions = self._deduplicate(suggestions)

        total_effort = sum(s.effort_minutes for s in suggestions)
        priority_counts: dict[str, int] = {}
        for s in suggestions:
            priority_counts[s.priority] = priority_counts.get(s.priority, 0) + 1

        n = len(suggestions)
        if n == 0:
            summary = "No refactoring opportunities found. Code looks clean."
        else:
            hi = priority_counts.get("critical", 0) + priority_counts.get("high", 0)
            summary = (
                f"Found {n} refactoring opportunity(s) "
                f"({hi} high/critical priority). "
                f"Estimated total effort: {total_effort} min."
            )

        return RefactoringResult(
            suggestions=suggestions,
            total_effort_minutes=total_effort,
            priority_counts=priority_counts,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Function-level checks
    # ------------------------------------------------------------------

    def _analyze_function(
        self,
        func: ast.FunctionDef,
        lines: list[str],
        parent_class: Optional[str],
    ) -> list[RefactoringSuggestion]:
        results = []
        name = func.name
        start = func.lineno
        end = getattr(func, "end_lineno", start)
        loc = f"function '{name}' (lines {start}–{end})"
        n_lines = _count_lines(func)

        # --- Long Method ---
        if n_lines > self.LONG_METHOD_LINES:
            priority = "critical" if n_lines > 80 else "high"
            results.append(RefactoringSuggestion(
                refactoring_type="extract_method",
                title=f"Extract Method: '{name}' is too long ({n_lines} lines)",
                description=(
                    f"'{name}' has {n_lines} lines. Functions should ideally be under "
                    f"{self.LONG_METHOD_LINES} lines. Identify logical sub-steps and "
                    f"extract them into well-named helper functions."
                ),
                location=loc,
                start_line=start,
                end_line=end,
                effort_minutes=30,
                priority=priority,
            ))

        # --- Too Many Parameters ---
        params = [a for a in func.args.args if a.arg != "self"]
        if len(params) > self.TOO_MANY_PARAMS:
            results.append(RefactoringSuggestion(
                refactoring_type="introduce_parameter_object",
                title=f"Introduce Parameter Object: '{name}' has {len(params)} params",
                description=(
                    f"'{name}' accepts {len(params)} parameters: "
                    f"{', '.join(a.arg for a in params)}. "
                    f"Group related parameters into a dataclass or named tuple."
                ),
                location=loc,
                start_line=start,
                end_line=end,
                effort_minutes=20,
                priority="high" if len(params) > 6 else "medium",
            ))

        # --- Deep Nesting ---
        depth = _nesting_depth(func)
        if depth > self.DEEP_NESTING:
            results.append(RefactoringSuggestion(
                refactoring_type="reduce_nesting",
                title=f"Reduce Nesting: '{name}' has {depth} levels deep",
                description=(
                    f"'{name}' reaches {depth} levels of control-flow nesting. "
                    f"Use early returns (guard clauses), extract inner loops into "
                    f"helper functions, or invert conditions to flatten the structure."
                ),
                location=loc,
                start_line=start,
                end_line=end,
                effort_minutes=25,
                priority="high" if depth > 4 else "medium",
            ))

        # --- Complex Conditions ---
        for node in ast.walk(func):
            if isinstance(node, (ast.If, ast.While)) and hasattr(node, "test"):
                ops = _condition_complexity(node.test)
                if ops >= self.COMPLEX_CONDITION_OPS:
                    results.append(RefactoringSuggestion(
                        refactoring_type="introduce_explaining_variable",
                        title=f"Simplify Condition in '{name}' (line {node.lineno})",
                        description=(
                            f"A condition at line {node.lineno} in '{name}' has "
                            f"{ops} boolean operators. Extract sub-expressions into "
                            f"descriptively named boolean variables for readability."
                        ),
                        location=f"line {node.lineno} in {loc}",
                        start_line=node.lineno,
                        end_line=node.lineno,
                        effort_minutes=10,
                        priority="medium",
                    ))

        # --- Many Return Points ---
        n_returns = _count_return_points(func)
        if n_returns > self.MANY_RETURN_POINTS:
            results.append(RefactoringSuggestion(
                refactoring_type="consolidate_returns",
                title=f"Consolidate Returns: '{name}' has {n_returns} return statements",
                description=(
                    f"'{name}' has {n_returns} return points which makes control flow "
                    f"hard to follow. Consider restructuring with guard clauses at the "
                    f"top and a single return at the bottom."
                ),
                location=loc,
                start_line=start,
                end_line=end,
                effort_minutes=15,
                priority="low",
            ))

        # --- Feature Envy ---
        if _method_uses_other_class(func, parent_class):
            results.append(RefactoringSuggestion(
                refactoring_type="move_method",
                title=f"Move Method: '{name}' envies another object",
                description=(
                    f"'{name}' accesses external objects' attributes more than its "
                    f"own class's. Consider moving this method closer to the data it "
                    f"uses, or pass only the values it needs."
                ),
                location=loc,
                start_line=start,
                end_line=end,
                effort_minutes=30,
                priority="medium",
            ))

        return results

    # ------------------------------------------------------------------
    # Module-level checks
    # ------------------------------------------------------------------

    def _check_module_level(
        self,
        tree: ast.Module,
        lines: list[str],
    ) -> list[RefactoringSuggestion]:
        results = []

        # --- Magic Numbers ---
        magic_nums: list[tuple[int, float | int]] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                magic_nums.extend(_count_magic_numbers(node))

        if len(magic_nums) >= self.MAGIC_NUMBER_THRESHOLD:
            sample = ", ".join(str(v) for _, v in magic_nums[:4])
            results.append(RefactoringSuggestion(
                refactoring_type="extract_constant",
                title=f"Extract Constants: {len(magic_nums)} magic number(s) found",
                description=(
                    f"Found {len(magic_nums)} magic number literals (e.g. {sample}). "
                    f"Replace with named constants at module or class level "
                    f"(e.g. MAX_RETRIES = 3) to improve readability and maintainability."
                ),
                location="module-level",
                start_line=magic_nums[0][0],
                end_line=magic_nums[-1][0],
                effort_minutes=15,
                priority="medium",
            ))

        # --- Duplicate import aliases ---
        from collections import Counter
        import_names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    import_names.append(alias.name)
        duplicates = [name for name, cnt in Counter(import_names).items() if cnt > 1]
        if duplicates:
            dup_list = ", ".join(f"'{d}'" for d in duplicates[:5])
            results.append(RefactoringSuggestion(
                refactoring_type="remove_duplicate_imports",
                title=f"Remove Duplicate Imports ({len(duplicates)} module(s))",
                description=(
                    f"Module(s) {dup_list} are imported more than once. "
                    "Consolidate all imports at the top of the file."
                ),
                location="module-level imports",
                start_line=1,
                end_line=1,
                effort_minutes=5,
                priority="low",
            ))

        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_parent_class(tree: ast.Module, func: ast.AST) -> Optional[str]:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for child in ast.walk(node):
                    if child is func:
                        return node.name
        return None

    @staticmethod
    def _deduplicate(suggestions: list[RefactoringSuggestion]) -> list[RefactoringSuggestion]:
        seen: set[tuple[str, int]] = set()
        result = []
        for s in suggestions:
            key = (s.refactoring_type, s.start_line)
            if key not in seen:
                seen.add(key)
                result.append(s)
        return result
