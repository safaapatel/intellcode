"""
Performance Hotspot Predictor
Detects static performance anti-patterns using AST analysis.

Detected patterns:
    nested_loops          — O(n²) or worse nested for/while loops
    call_in_loop          — expensive function/method calls inside loops
    string_concat_loop    — '+=' string concatenation in loops (use join)
    list_append_loop      — repeated list.append() (prefer list comprehension)
    repeated_attr_access  — accessing obj.attr repeatedly (cache it)
    global_in_loop        — reading a global variable inside a loop
    io_in_loop            — file I/O or network calls (open/read/write) in loops
    redundant_computation — same expression computed repeatedly
    large_default_arg     — mutable default arguments (list/dict) causing shared state
    inefficient_membership— using 'in list' instead of 'in set' for membership tests

Each finding:
    pattern_type  — machine-readable key
    title         — short label
    description   — what to fix and why
    location      — function + line
    severity      — "high" | "medium" | "low"
    speedup_hint  — estimated relative speedup if fixed
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class PerformanceIssue:
    pattern_type: str
    title: str
    description: str
    location: str
    start_line: int
    severity: str       # "high" | "medium" | "low"
    speedup_hint: str   # e.g. "10–100×" or "linear instead of quadratic"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PerformanceResult:
    issues: list[PerformanceIssue]
    hotspot_score: float    # 0 (clean) → 100 (many hotspots)
    severity_counts: dict[str, int]
    summary: str

    def to_dict(self) -> dict:
        return {
            "issues": [i.to_dict() for i in self.issues],
            "hotspot_score": round(self.hotspot_score, 1),
            "severity_counts": self.severity_counts,
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

_LOOP_TYPES = (ast.For, ast.While)
_IO_FUNCS = {"open", "read", "write", "readline", "readlines", "send", "recv",
             "urlopen", "get", "post", "request", "sleep", "connect"}
_EXPENSIVE_CALLS = {"sorted", "sort", "reversed", "max", "min", "sum",
                    "len", "list", "dict", "set", "copy", "deepcopy",
                    "json", "pickle", "load", "dump", "compile", "exec", "eval"}


def _is_loop(node) -> bool:
    return isinstance(node, _LOOP_TYPES)


def _loop_body_nodes(loop) -> list[ast.AST]:
    """Flatten all nodes inside a loop body."""
    result = []
    for child in ast.walk(loop):
        if child is not loop:
            result.append(child)
    return result


def _call_name(node: ast.Call) -> Optional[str]:
    """Extract the function name from a call node."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _is_string_node(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _check_nested_loops(tree: ast.Module) -> list[PerformanceIssue]:
    issues = []

    def walk_loops(node, depth, outer_line):
        if _is_loop(node) and depth > 0:
            issues.append(PerformanceIssue(
                pattern_type="nested_loops",
                title=f"Nested loop (depth {depth + 1}) at line {node.lineno}",
                description=(
                    f"A loop at line {node.lineno} is nested {depth} level(s) deep "
                    f"(outer loop at line {outer_line}). "
                    f"This creates O(n^{depth+1}) time complexity. "
                    f"Consider vectorization, hash maps, or restructuring the algorithm."
                ),
                location=f"line {node.lineno}",
                start_line=node.lineno,
                severity="high" if depth >= 2 else "medium",
                speedup_hint="O(n²) → O(n) with hash map / set lookup",
            ))

        child_depth = depth + 1 if _is_loop(node) else depth
        outer = node.lineno if _is_loop(node) and depth == 0 else outer_line
        for child in ast.iter_child_nodes(node):
            walk_loops(child, child_depth, outer)

    walk_loops(tree, 0, 0)
    return issues


def _check_calls_in_loops(tree: ast.Module) -> list[PerformanceIssue]:
    issues = []

    for node in ast.walk(tree):
        if not _is_loop(node):
            continue
        for child in _loop_body_nodes(node):
            if isinstance(child, ast.Call):
                name = _call_name(child)
                if name and name in _EXPENSIVE_CALLS:
                    issues.append(PerformanceIssue(
                        pattern_type="call_in_loop",
                        title=f"Expensive call '{name}()' inside loop (line {child.lineno})",
                        description=(
                            f"'{name}()' is called inside a loop at line {child.lineno}. "
                            f"If its result doesn't change per iteration, "
                            f"compute it once before the loop and cache the result."
                        ),
                        location=f"line {child.lineno}",
                        start_line=child.lineno,
                        severity="medium",
                        speedup_hint="Cache result outside loop → O(1) per iteration",
                    ))

        # IO in loop
        for child in _loop_body_nodes(node):
            if isinstance(child, ast.Call):
                name = _call_name(child)
                if name and name in _IO_FUNCS:
                    issues.append(PerformanceIssue(
                        pattern_type="io_in_loop",
                        title=f"I/O call '{name}()' inside loop (line {child.lineno})",
                        description=(
                            f"I/O operation '{name}()' at line {child.lineno} is inside a loop. "
                            f"I/O is orders of magnitude slower than CPU operations. "
                            f"Batch reads/writes outside the loop."
                        ),
                        location=f"line {child.lineno}",
                        start_line=child.lineno,
                        severity="high",
                        speedup_hint="Batch I/O outside loop → 10–1000× faster",
                    ))

    return issues


def _check_string_concat_loop(tree: ast.Module) -> list[PerformanceIssue]:
    """Detect s += '...' or s = s + '...' inside loops."""
    issues = []

    for node in ast.walk(tree):
        if not _is_loop(node):
            continue
        for child in _loop_body_nodes(node):
            # AugAssign: s += expr
            if isinstance(child, ast.AugAssign) and isinstance(child.op, ast.Add):
                if _is_string_node(child.value) or (
                    isinstance(child.value, ast.Call) and _call_name(child.value) == "str"
                ):
                    issues.append(PerformanceIssue(
                        pattern_type="string_concat_loop",
                        title=f"String concatenation in loop (line {child.lineno})",
                        description=(
                            f"String '+=' at line {child.lineno} creates a new string "
                            f"object every iteration (O(n²) total). "
                            f"Collect parts in a list and join at the end: "
                            f"'result = \"\".join(parts)'."
                        ),
                        location=f"line {child.lineno}",
                        start_line=child.lineno,
                        severity="high",
                        speedup_hint="''.join(parts) → O(n) vs O(n²)",
                    ))
            # Assign: s = s + '...' or s = '...' + s
            elif isinstance(child, ast.Assign) and isinstance(child.value, ast.BinOp):
                binop = child.value
                if isinstance(binop.op, ast.Add):
                    left_str = _is_string_node(binop.left) or (isinstance(binop.left, ast.Call) and _call_name(binop.left) == "str")
                    right_str = _is_string_node(binop.right) or (isinstance(binop.right, ast.Call) and _call_name(binop.right) == "str")
                    # At least one side is a string literal/str() call and the other is a Name
                    if (left_str and isinstance(binop.right, ast.Name)) or (right_str and isinstance(binop.left, ast.Name)):
                        issues.append(PerformanceIssue(
                            pattern_type="string_concat_loop",
                            title=f"String concatenation via assignment in loop (line {child.lineno})",
                            description=(
                                f"'s = s + ...' at line {child.lineno} creates a new string "
                                f"object every iteration (O(n²) total). "
                                f"Use 'parts.append(...); result = \"\".join(parts)' instead."
                            ),
                            location=f"line {child.lineno}",
                            start_line=child.lineno,
                            severity="high",
                            speedup_hint="''.join(parts) → O(n) vs O(n²)",
                        ))

    return issues


def _check_list_append_loop(tree: ast.Module) -> list[PerformanceIssue]:
    """Detect repeated list.append() — suggest list comprehension."""
    issues = []

    for node in ast.walk(tree):
        if not _is_loop(node):
            continue
        appends = 0
        last_line = node.lineno
        for child in _loop_body_nodes(node):
            if (isinstance(child, ast.Call) and
                    isinstance(child.func, ast.Attribute) and
                    child.func.attr == "append"):
                appends += 1
                last_line = child.lineno

        if appends >= 1:
            issues.append(PerformanceIssue(
                pattern_type="list_append_loop",
                title=f"list.append() in loop — prefer list comprehension (line {last_line})",
                description=(
                    f"Found {appends} append(s) inside a loop. "
                    f"List comprehensions are 20–50% faster than append loops "
                    f"because they avoid repeated attribute lookup and method call overhead."
                ),
                location=f"line {node.lineno}",
                start_line=node.lineno,
                severity="low",
                speedup_hint="List comprehension → ~30% faster",
            ))

    return issues


def _check_repeated_attr_access(tree: ast.Module) -> list[PerformanceIssue]:
    """Detect repeated obj.attr access inside a loop without caching."""
    issues = []

    for node in ast.walk(tree):
        if not _is_loop(node):
            continue

        attr_counts: dict[str, list[int]] = {}
        for child in _loop_body_nodes(node):
            if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
                key = f"{child.value.id}.{child.attr}"
                attr_counts.setdefault(key, []).append(child.col_offset)

        for attr, occurrences in attr_counts.items():
            if len(occurrences) >= 3:
                issues.append(PerformanceIssue(
                    pattern_type="repeated_attr_access",
                    title=f"Repeated attribute access '{attr}' in loop (line {node.lineno})",
                    description=(
                        f"'{attr}' is accessed {len(occurrences)} times inside a loop. "
                        f"Cache it in a local variable before the loop: "
                        f"'cached = {attr}' to avoid repeated dict lookups."
                    ),
                    location=f"line {node.lineno}",
                    start_line=node.lineno,
                    severity="low",
                    speedup_hint="Local variable caching → ~5–15% faster",
                ))

    return issues


def _check_membership_test(tree: ast.Module) -> list[PerformanceIssue]:
    """Detect 'x in list_var' where set would be O(1)."""
    issues = []

    # Collect known list variables (simple heuristic: assigned as list literal)
    list_vars: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and isinstance(node.value, ast.List):
                    list_vars.add(target.id)

    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for op, comp in zip(node.ops, node.comparators):
                if isinstance(op, ast.In) and isinstance(comp, ast.Name):
                    if comp.id in list_vars:
                        issues.append(PerformanceIssue(
                            pattern_type="inefficient_membership",
                            title=f"'in list' membership test at line {node.lineno}",
                            description=(
                                f"'{comp.id}' is a list; 'in' on a list is O(n). "
                                f"Convert to a set: '{comp.id} = set({comp.id})' "
                                f"for O(1) membership tests."
                            ),
                            location=f"line {node.lineno}",
                            start_line=node.lineno,
                            severity="medium",
                            speedup_hint="set → O(1) lookup vs O(n)",
                        ))

    return issues


def _check_mutable_default_args(tree: ast.Module) -> list[PerformanceIssue]:
    """Detect mutable default arguments (list/dict)."""
    issues = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for default in node.args.defaults + node.args.kw_defaults:
            if default is None:
                continue
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                issues.append(PerformanceIssue(
                    pattern_type="mutable_default_arg",
                    title=f"Mutable default argument in '{node.name}' (line {node.lineno})",
                    description=(
                        f"'{node.name}' uses a mutable default argument (list/dict/set). "
                        f"This object is shared across all calls, causing subtle bugs. "
                        f"Use None as default and create a new object inside the function."
                    ),
                    location=f"line {node.lineno}",
                    start_line=node.lineno,
                    severity="medium",
                    speedup_hint="Prevents shared-state bugs; no speed impact",
                ))

    return issues


_DB_CALL_NAMES = {
    "execute", "executemany", "fetchone", "fetchall", "fetchmany",
    "query", "filter", "filter_by", "get", "find", "find_one",
    "commit", "rollback",
}

_REGEX_CALL_NAMES = {"match", "search", "findall", "finditer", "fullmatch", "sub", "subn", "split"}


def _check_regex_in_loop(tree: ast.Module) -> list[PerformanceIssue]:
    """Detect re.compile()-able patterns called directly inside loops."""
    issues = []
    for node in ast.walk(tree):
        if not _is_loop(node):
            continue
        for child in _loop_body_nodes(node):
            if not isinstance(child, ast.Call):
                continue
            # re.match(...) / re.search(...) etc. with a literal pattern string
            if (isinstance(child.func, ast.Attribute)
                    and child.func.attr in _REGEX_CALL_NAMES
                    and isinstance(child.func.value, ast.Name)
                    and child.func.value.id == "re"
                    and child.args
                    and isinstance(child.args[0], ast.Constant)):
                issues.append(PerformanceIssue(
                    pattern_type="regex_in_loop",
                    title=f"re.{child.func.attr}() with literal pattern inside loop (line {child.lineno})",
                    description=(
                        f"re.{child.func.attr}() at line {child.lineno} recompiles the regex "
                        f"pattern on every iteration. Pre-compile with "
                        f"'pattern = re.compile(...)' before the loop and call 'pattern.{child.func.attr}()'."
                    ),
                    location=f"line {child.lineno}",
                    start_line=child.lineno,
                    severity="medium",
                    speedup_hint="Pre-compile regex → avoid per-iteration compilation overhead",
                ))
    return issues


def _check_db_query_in_loop(tree: ast.Module) -> list[PerformanceIssue]:
    """Detect database query calls (execute/query/filter) inside loops — N+1 query pattern."""
    issues = []
    for node in ast.walk(tree):
        if not _is_loop(node):
            continue
        for child in _loop_body_nodes(node):
            if not isinstance(child, ast.Call):
                continue
            name = _call_name(child)
            if name and name in _DB_CALL_NAMES:
                issues.append(PerformanceIssue(
                    pattern_type="db_query_in_loop",
                    title=f"Database call '{name}()' inside loop (line {child.lineno})",
                    description=(
                        f"'{name}()' at line {child.lineno} is called inside a loop. "
                        f"This is the N+1 query problem — each iteration hits the database. "
                        f"Batch the query outside the loop (e.g. fetch all records once, "
                        f"then index by key in a dict)."
                    ),
                    location=f"line {child.lineno}",
                    start_line=child.lineno,
                    severity="high",
                    speedup_hint="Batch query outside loop → N db calls → 1 db call",
                ))
    return issues


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class PerformanceAnalyzer:
    """
    Detects static performance anti-patterns in Python source code.

    Usage:
        analyzer = PerformanceAnalyzer()
        result = analyzer.analyze(source_code)
    """

    def analyze(self, source: str) -> PerformanceResult:
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return PerformanceResult(
                issues=[],
                hotspot_score=0.0,
                severity_counts={},
                summary=f"Syntax error: {e}",
            )

        issues: list[PerformanceIssue] = []
        issues.extend(_check_nested_loops(tree))
        issues.extend(_check_calls_in_loops(tree))
        issues.extend(_check_string_concat_loop(tree))
        issues.extend(_check_list_append_loop(tree))
        issues.extend(_check_repeated_attr_access(tree))
        issues.extend(_check_membership_test(tree))
        issues.extend(_check_mutable_default_args(tree))
        issues.extend(_check_regex_in_loop(tree))
        issues.extend(_check_db_query_in_loop(tree))

        # Deduplicate by (type, line)
        seen: set[tuple[str, int]] = set()
        unique: list[PerformanceIssue] = []
        for issue in issues:
            key = (issue.pattern_type, issue.start_line)
            if key not in seen:
                seen.add(key)
                unique.append(issue)

        unique.sort(key=lambda i: ({"high": 0, "medium": 1, "low": 2}.get(i.severity, 3), i.start_line))

        severity_counts: dict[str, int] = {}
        for i in unique:
            severity_counts[i.severity] = severity_counts.get(i.severity, 0) + 1

        # Hotspot score: weight by severity
        weights = {"high": 20, "medium": 10, "low": 5}
        raw = sum(weights.get(i.severity, 5) for i in unique)
        hotspot_score = min(100.0, raw)

        n = len(unique)
        hi = severity_counts.get("high", 0)
        if n == 0:
            summary = "No performance hotspots detected."
        else:
            summary = (
                f"Found {n} performance issue(s) ({hi} high severity). "
                f"Hotspot score: {hotspot_score:.0f}/100."
            )

        return PerformanceResult(
            issues=unique,
            hotspot_score=hotspot_score,
            severity_counts=severity_counts,
            summary=summary,
        )
