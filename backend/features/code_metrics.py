"""
Code Metrics
Computes static code quality metrics used by the complexity prediction model.

Metrics implemented:
  - Cyclomatic complexity (McCabe)
  - Cognitive complexity
  - Halstead metrics (volume, difficulty, effort, estimated bugs)
  - Lines-of-code breakdown
  - Unique operators/operands
"""

import ast
import tokenize
import io
import re
import math
from typing import Any
from dataclasses import dataclass, field, asdict


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LinesOfCode:
    total: int = 0
    sloc: int = 0       # Source lines (non-blank, non-comment)
    comments: int = 0
    blank: int = 0
    docstrings: int = 0


@dataclass
class HalsteadMetrics:
    n1: int = 0         # Distinct operators
    n2: int = 0         # Distinct operands
    N1: int = 0         # Total operators
    N2: int = 0         # Total operands
    vocabulary: int = 0
    length: int = 0
    volume: float = 0.0
    difficulty: float = 0.0
    effort: float = 0.0
    time_to_program: float = 0.0    # seconds
    bugs_delivered: float = 0.0     # estimated defects


@dataclass
class CodeMetricsResult:
    cyclomatic_complexity: int = 1
    cognitive_complexity: int = 0
    max_function_complexity: int = 1
    avg_function_complexity: float = 1.0
    lines: LinesOfCode = field(default_factory=LinesOfCode)
    halstead: HalsteadMetrics = field(default_factory=HalsteadMetrics)
    maintainability_index: float = 100.0
    n_long_functions: int = 0       # functions > 50 lines
    n_complex_functions: int = 0    # functions with CC > 10
    max_line_length: int = 0
    avg_line_length: float = 0.0
    n_lines_over_80: int = 0


# ---------------------------------------------------------------------------
# Cyclomatic Complexity (McCabe)
# ---------------------------------------------------------------------------

_DECISION_NODES = (
    ast.If, ast.While, ast.For, ast.ExceptHandler,
    ast.With, ast.Assert, ast.comprehension,
)

# BoolOp has 'op': And/Or — each adds a branch
_BOOL_OPS = (ast.And, ast.Or)


def cyclomatic_complexity(source: str) -> int:
    """
    Compute the McCabe cyclomatic complexity of an entire module.
    CC = number of decision points + 1
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 1

    count = 1  # base path
    for node in ast.walk(tree):
        if isinstance(node, _DECISION_NODES):
            count += 1
        elif isinstance(node, ast.BoolOp):
            # Each extra operand in a boolean expression adds a branch
            count += len(node.values) - 1
        elif isinstance(node, (ast.IfExp,)):  # ternary
            count += 1

    return count


def _function_cyclomatic(func_node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """CC for a single function node."""
    count = 1
    for node in ast.walk(func_node):
        if isinstance(node, _DECISION_NODES):
            count += 1
        elif isinstance(node, ast.BoolOp):
            count += len(node.values) - 1
        elif isinstance(node, ast.IfExp):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Cognitive Complexity
# ---------------------------------------------------------------------------

class _CognitiveComplexityVisitor(ast.NodeVisitor):
    """
    Approximates SonarSource's Cognitive Complexity metric.
    Rules:
      B1 — control flow structures increment by 1 + current nesting level
      B2 — sequences of logical operators increment by 1 each
      B3 — recursion increments by 1
    """

    def __init__(self):
        self.score = 0
        self._nesting = 0
        self._func_names: set[str] = set()

    def _increment(self, n: int = 1):
        self.score += n

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._func_names.add(node.name)
        self._nesting += 1
        self.generic_visit(node)
        self._nesting -= 1

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore

    def visit_If(self, node: ast.If):
        self._increment(1 + self._nesting)
        self._nesting += 1
        self.generic_visit(node)
        self._nesting -= 1

    def visit_For(self, node: ast.For):
        self._increment(1 + self._nesting)
        self._nesting += 1
        self.generic_visit(node)
        self._nesting -= 1

    visit_AsyncFor = visit_For  # type: ignore

    def visit_While(self, node: ast.While):
        self._increment(1 + self._nesting)
        self._nesting += 1
        self.generic_visit(node)
        self._nesting -= 1

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        self._increment(1 + self._nesting)
        self._nesting += 1
        self.generic_visit(node)
        self._nesting -= 1

    def visit_With(self, node: ast.With):
        self._increment(self._nesting)   # no structural increment, only nesting
        self._nesting += 1
        self.generic_visit(node)
        self._nesting -= 1

    def visit_BoolOp(self, node: ast.BoolOp):
        # Each operator token (and/or) adds 1
        self._increment(len(node.values) - 1)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Recursion
        if isinstance(node.func, ast.Name) and node.func.id in self._func_names:
            self._increment(1)
        self.generic_visit(node)


def cognitive_complexity(source: str) -> int:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 0
    visitor = _CognitiveComplexityVisitor()
    visitor.visit(tree)
    return visitor.score


# ---------------------------------------------------------------------------
# Lines of Code
# ---------------------------------------------------------------------------

def count_lines(source: str) -> LinesOfCode:
    result = LinesOfCode()
    lines = source.splitlines()
    result.total = len(lines)

    # Collect docstring line ranges
    docstring_lines: set[int] = set()
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                body = node.body
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                    ds = body[0]
                    start = ds.lineno - 1
                    end = (ds.end_lineno or ds.lineno)
                    for i in range(start, end):
                        docstring_lines.add(i)
    except SyntaxError:
        pass

    # Identify comment lines via tokenizer
    comment_lines: set[int] = set()
    try:
        reader = io.StringIO(source).readline
        for tok in tokenize.generate_tokens(reader):
            if tok.type == tokenize.COMMENT:
                comment_lines.add(tok.start[0] - 1)
    except tokenize.TokenError:
        pass

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            result.blank += 1
        elif idx in docstring_lines:
            result.docstrings += 1
        elif idx in comment_lines:
            result.comments += 1
        else:
            result.sloc += 1

    return result


# ---------------------------------------------------------------------------
# Halstead Metrics
# ---------------------------------------------------------------------------

_OPERATOR_TOKENS = {
    tokenize.OP,
    tokenize.NAME,   # keywords: if, for, while, return, import, ...
}

_KEYWORD_OPERATORS = {
    "if", "else", "elif", "for", "while", "with", "return", "yield",
    "import", "from", "as", "in", "not", "and", "or", "is", "lambda",
    "raise", "try", "except", "finally", "pass", "break", "continue",
    "del", "global", "nonlocal", "assert", "class", "def",
}

_OPERAND_TOKEN_TYPES = {
    tokenize.STRING, tokenize.NUMBER,
}


def halstead_metrics(source: str) -> HalsteadMetrics:
    operators: dict[str, int] = {}
    operands: dict[str, int] = {}

    try:
        reader = io.StringIO(source).readline
        for tok_type, tok_string, *_ in tokenize.generate_tokens(reader):
            if tok_type == tokenize.NAME and tok_string in _KEYWORD_OPERATORS:
                operators[tok_string] = operators.get(tok_string, 0) + 1
            elif tok_type == tokenize.OP:
                operators[tok_string] = operators.get(tok_string, 0) + 1
            elif tok_type in _OPERAND_TOKEN_TYPES:
                operands[tok_string] = operands.get(tok_string, 0) + 1
            elif tok_type == tokenize.NAME and tok_string not in _KEYWORD_OPERATORS:
                operands[tok_string] = operands.get(tok_string, 0) + 1
    except tokenize.TokenError:
        pass

    m = HalsteadMetrics()
    m.n1 = len(operators)
    m.n2 = len(operands)
    m.N1 = sum(operators.values())
    m.N2 = sum(operands.values())
    m.vocabulary = m.n1 + m.n2
    m.length = m.N1 + m.N2

    if m.vocabulary > 0 and m.length > 0:
        m.volume = m.length * math.log2(m.vocabulary)
        m.difficulty = (m.n1 / 2) * (m.N2 / m.n2) if m.n2 > 0 else 0.0
        m.effort = m.difficulty * m.volume
        m.time_to_program = m.effort / 18.0
        m.bugs_delivered = m.volume / 3000.0

    return m


# ---------------------------------------------------------------------------
# Maintainability Index
# ---------------------------------------------------------------------------

def maintainability_index(
    halstead_volume: float,
    cyclomatic: int,
    sloc: int,
) -> float:
    """
    Original Oman & Hagemeister (1992) formula, scaled to 0–100.
    MI = 171 - 5.2*ln(HV) - 0.23*CC - 16.2*ln(LOC)
    Clamped to [0, 100].
    """
    if halstead_volume <= 0 or sloc <= 0:
        return 100.0
    raw = (
        171.0
        - 5.2 * math.log(halstead_volume)
        - 0.23 * cyclomatic
        - 16.2 * math.log(sloc)
    )
    # Scale to 0-100
    return max(0.0, min(100.0, raw / 171.0 * 100.0))


# ---------------------------------------------------------------------------
# Line-length metrics
# ---------------------------------------------------------------------------

def line_length_metrics(source: str) -> tuple[int, float, int]:
    """Returns (max_length, avg_length, n_lines_over_80)."""
    lines = [ln for ln in source.splitlines() if ln.strip()]
    if not lines:
        return 0, 0.0, 0
    lengths = [len(ln) for ln in lines]
    return max(lengths), sum(lengths) / len(lengths), sum(1 for l in lengths if l > 80)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_all_metrics(source: str) -> CodeMetricsResult:
    """
    Compute every metric for *source* and return a CodeMetricsResult.
    This is the primary function called by the complexity prediction model.
    """
    result = CodeMetricsResult()

    # Lines
    result.lines = count_lines(source)

    # Cyclomatic
    result.cyclomatic_complexity = cyclomatic_complexity(source)
    result.cognitive_complexity = cognitive_complexity(source)

    # Per-function CC
    try:
        tree = ast.parse(source)
        func_ccs = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fc = _function_cyclomatic(node)
                func_ccs.append(fc)
                body_lines = (node.end_lineno or node.lineno) - node.lineno + 1
                if body_lines > 50:
                    result.n_long_functions += 1
                if fc > 10:
                    result.n_complex_functions += 1
        if func_ccs:
            result.max_function_complexity = max(func_ccs)
            result.avg_function_complexity = sum(func_ccs) / len(func_ccs)
    except SyntaxError:
        pass

    # Halstead
    result.halstead = halstead_metrics(source)

    # Maintainability Index
    result.maintainability_index = maintainability_index(
        result.halstead.volume,
        result.cyclomatic_complexity,
        result.lines.sloc,
    )

    # Line lengths
    result.max_line_length, result.avg_line_length, result.n_lines_over_80 = (
        line_length_metrics(source)
    )

    return result


def metrics_to_feature_vector(m: CodeMetricsResult) -> list[float]:
    """
    Flatten a CodeMetricsResult into a 16-element numeric feature vector.
    Order must stay consistent with training.

    NOTE: maintainability_index is intentionally EXCLUDED — it is used as the
    training target for the complexity model, so including it here would cause
    direct target leakage (previously caused R²=1.000 on training data).
    """
    return [
        float(m.cyclomatic_complexity),
        float(m.cognitive_complexity),
        float(m.max_function_complexity),
        float(m.avg_function_complexity),
        float(m.lines.sloc),
        float(m.lines.comments),
        float(m.lines.blank),
        float(m.halstead.volume),
        float(m.halstead.difficulty),
        float(m.halstead.effort),
        float(m.halstead.bugs_delivered),
        float(m.n_long_functions),
        float(m.n_complex_functions),
        float(m.max_line_length),
        float(m.avg_line_length),
        float(m.n_lines_over_80),
    ]
