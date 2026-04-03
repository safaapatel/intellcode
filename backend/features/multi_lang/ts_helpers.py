"""
Tree-sitter helper utilities shared by JS/TS and Java adapters.
"""

import re
import math
from typing import Any


# ---------------------------------------------------------------------------
# LOC counting for C-style commented languages (JS, TS, Java)
# ---------------------------------------------------------------------------

def loc_c_style(source: str):
    """Returns (sloc, comment_lines, blank_lines) for C-style commented code."""
    lines = source.splitlines()
    sloc = blank = comments = 0
    in_block = False
    for line in lines:
        s = line.strip()
        if not s:
            blank += 1
            continue
        if in_block:
            comments += 1
            if "*/" in s:
                in_block = False
            continue
        # Block comment start
        if s.startswith("/*") or s.startswith("/**"):
            comments += 1
            if "*/" not in s[2:]:
                in_block = True
            continue
        # Line comment
        if s.startswith("//"):
            comments += 1
            continue
        sloc += 1
    return sloc, comments, blank


def line_length_stats(source: str):
    """Returns (max_length, avg_length, n_over_80)."""
    lines = source.splitlines()
    if not lines:
        return 0, 0.0, 0
    lengths = [len(l) for l in lines]
    return max(lengths), sum(lengths) / len(lengths), sum(1 for l in lengths if l > 80)


# ---------------------------------------------------------------------------
# Halstead approximation via regex tokenisation
# ---------------------------------------------------------------------------

# Strip string/template literals and comments before tokenising
_STR_COMMENT_RE = re.compile(
    r'(?s)"(?:[^"\\]|\\.)*"'
    r"|'(?:[^'\\]|\\.)*'"
    r"|`(?:[^`\\]|\\.)*`"
    r"|//[^\n]*"
    r"|/\*.*?\*/",
)

_OPERATOR_RE = re.compile(
    r"(?:\+\+|--|&&|\|\||===|!==|==|!=|<=|>=|\?\?|>>>|>>|<<"
    r"|[-+*/%&|^]=?|[!~]|[<>]=?|[?:.,;()\[\]{}]|=>)",
)
_OPERAND_RE = re.compile(
    r"\b(?:0[xX][0-9a-fA-F]+|\d+\.?\d*(?:[eE][+-]?\d+)?)\b"
    r"|\b[a-zA-Z_$][a-zA-Z0-9_$]*\b",
)

# Keywords excluded from operands (they are operators/structure)
_KEYWORDS = frozenset([
    "if", "else", "for", "while", "do", "switch", "case", "break", "continue",
    "return", "try", "catch", "finally", "throw", "new", "delete", "typeof",
    "instanceof", "in", "of", "void", "class", "extends", "super", "import",
    "export", "default", "const", "let", "var", "function", "async", "await",
    "yield", "static", "this", "null", "undefined", "true", "false",
    # Java keywords
    "public", "private", "protected", "final", "abstract", "interface",
    "enum", "implements", "package", "throws", "native", "synchronized",
    "transient", "volatile", "strictfp", "assert",
])


def halstead_approx(source: str):
    """
    Approximate Halstead metrics via regex tokenisation.
    Returns HalsteadMetrics-compatible dict.
    """
    stripped = _STR_COMMENT_RE.sub(" ", source)

    ops = _OPERATOR_RE.findall(stripped)
    raw_operands = _OPERAND_RE.findall(stripped)
    operands = [t for t in raw_operands if t not in _KEYWORDS]

    n1 = len(set(ops))
    n2 = len(set(operands))
    N1 = len(ops)
    N2 = len(operands)

    vocab = n1 + n2
    length = N1 + N2
    volume = length * math.log2(vocab) if vocab > 1 else 0.0
    difficulty = (n1 / 2.0) * (N2 / n2) if n2 > 0 else 0.0
    effort = difficulty * volume
    bugs = volume / 3000.0

    return {
        "n1": n1, "n2": n2, "N1": N1, "N2": N2,
        "vocabulary": vocab, "length": length,
        "volume": volume, "difficulty": difficulty,
        "effort": effort, "time_to_program": effort / 18.0,
        "bugs_delivered": bugs,
    }


# ---------------------------------------------------------------------------
# Generic tree-sitter complexity walkers
# ---------------------------------------------------------------------------

def tree_cyclomatic(root_node: Any, branch_types: frozenset, bool_op_texts: frozenset) -> int:
    """
    McCabe cyclomatic complexity via iterative tree-sitter walk.

    branch_types     -- node.type values that add +1 (if, for, while, catch, ...)
    bool_op_texts    -- operator texts that add +1 (&&, ||, ??)
    """
    count = 1
    stack = [root_node]
    while stack:
        node = stack.pop()
        if node.type in branch_types:
            count += 1
        else:
            # Check children for boolean operators inside binary expressions
            for child in node.children:
                if child.type in bool_op_texts:
                    count += 1
        for child in node.children:
            stack.append(child)
    return count


def tree_cognitive(
    root_node: Any,
    structural_types: frozenset,
    nesting_raiser_types: frozenset,
    func_types: frozenset,
    bool_op_texts: frozenset,
) -> int:
    """
    SonarSource cognitive complexity via recursive tree-sitter walk.

    structural_types    -- node types that add 1 + nesting
    nesting_raiser_types -- node types that increment nesting for their children
    func_types          -- node types for function definitions
    bool_op_texts       -- operator texts counted as +1 (each &&, ||, ??)
    """
    score = 0
    func_depth = 0

    def _walk(node: Any, nesting: int) -> None:
        nonlocal score, func_depth

        ntype = node.type

        # Functions: only nested ones raise nesting depth
        if ntype in func_types:
            nested = func_depth > 0
            if nested:
                nesting += 1
            func_depth += 1
            for child in node.children:
                _walk(child, nesting)
            func_depth -= 1
            if nested:
                nesting -= 1
            return

        # Structural control flow: +1 + nesting, then increase nesting for body
        if ntype in structural_types:
            score += 1 + nesting
            for child in node.children:
                _walk(child, nesting + 1)
            return

        # Ternary: flat +1
        if ntype == "ternary_expression":
            score += 1
            for child in node.children:
                _walk(child, nesting)
            return

        # Boolean operators inline in any expression
        for child in node.children:
            if child.type in bool_op_texts:
                score += 1

        for child in node.children:
            _walk(child, nesting)

    for child in root_node.children:
        _walk(child, 0)

    return score
