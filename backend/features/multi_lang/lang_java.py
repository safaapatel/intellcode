"""
Java metric extraction via tree-sitter.
"""

from __future__ import annotations

from features.multi_lang.ts_helpers import (
    loc_c_style, line_length_stats, halstead_approx,
    tree_cyclomatic, tree_cognitive,
)

# ---------------------------------------------------------------------------
# tree-sitter setup
# ---------------------------------------------------------------------------

_java_parser = None


def _get_parser():
    global _java_parser
    if _java_parser is None:
        from tree_sitter import Language, Parser
        import tree_sitter_java as tsjava
        _java_parser = Parser(Language(tsjava.language()))
    return _java_parser


# ---------------------------------------------------------------------------
# Node type sets for Java
# ---------------------------------------------------------------------------

_JAVA_BRANCH_TYPES = frozenset([
    "if_statement", "while_statement", "do_statement",
    "for_statement", "enhanced_for_statement",
    "switch_label", "catch_clause",
])

_JAVA_BOOL_OPS = frozenset(["&&", "||"])

_JAVA_STRUCTURAL = frozenset([
    "if_statement", "while_statement", "do_statement",
    "for_statement", "enhanced_for_statement",
    "switch_expression", "switch_statement", "catch_clause",
])

_JAVA_NESTING_RAISERS = frozenset([
    "if_statement", "while_statement", "do_statement",
    "for_statement", "enhanced_for_statement",
    "switch_expression", "switch_statement", "catch_clause",
    "method_declaration", "constructor_declaration",
    "lambda_expression", "class_declaration", "interface_declaration",
])

_JAVA_FUNC_TYPES = frozenset([
    "method_declaration", "constructor_declaration", "lambda_expression",
])


# ---------------------------------------------------------------------------
# Per-function analysis
# ---------------------------------------------------------------------------

def _iter_functions(root_node):
    """Yield (function_node, start_line, end_line)."""
    stack = [root_node]
    while stack:
        node = stack.pop()
        if node.type in _JAVA_FUNC_TYPES:
            yield node, node.start_point[0] + 1, node.end_point[0] + 1
        for child in node.children:
            stack.append(child)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_java_metrics(source: str):
    """Return a CodeMetricsResult for Java source."""
    from features.code_metrics import (
        CodeMetricsResult, LinesOfCode, HalsteadMetrics, maintainability_index,
    )

    result = CodeMetricsResult()

    # Lines
    sloc, comments, blank = loc_c_style(source)
    result.lines = LinesOfCode(
        total=sloc + comments + blank,
        sloc=sloc,
        comments=comments,
        blank=blank,
        docstrings=0,
    )

    # Line length stats
    result.max_line_length, result.avg_line_length, result.n_lines_over_80 = (
        line_length_stats(source)
    )

    # Halstead
    h = halstead_approx(source)
    result.halstead = HalsteadMetrics(**h)

    # Parse with tree-sitter
    try:
        parser = _get_parser()
        tree = parser.parse(bytes(source, "utf-8"))
        root = tree.root_node

        result.cyclomatic_complexity = tree_cyclomatic(root, _JAVA_BRANCH_TYPES, _JAVA_BOOL_OPS)
        result.cognitive_complexity = tree_cognitive(
            root, _JAVA_STRUCTURAL, _JAVA_NESTING_RAISERS, _JAVA_FUNC_TYPES, _JAVA_BOOL_OPS,
        )

        # Per-function stats
        func_ccs = []
        for fn_node, start, end in _iter_functions(root):
            fc = tree_cyclomatic(fn_node, _JAVA_BRANCH_TYPES, _JAVA_BOOL_OPS)
            func_ccs.append(fc)
            if (end - start + 1) > 50:
                result.n_long_functions += 1
            if fc > 10:
                result.n_complex_functions += 1

        if func_ccs:
            result.max_function_complexity = max(func_ccs)
            result.avg_function_complexity = sum(func_ccs) / len(func_ccs)

    except Exception:
        # Regex fallback
        import re
        keywords = ("if ", "else if ", "for ", "while ", "case ", "catch ")
        result.cyclomatic_complexity = 1 + sum(
            len(re.findall(r"\b" + re.escape(kw.strip()) + r"\b", source))
            for kw in keywords
        )
        result.cognitive_complexity = result.cyclomatic_complexity

    # Maintainability index
    result.maintainability_index = maintainability_index(
        result.halstead.volume,
        result.cyclomatic_complexity,
        result.lines.sloc,
    )

    return result
