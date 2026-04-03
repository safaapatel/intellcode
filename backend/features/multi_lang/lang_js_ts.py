"""
JavaScript and TypeScript metric extraction via tree-sitter.
"""

from __future__ import annotations

from features.multi_lang.ts_helpers import (
    loc_c_style, line_length_stats, halstead_approx,
    tree_cyclomatic, tree_cognitive,
)

# ---------------------------------------------------------------------------
# tree-sitter setup (lazy import so missing deps degrade gracefully)
# ---------------------------------------------------------------------------

_js_parser = None
_ts_parser = None


def _get_parser(lang: str):
    global _js_parser, _ts_parser
    from tree_sitter import Language, Parser
    if lang == "typescript":
        if _ts_parser is None:
            import tree_sitter_typescript as tsts
            _ts_parser = Parser(Language(tsts.language_typescript()))
        return _ts_parser
    else:
        if _js_parser is None:
            import tree_sitter_javascript as tsjs
            _js_parser = Parser(Language(tsjs.language()))
        return _js_parser


# ---------------------------------------------------------------------------
# Node type sets for JavaScript / TypeScript
# ---------------------------------------------------------------------------

_JS_BRANCH_TYPES = frozenset([
    "if_statement", "while_statement", "do_statement",
    "for_statement", "for_in_statement",
    "switch_case", "catch_clause",
])

_JS_BOOL_OPS = frozenset(["&&", "||", "??"])

_JS_STRUCTURAL = frozenset([
    "if_statement", "while_statement", "do_statement",
    "for_statement", "for_in_statement",
    "switch_statement", "catch_clause",
])

_JS_NESTING_RAISERS = frozenset([
    "if_statement", "while_statement", "do_statement",
    "for_statement", "for_in_statement",
    "switch_statement", "catch_clause",
    "function_declaration", "function", "arrow_function", "method_definition",
    "generator_function", "generator_function_declaration",
])

_JS_FUNC_TYPES = frozenset([
    "function_declaration", "function", "arrow_function",
    "method_definition", "generator_function", "generator_function_declaration",
])


# ---------------------------------------------------------------------------
# Per-function analysis
# ---------------------------------------------------------------------------

def _iter_functions(root_node):
    """Yield (function_node, start_line, end_line) for each function in the tree."""
    stack = [root_node]
    while stack:
        node = stack.pop()
        if node.type in _JS_FUNC_TYPES:
            yield node, node.start_point[0] + 1, node.end_point[0] + 1
        for child in node.children:
            stack.append(child)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_js_metrics(source: str, language: str = "javascript"):
    """Return a CodeMetricsResult for JavaScript or TypeScript source."""
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
        parser = _get_parser(language)
        tree = parser.parse(bytes(source, "utf-8"))
        root = tree.root_node

        result.cyclomatic_complexity = tree_cyclomatic(root, _JS_BRANCH_TYPES, _JS_BOOL_OPS)
        result.cognitive_complexity = tree_cognitive(
            root, _JS_STRUCTURAL, _JS_NESTING_RAISERS, _JS_FUNC_TYPES, _JS_BOOL_OPS,
        )

        # Per-function stats
        func_ccs = []
        for fn_node, start, end in _iter_functions(root):
            fc = tree_cyclomatic(fn_node, _JS_BRANCH_TYPES, _JS_BOOL_OPS)
            func_ccs.append(fc)
            if (end - start + 1) > 50:
                result.n_long_functions += 1
            if fc > 10:
                result.n_complex_functions += 1

        if func_ccs:
            result.max_function_complexity = max(func_ccs)
            result.avg_function_complexity = sum(func_ccs) / len(func_ccs)

    except Exception:
        # If tree-sitter fails, fall back to simple heuristic counts
        import re
        keywords = ("if ", "else if ", "for ", "while ", "case ", "catch ", " ? ")
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
