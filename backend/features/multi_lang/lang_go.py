"""
Go metric extraction via tree-sitter.
"""
from __future__ import annotations
from features.multi_lang.ts_helpers import (
    loc_c_style, line_length_stats, halstead_approx,
    tree_cyclomatic, tree_cognitive,
)

_go_parser = None


def _get_parser():
    global _go_parser
    if _go_parser is None:
        from tree_sitter import Language, Parser
        import tree_sitter_go as tsgo
        _go_parser = Parser(Language(tsgo.language()))
    return _go_parser


_GO_BRANCH_TYPES = frozenset([
    "if_statement", "for_statement",
    "expression_switch_statement", "type_switch_statement",
    "select_statement", "expression_case", "default_case",
    "communication_case", "type_case",
])

_GO_BOOL_OPS = frozenset(["&&", "||"])

_GO_STRUCTURAL = frozenset([
    "if_statement", "for_statement",
    "expression_switch_statement", "type_switch_statement",
    "select_statement",
])

_GO_NESTING_RAISERS = frozenset([
    "if_statement", "for_statement",
    "expression_switch_statement", "type_switch_statement",
    "select_statement", "func_literal",
    "function_declaration", "method_declaration",
])

_GO_FUNC_TYPES = frozenset([
    "function_declaration", "method_declaration", "func_literal",
])


def _iter_functions(root_node):
    stack = [root_node]
    while stack:
        node = stack.pop()
        if node.type in _GO_FUNC_TYPES:
            yield node, node.start_point[0] + 1, node.end_point[0] + 1
        for child in node.children:
            stack.append(child)


def compute_go_metrics(source: str):
    from features.code_metrics import (
        CodeMetricsResult, LinesOfCode, HalsteadMetrics, maintainability_index,
    )
    result = CodeMetricsResult()

    sloc, comments, blank = loc_c_style(source)
    result.lines = LinesOfCode(total=sloc + comments + blank, sloc=sloc,
                               comments=comments, blank=blank, docstrings=0)
    result.max_line_length, result.avg_line_length, result.n_lines_over_80 = line_length_stats(source)
    h = halstead_approx(source)
    result.halstead = HalsteadMetrics(**h)

    try:
        parser = _get_parser()
        tree = parser.parse(bytes(source, "utf-8"))
        root = tree.root_node

        result.cyclomatic_complexity = tree_cyclomatic(root, _GO_BRANCH_TYPES, _GO_BOOL_OPS)
        result.cognitive_complexity = tree_cognitive(
            root, _GO_STRUCTURAL, _GO_NESTING_RAISERS, _GO_FUNC_TYPES, _GO_BOOL_OPS,
        )

        func_ccs = []
        for fn_node, start, end in _iter_functions(root):
            fc = tree_cyclomatic(fn_node, _GO_BRANCH_TYPES, _GO_BOOL_OPS)
            func_ccs.append(fc)
            if (end - start + 1) > 50:
                result.n_long_functions += 1
            if fc > 10:
                result.n_complex_functions += 1

        if func_ccs:
            result.max_function_complexity = max(func_ccs)
            result.avg_function_complexity = sum(func_ccs) / len(func_ccs)

    except Exception:
        import re
        keywords = ("if ", "for ", "case ", "select ")
        result.cyclomatic_complexity = 1 + sum(
            len(re.findall(r"\b" + re.escape(kw.strip()) + r"\b", source))
            for kw in keywords
        )
        result.cognitive_complexity = result.cyclomatic_complexity

    result.maintainability_index = maintainability_index(
        result.halstead.volume, result.cyclomatic_complexity, result.lines.sloc,
    )
    return result
