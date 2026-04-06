"""
PHP metric extraction via tree-sitter.
"""
from __future__ import annotations
import re
from features.multi_lang.ts_helpers import line_length_stats, halstead_approx, tree_cyclomatic, tree_cognitive

_php_parser = None


def _get_parser():
    global _php_parser
    if _php_parser is None:
        from tree_sitter import Language, Parser
        import tree_sitter_php as tsphp
        _php_parser = Parser(Language(tsphp.language_php()))
    return _php_parser


def _loc_php(source: str):
    """LOC counting for PHP (// # /* */ comments)."""
    sloc = blank = comments = 0
    in_block = False
    for line in source.splitlines():
        s = line.strip()
        if not s:
            blank += 1
            continue
        if in_block:
            comments += 1
            if "*/" in s:
                in_block = False
            continue
        if s.startswith("/*") or s.startswith("/**"):
            comments += 1
            if "*/" not in s[2:]:
                in_block = True
            continue
        if s.startswith("//") or s.startswith("#"):
            comments += 1
            continue
        sloc += 1
    return sloc, comments, blank


_PHP_BRANCH_TYPES = frozenset([
    "if_statement", "while_statement", "do_statement",
    "for_statement", "foreach_statement",
    "switch_statement", "catch_clause", "match_expression",
    "elseif_clause",
])

_PHP_BOOL_OPS = frozenset(["&&", "||", "and", "or"])

_PHP_STRUCTURAL = frozenset([
    "if_statement", "while_statement", "do_statement",
    "for_statement", "foreach_statement",
    "switch_statement", "catch_clause", "match_expression",
])

_PHP_NESTING_RAISERS = frozenset([
    "if_statement", "while_statement", "do_statement",
    "for_statement", "foreach_statement",
    "switch_statement", "catch_clause", "match_expression",
    "function_definition", "method_declaration",
    "anonymous_function", "arrow_function",
    "class_declaration", "interface_declaration",
])

_PHP_FUNC_TYPES = frozenset([
    "function_definition", "method_declaration",
    "anonymous_function", "arrow_function",
])


def _iter_functions(root_node):
    stack = [root_node]
    while stack:
        node = stack.pop()
        if node.type in _PHP_FUNC_TYPES:
            yield node, node.start_point[0] + 1, node.end_point[0] + 1
        for child in node.children:
            stack.append(child)


def compute_php_metrics(source: str):
    from features.code_metrics import (
        CodeMetricsResult, LinesOfCode, HalsteadMetrics, maintainability_index,
    )
    result = CodeMetricsResult()

    sloc, comments, blank = _loc_php(source)
    result.lines = LinesOfCode(total=sloc + comments + blank, sloc=sloc,
                               comments=comments, blank=blank, docstrings=0)
    result.max_line_length, result.avg_line_length, result.n_lines_over_80 = line_length_stats(source)
    h = halstead_approx(source)
    result.halstead = HalsteadMetrics(**h)

    try:
        parser = _get_parser()
        tree = parser.parse(bytes(source, "utf-8"))
        root = tree.root_node

        result.cyclomatic_complexity = tree_cyclomatic(root, _PHP_BRANCH_TYPES, _PHP_BOOL_OPS)
        result.cognitive_complexity = tree_cognitive(
            root, _PHP_STRUCTURAL, _PHP_NESTING_RAISERS, _PHP_FUNC_TYPES, _PHP_BOOL_OPS,
        )

        func_ccs = []
        for fn_node, start, end in _iter_functions(root):
            fc = tree_cyclomatic(fn_node, _PHP_BRANCH_TYPES, _PHP_BOOL_OPS)
            func_ccs.append(fc)
            if (end - start + 1) > 50:
                result.n_long_functions += 1
            if fc > 10:
                result.n_complex_functions += 1

        if func_ccs:
            result.max_function_complexity = max(func_ccs)
            result.avg_function_complexity = sum(func_ccs) / len(func_ccs)

    except Exception:
        keywords = ("if ", "elseif ", "while ", "for ", "foreach ", "case ", "catch ")
        result.cyclomatic_complexity = 1 + sum(
            len(re.findall(r"\b" + re.escape(kw.strip()) + r"\b", source))
            for kw in keywords
        )
        result.cognitive_complexity = result.cyclomatic_complexity

    result.maintainability_index = maintainability_index(
        result.halstead.volume, result.cyclomatic_complexity, result.lines.sloc,
    )
    return result
