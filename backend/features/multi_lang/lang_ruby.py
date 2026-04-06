"""
Ruby metric extraction via tree-sitter.
"""
from __future__ import annotations
import re
from features.multi_lang.ts_helpers import line_length_stats, halstead_approx, tree_cyclomatic, tree_cognitive

_ruby_parser = None


def _get_parser():
    global _ruby_parser
    if _ruby_parser is None:
        from tree_sitter import Language, Parser
        import tree_sitter_ruby as tsruby
        _ruby_parser = Parser(Language(tsruby.language()))
    return _ruby_parser


def _loc_ruby(source: str):
    """LOC counting for Ruby (#-commented language)."""
    sloc = blank = comments = 0
    in_heredoc = False
    for line in source.splitlines():
        s = line.strip()
        if not s:
            blank += 1
        elif s.startswith("#"):
            comments += 1
        elif s.startswith("=begin"):
            in_heredoc = True
            comments += 1
        elif s.startswith("=end"):
            in_heredoc = False
            comments += 1
        elif in_heredoc:
            comments += 1
        else:
            sloc += 1
    return sloc, comments, blank


_RUBY_BRANCH_TYPES = frozenset([
    "if", "unless", "while", "until", "for",
    "when", "rescue", "elsif",
    "if_modifier", "unless_modifier", "while_modifier", "until_modifier",
])

_RUBY_BOOL_OPS = frozenset(["&&", "||", "and", "or"])

_RUBY_STRUCTURAL = frozenset([
    "if", "unless", "while", "until", "for",
    "case", "rescue",
])

_RUBY_NESTING_RAISERS = frozenset([
    "if", "unless", "while", "until", "for",
    "case", "rescue", "method", "singleton_method",
    "block", "lambda", "do_block",
    "class", "module",
])

_RUBY_FUNC_TYPES = frozenset([
    "method", "singleton_method", "lambda", "block", "do_block",
])


def _iter_functions(root_node):
    stack = [root_node]
    while stack:
        node = stack.pop()
        if node.type in _RUBY_FUNC_TYPES:
            yield node, node.start_point[0] + 1, node.end_point[0] + 1
        for child in node.children:
            stack.append(child)


def compute_ruby_metrics(source: str):
    from features.code_metrics import (
        CodeMetricsResult, LinesOfCode, HalsteadMetrics, maintainability_index,
    )
    result = CodeMetricsResult()

    sloc, comments, blank = _loc_ruby(source)
    result.lines = LinesOfCode(total=sloc + comments + blank, sloc=sloc,
                               comments=comments, blank=blank, docstrings=0)
    result.max_line_length, result.avg_line_length, result.n_lines_over_80 = line_length_stats(source)
    h = halstead_approx(source)
    result.halstead = HalsteadMetrics(**h)

    try:
        parser = _get_parser()
        tree = parser.parse(bytes(source, "utf-8"))
        root = tree.root_node

        result.cyclomatic_complexity = tree_cyclomatic(root, _RUBY_BRANCH_TYPES, _RUBY_BOOL_OPS)
        result.cognitive_complexity = tree_cognitive(
            root, _RUBY_STRUCTURAL, _RUBY_NESTING_RAISERS, _RUBY_FUNC_TYPES, _RUBY_BOOL_OPS,
        )

        func_ccs = []
        for fn_node, start, end in _iter_functions(root):
            fc = tree_cyclomatic(fn_node, _RUBY_BRANCH_TYPES, _RUBY_BOOL_OPS)
            func_ccs.append(fc)
            if (end - start + 1) > 50:
                result.n_long_functions += 1
            if fc > 10:
                result.n_complex_functions += 1

        if func_ccs:
            result.max_function_complexity = max(func_ccs)
            result.avg_function_complexity = sum(func_ccs) / len(func_ccs)

    except Exception:
        keywords = ("if ", "unless ", "while ", "until ", "for ", "when ", "rescue ")
        result.cyclomatic_complexity = 1 + sum(
            len(re.findall(r"\b" + re.escape(kw.strip()) + r"\b", source))
            for kw in keywords
        )
        result.cognitive_complexity = result.cyclomatic_complexity

    result.maintainability_index = maintainability_index(
        result.halstead.volume, result.cyclomatic_complexity, result.lines.sloc,
    )
    return result
