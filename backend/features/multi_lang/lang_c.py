"""
C and C++ metric extraction via tree-sitter.
"""
from __future__ import annotations
import re
from features.multi_lang.ts_helpers import (
    loc_c_style, line_length_stats, halstead_approx,
    tree_cyclomatic, tree_cognitive,
)

_c_parser = None
_cpp_parser = None


def _get_c_parser():
    global _c_parser
    if _c_parser is None:
        from tree_sitter import Language, Parser
        import tree_sitter_c as tsc
        _c_parser = Parser(Language(tsc.language()))
    return _c_parser


def _get_cpp_parser():
    global _cpp_parser
    if _cpp_parser is None:
        from tree_sitter import Language, Parser
        import tree_sitter_cpp as tscpp
        _cpp_parser = Parser(Language(tscpp.language()))
    return _cpp_parser


_C_BRANCH_TYPES = frozenset([
    "if_statement", "while_statement", "do_statement",
    "for_statement", "switch_statement", "case_statement",
])

_C_BOOL_OPS = frozenset(["&&", "||"])

_C_STRUCTURAL = frozenset([
    "if_statement", "while_statement", "do_statement",
    "for_statement", "switch_statement",
])

_C_NESTING_RAISERS = frozenset([
    "if_statement", "while_statement", "do_statement",
    "for_statement", "switch_statement",
    "function_definition",
])

_C_FUNC_TYPES = frozenset(["function_definition"])

# C++ adds lambdas and constructors
_CPP_BRANCH_TYPES = _C_BRANCH_TYPES | frozenset(["try_statement"])
_CPP_STRUCTURAL   = _C_STRUCTURAL
_CPP_NESTING_RAISERS = _C_NESTING_RAISERS | frozenset([
    "lambda_expression", "constructor_or_destructor_definition",
])
_CPP_FUNC_TYPES   = _C_FUNC_TYPES | frozenset([
    "lambda_expression", "constructor_or_destructor_definition",
])


def _iter_functions(root_node, func_types):
    stack = [root_node]
    while stack:
        node = stack.pop()
        if node.type in func_types:
            yield node, node.start_point[0] + 1, node.end_point[0] + 1
        for child in node.children:
            stack.append(child)


def _compute(source: str, parser, branch_types, bool_ops, structural,
             nesting_raisers, func_types):
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
        tree = parser.parse(bytes(source, "utf-8"))
        root = tree.root_node

        result.cyclomatic_complexity = tree_cyclomatic(root, branch_types, bool_ops)
        result.cognitive_complexity = tree_cognitive(
            root, structural, nesting_raisers, func_types, bool_ops,
        )

        func_ccs = []
        for fn_node, start, end in _iter_functions(root, func_types):
            fc = tree_cyclomatic(fn_node, branch_types, bool_ops)
            func_ccs.append(fc)
            if (end - start + 1) > 50:
                result.n_long_functions += 1
            if fc > 10:
                result.n_complex_functions += 1

        if func_ccs:
            result.max_function_complexity = max(func_ccs)
            result.avg_function_complexity = sum(func_ccs) / len(func_ccs)

    except Exception:
        keywords = ("if ", "while ", "for ", "case ", "switch ")
        result.cyclomatic_complexity = 1 + sum(
            len(re.findall(r"\b" + re.escape(kw.strip()) + r"\b", source))
            for kw in keywords
        )
        result.cognitive_complexity = result.cyclomatic_complexity

    result.maintainability_index = maintainability_index(
        result.halstead.volume, result.cyclomatic_complexity, result.lines.sloc,
    )
    return result


def compute_c_metrics(source: str):
    return _compute(source, _get_c_parser(),
                    _C_BRANCH_TYPES, _C_BOOL_OPS, _C_STRUCTURAL,
                    _C_NESTING_RAISERS, _C_FUNC_TYPES)


def compute_cpp_metrics(source: str):
    return _compute(source, _get_cpp_parser(),
                    _CPP_BRANCH_TYPES, _C_BOOL_OPS, _CPP_STRUCTURAL,
                    _CPP_NESTING_RAISERS, _CPP_FUNC_TYPES)
