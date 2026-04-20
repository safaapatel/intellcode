"""
AST Feature Extractor
Parses Python source code using the built-in `ast` module and extracts
structural features used by the pattern recognition and security models.
"""

import ast
import tokenize
import io
import re
from typing import Any
from collections import defaultdict


class ASTExtractor(ast.NodeVisitor):
    """
    Walks a Python AST and accumulates structural features.

    Usage:
        extractor = ASTExtractor()
        features = extractor.extract(source_code)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._functions: list[dict] = []
        self._classes: list[dict] = []
        self._imports: list[str] = []
        self._calls: list[dict] = []
        self._assignments: list[dict] = []
        self._string_literals: list[dict] = []
        self._comparisons: list[dict] = []
        self._current_depth: int = 0
        self._max_depth: int = 0
        self._node_counts: dict[str, int] = defaultdict(int)
        self._global_vars: list[str] = []
        self._returns: int = 0
        self._raises: int = 0
        self._try_blocks: int = 0
        self._with_blocks: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, source: str) -> dict[str, Any]:
        """
        Parse *source* and return a feature dict.
        Returns an empty-features dict if parsing fails.
        """
        self.reset()
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return self._empty_features()

        self.visit(tree)

        return {
            # Counts
            "n_functions": len(self._functions),
            "n_classes": len(self._classes),
            "n_imports": len(self._imports),
            "n_calls": len(self._calls),
            "n_assignments": len(self._assignments),
            "n_string_literals": len(self._string_literals),
            "n_returns": self._returns,
            "n_raises": self._raises,
            "n_try_blocks": self._try_blocks,
            "n_with_blocks": self._with_blocks,
            # Depth / complexity
            "max_nesting_depth": self._max_depth,
            # Function-level features
            "max_params": max((f["n_params"] for f in self._functions), default=0),
            "avg_params": (
                sum(f["n_params"] for f in self._functions) / len(self._functions)
                if self._functions else 0.0
            ),
            "max_function_body_lines": max(
                (f["body_lines"] for f in self._functions), default=0
            ),
            "avg_function_body_lines": (
                sum(f["body_lines"] for f in self._functions) / len(self._functions)
                if self._functions else 0.0
            ),
            "n_decorated_functions": sum(
                1 for f in self._functions if f["n_decorators"] > 0
            ),
            # Class-level features
            "max_class_methods": max(
                (c["n_methods"] for c in self._classes), default=0
            ),
            "avg_class_methods": (
                sum(c["n_methods"] for c in self._classes) / len(self._classes)
                if self._classes else 0.0
            ),
            # Raw lists (for detailed analysis)
            "functions": self._functions,
            "classes": self._classes,
            "imports": self._imports,
            "calls": self._calls,
            "string_literals": self._string_literals,
            # Node type histogram (for ML feature vectors)
            "node_counts": dict(self._node_counts),
        }

    # ------------------------------------------------------------------
    # Visitor methods
    # ------------------------------------------------------------------

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._current_depth += 1
        self._max_depth = max(self._max_depth, self._current_depth)

        body_lines = (node.end_lineno or node.lineno) - node.lineno + 1

        self._functions.append({
            "name": node.name,
            "lineno": node.lineno,
            "n_params": (len(node.args.args) + len(node.args.posonlyargs)
                         + len(node.args.kwonlyargs)),
            "body_lines": body_lines,
            "n_decorators": len(node.decorator_list),
            "is_async": False,
        })
        self._node_counts["FunctionDef"] += 1
        self.generic_visit(node)
        self._current_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._current_depth += 1
        self._max_depth = max(self._max_depth, self._current_depth)

        body_lines = (node.end_lineno or node.lineno) - node.lineno + 1
        self._functions.append({
            "name": node.name,
            "lineno": node.lineno,
            "n_params": (len(node.args.args) + len(node.args.posonlyargs)
                         + len(node.args.kwonlyargs)),
            "body_lines": body_lines,
            "n_decorators": len(node.decorator_list),
            "is_async": True,
        })
        self._node_counts["AsyncFunctionDef"] += 1
        self.generic_visit(node)
        self._current_depth -= 1

    def visit_ClassDef(self, node: ast.ClassDef):
        self._current_depth += 1
        self._max_depth = max(self._max_depth, self._current_depth)

        # Use direct body children only — ast.walk would double-count nested class methods
        methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        self._classes.append({
            "name": node.name,
            "lineno": node.lineno,
            "n_methods": len(methods),
            "n_bases": len(node.bases),
        })
        self._node_counts["ClassDef"] += 1
        self.generic_visit(node)
        self._current_depth -= 1

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self._imports.append(alias.name)
        self._node_counts["Import"] += 1
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        module = node.module or ""
        for alias in node.names:
            self._imports.append(f"{module}.{alias.name}")
        self._node_counts["ImportFrom"] += 1
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        self._calls.append({
            "name": func_name,
            "lineno": node.lineno,
            "n_args": len(node.args) + len(node.keywords),
        })
        self._node_counts["Call"] += 1
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._assignments.append({
                    "name": target.id,
                    "lineno": node.lineno,
                })
        self._node_counts["Assign"] += 1
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, str) and len(node.value) > 3:
            self._string_literals.append({
                "value": node.value,
                "lineno": getattr(node, "lineno", 0),
            })
        self._node_counts["Constant"] += 1
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare):
        self._comparisons.append({"lineno": node.lineno})
        self._node_counts["Compare"] += 1
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return):
        self._returns += 1
        self._node_counts["Return"] += 1
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise):
        self._raises += 1
        self._node_counts["Raise"] += 1
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try):
        self._try_blocks += 1
        self._node_counts["Try"] += 1
        self._current_depth += 1
        self._max_depth = max(self._max_depth, self._current_depth)
        self.generic_visit(node)
        self._current_depth -= 1

    def visit_With(self, node: ast.With):
        self._with_blocks += 1
        self._node_counts["With"] += 1
        self._current_depth += 1
        self._max_depth = max(self._max_depth, self._current_depth)
        self.generic_visit(node)
        self._current_depth -= 1

    def visit_For(self, node: ast.For):
        self._node_counts["For"] += 1
        self._current_depth += 1
        self._max_depth = max(self._max_depth, self._current_depth)
        self.generic_visit(node)
        self._current_depth -= 1

    def visit_While(self, node: ast.While):
        self._node_counts["While"] += 1
        self._current_depth += 1
        self._max_depth = max(self._max_depth, self._current_depth)
        self.generic_visit(node)
        self._current_depth -= 1

    def visit_If(self, node: ast.If):
        self._node_counts["If"] += 1
        self._current_depth += 1
        self._max_depth = max(self._max_depth, self._current_depth)
        self.generic_visit(node)
        self._current_depth -= 1

    def visit_Lambda(self, node: ast.Lambda):
        self._node_counts["Lambda"] += 1
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp):
        self._node_counts["ListComp"] += 1
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp):
        self._node_counts["DictComp"] += 1
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp):
        self._node_counts["SetComp"] += 1
        self.generic_visit(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        self._node_counts["GeneratorExp"] += 1
        self.generic_visit(node)

    def visit_Yield(self, node: ast.Yield):
        self._node_counts["Yield"] += 1
        self.generic_visit(node)

    def visit_YieldFrom(self, node: ast.YieldFrom):
        self._node_counts["YieldFrom"] += 1
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert):
        self._node_counts["Assert"] += 1
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global):
        self._node_counts["Global"] += 1
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign):
        self._node_counts["AnnAssign"] += 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp):
        self._node_counts["IfExp"] += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node: ast.BoolOp):
        self._node_counts["BoolOp"] += 1
        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_features() -> dict[str, Any]:
        return {
            "n_functions": 0, "n_classes": 0, "n_imports": 0,
            "n_calls": 0, "n_assignments": 0, "n_string_literals": 0,
            "n_returns": 0, "n_raises": 0, "n_try_blocks": 0, "n_with_blocks": 0,
            "max_nesting_depth": 0, "max_params": 0, "avg_params": 0.0,
            "max_function_body_lines": 0, "avg_function_body_lines": 0.0,
            "n_decorated_functions": 0, "max_class_methods": 0, "avg_class_methods": 0.0,
            "functions": [], "classes": [], "imports": [], "calls": [],
            "string_literals": [], "node_counts": {},
        }


def extract_ast_features(source: str) -> dict[str, Any]:
    """Convenience wrapper around ASTExtractor."""
    return ASTExtractor().extract(source)


def tokenize_code(source: str) -> list[str]:
    """
    Tokenize Python source into a flat list of token strings.
    Used by the CNN-based security model.
    """
    tokens = []
    try:
        reader = io.StringIO(source).readline
        for tok in tokenize.generate_tokens(reader):
            if tok.type not in (tokenize.COMMENT, tokenize.NEWLINE,
                                tokenize.NL, tokenize.ENCODING):
                tokens.append(tok.string)
    except tokenize.TokenError:
        # Fall back to simple whitespace split if tokenizer fails
        tokens = source.split()
    return tokens


def build_token_vocab(corpus: list[str], max_vocab: int = 10_000) -> dict[str, int]:
    """
    Build a token → integer vocabulary from a list of source files.
    Reserved: 0 = <PAD>, 1 = <UNK>
    """
    from collections import Counter
    counter: Counter = Counter()
    for src in corpus:
        counter.update(tokenize_code(src))
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for token, _ in counter.most_common(max_vocab - 2):
        vocab[token] = len(vocab)
    return vocab


def tokens_to_ids(tokens: list[str], vocab: dict[str, int],
                  max_len: int = 512) -> list[int]:
    """Map token strings to integer IDs, truncating/padding to max_len."""
    unk = vocab.get("<UNK>", 1)
    pad = vocab.get("<PAD>", 0)
    ids = [vocab.get(t, unk) for t in tokens[:max_len]]
    ids += [pad] * (max_len - len(ids))
    return ids
