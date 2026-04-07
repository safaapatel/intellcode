"""
Lightweight AST-based Taint Path Tracker  (v2)
================================================
Detects whether user-controlled data flows from a taint source to a dangerous
sink within a module, including simple inter-procedural propagation (1-2 hops).

v2 improvements (from audit follow-up):
  - Inter-procedural: tracks functions that return tainted data and treats
    calls to them as taint sources in the caller.
  - Graded sanitizer trust: replace()/strip() -> 0.15, escape() -> 0.65,
    parameterized queries -> 1.0.  Risk = source_strength * (1 - san_strength).
  - Argument-level semantics: classifies sink arguments as
    constant / parameterized / fstring / concatenation / variable.
  - Path-sensitive (shallow): counts fraction of if/else branches that
    carry taint to a sink, giving branch_taint_fraction in [0, 1].

Why this matters:
  The original RF features counted API presence (injection_api_count,
  taint_source_count) independently. Under LOPO those counts had zero
  correlation with actual vulnerability. The causal signal is the PATH:
  does user input reach the sink unsanitized?

  The inter-procedural case that was previously missed:

      # file-level, two functions
      def get_q(req): return req.GET["q"]   # returns taint
      def run(req):
          x = get_q(req)                    # x is now tainted
          cursor.execute("SELECT " + x)     # ← detected in v2

Output: TaintAnalysisResult with a 16-dim feature vector.

Limitations (intentional — keep it fast):
  - No aliasing analysis (a = b; sink(b) -- not followed)
  - No container flows (d["k"] = tainted; sink(d) -- missed)
  - Max 2 inter-procedural hops to avoid combinatorial explosion
  - No type inference
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Taint sources
# ---------------------------------------------------------------------------

_TAINT_SOURCE_ATTRS = {
    "GET", "POST", "FILES", "COOKIES", "META", "headers", "data", "json",
    "form", "args", "values", "query_string", "params", "path_params",
    "query_params", "body",
}

_TAINT_SOURCE_CALLS = {
    "input", "raw_input", "getenv", "environ", "recv", "recvfrom",
    "readline", "read", "readlines",
}

_TAINT_SOURCE_NAMES = {
    "request", "req", "query", "user_input", "user_data", "raw_input",
    "argv", "sys_argv", "environ", "env",
}

_TAINT_ATTR_CHAINS = {
    ("request", "GET"), ("request", "POST"), ("request", "data"),
    ("request", "json"), ("request", "form"), ("request", "args"),
    ("request", "values"), ("request", "cookies"), ("request", "headers"),
    ("req", "GET"), ("req", "POST"),
}

# ---------------------------------------------------------------------------
# Taint sinks
# ---------------------------------------------------------------------------

_SINK_CALLS = {
    "execute", "executemany", "executescript", "raw", "RawSQL", "extra",
    "system", "popen", "Popen", "call", "run", "check_call", "check_output",
    "spawn", "spawnl", "spawnlp",
    "eval", "exec", "compile", "__import__", "execfile",
    "loads", "load", "unsafe_load", "Unpickler",
    "open",
    "render_string", "Template",
    "search", "query",
}

# ---------------------------------------------------------------------------
# Sanitizer strengths: graded trust instead of binary (v2)
#
# Evidence from audit: replace("'", "''") is NOT safe for all databases.
# We grade sanitizers to allow risk = source_strength * (1 - san_strength).
# ---------------------------------------------------------------------------

_SANITIZER_STRENGTHS: dict[str, float] = {
    # Parameterised SQL (DB handles all escaping)
    "parameterize":   1.00,
    "mogrify":        1.00,
    "literal":        1.00,
    "adapt":          1.00,
    # ORM safe bindings
    "Q":              0.95,
    "F":              0.95,
    # HTML escaping
    "bleach_clean":   0.92,
    "html_escape":    0.88,
    "escape_html":    0.88,
    "strip_tags":     0.75,
    # Shell quoting
    "shlex_quote":    0.90,
    # General escaping (context-dependent — not always safe)
    "quote":          0.70,
    "escape":         0.65,
    "sanitize":       0.60,
    "clean":          0.58,
    "cleanse":        0.55,
    "validate":       0.45,
    "check":          0.35,
    "is_valid":       0.35,
    # WEAK sanitizers (produce false negatives if treated as safe)
    "encode":         0.20,
    "replace":        0.15,  # replace("'", "''") is DB-dialect-specific, NOT universal
    "strip":          0.10,
    "lstrip":         0.10,
    "rstrip":         0.10,
}

# Parameterized SQL pattern (safe regardless of what's in the query string)
_PARAMETERIZED_SQL_RE = re.compile(
    r"\.execute\s*\(\s*['\"].*?['\"],\s*[(\[]"
)
_STRING_CONCAT_RE = re.compile(
    r'["\'].*?["\'].*?\+\s*\w|["\'].*?%[sd].*?["\'].*?%'
)
_FSTRING_RE = re.compile(r'f["\'].*?\{[^}]+\}.*?["\']')


# ---------------------------------------------------------------------------
# Argument-level semantics (v2)
# ---------------------------------------------------------------------------

def classify_sink_argument(arg: ast.expr) -> str:
    """
    Classify the structural type of an argument passed to a sink call.

    This is the argument-level semantic feature identified in the audit:
    two calls can look identical at the function-name level but differ
    completely in risk based on how the query string is constructed.

    Returns one of:
        "constant"       -- string/number literal; safe
        "parameterized"  -- variable in params tuple, not in query; safe
        "fstring"        -- f"...{var}..." construct; dangerous
        "concatenation"  -- "..." + var; dangerous
        "format_call"    -- "...%s..." % var or "...".format(...); dangerous
        "variable"       -- bare variable reference; depends on taint
        "call_result"    -- result of another function call; unknown
        "unknown"        -- anything else
    """
    if isinstance(arg, ast.Constant):
        return "constant"

    if isinstance(arg, ast.JoinedStr):
        # f-string: check if any value slot is non-trivial
        for val in arg.values:
            if isinstance(val, ast.FormattedValue):
                return "fstring"
        return "constant"   # f-string with only constants

    if isinstance(arg, ast.BinOp):
        if isinstance(arg.op, ast.Add):
            return "concatenation"
        if isinstance(arg.op, ast.Mod):
            return "format_call"

    if isinstance(arg, ast.Call):
        # "...".format(...) pattern
        if (isinstance(arg.func, ast.Attribute)
                and arg.func.attr in ("format", "format_map")):
            return "format_call"
        return "call_result"

    if isinstance(arg, ast.Name):
        return "variable"

    if isinstance(arg, ast.Subscript):
        return "variable"   # dictionary/list access

    return "unknown"


# Argument type risk scores (used in composite risk calculation)
_ARGUMENT_TYPE_RISK: dict[str, float] = {
    "constant":      0.00,
    "parameterized": 0.05,
    "variable":      0.50,   # depends on taint tracking
    "call_result":   0.40,
    "fstring":       0.90,
    "concatenation": 0.85,
    "format_call":   0.80,
    "unknown":       0.30,
}


# ---------------------------------------------------------------------------
# Inter-procedural call graph (v2): 1-2 hop propagation
# ---------------------------------------------------------------------------

class CallGraph:
    """
    Shallow intra-module call graph for inter-procedural taint propagation.

    For each function in the module, records:
      - Does it return tainted data? (returns_taint)
      - What functions does it call? (callees)

    Limited to 2 hops to prevent combinatorial blowup:
      hop 0: function directly accesses taint source
      hop 1: function calls a hop-0 function and returns it
      hop 2: (max) function calls a hop-1 function

    Example that this catches (v1 missed):
        def get_q(req): return req.GET["q"]   # hop 0
        def process(req): return get_q(req)   # hop 1 -- returns_taint=True
        def run():
            x = process(request)              # x is tainted
            cursor.execute("SELECT " + x)     # detected!
    """

    def __init__(self, source: str, max_hops: int = 2):
        self._max_hops = max_hops
        self._direct_taint: set[str] = set()     # functions that directly access taint
        self._callees: dict[str, set[str]] = {}  # func -> set of called function names
        self.returns_taint: dict[str, bool] = {}

        try:
            tree = ast.parse(source)
            self._collect_functions(tree)
            self._propagate()
        except SyntaxError:
            pass

    def is_taint_returning(self, func_name: str) -> bool:
        return self.returns_taint.get(func_name, False)

    def _collect_functions(self, tree: ast.Module) -> None:
        for func_node in ast.walk(tree):
            if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            name = func_node.name
            self._callees[name] = set()

            for node in ast.walk(func_node):
                # Track what this function calls
                if isinstance(node, ast.Call):
                    callee = self._extract_call_name(node)
                    if callee:
                        self._callees[name].add(callee)

                # Check if any return value is a taint source
                if isinstance(node, ast.Return) and node.value:
                    if self._expr_is_direct_source(node.value):
                        self._direct_taint.add(name)

    def _propagate(self) -> None:
        """Propagate taint-returning status through the call graph (BFS)."""
        # Start: all functions that directly access a taint source
        frontier = set(self._direct_taint)
        self.returns_taint = {f: True for f in frontier}

        for _hop in range(self._max_hops):
            next_frontier: set[str] = set()
            for func, callees in self._callees.items():
                if func in self.returns_taint:
                    continue
                # Does this function call a taint-returning function and return it?
                if any(c in self.returns_taint for c in callees):
                    # Heuristic: if it calls a taint-returning function, assume
                    # it *may* return taint. Conservative but prevents missed paths.
                    self.returns_taint[func] = True
                    next_frontier.add(func)
            if not next_frontier:
                break

    @staticmethod
    def _extract_call_name(node: ast.Call) -> Optional[str]:
        """Extract the name of a called function (simple cases only)."""
        func = node.func
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return None

    @staticmethod
    def _expr_is_direct_source(node: ast.expr) -> bool:
        """Check if expr directly accesses a known taint source."""
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute) and sub.attr in _TAINT_SOURCE_ATTRS:
                return True
            if isinstance(sub, ast.Name) and sub.id in _TAINT_SOURCE_NAMES:
                return True
            if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name)
                    and sub.func.id in _TAINT_SOURCE_CALLS):
                return True
        return False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TaintPath:
    """One discovered source-to-sink taint path within a function."""
    source_var:         str            # variable/call that carries taint
    sink_call:          str            # dangerous function receiving tainted data
    sanitizer_strength: float = 0.0   # [0,1]; 0 = no sanitizer, 1 = fully safe
    argument_type:      str = "unknown"  # classify_sink_argument() result
    is_direct:          bool = False   # source passed directly (no intermediary var)
    is_interprocedural: bool = False   # taint came via a function call (CallGraph)
    line_approx:        int = 0

    @property
    def is_sanitized(self) -> bool:
        """Backward-compatible boolean: True if sanitizer_strength > 0.5."""
        return self.sanitizer_strength > 0.5

    @property
    def risk_contribution(self) -> float:
        """
        Risk contribution of this path in [0, 1].

        risk = argument_type_risk * (1 - sanitizer_strength)

        A high-risk argument type (fstring) with a strong sanitizer (parameterize)
        still yields low risk. A low-risk argument type (constant) with no sanitizer
        yields zero risk.
        """
        arg_risk = _ARGUMENT_TYPE_RISK.get(self.argument_type, 0.3)
        return round(arg_risk * max(0.0, 1.0 - self.sanitizer_strength), 4)


@dataclass
class TaintAnalysisResult:
    """
    Full taint analysis result for one function or file.

    Feature vector (16-dim):
        [0]  has_taint_source
        [1]  has_taint_sink
        [2]  direct_flows              -- source directly to sink
        [3]  unsanitized_paths         -- paths with sanitizer_strength < 0.5
        [4]  sanitized_paths           -- paths with sanitizer_strength >= 0.5
        [5]  string_concat_to_sink     -- "SELECT " + var pattern
        [6]  fstring_to_sink           -- f"SELECT {var}" pattern
        [7]  parameterized_sql         -- execute("...", (params,)) pattern
        [8]  n_tainted_vars
        [9]  n_sinks_reached
        [10] sanitizer_coverage        -- sanitized / (sanitized + unsanitized)
        [11] risk_score                -- composite 0-1 risk score
        [12] max_sanitizer_strength    -- strongest sanitizer seen (v2)
        [13] branch_taint_fraction     -- fraction of branches with tainted path (v2)
        [14] interprocedural_paths     -- paths crossing function boundaries (v2)
        [15] argument_type_risk        -- mean argument-level risk (v2)
    """
    has_taint_source:      bool = False
    has_taint_sink:        bool = False
    paths:                 list[TaintPath] = field(default_factory=list)
    direct_flows:          int = 0
    unsanitized_paths:     int = 0
    sanitized_paths:       int = 0
    string_concat_to_sink: int = 0
    fstring_to_sink:       int = 0
    parameterized_sql:     int = 0
    n_tainted_vars:        int = 0
    n_sinks_reached:       int = 0
    branch_taint_fraction: float = 0.0   # v2: path sensitivity
    interprocedural_paths: int = 0       # v2: inter-procedural count
    _branch_total:         int = field(default=0, repr=False)
    _branch_tainted:       int = field(default=0, repr=False)

    @property
    def sanitizer_coverage(self) -> float:
        total = self.sanitized_paths + self.unsanitized_paths
        return float(self.sanitized_paths / total) if total > 0 else 0.0

    @property
    def max_sanitizer_strength(self) -> float:
        if not self.paths:
            return 0.0
        return max(p.sanitizer_strength for p in self.paths)

    @property
    def argument_type_risk(self) -> float:
        """Mean argument-level risk across all paths (v2)."""
        if not self.paths:
            return 0.0
        return round(sum(p.risk_contribution for p in self.paths) / len(self.paths), 4)

    @property
    def risk_score(self) -> float:
        """
        Composite risk in [0, 1].

        v2: uses graded sanitizer strength and argument-level semantics:
            risk = max(path.risk_contribution for unsanitized paths)

        Also uses branch_taint_fraction: if only some branches are tainted,
        risk is scaled down proportionally.
        """
        if not self.has_taint_source or not self.has_taint_sink:
            return 0.10 if self.has_taint_sink else 0.0

        # String-level signals (fast, high precision)
        score = 0.0
        if self.string_concat_to_sink > 0:
            score = max(score, 0.85)
        if self.fstring_to_sink > 0:
            score = max(score, 0.80)
        if self.direct_flows > 0:
            score = max(score, 0.90)

        # Per-path risk (argument type × sanitizer strength)
        for path in self.paths:
            score = max(score, path.risk_contribution)

        # Parameterized SQL reduces risk
        if self.parameterized_sql > 0:
            score = max(0.0, score - 0.20)

        # Path sensitivity: scale by fraction of branches that are tainted
        if self.branch_taint_fraction > 0 and score > 0:
            score *= (0.5 + 0.5 * self.branch_taint_fraction)

        return round(min(1.0, max(0.0, score)), 4)

    @property
    def vector(self) -> list[float]:
        """16-dim causal feature vector."""
        return [
            float(self.has_taint_source),
            float(self.has_taint_sink),
            float(self.direct_flows),
            float(self.unsanitized_paths),
            float(self.sanitized_paths),
            float(self.string_concat_to_sink),
            float(self.fstring_to_sink),
            float(self.parameterized_sql),
            float(self.n_tainted_vars),
            float(self.n_sinks_reached),
            float(self.sanitizer_coverage),
            float(self.risk_score),
            float(self.max_sanitizer_strength),       # v2
            float(self.branch_taint_fraction),        # v2
            float(self.interprocedural_paths),        # v2
            float(self.argument_type_risk),           # v2
        ]


# ---------------------------------------------------------------------------
# AST visitor — intra-function taint propagation with path sensitivity (v2)
# ---------------------------------------------------------------------------

class _TaintVisitor(ast.NodeVisitor):
    """
    Statement-ordered AST taint propagation with:
      - Graded sanitizer strength (v2)
      - Argument-level semantics per path (v2)
      - Branch-aware taint tracking (v2 path sensitivity)
      - Inter-procedural source detection via CallGraph (v2)
    """

    def __init__(self, call_graph: Optional[CallGraph] = None):
        self._tainted: set[str] = set()
        self._sanitized: dict[str, float] = {}  # var -> sanitizer_strength
        self._call_graph = call_graph or CallGraph.__new__(CallGraph)
        self.result = TaintAnalysisResult()

    # ── Assignment tracking ───────────────────────────────────────────────────

    def visit_Assign(self, node: ast.Assign):
        rhs = node.value
        source = self._detect_source(rhs)

        if source:
            self.result.has_taint_source = True
            for tgt in node.targets:
                for name in self._lhs_names(tgt):
                    self._tainted.add(name)
                    self._sanitized.pop(name, None)

        elif self._is_sanitizer_call(rhs):
            strength = self._sanitizer_call_strength(rhs)
            for tgt in node.targets:
                for name in self._lhs_names(tgt):
                    if strength >= 0.5:
                        self._tainted.discard(name)
                        self._sanitized[name] = strength
                    else:
                        # Weak sanitizer: still tainted but strength recorded
                        self._sanitized[name] = strength

        elif self._expr_is_tainted(rhs):
            for tgt in node.targets:
                for name in self._lhs_names(tgt):
                    self._tainted.add(name)
                    self._sanitized.pop(name, None)

        self.result.n_tainted_vars = len(self._tainted)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign):
        if self._expr_is_tainted(node.value):
            if isinstance(node.target, ast.Name):
                self._tainted.add(node.target.id)
        self.generic_visit(node)

    # ── Branch-aware taint (v2 path sensitivity) ──────────────────────────────

    def visit_If(self, node: ast.If):
        """Track taint separately for each branch, record fraction reaching sinks."""
        saved_tainted = set(self._tainted)
        saved_sanitized = dict(self._sanitized)
        saved_paths_len = len(self.result.paths)

        # Visit 'if' body
        for stmt in node.body:
            self.visit(stmt)
        if_paths = len(self.result.paths) - saved_paths_len
        self.result._branch_total += 1
        if if_paths > 0:
            self.result._branch_tainted += 1

        # Reset taint state for 'else' body
        self._tainted = set(saved_tainted)
        self._sanitized = dict(saved_sanitized)
        else_paths_start = len(self.result.paths)

        if node.orelse:
            for stmt in node.orelse:
                self.visit(stmt)
            else_paths = len(self.result.paths) - else_paths_start
            self.result._branch_total += 1
            if else_paths > 0:
                self.result._branch_tainted += 1

        # Merge taint state (conservative: union of both branches)
        self._tainted = self._tainted | saved_tainted

        # Update branch_taint_fraction
        if self.result._branch_total > 0:
            self.result.branch_taint_fraction = round(
                self.result._branch_tainted / self.result._branch_total, 4
            )

    # ── Sink detection ────────────────────────────────────────────────────────

    def visit_Call(self, node: ast.Call):
        sink_name = self._detect_sink(node)
        if not sink_name:
            self.generic_visit(node)
            return

        self.result.has_taint_sink = True
        all_args = list(node.args) + [kw.value for kw in node.keywords]
        tainted_args = [a for a in all_args if self._expr_is_tainted(a)]

        if not tainted_args:
            self.generic_visit(node)
            return

        # Determine sanitizer strength for this path
        san_strength = 0.0
        for a in tainted_args:
            # Check if argument itself went through a sanitizer
            if isinstance(a, ast.Name) and a.id in self._sanitized:
                san_strength = max(san_strength, self._sanitized[a.id])
            if self._is_sanitizer_call(a):
                san_strength = max(san_strength, self._sanitizer_call_strength(a))

        # Argument-level semantics (first tainted arg determines type)
        arg_type = classify_sink_argument(tainted_args[0]) if tainted_args else "unknown"

        is_direct = any(
            isinstance(a, ast.Name) and a.id in self._tainted
            for a in tainted_args
        )
        is_interprocedural = any(
            isinstance(a, ast.Name) and a.id not in self._tainted
            # (came from call graph propagation into _tainted earlier)
            for a in tainted_args
        )

        path = TaintPath(
            source_var=next(
                (a.id for a in tainted_args if isinstance(a, ast.Name)),
                "unknown",
            ),
            sink_call=sink_name,
            sanitizer_strength=san_strength,
            argument_type=arg_type,
            is_direct=is_direct,
            is_interprocedural=is_interprocedural,
            line_approx=getattr(node, "lineno", 0),
        )
        self.result.paths.append(path)

        if is_direct:
            self.result.direct_flows += 1
        if is_interprocedural:
            self.result.interprocedural_paths += 1
        if san_strength >= 0.5:
            self.result.sanitized_paths += 1
        else:
            self.result.unsanitized_paths += 1

        self.result.n_sinks_reached = len({p.sink_call for p in self.result.paths})
        self.generic_visit(node)

    # ── Source detection ──────────────────────────────────────────────────────

    def _detect_source(self, node: ast.expr) -> Optional[str]:
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                obj = func.value
                if isinstance(obj, ast.Attribute) and obj.attr in _TAINT_SOURCE_ATTRS:
                    return f"attr:{obj.attr}"
                if isinstance(obj, ast.Name) and obj.id in _TAINT_SOURCE_NAMES:
                    return f"call:{func.attr}"
                # Inter-procedural (v2): call to taint-returning function
                if isinstance(obj, ast.Name) and hasattr(self._call_graph, 'returns_taint'):
                    if self._call_graph.is_taint_returning(func.attr):
                        return f"interprocedural:{func.attr}"
            if isinstance(func, ast.Name):
                if func.id in _TAINT_SOURCE_CALLS:
                    return f"call:{func.id}"
                # Inter-procedural (v2)
                if hasattr(self._call_graph, 'returns_taint'):
                    if self._call_graph.is_taint_returning(func.id):
                        return f"interprocedural:{func.id}"

        if isinstance(node, ast.Subscript):
            val = node.value
            if isinstance(val, ast.Attribute) and val.attr in _TAINT_SOURCE_ATTRS:
                return f"subscript:{val.attr}"
            if isinstance(val, ast.Name) and val.id in _TAINT_SOURCE_NAMES:
                return f"subscript:{val.id}"

        if isinstance(node, ast.Name) and node.id in _TAINT_SOURCE_NAMES:
            return f"name:{node.id}"

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            attr_chain = self._attr_chain(node.func.value)
            if attr_chain in ("os.environ", "environ"):
                return "environ"

        return None

    def _detect_sink(self, node: ast.Call) -> Optional[str]:
        func = node.func
        if isinstance(func, ast.Name) and func.id in _SINK_CALLS:
            return func.id
        if isinstance(func, ast.Attribute) and func.attr in _SINK_CALLS:
            return func.attr
        return None

    def _is_sanitizer_call(self, node: ast.expr) -> bool:
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        if isinstance(func, ast.Name):
            return func.id in _SANITIZER_STRENGTHS
        if isinstance(func, ast.Attribute):
            return func.attr in _SANITIZER_STRENGTHS
        return False

    def _sanitizer_call_strength(self, node: ast.expr) -> float:
        """Return the graded trust strength of a sanitizer call."""
        if not isinstance(node, ast.Call):
            return 0.0
        func = node.func
        if isinstance(func, ast.Name):
            return _SANITIZER_STRENGTHS.get(func.id, 0.0)
        if isinstance(func, ast.Attribute):
            return _SANITIZER_STRENGTHS.get(func.attr, 0.0)
        return 0.0

    def _expr_is_tainted(self, node: ast.expr) -> bool:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and sub.id in self._tainted:
                return True
        return False

    @staticmethod
    def _lhs_names(target: ast.expr) -> list[str]:
        if isinstance(target, ast.Name):
            return [target.id]
        if isinstance(target, (ast.Tuple, ast.List)):
            return [e.id for e in target.elts if isinstance(e, ast.Name)]
        return []

    @staticmethod
    def _attr_chain(node: ast.expr) -> str:
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts))


# ---------------------------------------------------------------------------
# String-level pattern checks
# ---------------------------------------------------------------------------

def _string_level_checks(source: str) -> dict:
    return {
        "parameterized_sql": int(bool(_PARAMETERIZED_SQL_RE.search(source))),
        "string_concat":     int(bool(_STRING_CONCAT_RE.search(source))),
        "fstring":           int(bool(_FSTRING_RE.search(source))),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_taint_paths(source: str) -> TaintAnalysisResult:
    """
    Analyse *source* for taint paths, including inter-procedural propagation.

    v2: builds a CallGraph first, then visits each function with that context.

    Args:
        source: Python source code (module or function level).

    Returns:
        TaintAnalysisResult with 16-dim feature vector.
    """
    combined = TaintAnalysisResult()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return combined

    # Build call graph for inter-procedural propagation (v2)
    call_graph = CallGraph(source)

    # String-level signals
    str_checks = _string_level_checks(source)
    combined.parameterized_sql    = str_checks["parameterized_sql"]
    combined.string_concat_to_sink = str_checks["string_concat"]
    combined.fstring_to_sink       = str_checks["fstring"]

    func_nodes = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    if not func_nodes:
        visitor = _TaintVisitor(call_graph=call_graph)
        visitor.visit(tree)
        _merge(combined, visitor.result)
    else:
        for func in func_nodes:
            visitor = _TaintVisitor(call_graph=call_graph)
            visitor.visit(func)
            _merge(combined, visitor.result)

    # Reconcile counts
    if combined.parameterized_sql > 0:
        combined.sanitized_paths = max(combined.sanitized_paths, 1)

    combined.n_tainted_vars  = len({p.source_var for p in combined.paths if p.source_var != "unknown"})
    combined.n_sinks_reached = len({p.sink_call for p in combined.paths})

    return combined


def _merge(dest: TaintAnalysisResult, src: TaintAnalysisResult) -> None:
    dest.has_taint_source      = dest.has_taint_source or src.has_taint_source
    dest.has_taint_sink        = dest.has_taint_sink   or src.has_taint_sink
    dest.paths                 += src.paths
    dest.direct_flows          += src.direct_flows
    dest.unsanitized_paths     += src.unsanitized_paths
    dest.sanitized_paths       += src.sanitized_paths
    dest.interprocedural_paths += src.interprocedural_paths
    dest._branch_total         += src._branch_total
    dest._branch_tainted       += src._branch_tainted
    if dest._branch_total > 0:
        dest.branch_taint_fraction = round(dest._branch_tainted / dest._branch_total, 4)


def augment_security_features(base_features: "np.ndarray", source: str) -> "np.ndarray":
    """
    Concatenate 16-dim taint features (v2) to an existing security feature vector.
    Replaces the broken count-based taint_source_count and injection_api_count.
    """
    import numpy as np
    result = extract_taint_paths(source)
    return np.concatenate([base_features, np.array(result.vector, dtype=np.float32)])


# ---------------------------------------------------------------------------
# Feature names (for model interpretability)
# ---------------------------------------------------------------------------

TAINT_FEATURE_NAMES = [
    "taint_has_source",
    "taint_has_sink",
    "taint_direct_flows",
    "taint_unsanitized_paths",
    "taint_sanitized_paths",
    "taint_string_concat_to_sink",
    "taint_fstring_to_sink",
    "taint_parameterized_sql",
    "taint_n_tainted_vars",
    "taint_n_sinks_reached",
    "taint_sanitizer_coverage",
    "taint_risk_score",
    "taint_max_sanitizer_strength",    # v2
    "taint_branch_taint_fraction",     # v2
    "taint_interprocedural_paths",     # v2
    "taint_argument_type_risk",        # v2
]
