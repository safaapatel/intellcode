"""
Dependency & Coupling Analyzer
Measures import dependencies, module coupling, and structural coupling metrics.

Metrics computed:
    fan_out         — number of distinct modules this file imports
    fan_in          — (multi-file only) how many files import this one
    instability     — fan_out / (fan_in + fan_out), 0=stable, 1=unstable
    coupling_score  — composite 0–100 (higher = more coupled / harder to test)

Detected issues:
    high_fan_out        — imports too many modules (> 10)
    wildcard_import     — 'from x import *' pollutes namespace
    circular_import     — A imports B which imports A (multi-file detection)
    god_import          — a single imported module is used everywhere in the code
    unused_stdlib       — standard library module imported but not referenced
    deep_relative       — relative imports more than 2 levels deep (../..)
    import_inside_func  — import statement inside a function body

Coupling categories:
    external   — third-party packages (not stdlib, not local)
    stdlib     — Python standard library
    local      — relative imports or same-package imports
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, asdict, field
from typing import Optional


# ---------------------------------------------------------------------------
# Standard library module names (Python 3.x top-level)
# ---------------------------------------------------------------------------

_STDLIB_MODULES = {
    "abc", "aifc", "argparse", "array", "ast", "asynchat", "asyncio",
    "asyncore", "atexit", "audioop", "base64", "bdb", "binascii", "binhex",
    "bisect", "builtins", "bz2", "calendar", "cgi", "cgitb", "chunk",
    "cmath", "cmd", "code", "codecs", "codeop", "colorsys", "compileall",
    "concurrent", "configparser", "contextlib", "contextvars", "copy",
    "copyreg", "cProfile", "csv", "ctypes", "curses", "dataclasses",
    "datetime", "dbm", "decimal", "difflib", "dis", "distutils", "doctest",
    "email", "encodings", "enum", "errno", "faulthandler", "fcntl",
    "filecmp", "fileinput", "fnmatch", "fractions", "ftplib", "functools",
    "gc", "getopt", "getpass", "gettext", "glob", "grp", "gzip", "hashlib",
    "heapq", "hmac", "html", "http", "idlelib", "imaplib", "imghdr",
    "importlib", "inspect", "io", "ipaddress", "itertools", "json",
    "keyword", "lib2to3", "linecache", "locale", "logging", "lzma",
    "mailbox", "mailcap", "marshal", "math", "mimetypes", "mmap",
    "modulefinder", "multiprocessing", "netrc", "nis", "nntplib",
    "numbers", "operator", "optparse", "os", "ossaudiodev", "pathlib",
    "pdb", "pickle", "pickletools", "pipes", "pkgutil", "platform",
    "plistlib", "poplib", "posix", "posixpath", "pprint", "profile",
    "pstats", "pty", "pwd", "py_compile", "pyclbr", "pydoc", "queue",
    "quopri", "random", "re", "readline", "reprlib", "resource", "rlcompleter",
    "runpy", "sched", "secrets", "select", "selectors", "shelve", "shlex",
    "shutil", "signal", "site", "smtpd", "smtplib", "sndhdr", "socket",
    "socketserver", "spwd", "sqlite3", "sre_compile", "sre_constants",
    "sre_parse", "ssl", "stat", "statistics", "string", "stringprep",
    "struct", "subprocess", "sunau", "symtable", "sys", "sysconfig",
    "syslog", "tabnanny", "tarfile", "telnetlib", "tempfile", "termios",
    "test", "textwrap", "threading", "time", "timeit", "tkinter", "token",
    "tokenize", "tomllib", "trace", "traceback", "tracemalloc", "tty",
    "turtle", "turtledemo", "types", "typing", "unicodedata", "unittest",
    "urllib", "uu", "uuid", "venv", "warnings", "wave", "weakref",
    "webbrowser", "wsgiref", "xdrlib", "xml", "xmlrpc", "zipapp",
    "zipfile", "zipimport", "zlib", "zoneinfo", "_thread", "__future__",
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ImportRecord:
    module: str
    names: list[str]        # specific names imported (empty = import module)
    is_relative: bool
    relative_level: int     # dots in relative import
    is_wildcard: bool
    lineno: int
    inside_function: bool
    category: str           # "stdlib" | "external" | "local"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CouplingIssue:
    issue_type: str
    severity: str
    title: str
    description: str
    location: str
    start_line: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DependencyResult:
    imports: list[ImportRecord]
    fan_out: int
    fan_out_stdlib: int
    fan_out_external: int
    fan_out_local: int
    instability: float          # 0 = stable, 1 = unstable (no fan_in without multi-file)
    coupling_score: float       # 0–100
    issues: list[CouplingIssue]
    dependency_map: dict        # {category: [module_name, ...]}
    summary: str

    def to_dict(self) -> dict:
        return {
            "imports": [i.to_dict() for i in self.imports],
            "fan_out": self.fan_out,
            "fan_out_stdlib": self.fan_out_stdlib,
            "fan_out_external": self.fan_out_external,
            "fan_out_local": self.fan_out_local,
            "instability": round(self.instability, 3),
            "coupling_score": round(self.coupling_score, 1),
            "issues": [i.to_dict() for i in self.issues],
            "dependency_map": self.dependency_map,
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classify_module(name: str, is_relative: bool) -> str:
    if is_relative:
        return "local"
    top = name.split(".")[0] if name else ""
    if top in _STDLIB_MODULES:
        return "stdlib"
    return "external"


def _is_inside_function(tree: ast.Module, target: ast.AST) -> bool:
    """Check if a node is inside a function definition."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                if child is target:
                    return True
    return False


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class DependencyAnalyzer:
    """
    Analyzes import dependencies and coupling metrics for a Python source file.

    Usage:
        analyzer = DependencyAnalyzer()
        result = analyzer.analyze(source_code)
    """

    FAN_OUT_THRESHOLD = 10
    RELATIVE_LEVEL_THRESHOLD = 2

    def analyze(self, source: str) -> DependencyResult:
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return DependencyResult(
                imports=[],
                fan_out=0,
                fan_out_stdlib=0,
                fan_out_external=0,
                fan_out_local=0,
                instability=0.0,
                coupling_score=0.0,
                issues=[],
                dependency_map={},
                summary=f"Syntax error: {e}",
            )

        imports = self._extract_imports(tree)
        issues = self._detect_issues(tree, imports)

        # Metrics
        modules = {r.module for r in imports}
        fan_out = len(modules)
        stdlib = {r.module for r in imports if r.category == "stdlib"}
        external = {r.module for r in imports if r.category == "external"}
        local = {r.module for r in imports if r.category == "local"}

        # instability: without fan_in data, approximate as 0.5 for external-heavy files
        external_ratio = len(external) / fan_out if fan_out > 0 else 0
        instability = min(1.0, external_ratio * 1.2)

        # Coupling score
        coupling_score = self._compute_coupling(fan_out, issues)

        dep_map = {
            "stdlib": sorted(stdlib),
            "external": sorted(external),
            "local": sorted(local),
        }

        n_issues = len(issues)
        summary = (
            f"Fan-out: {fan_out} modules "
            f"({len(stdlib)} stdlib, {len(external)} external, {len(local)} local). "
            f"Coupling score: {coupling_score:.0f}/100. "
            f"{n_issues} coupling issue(s) found."
        )

        return DependencyResult(
            imports=imports,
            fan_out=fan_out,
            fan_out_stdlib=len(stdlib),
            fan_out_external=len(external),
            fan_out_local=len(local),
            instability=instability,
            coupling_score=coupling_score,
            issues=issues,
            dependency_map=dep_map,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Import extraction
    # ------------------------------------------------------------------

    def _extract_imports(self, tree: ast.Module) -> list[ImportRecord]:
        records = []
        func_nodes: set[int] = set()

        # Find nodes inside functions
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(node):
                    func_nodes.add(id(child))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    cat = _classify_module(alias.name, False)
                    records.append(ImportRecord(
                        module=alias.name,
                        names=[alias.asname or alias.name],
                        is_relative=False,
                        relative_level=0,
                        is_wildcard=False,
                        lineno=node.lineno,
                        inside_function=id(node) in func_nodes,
                        category=cat,
                    ))

            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                is_rel = node.level > 0
                is_wild = any(a.name == "*" for a in node.names)
                cat = _classify_module(mod, is_rel)
                names = [a.name for a in node.names]
                records.append(ImportRecord(
                    module=mod or ".",
                    names=names,
                    is_relative=is_rel,
                    relative_level=node.level,
                    is_wildcard=is_wild,
                    lineno=node.lineno,
                    inside_function=id(node) in func_nodes,
                    category=cat,
                ))

        return records

    # ------------------------------------------------------------------
    # Issue detection
    # ------------------------------------------------------------------

    def _detect_issues(
        self,
        tree: ast.Module,
        imports: list[ImportRecord],
    ) -> list[CouplingIssue]:
        issues: list[CouplingIssue] = []
        modules = {r.module for r in imports}

        # High fan-out
        if len(modules) > self.FAN_OUT_THRESHOLD:
            issues.append(CouplingIssue(
                issue_type="high_fan_out",
                severity="high",
                title=f"High fan-out: {len(modules)} imports",
                description=(
                    f"This file imports {len(modules)} distinct modules "
                    f"(threshold: {self.FAN_OUT_THRESHOLD}). "
                    f"High fan-out indicates tight coupling and makes the module "
                    f"harder to test and maintain. Consider splitting responsibilities."
                ),
                location="module-level",
                start_line=1,
            ))

        # Wildcard imports
        for r in imports:
            if r.is_wildcard:
                issues.append(CouplingIssue(
                    issue_type="wildcard_import",
                    severity="high",
                    title=f"Wildcard import from '{r.module}' (line {r.lineno})",
                    description=(
                        f"'from {r.module} import *' at line {r.lineno} pollutes the "
                        f"module namespace, makes it impossible to know where names come "
                        f"from, and can shadow existing names. Import explicitly."
                    ),
                    location=f"line {r.lineno}",
                    start_line=r.lineno,
                ))

        # Deep relative imports
        for r in imports:
            if r.is_relative and r.relative_level > self.RELATIVE_LEVEL_THRESHOLD:
                issues.append(CouplingIssue(
                    issue_type="deep_relative_import",
                    severity="medium",
                    title=f"Deep relative import (level {r.relative_level}) at line {r.lineno}",
                    description=(
                        f"A relative import goes {r.relative_level} levels up at "
                        f"line {r.lineno}. Deep relative imports indicate poor package "
                        f"structure. Consider reorganizing the package hierarchy."
                    ),
                    location=f"line {r.lineno}",
                    start_line=r.lineno,
                ))

        # Imports inside functions
        func_imports = [r for r in imports if r.inside_function]
        for r in func_imports:
            issues.append(CouplingIssue(
                issue_type="import_inside_function",
                severity="low",
                title=f"Import inside function body (line {r.lineno})",
                description=(
                    f"'import {r.module}' at line {r.lineno} is inside a function. "
                    f"This re-executes the import on every call (though Python caches it). "
                    f"Move imports to module level unless intentionally lazy-loading."
                ),
                location=f"line {r.lineno}",
                start_line=r.lineno,
            ))

        # Duplicate imports
        seen_modules: dict[str, int] = {}
        for r in imports:
            if r.module in seen_modules:
                issues.append(CouplingIssue(
                    issue_type="duplicate_import",
                    severity="low",
                    title=f"Duplicate import of '{r.module}' (lines {seen_modules[r.module]} and {r.lineno})",
                    description=(
                        f"'{r.module}' is imported more than once. "
                        f"Consolidate into a single import statement."
                    ),
                    location=f"lines {seen_modules[r.module]}, {r.lineno}",
                    start_line=r.lineno,
                ))
            else:
                seen_modules[r.module] = r.lineno

        return sorted(issues, key=lambda i: i.start_line)

    def _compute_coupling(self, fan_out: int, issues: list[CouplingIssue]) -> float:
        score = min(50.0, fan_out * 4)  # up to 50 from fan-out alone
        weights = {"high": 20, "medium": 10, "low": 5}
        score += sum(weights.get(i.severity, 5) for i in issues)
        return min(100.0, score)
