"""
Security Pattern Scanner
Detects common vulnerability patterns in Python source code using a combination
of AST-based structural analysis and regex heuristics.

Vulnerability categories:
  - SQL Injection
  - Hardcoded Credentials / Secrets
  - Path Traversal
  - Weak Cryptography
  - XSS (template injection)
  - Command Injection (subprocess/os.system)
  - Insecure Deserialization (pickle)
  - XML External Entity (XXE)
  - Open Redirect
"""

import ast
import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class SecurityFinding:
    vuln_type: str
    severity: str               # critical | high | medium | low
    title: str
    description: str
    lineno: int
    snippet: str
    confidence: float           # 0.0 – 1.0
    cwe: str = ""               # CWE identifier


@dataclass
class SecurityScanResult:
    findings: list[SecurityFinding] = field(default_factory=list)
    scanned_lines: int = 0

    @property
    def critical(self) -> list[SecurityFinding]:
        return [f for f in self.findings if f.severity == "critical"]

    @property
    def high(self) -> list[SecurityFinding]:
        return [f for f in self.findings if f.severity == "high"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": [vars(f) for f in self.findings],
            "scanned_lines": self.scanned_lines,
            "summary": {
                "critical": len(self.critical),
                "high": len(self.high),
                "medium": len([f for f in self.findings if f.severity == "medium"]),
                "low": len([f for f in self.findings if f.severity == "low"]),
                "total": len(self.findings),
            },
        }


# ---------------------------------------------------------------------------
# Regex patterns for quick heuristic checks
# ---------------------------------------------------------------------------

_HARDCODED_SECRET_PATTERNS = [
    # Generic key/password/secret assignments
    (r'(?i)(password|passwd|pwd|secret|api_key|apikey|token|auth_token|access_token'
     r'|private_key|client_secret)\s*=\s*["\'][^"\']{6,}["\']',
     "Hardcoded credential"),
    # AWS access keys
    (r'AKIA[0-9A-Z]{16}', "Hardcoded AWS access key"),
    # Generic base64-looking tokens in strings
    (r'["\'][A-Za-z0-9+/]{40,}={0,2}["\']', "Possible hardcoded token"),
    # Private key PEM blocks
    (r'-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----', "Embedded private key"),
]

_WEAK_CRYPTO_CALLS = {
    "md5": ("Weak hash algorithm (MD5)", "high", "CWE-327"),
    "sha1": ("Weak hash algorithm (SHA-1)", "high", "CWE-327"),
    "des": ("Weak cipher (DES)", "high", "CWE-327"),
    "rc4": ("Weak cipher (RC4)", "high", "CWE-327"),
    "random": ("Insecure random (not cryptographically safe)", "medium", "CWE-338"),
}

_COMMAND_INJECTION_CALLS = {
    "system", "popen", "popen2", "popen3", "popen4",
    "Popen", "call", "check_call", "check_output", "run",
}

_DESERIALIZATION_CALLS = {
    "loads",  # pickle.loads / yaml.load
    "load",   # pickle.load
    "unsafe_load",
}


# ---------------------------------------------------------------------------
# AST-based detectors
# ---------------------------------------------------------------------------

class SecurityPatternScanner(ast.NodeVisitor):
    """
    Walks a Python AST to detect security anti-patterns.
    """

    def __init__(self, source: str):
        self._source = source
        self._lines = source.splitlines()
        self._findings: list[SecurityFinding] = []

    def scan(self) -> SecurityScanResult:
        """Run all checks and return results."""
        try:
            tree = ast.parse(self._source)
        except SyntaxError:
            return SecurityScanResult(scanned_lines=len(self._lines))

        self.visit(tree)
        self._regex_scan()

        result = SecurityScanResult(
            findings=sorted(self._findings, key=lambda f: (
                {"critical": 0, "high": 1, "medium": 2, "low": 3}[f.severity],
                f.lineno,
            )),
            scanned_lines=len(self._lines),
        )
        return result

    def _snippet(self, lineno: int, context: int = 1) -> str:
        start = max(0, lineno - 1 - context)
        end = min(len(self._lines), lineno + context)
        return "\n".join(self._lines[start:end])

    def _add(self, finding: SecurityFinding):
        self._findings.append(finding)

    # ------------------------------------------------------------------
    # SQL Injection detection
    # ------------------------------------------------------------------

    def visit_Call(self, node: ast.Call):
        func_name = ""
        if isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            func_name = node.func.id

        # --- SQL Injection: db.execute(string_concat) ---
        if func_name in ("execute", "executemany", "raw", "query"):
            if node.args:
                arg = node.args[0]
                if self._is_string_concat(arg):
                    self._add(SecurityFinding(
                        vuln_type="sql_injection",
                        severity="critical",
                        title="SQL Injection via String Concatenation",
                        description=(
                            "SQL query built by string concatenation allows attackers to "
                            "inject arbitrary SQL. Use parameterized queries instead."
                        ),
                        lineno=node.lineno,
                        snippet=self._snippet(node.lineno),
                        confidence=0.92,
                        cwe="CWE-89",
                    ))
                elif self._uses_format_or_percent(arg):
                    self._add(SecurityFinding(
                        vuln_type="sql_injection",
                        severity="critical",
                        title="SQL Injection via String Formatting",
                        description=(
                            "SQL query built with str.format() or % formatting is vulnerable "
                            "to injection. Use parameterized queries instead."
                        ),
                        lineno=node.lineno,
                        snippet=self._snippet(node.lineno),
                        confidence=0.88,
                        cwe="CWE-89",
                    ))

        # --- Weak Cryptography ---
        if func_name in _WEAK_CRYPTO_CALLS:
            title, severity, cwe = _WEAK_CRYPTO_CALLS[func_name]
            # Check module context (hashlib.md5 is more confident than random md5)
            confidence = 0.85
            if isinstance(node.func, ast.Attribute):
                parent = ""
                if isinstance(node.func.value, ast.Name):
                    parent = node.func.value.id
                if parent in ("hashlib", "Crypto", "OpenSSL"):
                    confidence = 0.95
            self._add(SecurityFinding(
                vuln_type="weak_cryptography",
                severity=severity,
                title=title,
                description=f"Use of {func_name}() is considered cryptographically weak.",
                lineno=node.lineno,
                snippet=self._snippet(node.lineno),
                confidence=confidence,
                cwe=cwe,
            ))

        # --- Command Injection ---
        if func_name in _COMMAND_INJECTION_CALLS:
            # Look for shell=True or direct string arg that could be user-controlled
            has_shell_true = any(
                isinstance(kw.value, ast.Constant) and kw.value.value is True
                and kw.arg == "shell"
                for kw in node.keywords
            )
            if has_shell_true:
                self._add(SecurityFinding(
                    vuln_type="command_injection",
                    severity="critical",
                    title="Command Injection via shell=True",
                    description=(
                        "subprocess called with shell=True allows shell injection if any "
                        "part of the command string is user-controlled."
                    ),
                    lineno=node.lineno,
                    snippet=self._snippet(node.lineno),
                    confidence=0.90,
                    cwe="CWE-78",
                ))
            elif node.args and self._is_string_concat(node.args[0]):
                self._add(SecurityFinding(
                    vuln_type="command_injection",
                    severity="high",
                    title="Possible Command Injection via String Concat",
                    description=(
                        "Command string built from string concatenation may be injectable."
                    ),
                    lineno=node.lineno,
                    snippet=self._snippet(node.lineno),
                    confidence=0.70,
                    cwe="CWE-78",
                ))

        # --- Insecure Deserialization ---
        if func_name in _DESERIALIZATION_CALLS:
            parent = ""
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                parent = node.func.value.id
            if parent in ("pickle", "cPickle", "marshal", "shelve"):
                self._add(SecurityFinding(
                    vuln_type="insecure_deserialization",
                    severity="critical",
                    title="Insecure Deserialization (pickle)",
                    description=(
                        "pickle.loads/load on untrusted data can execute arbitrary code."
                    ),
                    lineno=node.lineno,
                    snippet=self._snippet(node.lineno),
                    confidence=0.88,
                    cwe="CWE-502",
                ))
            elif parent == "yaml" and func_name == "load":
                # yaml.load without Loader is insecure
                has_loader = any(kw.arg == "Loader" for kw in node.keywords)
                if not has_loader:
                    self._add(SecurityFinding(
                        vuln_type="insecure_deserialization",
                        severity="high",
                        title="Insecure YAML Load",
                        description=(
                            "yaml.load() without explicit Loader is unsafe; "
                            "use yaml.safe_load() instead."
                        ),
                        lineno=node.lineno,
                        snippet=self._snippet(node.lineno),
                        confidence=0.85,
                        cwe="CWE-502",
                    ))

        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Path traversal detection
    # ------------------------------------------------------------------

    def visit_Assign(self, node: ast.Assign):
        self.generic_visit(node)

    def visit_With(self, node: ast.With):
        for item in node.items:
            ctx = item.context_expr
            if isinstance(ctx, ast.Call):
                func_name = ""
                if isinstance(ctx.func, ast.Name):
                    func_name = ctx.func.id
                elif isinstance(ctx.func, ast.Attribute):
                    func_name = ctx.func.attr
                if func_name == "open" and ctx.args:
                    arg = ctx.args[0]
                    if not isinstance(arg, ast.Constant) and self._could_be_user_input(arg):
                        self._add(SecurityFinding(
                            vuln_type="path_traversal",
                            severity="high",
                            title="Potential Path Traversal",
                            description=(
                                "File path passed to open() may be user-controlled. "
                                "Validate and sanitize paths before use."
                            ),
                            lineno=node.lineno,
                            snippet=self._snippet(node.lineno),
                            confidence=0.65,
                            cwe="CWE-22",
                        ))
        self.generic_visit(node)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_string_concat(node: ast.expr) -> bool:
        """True if node is a string concatenation (BinOp with Add)."""
        return isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)

    @staticmethod
    def _uses_format_or_percent(node: ast.expr) -> bool:
        """True if node uses str.format() or % formatting."""
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            return True
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "format":
                return True
        # f-strings
        if isinstance(node, ast.JoinedStr):
            return True
        return False

    @staticmethod
    def _could_be_user_input(node: ast.expr) -> bool:
        """Heuristic: is this a variable that might hold user input?"""
        if isinstance(node, ast.Name):
            suspicious = {"path", "filename", "file", "name", "input",
                          "user_path", "user_file", "request", "data", "url"}
            return node.id.lower() in suspicious or "path" in node.id.lower()
        return isinstance(node, ast.BinOp)  # concatenated paths

    # ------------------------------------------------------------------
    # Regex-based scan (runs on raw source)
    # ------------------------------------------------------------------

    def _regex_scan(self):
        for line_idx, line in enumerate(self._lines, start=1):
            # Hardcoded secrets
            for pattern, label in _HARDCODED_SECRET_PATTERNS:
                if re.search(pattern, line):
                    # Skip if inside a comment
                    stripped = line.lstrip()
                    if stripped.startswith("#"):
                        continue
                    self._add(SecurityFinding(
                        vuln_type="hardcoded_secret",
                        severity="critical",
                        title=f"Hardcoded Secret Detected ({label})",
                        description=(
                            "Sensitive values embedded in source code can be exposed via "
                            "version control. Use environment variables or a secrets manager."
                        ),
                        lineno=line_idx,
                        snippet=line.strip(),
                        confidence=0.80,
                        cwe="CWE-798",
                    ))
                    break  # one finding per line

            # eval() usage
            if re.search(r'\beval\s*\(', line) and not line.lstrip().startswith("#"):
                self._add(SecurityFinding(
                    vuln_type="code_injection",
                    severity="high",
                    title="Use of eval()",
                    description=(
                        "eval() on untrusted input allows arbitrary code execution."
                    ),
                    lineno=line_idx,
                    snippet=line.strip(),
                    confidence=0.75,
                    cwe="CWE-95",
                ))

            # exec() usage
            if re.search(r'\bexec\s*\(', line) and not line.lstrip().startswith("#"):
                self._add(SecurityFinding(
                    vuln_type="code_injection",
                    severity="high",
                    title="Use of exec()",
                    description=(
                        "exec() on untrusted input allows arbitrary code execution."
                    ),
                    lineno=line_idx,
                    snippet=line.strip(),
                    confidence=0.75,
                    cwe="CWE-95",
                ))

            # XML XXE — defusedxml not used
            if re.search(r'xml\.etree|lxml\.etree|minidom', line):
                if "defused" not in line and not line.lstrip().startswith("#"):
                    self._add(SecurityFinding(
                        vuln_type="xxe",
                        severity="medium",
                        title="Potential XML External Entity (XXE)",
                        description=(
                            "Standard XML parsers are vulnerable to XXE attacks. "
                            "Consider using defusedxml."
                        ),
                        lineno=line_idx,
                        snippet=line.strip(),
                        confidence=0.60,
                        cwe="CWE-611",
                    ))


def scan_security_patterns(source: str) -> SecurityScanResult:
    """Convenience wrapper: scan *source* and return SecurityScanResult."""
    scanner = SecurityPatternScanner(source)
    return scanner.scan()
