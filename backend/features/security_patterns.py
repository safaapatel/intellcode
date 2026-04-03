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


# ---------------------------------------------------------------------------
# JavaScript / TypeScript security scanner (regex-based)
# ---------------------------------------------------------------------------

# (pattern, vuln_type, severity, title, description, confidence, cwe)
_JS_SECURITY_RULES: list[tuple] = [
    # XSS
    (r'\.innerHTML\s*=', "xss", "critical",
     "DOM-based XSS via innerHTML",
     "Assigning to innerHTML with unsanitised input allows XSS. Use textContent or DOMPurify.",
     0.85, "CWE-79"),
    (r'\.outerHTML\s*=', "xss", "critical",
     "DOM-based XSS via outerHTML",
     "Assigning to outerHTML with unsanitised input allows XSS.", 0.85, "CWE-79"),
    (r'document\.write\s*\(', "xss", "high",
     "XSS risk via document.write()",
     "document.write() with user-controlled input enables XSS attacks.", 0.80, "CWE-79"),
    (r'\beval\s*\(', "code_injection", "critical",
     "Use of eval()",
     "eval() on untrusted input allows arbitrary code execution.", 0.90, "CWE-95"),
    (r'Function\s*\(', "code_injection", "high",
     "Dynamic Function() constructor",
     "new Function() with user input is equivalent to eval().", 0.80, "CWE-95"),
    (r'setTimeout\s*\(\s*["\']', "code_injection", "high",
     "setTimeout with string argument",
     "Passing a string to setTimeout is equivalent to eval().", 0.85, "CWE-95"),
    (r'setInterval\s*\(\s*["\']', "code_injection", "high",
     "setInterval with string argument",
     "Passing a string to setInterval is equivalent to eval().", 0.85, "CWE-95"),
    # SQL injection
    (r'(?i)(query|execute)\s*\(\s*["\'].*\+', "sql_injection", "critical",
     "SQL Injection via string concatenation",
     "SQL query built from string concatenation. Use parameterised queries.", 0.88, "CWE-89"),
    # Hardcoded credentials (shared with Python patterns — already in _HARDCODED_SECRET_PATTERNS)
    # Weak crypto
    (r'Math\.random\s*\(\)', "weak_crypto", "medium",
     "Insecure random (Math.random)",
     "Math.random() is not cryptographically secure. Use crypto.getRandomValues().",
     0.80, "CWE-338"),
    (r'createCipher\s*\(', "weak_crypto", "high",
     "Deprecated crypto.createCipher()",
     "crypto.createCipher() is deprecated; use createCipheriv() with a random IV.",
     0.85, "CWE-327"),
    (r'(?i)(md5|sha1)\s*\(', "weak_crypto", "high",
     "Weak hash algorithm",
     "MD5/SHA-1 are cryptographically broken. Use SHA-256 or stronger.", 0.75, "CWE-327"),
    # Prototype pollution
    (r'__proto__\s*[=\[]', "prototype_pollution", "high",
     "Prototype pollution risk",
     "Assigning to __proto__ can pollute the Object prototype.", 0.80, "CWE-1321"),
    # Path traversal
    (r'(?:readFile|readFileSync|createReadStream)\s*\([^)]*\+', "path_traversal", "high",
     "Potential path traversal",
     "File path built from concatenation may allow directory traversal.", 0.70, "CWE-22"),
    # Open redirect
    (r'res\.redirect\s*\([^)]*req\.(query|body|params)', "open_redirect", "medium",
     "Potential open redirect",
     "Redirect target derived from request input. Validate the URL.", 0.70, "CWE-601"),
    # NoSQL injection (MongoDB)
    (r'\$where\s*:', "nosql_injection", "high",
     "MongoDB $where injection risk",
     "$where evaluates JavaScript; avoid with user-supplied values.", 0.80, "CWE-943"),
]

# ---------------------------------------------------------------------------
# Java security scanner (regex-based)
# ---------------------------------------------------------------------------

_JAVA_SECURITY_RULES: list[tuple] = [
    # SQL injection
    (r'Statement.*execute\s*\(\s*"[^"]*"\s*\+', "sql_injection", "critical",
     "SQL Injection via string concatenation",
     "Build SQL with PreparedStatement and bind parameters instead.", 0.88, "CWE-89"),
    (r'createStatement\s*\(\)', "sql_injection", "medium",
     "Use of createStatement (prefer PreparedStatement)",
     "createStatement() with concatenated SQL is vulnerable to injection.", 0.65, "CWE-89"),
    # Command injection
    (r'Runtime\.getRuntime\s*\(\)\s*\.exec\s*\(', "command_injection", "critical",
     "OS command execution via Runtime.exec()",
     "Runtime.exec() with user-controlled input allows command injection.", 0.90, "CWE-78"),
    (r'ProcessBuilder\s*\(', "command_injection", "high",
     "ProcessBuilder command execution",
     "Ensure no user-controlled values are passed to ProcessBuilder.", 0.70, "CWE-78"),
    # Weak crypto
    (r'MessageDigest\.getInstance\s*\(\s*"(?:MD5|SHA-1|SHA1)"', "weak_crypto", "high",
     "Weak hash algorithm (MD5/SHA-1)",
     "MD5 and SHA-1 are broken. Use SHA-256 or stronger.", 0.90, "CWE-327"),
    (r'Cipher\.getInstance\s*\(\s*"(?:DES|RC4|RC2)', "weak_crypto", "high",
     "Weak cipher (DES/RC4/RC2)",
     "DES and RC4 are broken. Use AES-GCM.", 0.90, "CWE-327"),
    (r'new\s+Random\s*\(\)', "weak_crypto", "medium",
     "Insecure random (java.util.Random)",
     "java.util.Random is not cryptographically secure. Use SecureRandom.", 0.80, "CWE-338"),
    # Deserialization
    (r'ObjectInputStream\s*\(', "insecure_deserialization", "critical",
     "Unsafe Java deserialization",
     "ObjectInputStream on untrusted data allows remote code execution.", 0.85, "CWE-502"),
    # Path traversal
    (r'new\s+File\s*\([^)]*request\.(getParameter|getAttribute)', "path_traversal", "high",
     "Potential path traversal via request parameter",
     "File path derived from request input. Validate and canonicalize paths.", 0.80, "CWE-22"),
    # XXE
    (r'DocumentBuilderFactory\.newInstance\s*\(\)', "xxe", "medium",
     "XML parser XXE risk (DocumentBuilderFactory)",
     "Disable external entity processing: setFeature(XMLConstants.FEATURE_SECURE_PROCESSING, true).",
     0.70, "CWE-611"),
    (r'SAXParserFactory\.newInstance\s*\(\)', "xxe", "medium",
     "XML parser XXE risk (SAXParserFactory)",
     "Disable external DTD and entity processing on the SAX parser.", 0.70, "CWE-611"),
    # Hardcoded passwords
    (r'(?i)(password|passwd|secret)\s*=\s*"[^"]{4,}"', "hardcoded_credential", "high",
     "Hardcoded credential",
     "Credentials should not be stored in source code.", 0.80, "CWE-798"),
]


class _RegexSecurityScanner:
    """Generic regex-based security scanner for JS/TS/Java."""

    def __init__(self, source: str, rules: list[tuple]):
        self._lines = source.splitlines()
        self._rules = rules
        self._findings: list[SecurityFinding] = []

    def scan(self) -> SecurityScanResult:
        # Language-specific rules
        for lineno, line in enumerate(self._lines, start=1):
            s = line.strip()
            if not s or s.startswith("//") or s.startswith("*"):
                continue
            for pattern, vuln_type, severity, title, desc, confidence, cwe in self._rules:
                if re.search(pattern, line):
                    self._findings.append(SecurityFinding(
                        vuln_type=vuln_type,
                        severity=severity,
                        title=title,
                        description=desc,
                        lineno=lineno,
                        snippet=line.strip()[:200],
                        confidence=confidence,
                        cwe=cwe,
                    ))

        # Shared: hardcoded secrets (language-agnostic regex)
        for lineno, line in enumerate(self._lines, start=1):
            for pattern, label in _HARDCODED_SECRET_PATTERNS:
                if re.search(pattern, line):
                    self._findings.append(SecurityFinding(
                        vuln_type="hardcoded_credential",
                        severity="high",
                        title=label,
                        description="Sensitive value should not be hardcoded in source code.",
                        lineno=lineno,
                        snippet=line.strip()[:200],
                        confidence=0.75,
                        cwe="CWE-798",
                    ))

        # Deduplicate by (lineno, vuln_type)
        seen: set[tuple] = set()
        unique: list[SecurityFinding] = []
        for f in self._findings:
            key = (f.lineno, f.vuln_type)
            if key not in seen:
                seen.add(key)
                unique.append(f)

        return SecurityScanResult(
            findings=sorted(unique, key=lambda f: (
                {"critical": 0, "high": 1, "medium": 2, "low": 3}[f.severity], f.lineno,
            )),
            scanned_lines=len(self._lines),
        )


def scan_security_patterns_for_language(source: str, language: str) -> SecurityScanResult:
    """Scan source in the given language. Dispatches to the appropriate scanner."""
    lang = language.lower()
    if lang in ("javascript", "typescript"):
        return _RegexSecurityScanner(source, _JS_SECURITY_RULES).scan()
    if lang == "java":
        return _RegexSecurityScanner(source, _JAVA_SECURITY_RULES).scan()
    # Default: Python
    return scan_security_patterns(source)
