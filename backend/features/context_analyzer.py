"""
context_analyzer.py
Local, AST-based taint and context analysis for security findings.

No external API calls — purely structural/semantic analysis of Python source code.

Key exports:
  enrich_findings(code, findings) -> list[dict]
    Adds to each finding:
      - argument_name   : actual variable/expression passed to the dangerous call
      - taint_source    : "request_input" | "argv" | "user_input" | "constant" |
                          "internal" | "function_param" | "unknown"
      - taint_path      : human-readable trace, e.g. "request.args['q'] -> expr -> eval(expr)"
      - user_controlled : bool
      - false_positive  : bool  (True when argument is a literal or known-safe internal)
      - exploitability  : int 0-10
      - context_sentence: one-sentence description referencing real variable names
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TaintResult:
    argument_name: str = ""
    taint_source: str = "unknown"          # see module docstring
    taint_path: str = ""
    user_controlled: bool = False
    false_positive: bool = False
    exploitability: int = 5
    context_sentence: str = ""


# ---------------------------------------------------------------------------
# Source patterns that indicate user-controlled data
# ---------------------------------------------------------------------------

_USER_CONTROLLED_PATTERNS = [
    # Flask / Django / FastAPI request objects
    r"request\.(args|form|json|data|values|files|get_json|params)",
    r"request\.get\(",
    # FastAPI path/query parameters — typically function params named after routes
    r"\bBody\(|\bQuery\(|\bPath\(|\bForm\(",
    # WSGI environ
    r"environ\[|os\.environ\[",
    # CLI arguments
    r"sys\.argv",
    r"argparse|click\.",
    # Standard input
    r"\binput\s*\(",
    r"stdin\.read",
    # Socket / network
    r"\.recv\(|\.read\(\)",
    r"socket\.",
    # HTTP libraries (server-side handlers)
    r"self\.(get|post|put|delete|patch)\(",
]

_USER_SOURCE_LABELS = {
    r"request\.(args|form|json|data|values|files|get_json|params)": "request_input",
    r"request\.get\(": "request_input",
    r"\bBody\(|\bQuery\(|\bPath\(|\bForm\(": "request_input",
    r"environ\[|os\.environ\[": "environment",
    r"sys\.argv": "argv",
    r"argparse|click\.": "argv",
    r"\binput\s*\(": "user_input",
    r"stdin\.read": "user_input",
    r"\.recv\(|\.read\(\)": "network_input",
    r"socket\.": "network_input",
    r"self\.(get|post|put|delete|patch)\(": "request_input",
}

_SAFE_INTERNAL_PATTERNS = [
    # Config / settings files
    r"config\[|settings\[|CONFIG\[|SETTINGS\[",
    r"\.cfg|\.ini|\.yaml|\.json",
    # Known model/checkpoint files
    r"checkpoints?/|model\.|weights\.",
    r"open\(['\"][^'\"]*\.(pkl|json|yaml|cfg|ini|txt)['\"]",
]


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def _unparse(node: ast.AST) -> str:
    """Best-effort unparse a node to a string."""
    try:
        return ast.unparse(node)
    except Exception:
        return "<expr>"


def _get_line(code: str, lineno: int) -> str:
    lines = code.splitlines()
    if 0 < lineno <= len(lines):
        return lines[lineno - 1].strip()
    return ""


def _find_call_arg(code: str, lineno: int) -> str:
    """
    Parse the line at lineno and extract the first argument to the function call.
    Returns the argument as a string expression.
    """
    line = _get_line(code, lineno)
    try:
        tree = ast.parse(line, mode="eval")
    except SyntaxError:
        # Try wrapping as a statement
        try:
            tree = ast.parse(line)
        except SyntaxError:
            return ""

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.args:
            return _unparse(node.args[0])
    return ""


def _is_string_literal(expr: str) -> bool:
    """Return True if expr looks like a string or numeric literal."""
    stripped = expr.strip()
    if stripped.startswith(("'", '"', '"""', "'''")):
        return True
    if stripped.startswith(("f'", 'f"')):
        # f-string — may interpolate user data, not a FP
        return False
    try:
        ast.literal_eval(stripped)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Backward variable tracer
# ---------------------------------------------------------------------------

def _trace_variable(code: str, var_name: str, from_line: int, max_depth: int = 8) -> list[str]:
    """
    Walk backwards from from_line looking for assignments to var_name.
    Returns the list of right-hand-side expressions found (most recent first).
    """
    if not var_name or var_name in ("None", "True", "False"):
        return []

    lines = code.splitlines()
    results: list[str] = []

    # Simple patterns: var = ..., var, other = ..., var: type = ...
    patterns = [
        re.compile(r"^\s*" + re.escape(var_name) + r"\s*(?::\s*\S+)?\s*=\s*(.+)$"),
        re.compile(r"^\s*(?:\w+\s*,\s*)*" + re.escape(var_name) + r"\s*(?:,\s*\w+)*\s*=\s*(.+)$"),
    ]

    search_start = max(0, from_line - 1)
    for i in range(search_start, max(0, search_start - 80), -1):
        line = lines[i] if i < len(lines) else ""
        for pat in patterns:
            m = pat.match(line)
            if m:
                rhs = m.group(1).strip()
                results.append(rhs)
                if len(results) >= max_depth:
                    return results
                # Recurse one level: if rhs is itself a plain variable name
                if re.match(r"^\w+$", rhs) and rhs != var_name:
                    deeper = _trace_variable(code, rhs, i, max_depth - len(results))
                    results.extend(deeper)
                break

    return results


def _classify_source(traces: list[str], code_context: str) -> tuple[str, str]:
    """
    Given a list of traced RHS expressions and surrounding code context,
    return (taint_source, human_label).
    """
    all_text = " ".join(traces) + " " + code_context

    for pattern, label in _USER_SOURCE_LABELS.items():
        if re.search(pattern, all_text):
            # Extract the specific expression for a nice label
            m = re.search(pattern, all_text)
            snippet = m.group(0) if m else label
            return label, snippet

    # Heuristic: function parameter names that suggest user input
    user_param_re = re.compile(
        r"\b(user_\w+|client_\w+|payload|body|data|query|expr|formula|cmd|command"
        r"|user_input|raw_\w+|untrusted)\b"
    )
    if user_param_re.search(all_text):
        m = user_param_re.search(all_text)
        return "function_param", m.group(0)

    # Check for safe internal patterns
    for pat in _SAFE_INTERNAL_PATTERNS:
        if re.search(pat, all_text, re.IGNORECASE):
            return "internal", "internal source"

    return "unknown", "unknown source"


# ---------------------------------------------------------------------------
# Per-finding enrichment
# ---------------------------------------------------------------------------

_VULN_DANGER_FUNCS = {
    "code_injection": ["eval", "exec", "compile"],
    "eval_injection": ["eval", "exec", "compile"],
    "command_injection": ["os.system", "subprocess", "popen", "check_output"],
    "insecure_deserialization": ["pickle.loads", "pickle.load", "marshal.loads",
                                 "yaml.load", "shelve"],
    "path_traversal": ["open", "os.path.join", "os.listdir", "shutil"],
    "sql_injection": ["execute", "cursor.execute", "raw", "query"],
    "ssrf": ["requests.get", "requests.post", "urllib.request", "httpx.get"],
    "weak_cryptography": ["md5", "sha1", "hashlib.md5", "hashlib.sha1"],
    "hardcoded_secret": [],
    "hardcoded_credential": [],
}

_EXPLOITABILITY_BASE = {
    "critical": 9,
    "high": 7,
    "medium": 4,
    "low": 2,
}

_SOURCE_BOOST = {
    "request_input": 2,
    "user_input": 2,
    "network_input": 2,
    "argv": 1,
    "environment": 0,
    "function_param": 1,
    "internal": -4,
    "constant": -6,
    "unknown": 0,
}

_CONTEXT_TEMPLATES = {
    "code_injection": (
        "{func}() is called with `{arg}` which originates from {source_label} — "
        "an attacker can supply arbitrary Python code for immediate remote code execution"
    ),
    "eval_injection": (
        "{func}() is called with `{arg}` which originates from {source_label} — "
        "an attacker can supply arbitrary Python code for immediate remote code execution"
    ),
    "command_injection": (
        "{func}() passes `{arg}` directly to the shell from {source_label} — "
        "an attacker can inject shell metacharacters to execute arbitrary OS commands"
    ),
    "insecure_deserialization": (
        "`{arg}` is deserialized with pickle from {source_label} — "
        "pickle's __reduce__ protocol allows arbitrary code execution during load"
    ),
    "path_traversal": (
        "open() receives `{arg}` from {source_label} without path normalization — "
        "an attacker can use ../ sequences to read or write arbitrary files on the server"
    ),
    "sql_injection": (
        "SQL query string is constructed using `{arg}` from {source_label} — "
        "string concatenation into SQL allows an attacker to alter query logic"
    ),
    "ssrf": (
        "HTTP request target `{arg}` is supplied by {source_label} — "
        "the server may be directed to fetch internal resources or bypass firewalls"
    ),
    "weak_cryptography": (
        "`{arg}` is hashed with a broken algorithm — "
        "MD5/SHA-1 are collision-vulnerable and unsuitable for security-critical hashing"
    ),
    "hardcoded_secret": (
        "Credential `{arg}` is embedded in source code — "
        "visible in version control history even after removal"
    ),
    "hardcoded_credential": (
        "Credential `{arg}` is embedded in source code — "
        "visible in version control history even after removal"
    ),
}

_FALSE_POSITIVE_LABELS = {
    "constant": "argument is a string or numeric literal — not user-controlled",
    "internal": "data comes from internal config or model file, not user input",
}

_SOURCE_HUMAN = {
    "request_input": "HTTP request parameters",
    "user_input": "standard input / console",
    "network_input": "network socket",
    "argv": "command-line arguments",
    "environment": "environment variables",
    "function_param": "a function parameter",
    "internal": "internal source",
    "unknown": "an untraced variable",
    "constant": "a hardcoded literal",
}


def _enrich_single(code: str, finding: dict) -> dict:
    f = dict(finding)
    lineno = f.get("lineno", 0)
    vuln_type = f.get("vuln_type", "")
    severity = f.get("severity", "medium")

    line = _get_line(code, lineno)
    arg = _find_call_arg(code, lineno) or f.get("snippet", "").strip()

    # --- False positive: literal argument ---
    if arg and _is_string_literal(arg):
        f["false_positive"] = True
        f["user_controlled"] = False
        f["exploitability"] = 1
        f["argument_name"] = arg
        f["taint_source"] = "constant"
        f["taint_path"] = f"{arg} -> (literal constant)"
        f["context_sentence"] = (
            f"Line {lineno}: argument is the literal {arg!r} — "
            "not user-controlled; low risk unless the literal itself is malformed"
        )
        return f

    # --- Backward trace ---
    # For hardcoded secrets, the "arg" is the value being assigned, not a call arg
    if vuln_type in ("hardcoded_secret", "hardcoded_credential"):
        # Extract variable name from the line instead
        m = re.match(r"^\s*([\w.]+)\s*=", line)
        arg = m.group(1) if m else arg
        taint_source, source_snippet = "constant", "hardcoded value"
        traces: list[str] = []
    else:
        traces = _trace_variable(code, arg, lineno)
        # Also search ±10 lines for user-input patterns
        start = max(0, lineno - 10)
        end = min(len(code.splitlines()), lineno + 2)
        local_context = "\n".join(code.splitlines()[start:end])
        taint_source, source_snippet = _classify_source(traces, local_context + " " + line)

    # --- Exploitability score ---
    base = _EXPLOITABILITY_BASE.get(severity, 5)
    # Hardcoded secrets are inherently high-risk regardless of taint source
    # (their "constant" source is the bug, not a safety signal)
    _no_boost_types = {"hardcoded_secret", "hardcoded_credential", "weak_cryptography", "debug_enabled"}
    boost = 0 if vuln_type in _no_boost_types else _SOURCE_BOOST.get(taint_source, 0)
    exploitability = max(1, min(10, base + boost))

    user_controlled = taint_source in (
        "request_input", "user_input", "network_input", "argv", "function_param"
    )
    false_positive = taint_source in ("constant", "internal") and vuln_type not in (
        "hardcoded_secret", "hardcoded_credential", "weak_cryptography", "debug_enabled"
    )

    # --- Taint path ---
    path_parts = [source_snippet or taint_source]
    for t in traces[:3]:
        path_parts.append(t[:60])
    path_parts.append(f"{arg} -> vulnerable call at L{lineno}")
    taint_path = " -> ".join(path_parts)

    # --- Context sentence ---
    template = _CONTEXT_TEMPLATES.get(vuln_type, "")
    if false_positive:
        context_sentence = (
            f"Line {lineno}: {_FALSE_POSITIVE_LABELS.get(taint_source, 'likely safe — verify manually')}"
        )
    elif template:
        source_label = _SOURCE_HUMAN.get(taint_source, taint_source)
        context_sentence = template.format(
            func=_guess_func_name(line, vuln_type),
            arg=arg or "?",
            source_label=source_label,
        )
    else:
        source_label = _SOURCE_HUMAN.get(taint_source, taint_source)
        context_sentence = (
            f"Line {lineno}: `{arg}` reaches a potentially dangerous call "
            f"from {source_label}"
        )

    f.update({
        "argument_name": arg,
        "taint_source": taint_source,
        "taint_path": taint_path,
        "user_controlled": user_controlled,
        "false_positive": false_positive,
        "exploitability": exploitability,
        "context_sentence": context_sentence,
    })
    return f


def _guess_func_name(line: str, vuln_type: str) -> str:
    for func in _VULN_DANGER_FUNCS.get(vuln_type, []):
        if func.split(".")[-1] in line:
            return func
    return vuln_type.replace("_", " ")


# ---------------------------------------------------------------------------
# Root cause grouping
# ---------------------------------------------------------------------------

_ROOT_CAUSE_LABELS = {
    "path_traversal": (
        "Unsanitized file path handling",
        "File operations accept user-controlled paths without os.path.realpath() validation. "
        "A single path-sanitization utility applied at all {count} call sites would eliminate this class of bug."
    ),
    "code_injection": (
        "Dynamic code evaluation on user input",
        "eval()/exec() is called with data that traces back to {source}. "
        "Replace with a safe expression evaluator (ast.literal_eval or a sandboxed interpreter)."
    ),
    "eval_injection": (
        "Dynamic code evaluation on user input",
        "eval()/exec() is called with data that traces back to {source}. "
        "Replace with a safe expression evaluator (ast.literal_eval or a sandboxed interpreter)."
    ),
    "insecure_deserialization": (
        "Untrusted deserialization",
        "pickle.loads() can execute arbitrary code during deserialization. "
        "Use JSON or a schema-validated format for data from {source}."
    ),
    "sql_injection": (
        "SQL string concatenation",
        "Query strings are built by concatenating {source} data. "
        "Switch to parameterized queries / ORM methods throughout."
    ),
    "hardcoded_secret": (
        "Credentials in source code",
        "Secrets are embedded in the codebase and visible in version control history. "
        "Move to environment variables or a secrets manager."
    ),
    "hardcoded_credential": (
        "Credentials in source code",
        "Secrets are embedded in the codebase and visible in version control history. "
        "Move to environment variables or a secrets manager."
    ),
    "command_injection": (
        "Shell command injection",
        "Shell commands are constructed using {source} data. "
        "Use subprocess with shell=False and pass arguments as a list."
    ),
    "ssrf": (
        "Server-side request forgery",
        "HTTP requests are made to URLs controlled by {source}. "
        "Validate URLs against an allowlist before fetching."
    ),
    "weak_cryptography": (
        "Broken cryptographic algorithm",
        "MD5 or SHA-1 is used for security-sensitive hashing. "
        "Replace with bcrypt, argon2, or SHA-256+ depending on the use case."
    ),
}


def group_findings_by_root_cause(findings: list[dict]) -> list[dict]:
    """
    Collapse findings into root-cause groups.
    Returns a list of group dicts with:
      title, root_cause_description, severity, linenos, count,
      representative_finding, fix_block
    """
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)
    for f in findings:
        vt = f.get("vuln_type", "unknown")
        groups[vt].append(f)

    result = []
    for vuln_type, group in groups.items():
        # Skip false positives if they all are
        real = [f for f in group if not f.get("false_positive")]
        if not real:
            continue

        rep = sorted(real, key=lambda x: (
            {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(x.get("severity", "low"), 4),
            -x.get("exploitability", 5),
        ))[0]

        linenos = sorted({f.get("lineno", 0) for f in real if f.get("lineno")})
        sources = {f.get("taint_source", "unknown") for f in real}
        dominant_source = _SOURCE_HUMAN.get(
            next(iter(sources - {"unknown", "internal"}), "unknown"), "an untraced source"
        )

        label_title, label_desc_tmpl = _ROOT_CAUSE_LABELS.get(
            vuln_type, (vuln_type.replace("_", " ").title(), "Review all {count} occurrences.")
        )
        label_desc = label_desc_tmpl.format(count=len(real), source=dominant_source)

        result.append({
            "vuln_type": vuln_type,
            "title": label_title,
            "root_cause_description": label_desc,
            "severity": rep.get("severity", "medium"),
            "linenos": linenos,
            "count": len(real),
            "false_positive_count": len(group) - len(real),
            "representative_finding": rep,
            "user_controlled": any(f.get("user_controlled") for f in real),
            "max_exploitability": max(f.get("exploitability", 5) for f in real),
            "taint_sources": list(sources),
        })

    # Sort by exploitability desc
    result.sort(key=lambda g: -g["max_exploitability"])
    return result


# ---------------------------------------------------------------------------
# File-level inference
# ---------------------------------------------------------------------------

def infer_file_purpose(code: str, filename: str) -> str:
    """
    Infer what the file does from imports, class names, function names.
    Returns a one-sentence description.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return f"Python module `{filename}`"

    imports: list[str] = []
    classes: list[str] = []
    top_funcs: list[str] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module.split(".")[0])
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
        elif isinstance(node, ast.FunctionDef):
            top_funcs.append(node.name)

    # Heuristics
    imp_set = set(imports)
    if "flask" in imp_set or "fastapi" in imp_set or "django" in imp_set:
        layer = "web API"
    elif "sklearn" in imp_set or "torch" in imp_set or "tensorflow" in imp_set:
        layer = "ML model"
    elif "sqlalchemy" in imp_set or "psycopg2" in imp_set or "sqlite3" in imp_set:
        layer = "database layer"
    elif "socket" in imp_set or "asyncio" in imp_set:
        layer = "network service"
    elif "pickle" in imp_set or "joblib" in imp_set:
        layer = "model serialization module"
    else:
        layer = "Python module"

    class_str = f" containing {', '.join(classes[:3])}" if classes else ""
    return f"A {layer}{class_str} (`{filename}`)"


# ---------------------------------------------------------------------------
# Priority action plan
# ---------------------------------------------------------------------------

def build_action_plan(groups: list[dict]) -> list[str]:
    """Build a numbered, prioritized remediation plan from root-cause groups."""
    plan: list[str] = []
    n = 1

    for g in groups:
        vt = g["vuln_type"]
        linenos = g["linenos"]
        loc = (
            f"L{linenos[0]}" if len(linenos) == 1
            else f"L{linenos[0]}-L{linenos[-1]}"
        )
        severity = g["severity"].upper()
        user_ctrl = " (user-controlled input confirmed)" if g["user_controlled"] else ""

        if vt in ("code_injection", "eval_injection"):
            plan.append(
                f"{n}. **[{severity}] Replace eval()/exec() at {loc}**{user_ctrl} — "
                "use `ast.literal_eval()` for safe literal parsing, or a sandboxed evaluator "
                "for expression evaluation. This is a remote code execution risk."
            )
        elif vt == "insecure_deserialization":
            plan.append(
                f"{n}. **[{severity}] Remove pickle deserialization at {loc}**{user_ctrl} — "
                "switch to `json.loads()` with schema validation, or `msgpack` if binary format is required."
            )
        elif vt == "path_traversal":
            plan.append(
                f"{n}. **[{severity}] Add path sanitization at {loc} ({g['count']} locations)**{user_ctrl} — "
                "create a single `safe_path(base, user_path)` utility using "
                "`os.path.realpath()` + prefix check and call it at every file open site."
            )
        elif vt == "command_injection":
            plan.append(
                f"{n}. **[{severity}] Replace shell=True / os.system at {loc}**{user_ctrl} — "
                "use `subprocess.run([cmd, *args], shell=False)` with arguments as a list, never as a string."
            )
        elif vt in ("hardcoded_secret", "hardcoded_credential"):
            plan.append(
                f"{n}. **[{severity}] Remove hardcoded credentials at {loc}** — "
                "rotate the affected secret immediately, then load from `os.environ` or a secrets manager. "
                "Add a pre-commit hook (e.g. `detect-secrets`) to prevent recurrence."
            )
        elif vt == "sql_injection":
            plan.append(
                f"{n}. **[{severity}] Parameterize SQL queries at {loc}**{user_ctrl} — "
                "replace string concatenation with `cursor.execute(sql, (param,))` or ORM query builders."
            )
        elif vt == "ssrf":
            plan.append(
                f"{n}. **[{severity}] Validate outbound request URLs at {loc}**{user_ctrl} — "
                "enforce an allowlist of permitted hosts; reject requests to RFC-1918 and loopback ranges."
            )
        elif vt == "weak_cryptography":
            plan.append(
                f"{n}. **[{severity}] Replace broken hash algorithm at {loc}** — "
                "use `bcrypt` or `argon2` for passwords, `hashlib.sha256()` or better for integrity checks."
            )
        else:
            plan.append(
                f"{n}. **[{severity}] Remediate {g['title']} at {loc}**{user_ctrl}."
            )
        n += 1

    return plan


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enrich_findings(code: str, findings: list[dict]) -> list[dict]:
    """Add taint analysis fields to each finding. Safe — never raises."""
    enriched = []
    for f in findings:
        try:
            enriched.append(_enrich_single(code, f))
        except Exception:
            enriched.append(f)
    return enriched


def build_pr_narrative(
    code: str,
    filename: str,
    findings: list[dict],
    overall_score: Optional[int],
    analysis_summary: str = "",
) -> str:
    """
    Build a structured, context-rich PR review comment in GitHub Markdown.
    Pure Python — no external API calls.
    """
    if not findings:
        return ""

    enriched = enrich_findings(code, findings)
    groups = group_findings_by_root_cause(enriched)
    purpose = infer_file_purpose(code, filename)

    real_findings = [f for f in enriched if not f.get("false_positive")]
    fp_count = len(enriched) - len(real_findings)
    user_ctrl_count = sum(1 for f in real_findings if f.get("user_controlled"))

    lines: list[str] = []

    # ── Header ─────────────────────────────────────────────────────────────
    lines.append("## IntelliCode Security Review\n")
    lines.append(f"**File:** {purpose}\n")

    # ── Score explanation ──────────────────────────────────────────────────
    if overall_score is not None:
        crit = sum(1 for g in groups if g["severity"] == "critical")
        high = sum(1 for g in groups if g["severity"] == "high")
        score_reason_parts = []
        if crit:
            score_reason_parts.append(f"{crit} critical finding{'s' if crit > 1 else ''} (-{crit * 15} pts)")
        if high:
            score_reason_parts.append(f"{high} high-severity pattern{'s' if high > 1 else ''} (-{high * 8} pts)")
        if fp_count:
            score_reason_parts.append(f"{fp_count} finding{'s' if fp_count > 1 else ''} likely false positive (excluded)")
        score_reason = "; ".join(score_reason_parts) if score_reason_parts else "multiple quality signals"
        lines.append(f"**Score: {overall_score}/100** — {score_reason}\n")
    elif analysis_summary:
        # Pull score line from summary
        m = re.search(r"(\d+)/100", analysis_summary)
        if m:
            lines.append(f"**Overall score: {m.group(1)}/100**\n")

    # ── Risk summary ───────────────────────────────────────────────────────
    if user_ctrl_count:
        lines.append(
            f"> **{user_ctrl_count} of {len(real_findings)} findings involve user-controlled input** — "
            f"these are confirmed attack vectors, not theoretical risks.\n"
        )
    if fp_count:
        lines.append(
            f"> {fp_count} scanner finding{'s were' if fp_count > 1 else ' was'} "
            f"identified as likely false positives (constant arguments or internal data) and excluded below.\n"
        )

    # ── Grouped findings ───────────────────────────────────────────────────
    if groups:
        lines.append("\n---\n\n## Findings by Root Cause\n")

    for g in groups:
        rep = g["representative_finding"]
        linenos = g["linenos"]
        loc = (
            f"L{linenos[0]}" if len(linenos) == 1
            else f"L{linenos[0]}–L{linenos[-1]}, {g['count']} locations"
        )
        sev_badge = {
            "critical": "[CRITICAL]",
            "high": "[HIGH]",
            "medium": "[MEDIUM]",
            "low": "[LOW]",
        }.get(g["severity"], g["severity"].upper())
        uc_tag = " — **user-controlled input confirmed**" if g["user_controlled"] else ""

        lines.append(f"### {sev_badge} — {g['title']} ({loc}){uc_tag}\n")
        lines.append(f"{g['root_cause_description']}\n")

        # Context sentence from the representative finding
        ctx = rep.get("context_sentence", "")
        if ctx:
            lines.append(f"\n**Context:** {ctx}\n")

        # Taint path
        tp = rep.get("taint_path", "")
        if tp and rep.get("taint_source") not in ("unknown", "constant", "internal"):
            lines.append(f"\n**Data flow:** `{tp}`\n")

        # Fix block using actual argument name
        arg = rep.get("argument_name", "")
        fix = _build_fix_block(g["vuln_type"], arg, rep)
        if fix:
            lines.append(f"\n**Fix:**\n{fix}\n")

    # ── Action plan ────────────────────────────────────────────────────────
    plan = build_action_plan(groups)
    if plan:
        lines.append("\n---\n\n## Recommended Action Plan\n")
        lines.extend(p + "\n" for p in plan)

    lines.append(
        "\n---\n_Reviewed by [IntelliCode](https://intellcode.onrender.com) — "
        "static analysis + taint-flow reasoning_"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fix block builder (uses actual argument name)
# ---------------------------------------------------------------------------

def _build_fix_block(vuln_type: str, arg: str, finding: dict) -> str:
    line_code = finding.get("snippet", "") or finding.get("argument_name", "") or arg

    templates: dict[str, tuple[str, str]] = {
        "code_injection": (
            f"result = eval({arg})",
            f"import ast\nresult = ast.literal_eval({arg})  # safe for literals; "
            "use a sandboxed evaluator for expressions",
        ),
        "eval_injection": (
            f"result = eval({arg})",
            f"import ast\nresult = ast.literal_eval({arg})  # safe for literals; "
            "use a sandboxed evaluator for expressions",
        ),
        "insecure_deserialization": (
            f"data = pickle.loads({arg})",
            f"import json, schema\ndata = schema.validate(json.loads({arg}))  "
            "# never deserialize pickle from untrusted sources",
        ),
        "path_traversal": (
            f"open({arg})",
            f"import os\n_safe = os.path.realpath(os.path.join(BASE_DIR, {arg}))\n"
            f"if not _safe.startswith(os.path.realpath(BASE_DIR)):\n"
            f"    raise ValueError('Path traversal blocked')\n"
            f"open(_safe)",
        ),
        "command_injection": (
            f"os.system({arg})",
            f"import subprocess, shlex\n"
            f"subprocess.run(shlex.split({arg}), shell=False, check=True)  "
            "# or pass args as list directly",
        ),
        "sql_injection": (
            f'cursor.execute("SELECT ... " + {arg})',
            f'cursor.execute("SELECT ... WHERE col = %s", ({arg},))  '
            "# parameterized — never concatenate",
        ),
        "hardcoded_secret": (
            f"{arg} = 'sk-...'  # hardcoded",
            f"import os\n{arg} = os.environ['{arg.upper()}']  "
            "# load from environment or secrets manager",
        ),
        "hardcoded_credential": (
            f"{arg} = 'password123'  # hardcoded",
            f"import os\n{arg} = os.environ['{arg.upper()}']",
        ),
        "weak_cryptography": (
            f"hashlib.md5({arg}.encode()).hexdigest()",
            f"import bcrypt\nhashed = bcrypt.hashpw({arg}.encode(), bcrypt.gensalt())  "
            "# use argon2 for new systems",
        ),
        "ssrf": (
            f"requests.get({arg})",
            f"ALLOWED = {{'api.example.com'}}\n"
            f"parsed = urllib.parse.urlparse({arg})\n"
            f"if parsed.hostname not in ALLOWED:\n"
            f"    raise ValueError('Blocked host')\n"
            f"requests.get({arg}, timeout=5)",
        ),
    }

    if vuln_type not in templates:
        return ""

    before, after = templates[vuln_type]
    return (
        f"```python\n# Before (vulnerable)\n{before}\n```\n\n"
        f"```python\n# After (fixed)\n{after}\n```"
    )
