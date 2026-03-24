"""
DEPRECATED — Synthetic Dataset Generator
=========================================
This file is kept for reference only.  The training pipeline now uses REAL
public datasets fetched by:

    python training/fetch_real_datasets.py --all --out data/

Or run everything in one step:

    python training/train_all.py

The synthetic generator produces artificially simple code patterns that lead
to over-fitted models (AUC 1.0 on training data, poor generalisation).
Real data from Bandit examples, PyDriller commit mining, and actual Python
open-source repos produces significantly more realistic model behaviour.
"""

import sys
print(
    "\n[DEPRECATED] generate_synthetic_data.py is no longer used.\n"
    "Run:  python training/fetch_real_datasets.py --all --out data/\n"
    "  or: python training/train_all.py\n"
)
sys.exit(0)

# ── Legacy code kept below for reference ────────────────────────────────────

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Callable

# Allow imports from backend/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Code snippet templates
# ─────────────────────────────────────────────────────────────────────────────

_NAMES = [
    "process", "compute", "validate", "fetch", "update", "delete", "create",
    "handle", "parse", "format", "transform", "filter", "merge", "check",
    "build", "load", "save", "send", "receive", "convert",
]
_OBJECTS = [
    "user", "order", "product", "record", "data", "item", "request",
    "response", "config", "report", "payload", "event", "transaction",
    "message", "file", "session", "token", "query", "result", "account",
]
_TYPES = ["str", "int", "float", "bool", "list", "dict", "Optional[str]", "Optional[int]"]


def _rname() -> str:
    return f"{random.choice(_NAMES)}_{random.choice(_OBJECTS)}"


def _rparam(n: int) -> str:
    params = [f"p{i}: {random.choice(_TYPES)}" for i in range(n)]
    return ", ".join(params)


# ─────────────────────────────────────────────────────────────────────────────
# Complexity dataset
# ─────────────────────────────────────────────────────────────────────────────

def _make_clean_function() -> str:
    name = _rname()
    n_params = random.randint(1, 3)
    body_lines = random.randint(3, 12)
    lines = [f"def {name}({_rparam(n_params)}) -> None:"]
    lines.append(f'    """Process {name.replace("_", " ")}."""')
    for _ in range(body_lines):
        lines.append(f"    result = p0 if p0 else None")
    lines.append("    return result")
    return "\n".join(lines)


def _make_complex_function(depth: int = 3) -> str:
    name = _rname()
    lines = [f"def {name}(data, config, mode, flag, extra=None):"]
    lines.append('    """Complex function with high cyclomatic complexity."""')
    indent = "    "
    for i in range(depth):
        lines.append(f"{indent}if data and config:")
        indent += "    "
        lines.append(f"{indent}for item in data:")
        indent += "    "
        lines.append(f"{indent}if item:")
        indent += "    "
    lines.append(f"{indent}result = item")
    # unwind
    for i in range(depth * 3):
        lines.append("    pass")
    lines.append("    return result")
    return "\n".join(lines)


def _make_long_function(n_lines: int = 60) -> str:
    name = _rname()
    lines = [f"def {name}(data):"]
    lines.append('    """Very long function."""')
    for i in range(n_lines):
        lines.append(f"    x{i} = data.get('key{i}', {i})")
    lines.append("    return x0")
    return "\n".join(lines)


def generate_complexity_samples(n: int) -> list[dict]:
    """
    Generate n Python code samples and extract feature vectors + maintainability score.
    """
    from features.code_metrics import compute_all_metrics, metrics_to_feature_vector

    samples = []
    generators: list[Callable[[], str]] = [
        _make_clean_function,
        lambda: _make_complex_function(random.randint(2, 4)),
        lambda: _make_long_function(random.randint(40, 100)),
        lambda: "\n\n".join(_make_clean_function() for _ in range(random.randint(2, 5))),
        lambda: _make_complex_function(2) + "\n\n" + _make_long_function(30),
    ]

    for i in range(n):
        gen = random.choice(generators)
        source = gen()
        try:
            metrics = compute_all_metrics(source)
            feat_vec = metrics_to_feature_vector(metrics)
            target = float(metrics.maintainability_index)
            if not (0 <= target <= 100):
                continue
            samples.append({
                "features": [float(x) for x in feat_vec],
                "target": round(target, 2),
                "source_len": len(source),
            })
        except Exception:
            continue

        if (i + 1) % 500 == 0:
            print(f"  Complexity: {i + 1}/{n}")

    print(f"  Generated {len(samples)} complexity samples")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Security dataset
# ─────────────────────────────────────────────────────────────────────────────

_VULN_TEMPLATES = [
    # SQL injection
    'def get_user(uid):\n    query = "SELECT * FROM users WHERE id = " + str(uid)\n    return db.execute(query)\n',
    'def search(term):\n    sql = f"SELECT * FROM products WHERE name LIKE \'%{term}%\'"\n    return cursor.execute(sql)\n',
    'def login(user, pwd):\n    q = "SELECT * FROM users WHERE username=\'"+user+"\' AND password=\'"+pwd+"\'"\n    return db.query(q)\n',
    # Hardcoded secrets
    'SECRET_KEY = "hardcoded-secret-abc123"\nAPI_TOKEN = "sk-prod-xyzabc"\nDB_PASSWORD = "admin1234"\n',
    'def connect():\n    password = "mypassword123"\n    host = "db.internal.com"\n    return connect(host, password)\n',
    # Eval / exec
    'def run_code(user_input):\n    eval(user_input)\n    return "done"\n',
    'def execute(cmd):\n    import os\n    os.system(cmd)\n',
    # Shell injection
    'def ping(host):\n    import subprocess\n    subprocess.call("ping " + host, shell=True)\n',
    # XSS-like (template injection)
    'def render(name):\n    return "<html>Hello " + name + "</html>"\n',
    # Insecure deserialization
    'def load_data(data):\n    import pickle\n    return pickle.loads(data)\n',
    # Path traversal
    'def read_file(filename):\n    with open("/var/data/" + filename) as f:\n        return f.read()\n',
    # Weak crypto
    'import md5\ndef hash_password(pwd):\n    return md5.new(pwd).hexdigest()\n',
]

_CLEAN_TEMPLATES = [
    # Parameterized queries
    'def get_user(uid):\n    cursor.execute("SELECT * FROM users WHERE id = %s", (uid,))\n    return cursor.fetchone()\n',
    'def search(term):\n    cursor.execute("SELECT * FROM products WHERE name LIKE %s", (f"%{term}%",))\n    return cursor.fetchall()\n',
    # Env vars for secrets
    'import os\ndef connect():\n    password = os.environ["DB_PASSWORD"]\n    return create_connection(password)\n',
    # Safe subprocess
    'import subprocess\ndef ping(host):\n    result = subprocess.run(["ping", "-c", "1", host], capture_output=True, text=True)\n    return result.stdout\n',
    # Input validation
    'def process(value: int) -> int:\n    if not isinstance(value, int):\n        raise TypeError("Expected int")\n    return value * 2\n',
    # Safe file ops
    'from pathlib import Path\ndef read_file(name: str) -> str:\n    base = Path("/var/data")\n    path = (base / name).resolve()\n    if not str(path).startswith(str(base)):\n        raise ValueError("Path traversal detected")\n    return path.read_text()\n',
    # Strong crypto
    'import hashlib\ndef hash_password(pwd: str) -> str:\n    return hashlib.sha256(pwd.encode()).hexdigest()\n',
    # Generic clean functions
    'def add(a: int, b: int) -> int:\n    """Add two numbers."""\n    return a + b\n',
    'def greet(name: str) -> str:\n    """Return a greeting."""\n    return f"Hello, {name}!"\n',
    'def is_valid(value) -> bool:\n    return value is not None and len(str(value)) > 0\n',
    'def compute_total(items: list) -> float:\n    return sum(item["price"] for item in items)\n',
    'def load_config(path: str) -> dict:\n    import json\n    with open(path) as f:\n        return json.load(f)\n',
]


def _add_noise(code: str) -> str:
    """Add random benign lines to make samples more realistic."""
    noise = [
        "    # TODO: add logging\n",
        "    result = None\n",
        f"    name = '{_rname()}'\n",
        "    count = 0\n",
    ]
    lines = code.splitlines(keepends=True)
    insert_at = random.randint(0, max(0, len(lines) - 1))
    lines.insert(insert_at, random.choice(noise))
    return "".join(lines)


def generate_security_samples(n: int) -> list[dict]:
    from features.ast_extractor import ASTExtractor, tokenize_code

    samples = []
    n_vuln = n // 2
    n_clean = n - n_vuln

    # Vulnerable samples
    for i in range(n_vuln):
        tmpl = random.choice(_VULN_TEMPLATES)
        source = _add_noise(tmpl) if random.random() > 0.3 else tmpl
        try:
            ast_feats = ASTExtractor().extract(source)
            tokens = tokenize_code(source)[:512]
            samples.append({
                "label": 1,
                "tokens": tokens,
                "n_calls": ast_feats.get("n_calls", 0),
                "n_imports": ast_feats.get("n_imports", 0),
                "source": source,
            })
        except Exception:
            continue

    # Clean samples
    for i in range(n_clean):
        tmpl = random.choice(_CLEAN_TEMPLATES)
        source = _add_noise(tmpl) if random.random() > 0.3 else tmpl
        # Add some extra clean functions to make it longer
        if random.random() > 0.5:
            source += "\n\n" + _make_clean_function()
        try:
            ast_feats = ASTExtractor().extract(source)
            tokens = tokenize_code(source)[:512]
            samples.append({
                "label": 0,
                "tokens": tokens,
                "n_calls": ast_feats.get("n_calls", 0),
                "n_imports": ast_feats.get("n_imports", 0),
                "source": source,
            })
        except Exception:
            continue

    random.shuffle(samples)
    print(f"  Generated {len(samples)} security samples ({n_vuln} vulnerable, {n_clean} clean)")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Pattern recognition dataset
# ─────────────────────────────────────────────────────────────────────────────

def _make_clean_snippet() -> str:
    name = _rname()
    n = random.randint(1, 3)
    lines = [f"def {name}({_rparam(n)}) -> None:"]
    lines.append(f'    """Handle {name.replace("_", " ")}."""')
    lines.append("    if p0 is None:")
    lines.append("        return None")
    lines.append("    result = str(p0)")
    lines.append("    return result")
    return "\n".join(lines)


def _make_code_smell_snippet() -> str:
    """Long function, many returns, no docstring."""
    name = _rname()
    params = ", ".join(f"p{i}" for i in range(7))
    lines = [f"def {name}({params}):"]
    for i in range(45):
        lines.append(f"    x{i} = p0 + {i}")
    lines.append("    if x0 > 10:")
    lines.append("        return x0")
    lines.append("    elif x1 > 5:")
    lines.append("        return x1")
    lines.append("    return None")
    return "\n".join(lines)


def _make_anti_pattern_snippet() -> str:
    """God class: does everything."""
    lines = [
        "class GodClass:",
        "    def do_everything(self, a, b, c, d, e, f, g):",
        "        # handles auth, db, email, logging, reporting",
        "        import smtplib, sqlite3, hashlib, logging, json, os, re",
        "        logging.info('starting')",
        "        conn = sqlite3.connect('db.sqlite')",
        "        conn.execute(\"SELECT * FROM users WHERE id=\" + str(a))",
        "        h = hashlib.md5(str(b).encode()).hexdigest()",
        "        os.system('rm -rf /tmp/*')",
        "        smtplib.SMTP('mail.server.com').sendmail(c, d, str(e))",
        "        return eval(f)",
    ]
    return "\n".join(lines)


def _make_style_violation_snippet() -> str:
    """Long lines, camelCase, magic numbers."""
    lines = [
        "def processUserDataAndComputeMetricsAndGenerateReportForAllActiveUsersInTheSystem(userData, configurationParameters, databaseConnectionObject, reportingEngine, emailService):",
        "    x=userData['total']*3.14159265358979*2.71828182845904523536+42/7-1337/1000000",
        "    if x>99.999999 and userData['active']==True and configurationParameters['enabled']!=False:",
        "        reportingEngine.generateReport(userData, format='PDF', pages=100, compress=True, encrypt=False, sign=True, validate=True, preview=False, download=True)",
        "    return x",
    ]
    return "\n".join(lines)


def generate_pattern_samples(n: int) -> list[dict]:
    label_generators = {
        "clean": _make_clean_snippet,
        "code_smell": _make_code_smell_snippet,
        "anti_pattern": _make_anti_pattern_snippet,
        "style_violation": _make_style_violation_snippet,
    }

    samples = []
    per_label = n // len(label_generators)

    for label, gen in label_generators.items():
        for _ in range(per_label):
            code = gen()
            # Add some noise
            if random.random() > 0.5:
                code = _make_clean_function() + "\n\n" + code
            samples.append({
                "code": code[:2000],
                "label": label,
            })

    random.shuffle(samples)
    print(f"  Generated {len(samples)} pattern samples ({per_label} per label)")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Bug prediction dataset
# ─────────────────────────────────────────────────────────────────────────────

def _make_buggy_code() -> str:
    """Code with bug-prone patterns: deep nesting, no error handling, many params."""
    name = _rname()
    n_lines = random.randint(50, 120)
    lines = [f"def {name}(a, b, c, d, e, f, g, h):"]
    for i in range(5):
        lines.append(f"    if a and b:")
        lines.append(f"        for item in c:")
        lines.append(f"            if item:")
        lines.append(f"                while d:")
        lines.append(f"                    x = a + b + {i}")
    for i in range(n_lines - 20):
        lines.append(f"    val{i} = a * {i} + b")
    lines.append("    return val0")
    return "\n".join(lines)


def _make_clean_code() -> str:
    """Well-structured code with error handling, short, simple."""
    name = _rname()
    n = random.randint(1, 3)
    lines = [f"def {name}({_rparam(n)}) -> str:"]
    lines.append(f'    """Process {name.replace("_", " ")} safely."""')
    lines.append("    try:")
    lines.append("        if p0 is None:")
    lines.append("            raise ValueError('p0 cannot be None')")
    lines.append("        result = str(p0)")
    lines.append("        return result")
    lines.append("    except (TypeError, ValueError) as e:")
    lines.append("        raise RuntimeError(f'Processing failed: {e}') from e")
    return "\n".join(lines)


def generate_bug_samples(n: int) -> list[dict]:
    from features.code_metrics import compute_all_metrics, metrics_to_feature_vector
    from features.ast_extractor import ASTExtractor

    samples = []
    n_buggy = n // 2
    n_clean = n - n_buggy

    for _ in range(n_buggy):
        source = _make_buggy_code()
        try:
            metrics = compute_all_metrics(source)
            ast_feats = ASTExtractor().extract(source)
            feat_vec = metrics_to_feature_vector(metrics)
            samples.append({
                "label": 1,
                "static_features": [float(x) for x in feat_vec],
                "git_features": {
                    "code_churn": random.randint(200, 800),
                    "author_count": random.randint(3, 10),
                    "file_age_days": random.randint(100, 500),
                    "n_past_bugs": random.randint(2, 8),
                    "commit_freq": random.uniform(1.0, 5.0),
                },
            })
        except Exception:
            continue

    for _ in range(n_clean):
        source = _make_clean_code()
        # Optionally combine multiple clean functions
        if random.random() > 0.4:
            source += "\n\n" + _make_clean_function()
        try:
            metrics = compute_all_metrics(source)
            ast_feats = ASTExtractor().extract(source)
            feat_vec = metrics_to_feature_vector(metrics)
            samples.append({
                "label": 0,
                "static_features": [float(x) for x in feat_vec],
                "git_features": {
                    "code_churn": random.randint(0, 100),
                    "author_count": random.randint(1, 3),
                    "file_age_days": random.randint(10, 200),
                    "n_past_bugs": random.randint(0, 1),
                    "commit_freq": random.uniform(0.1, 2.0),
                },
            })
        except Exception:
            continue

    random.shuffle(samples)
    print(f"  Generated {len(samples)} bug samples ({n_buggy} buggy, {n_clean} clean)")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Writers
# ─────────────────────────────────────────────────────────────────────────────

def write_jsonl(samples: list[dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"  Saved {len(samples)} records -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic training datasets (no GitHub token needed)"
    )
    parser.add_argument("--task", choices=["complexity", "security", "pattern", "bug"],
                        help="Which dataset to generate (omit with --all)")
    parser.add_argument("--all", action="store_true", help="Generate all datasets")
    parser.add_argument("--n", type=int, default=1500, help="Samples per dataset")
    parser.add_argument("--out", default="data", help="Output directory")
    args = parser.parse_args()

    if not args.all and not args.task:
        parser.error("Specify --task or --all")

    tasks = (
        ["complexity", "security", "pattern", "bug"]
        if args.all
        else [args.task]
    )

    for task in tasks:
        print(f"\n{'='*50}")
        print(f"Generating {task} dataset ({args.n} samples)...")
        print("=" * 50)

        if task == "complexity":
            samples = generate_complexity_samples(args.n)
            write_jsonl(samples, f"{args.out}/complexity_dataset.jsonl")

        elif task == "security":
            samples = generate_security_samples(args.n)
            write_jsonl(samples, f"{args.out}/security_dataset.jsonl")

        elif task == "pattern":
            samples = generate_pattern_samples(args.n)
            write_jsonl(samples, f"{args.out}/pattern_dataset.jsonl")

        elif task == "bug":
            samples = generate_bug_samples(args.n)
            write_jsonl(samples, f"{args.out}/bug_dataset.jsonl")

    print("\nAll datasets generated successfully.")


if __name__ == "__main__":
    main()
