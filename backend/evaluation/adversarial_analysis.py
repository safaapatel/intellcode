"""
Adversarial Analysis
Generates adversarial Python code snippets designed to fool the complexity and
security models, measures prediction degradation, and reports findings.

Results saved to evaluation/results/adversarial_analysis.json
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path

# Ensure backend root is on path
_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from features.code_metrics import compute_all_metrics, metrics_to_feature_vector, cognitive_complexity

# ---------------------------------------------------------------------------
# Adversarial snippets: complexity model
# ---------------------------------------------------------------------------

COMPLEXITY_SNIPPETS = {
    # ---- Category 1: deeply nested, low Halstead ----
    "deep_nesting_low_halstead_1": {
        "category": "deep_nesting_low_halstead",
        "description": "8 levels of if nesting, trivial bodies, minimal operators",
        "source": """\
def f(x):
    if x:
        if x:
            if x:
                if x:
                    if x:
                        if x:
                            if x:
                                if x:
                                    return 1
    return 0
""",
    },
    "deep_nesting_low_halstead_2": {
        "category": "deep_nesting_low_halstead",
        "description": "7-level for loop nesting, empty pass bodies",
        "source": """\
def g(a, b, c, d, e, f, h):
    for i in a:
        for j in b:
            for k in c:
                for l in d:
                    for m in e:
                        for n in f:
                            for o in h:
                                pass
""",
    },
    "deep_nesting_low_halstead_3": {
        "category": "deep_nesting_low_halstead",
        "description": "mixed if/while nesting, single-token bodies",
        "source": """\
def h(x):
    if x:
        while x:
            if x:
                while x:
                    if x:
                        while x:
                            x = 0
""",
    },
    "deep_nesting_low_halstead_4": {
        "category": "deep_nesting_low_halstead",
        "description": "try/except deeply nested, minimal expressions",
        "source": """\
def safe(x):
    try:
        try:
            try:
                try:
                    try:
                        return x
                    except:
                        pass
                except:
                    pass
            except:
                pass
        except:
            pass
    except:
        pass
""",
    },
    "deep_nesting_low_halstead_5": {
        "category": "deep_nesting_low_halstead",
        "description": "if/elif chains with no operators in bodies",
        "source": """\
def classify(v):
    if v == 1:
        if v == 1:
            if v == 1:
                if v == 1:
                    return v
            elif v == 2:
                return v
        elif v == 3:
            return v
    elif v == 4:
        return v
    return v
""",
    },
    # ---- Category 2: high Halstead, simple structure ----
    "high_halstead_simple_1": {
        "category": "high_halstead_simple",
        "description": "single expression with 8 lambda chains and many operators",
        "source": """\
result = (lambda x: x+1)(1) + (lambda x: x*2)(2) + (lambda x: x**3)(3) + (lambda x: x//4)(4) + (lambda x: x%5)(5) + (lambda x: x&6)(6) + (lambda x: x|7)(7) + (lambda x: x^8)(8)
""",
    },
    "high_halstead_simple_2": {
        "category": "high_halstead_simple",
        "description": "flat arithmetic expression with many unique operators",
        "source": """\
a = 1; b = 2; c = 3; d = 4; e = 5; f = 6; g = 7; h = 8
result = ((a + b) * (c - d)) / ((e ** f) % (g & h)) | (a ^ b) + (c << d) - (e >> f) + (~g) - (h // a) + (b | c) * (d & e) - (f ^ g)
""",
    },
    "high_halstead_simple_3": {
        "category": "high_halstead_simple",
        "description": "list comprehension with many embedded expressions",
        "source": """\
import math
data = [i for i in range(100)]
out = [math.sin(x) * math.cos(x) + math.tan(x) - math.sqrt(abs(x)) + math.log(x+1) * math.exp(-x/100.0) for x in data if x > 0 and x % 2 == 0 and x % 3 != 0 and x < 90]
""",
    },
    "high_halstead_simple_4": {
        "category": "high_halstead_simple",
        "description": "dict comprehension with heavy operator use, flat structure",
        "source": """\
keys = list(range(50))
lookup = {k: (k**2 + k*3 - k//2 + k%7 - k&3 + k|1 + k^5 - k<<1 + k>>1) for k in keys if k > 0}
""",
    },
    "high_halstead_simple_5": {
        "category": "high_halstead_simple",
        "description": "chained ternary expressions on a single line",
        "source": """\
def pick(x):
    return (1 if x < 0 else 2) + (3 if x > 0 else 4) + (5 if x == 0 else 6) + (7 if x != 1 else 8) + (9 if x <= 2 else 10) + (11 if x >= 3 else 12)
""",
    },
    # ---- Category 3: long file, single function ----
    "long_file_single_function_1": {
        "category": "long_file_single_function",
        "description": "500-line file, all simple assignments, no branches",
        "source": "\n".join(
            ["def build_config():"]
            + [f"    cfg_{i} = {i}" for i in range(1, 498)]
            + ["    return None", ""]
        ),
    },
    "long_file_single_function_2": {
        "category": "long_file_single_function",
        "description": "400-line module of constant definitions only",
        "source": "\n".join(
            [f"CONST_{i} = {i * 3 + 1}" for i in range(400)] + [""]
        ),
    },
    "long_file_single_function_3": {
        "category": "long_file_single_function",
        "description": "300-line function body of string assignments",
        "source": "\n".join(
            ["def build_strings():"]
            + [f"    s_{i} = 'value_{i}'" for i in range(298)]
            + ["    return None", ""]
        ),
    },
    # ---- Category 4: heavily commented ----
    "heavily_commented_1": {
        "category": "heavily_commented",
        "description": "real sorting function + 10x comment lines",
        "source": (
            "\n".join(
                [
                    "# This is comment line {}".format(i)
                    for i in range(100)
                ]
            )
            + """
def sort_list(items):
    # Sort the list using bubble sort
    n = len(items)
    for i in range(n):
        for j in range(n - i - 1):
            if items[j] > items[j + 1]:
                items[j], items[j + 1] = items[j + 1], items[j]
    return items
"""
            + "\n".join(
                [
                    "# Trailing comment line {}".format(i)
                    for i in range(100)
                ]
            )
        ),
    },
    "heavily_commented_2": {
        "category": "heavily_commented",
        "description": "simple class + 200 docstring-style comment lines",
        "source": (
            "\n".join(
                [
                    "# Documentation line {}".format(i)
                    for i in range(200)
                ]
            )
            + """
class Config:
    def __init__(self):
        self.value = 0

    def get(self):
        return self.value
"""
        ),
    },
}


# ---------------------------------------------------------------------------
# Adversarial snippets: security model
# ---------------------------------------------------------------------------

SECURITY_SNIPPETS = {
    "benign_with_keywords_1": {
        "category": "benign_with_keywords",
        "true_label": 0,
        "description": "Benign config handler with password/sql variable names",
        "source": """\
class AppConfig:
    def __init__(self):
        self.password = "placeholder"
        self.sql_query = "select * from docs"
        self.api_key = "demo_key_only"

    def display(self):
        print("Config loaded")
        return True
""",
    },
    "benign_with_keywords_2": {
        "category": "benign_with_keywords",
        "true_label": 0,
        "description": "Unit test file that mentions exec/eval in docstrings only",
        "source": """\
def test_eval_description():
    '''
    Tests that eval() and exec() are documented in the codebase.
    The string 'sql_injection' should be flagged by static tools.
    '''
    description = "eval and exec are dangerous"
    sql_injection_note = "sanitise your inputs"
    assert len(description) > 0
""",
    },
    "benign_with_keywords_3": {
        "category": "benign_with_keywords",
        "true_label": 0,
        "description": "Password hashing function (secure usage)",
        "source": """\
import hashlib
import os

def hash_password(password: str) -> str:
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    return salt.hex() + ':' + key.hex()

def verify_password(password: str, stored: str) -> bool:
    salt_hex, key_hex = stored.split(':')
    salt = bytes.fromhex(salt_hex)
    key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
    return key.hex() == key_hex
""",
    },
    "vulnerable_no_keywords_1": {
        "category": "vulnerable_no_keywords",
        "true_label": 1,
        "description": "SQL injection via string concat but variable names are innocuous",
        "source": """\
import sqlite3

def lookup(conn, x, y):
    z = x + y
    w = "SELECT * FROM records WHERE id = " + z
    cursor = conn.cursor()
    cursor.execute(w)
    return cursor.fetchall()
""",
    },
    "vulnerable_no_keywords_2": {
        "category": "vulnerable_no_keywords",
        "true_label": 1,
        "description": "OS command injection with neutral variable names",
        "source": """\
import os

def run(a, b):
    c = a + ' ' + b
    d = os.popen(c)
    return d.read()
""",
    },
    "vulnerable_no_keywords_3": {
        "category": "vulnerable_no_keywords",
        "true_label": 1,
        "description": "Pickle deserialization from user-supplied data with innocuous names",
        "source": """\
import pickle

def load_data(raw_bytes):
    obj = pickle.loads(raw_bytes)
    return obj
""",
    },
    "obfuscated_vulnerability_1": {
        "category": "obfuscated_vulnerability",
        "true_label": 1,
        "description": "Base64-encoded eval payload",
        "source": """\
import base64

def process(data):
    payload = base64.b64decode(data).decode()
    eval(payload)
""",
    },
    "obfuscated_vulnerability_2": {
        "category": "obfuscated_vulnerability",
        "true_label": 1,
        "description": "Indirect exec via compile and string building",
        "source": """\
def run_code(parts):
    combined = ''.join(parts)
    code_obj = compile(combined, '<string>', 'exec')
    exec(code_obj)
""",
    },
    "obfuscated_vulnerability_3": {
        "category": "obfuscated_vulnerability",
        "true_label": 1,
        "description": "Format-string based command injection",
        "source": """\
import subprocess

def deploy(host, cmd):
    template = 'ssh {}@production {}'
    full = template.format(host, cmd)
    subprocess.Popen(full, shell=True)
""",
    },
}


# ---------------------------------------------------------------------------
# Clean test-set baselines (for degradation comparison)
# ---------------------------------------------------------------------------

CLEAN_COMPLEXITY_EXAMPLES = [
    ("simple_func", """\
def add(a, b):
    return a + b
""", 0),
    ("medium_func", """\
def process(items):
    result = []
    for item in items:
        if item > 0:
            result.append(item * 2)
        else:
            result.append(0)
    return result
""", 3),
    ("class_with_methods", """\
class Calculator:
    def __init__(self):
        self.history = []

    def add(self, a, b):
        r = a + b
        self.history.append(r)
        return r

    def subtract(self, a, b):
        r = a - b
        self.history.append(r)
        return r
""", 2),
]

CLEAN_SECURITY_EXAMPLES = [
    ("parametrised_query", """\
def get_user(conn, user_id):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    return cursor.fetchone()
""", 0),
    ("safe_subprocess", """\
import subprocess
result = subprocess.run(['ls', '-la'], capture_output=True, text=True)
print(result.stdout)
""", 0),
    ("obvious_injection", """\
import os
user_input = input('Enter command: ')
os.system(user_input)
""", 1),
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _load_complexity_model():
    try:
        from models.complexity_prediction import ComplexityPredictionModel
        return ComplexityPredictionModel(
            checkpoint_path=str(_BACKEND / "checkpoints" / "complexity" / "model.pkl")
        )
    except Exception as exc:
        print(f"[WARN] Could not load complexity model: {exc}")
        return None


def _load_security_model():
    try:
        from models.security_detection import EnsembleSecurityModel
        return EnsembleSecurityModel(
            checkpoint_dir=str(_BACKEND / "checkpoints" / "security")
        )
    except Exception as exc:
        print(f"[WARN] Could not load security model: {exc}")
        return None


def run_complexity_adversarial(model) -> dict:
    results = []
    errors = []

    for name, info in COMPLEXITY_SNIPPETS.items():
        source = info["source"]
        true_cog = cognitive_complexity(source)
        if model is not None:
            try:
                predicted = model.predict_cognitive_complexity(source)
            except Exception as exc:
                predicted = float(true_cog)
                print(f"  [WARN] complexity predict failed for {name}: {exc}")
        else:
            predicted = float(true_cog)

        err = abs(predicted - true_cog)
        errors.append(err)
        error_magnitude = "large" if err > 10.0 else "moderate" if err > 5.0 else "small"
        results.append({
            "name": name,
            "category": info["category"],
            "description": info["description"],
            "true_cognitive_complexity": true_cog,
            "predicted_cognitive_complexity": round(predicted, 2),
            "absolute_error": round(err, 2),
            "error_magnitude": error_magnitude,
        })

    # Clean baseline
    clean_errors = []
    for name, src, _ in CLEAN_COMPLEXITY_EXAMPLES:
        true_c = cognitive_complexity(src)
        if model is not None:
            try:
                pred_c = model.predict_cognitive_complexity(src)
            except Exception:
                pred_c = float(true_c)
        else:
            pred_c = float(true_c)
        clean_errors.append(abs(pred_c - true_c))

    mean_adv = sum(errors) / len(errors) if errors else 0.0
    mean_clean = sum(clean_errors) / len(clean_errors) if clean_errors else 0.0

    # Per-category mean error — more informative than an aggregate "fooling rate"
    # which conflates categories with different error magnitudes
    categories: dict[str, list[float]] = {}
    for r in results:
        categories.setdefault(r["category"], []).append(r["absolute_error"])
    per_category = {cat: round(sum(v) / len(v), 2) for cat, v in categories.items()}

    return {
        "results": results,
        "summary": {
            "n_adversarial": len(results),
            "mean_adv_error": round(mean_adv, 2),
            "mean_clean_error": round(mean_clean, 2),
            "mean_error_degradation": round(mean_adv - mean_clean, 2),
            "per_category_mean_error": per_category,
            "note": (
                "No aggregate fooling rate reported. These cases were constructed "
                "to target known feature weaknesses; any percentage derived from "
                "them estimates targeted-attack success, not production error frequency."
            ),
        },
    }


def run_security_adversarial(model) -> dict:
    results = []
    errors = []
    clean_errors = []

    threshold = 0.5
    if model is not None and hasattr(model, "_rf_threshold"):
        threshold = model._rf_threshold

    for name, info in SECURITY_SNIPPETS.items():
        source = info["source"]
        true_label = info["true_label"]
        if model is not None:
            try:
                prob = model.vulnerability_score(source)
            except Exception as exc:
                prob = 0.0
                print(f"  [WARN] security predict failed for {name}: {exc}")
        else:
            prob = 0.0

        pred_label = 1 if prob >= threshold else 0
        wrong = pred_label != true_label
        if true_label == 1 and pred_label == 0:
            error_type = "false_negative"
        elif true_label == 0 and pred_label == 1:
            error_type = "false_positive"
        else:
            error_type = "correct"
        errors.append(1 if wrong else 0)

        results.append({
            "name": name,
            "category": info["category"],
            "description": info["description"],
            "true_label": true_label,
            "predicted_prob": round(prob, 4),
            "predicted_label": pred_label,
            "error_type": error_type,
        })

    # Clean baseline
    for name, src, true_lbl in CLEAN_SECURITY_EXAMPLES:
        if model is not None:
            try:
                prob_c = model.vulnerability_score(src)
            except Exception:
                prob_c = 0.0
        else:
            prob_c = 0.0
        pred_lbl = 1 if prob_c >= threshold else 0
        clean_errors.append(1 if pred_lbl != true_lbl else 0)

    clean_err_rate = sum(clean_errors) / len(clean_errors) if clean_errors else 0.0
    n_fn = sum(1 for r in results if r["error_type"] == "false_negative")
    n_fp = sum(1 for r in results if r["error_type"] == "false_positive")

    # Per-category breakdown — more informative than a single aggregate rate
    categories: dict[str, dict] = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "false_negative": 0, "false_positive": 0, "correct": 0}
        categories[cat]["total"] += 1
        categories[cat][r["error_type"]] += 1

    return {
        "results": results,
        "summary": {
            "n_adversarial": len(results),
            "false_negatives": n_fn,
            "false_positives": n_fp,
            "correct": len(results) - n_fn - n_fp,
            "clean_error_rate": round(clean_err_rate, 4),
            "per_category": categories,
            "note": (
                "No aggregate fooling rate reported. Cases were constructed to target "
                "specific failure modes; per-category breakdown is more informative "
                "than an aggregate rate across purposely adversarial inputs."
            ),
        },
    }


def print_complexity_table(data: dict):
    results = data["results"]
    summary = data["summary"]
    print()
    print("=== Adversarial Complexity Diagnostic Results ===")
    col_w = [32, 24, 8, 10, 8, 10]
    header = (
        f"{'Name':<{col_w[0]}} {'Category':<{col_w[1]}} {'TrueCog':>{col_w[2]}} "
        f"{'Predicted':>{col_w[3]}} {'AbsErr':>{col_w[4]}} {'ErrSize':>{col_w[5]}}"
    )
    sep = "-" * sum(col_w + [5])
    print(sep)
    print(header)
    print(sep)
    for r in results:
        name = r["name"][:col_w[0]]
        cat = r["category"][:col_w[1]]
        print(
            f"{name:<{col_w[0]}} {cat:<{col_w[1]}} {r['true_cognitive_complexity']:>{col_w[2]}} "
            f"{r['predicted_cognitive_complexity']:>{col_w[3]}.2f} "
            f"{r['absolute_error']:>{col_w[4]}.2f} {r['error_magnitude']:>{col_w[5]}}"
        )
    print(sep)
    print(
        f"Mean adv error: {summary['mean_adv_error']:.2f}  "
        f"Mean clean error: {summary['mean_clean_error']:.2f}  "
        f"Error degradation: {summary['mean_error_degradation']:.2f}"
    )
    print("Per-category mean error:")
    for cat, mean_err in summary["per_category_mean_error"].items():
        print(f"  {cat}: {mean_err:.2f}")


def print_security_table(data: dict):
    results = data["results"]
    summary = data["summary"]
    print()
    print("=== Adversarial Security Diagnostic Results ===")
    col_w = [36, 28, 6, 8, 6, 18]
    header = (
        f"{'Name':<{col_w[0]}} {'Category':<{col_w[1]}} {'TrueL':>{col_w[2]}} "
        f"{'Prob':>{col_w[3]}} {'PredL':>{col_w[4]}} {'ErrorType':<{col_w[5]}}"
    )
    sep = "-" * sum(col_w + [5])
    print(sep)
    print(header)
    print(sep)
    for r in results:
        name = r["name"][:col_w[0]]
        cat = r["category"][:col_w[1]]
        print(
            f"{name:<{col_w[0]}} {cat:<{col_w[1]}} {r['true_label']:>{col_w[2]}} "
            f"{r['predicted_prob']:>{col_w[3]}.4f} {r['predicted_label']:>{col_w[4]}} "
            f"{r['error_type']:<{col_w[5]}}"
        )
    print(sep)
    print(
        f"FN: {summary['false_negatives']}  FP: {summary['false_positives']}  "
        f"Correct: {summary['correct']}  Clean error rate: {summary['clean_error_rate']:.1%}"
    )
    print("Per-category breakdown:")
    for cat, counts in summary["per_category"].items():
        print(f"  {cat}: {counts}")


def main():
    out_dir = _BACKEND / "evaluation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    complexity_model = _load_complexity_model()
    security_model = _load_security_model()

    print("Running complexity adversarial analysis...")
    complexity_data = run_complexity_adversarial(complexity_model)

    print("Running security adversarial analysis...")
    security_data = run_security_adversarial(security_model)

    output = {
        "complexity": complexity_data,
        "security": security_data,
    }

    out_path = out_dir / "adversarial_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print_complexity_table(complexity_data)
    print_security_table(security_data)

    c_summ = complexity_data["summary"]
    s_summ = security_data["summary"]
    print()
    print("=== Overall Diagnostic Summary ===")
    print(f"Complexity mean error (adversarial): {c_summ['mean_adv_error']:.2f}")
    print(f"Complexity mean error (clean):       {c_summ['mean_clean_error']:.2f}")
    print(f"Complexity error degradation:        {c_summ['mean_error_degradation']:.2f}")
    print(f"Security FN (targeted misses):       {s_summ['false_negatives']}")
    print(f"Security FP (targeted false alarms): {s_summ['false_positives']}")


if __name__ == "__main__":
    main()
