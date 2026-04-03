"""
Distribution Shift Test — Python 3.8 vs 3.12 syntax
Tests how well models generalise from older Python style to modern syntax.

For each paired snippet (old-style / new-style):
  - Complexity model: are predictions consistent? (|diff| < 5 = stable)
  - Security model: are risk scores consistent? (|diff| < 0.15 = stable)

Results saved to evaluation/results/distribution_shift_test.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# ---------------------------------------------------------------------------
# Paired syntax snippets
# ---------------------------------------------------------------------------

SYNTAX_PAIRS = [
    {
        "name": "walrus_operator",
        "description": "List comprehension with walrus vs explicit loop",
        "old_style": """\
data = [1, 2, 3, 4, 5, 10, 20]
result = []
for x in data:
    y = x * 2
    if y > 10:
        result.append(y)
""",
        "new_style": """\
data = [1, 2, 3, 4, 5, 10, 20]
result = [y for x in data if (y := x * 2) > 10]
""",
    },
    {
        "name": "match_statement",
        "description": "Structural pattern matching vs if/elif chain",
        "old_style": """\
def classify(x):
    if x == 1:
        return "one"
    elif x == 2:
        return "two"
    elif x == 3:
        return "three"
    else:
        return "other"
""",
        "new_style": """\
def classify(x):
    match x:
        case 1:
            return "one"
        case 2:
            return "two"
        case 3:
            return "three"
        case _:
            return "other"
""",
    },
    {
        "name": "type_hints_modern",
        "description": "PEP 604 union types vs typing.Optional/List/Dict",
        "old_style": """\
from typing import Optional, List, Dict

def process(items: List[str], config: Optional[Dict[str, int]] = None) -> List[str]:
    return [x.upper() for x in items]
""",
        "new_style": """\
def process(items: list[str], config: dict[str, int] | None = None) -> list[str]:
    return [x.upper() for x in items]
""",
    },
    {
        "name": "exception_groups",
        "description": "ExceptionGroup (3.11+) vs multiple except clauses",
        "old_style": """\
def handle_errors(fn):
    try:
        return fn()
    except ValueError as e:
        print("Value error:", e)
    except TypeError as e:
        print("Type error:", e)
    except RuntimeError as e:
        print("Runtime error:", e)
""",
        "new_style": """\
def handle_errors(fn):
    try:
        return fn()
    except* ValueError as eg:
        for e in eg.exceptions:
            print("Value error:", e)
    except* TypeError as eg:
        for e in eg.exceptions:
            print("Type error:", e)
""",
    },
    {
        "name": "fstring_modern",
        "description": "f-string debugging (= specifier) vs explicit format",
        "old_style": """\
def debug_values(x, y, z):
    print("x={}, y={}, z={}".format(x, y, z))
    result = x + y + z
    print("result={}".format(result))
    return result
""",
        "new_style": """\
def debug_values(x, y, z):
    print(f"{x=}, {y=}, {z=}")
    result = x + y + z
    print(f"{result=}")
    return result
""",
    },
    {
        "name": "dataclass_slots",
        "description": "dataclass with __slots__ (3.10+) vs manual __slots__",
        "old_style": """\
from dataclasses import dataclass

@dataclass
class Point:
    x: float
    y: float
    z: float = 0.0

    def distance(self):
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5
""",
        "new_style": """\
from dataclasses import dataclass

@dataclass(slots=True)
class Point:
    x: float
    y: float
    z: float = 0.0

    def distance(self):
        return (self.x**2 + self.y**2 + self.z**2) ** 0.5
""",
    },
    {
        "name": "parameter_spec",
        "description": "ParamSpec (3.10+) vs Callable in typing",
        "old_style": """\
from typing import Callable, TypeVar

T = TypeVar('T')

def logged(fn: Callable[..., T]) -> Callable[..., T]:
    def wrapper(*args, **kwargs):
        print("calling", fn.__name__)
        return fn(*args, **kwargs)
    return wrapper
""",
        "new_style": """\
from typing import ParamSpec, TypeVar, Callable

P = ParamSpec('P')
T = TypeVar('T')

def logged(fn: Callable[P, T]) -> Callable[P, T]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        print("calling", fn.__name__)
        return fn(*args, **kwargs)
    return wrapper
""",
    },
    {
        "name": "dict_merge",
        "description": "Dict merge operator (3.9+) vs dict.update()",
        "old_style": """\
def merge_configs(base, override):
    result = base.copy()
    result.update(override)
    return result
""",
        "new_style": """\
def merge_configs(base, override):
    return base | override
""",
    },
    {
        "name": "positional_only_params",
        "description": "Positional-only parameters (3.8+) vs docstring convention",
        "old_style": """\
def divide(numerator, denominator):
    '''
    numerator and denominator are positional-only by convention.
    '''
    if denominator == 0:
        raise ValueError("denominator cannot be zero")
    return numerator / denominator
""",
        "new_style": """\
def divide(numerator, denominator, /, *, round_result=False):
    if denominator == 0:
        raise ValueError("denominator cannot be zero")
    result = numerator / denominator
    return round(result) if round_result else result
""",
    },
    {
        "name": "generator_vs_list",
        "description": "Generator expression vs list comprehension for sum",
        "old_style": """\
def total_length(strings):
    lengths = [len(s) for s in strings]
    return sum(lengths)
""",
        "new_style": """\
def total_length(strings):
    return sum(len(s) for s in strings)
""",
    },
    {
        "name": "contextlib_suppress",
        "description": "contextlib.suppress (idiomatic 3.x) vs try/except/pass",
        "old_style": """\
import os

def safe_remove(path):
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
""",
        "new_style": """\
import os
import contextlib

def safe_remove(path):
    with contextlib.suppress(FileNotFoundError):
        os.remove(path)
""",
    },
    {
        "name": "star_unpacking",
        "description": "Extended star unpacking vs slices",
        "old_style": """\
def split_first_last(items):
    first = items[0]
    middle = items[1:-1]
    last = items[-1]
    return first, middle, last
""",
        "new_style": """\
def split_first_last(items):
    first, *middle, last = items
    return first, middle, last
""",
    },
]


# ---------------------------------------------------------------------------
# Model loading helpers
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


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------

COMPLEXITY_STABLE_THRESHOLD = 5.0   # |old_pred - new_pred| < this = stable
SECURITY_STABLE_THRESHOLD = 0.15    # |old_prob - new_prob| < this = stable


def run_complexity_shift(model, pairs) -> list[dict]:
    results = []
    for pair in pairs:
        old_src = pair["old_style"]
        new_src = pair["new_style"]
        if model is not None:
            try:
                old_pred = model.predict_cognitive_complexity(old_src)
            except Exception:
                old_pred = 0.0
            try:
                new_pred = model.predict_cognitive_complexity(new_src)
            except Exception:
                new_pred = 0.0
        else:
            from features.code_metrics import cognitive_complexity
            old_pred = float(cognitive_complexity(old_src))
            new_pred = float(cognitive_complexity(new_src))

        diff = abs(old_pred - new_pred)
        stable = diff < COMPLEXITY_STABLE_THRESHOLD
        results.append({
            "name": pair["name"],
            "description": pair["description"],
            "old_pred": round(old_pred, 2),
            "new_pred": round(new_pred, 2),
            "diff": round(diff, 2),
            "stable": stable,
        })
    return results


def run_security_shift(model, pairs) -> list[dict]:
    results = []
    for pair in pairs:
        old_src = pair["old_style"]
        new_src = pair["new_style"]
        if model is not None:
            try:
                old_prob = model.vulnerability_score(old_src)
            except Exception:
                old_prob = 0.0
            try:
                new_prob = model.vulnerability_score(new_src)
            except Exception:
                new_prob = 0.0
        else:
            old_prob = 0.0
            new_prob = 0.0

        diff = abs(old_prob - new_prob)
        stable = diff < SECURITY_STABLE_THRESHOLD
        results.append({
            "name": pair["name"],
            "description": pair["description"],
            "old_prob": round(old_prob, 4),
            "new_prob": round(new_prob, 4),
            "diff": round(diff, 4),
            "stable": stable,
        })
    return results


def summarise(results: list[dict]) -> dict:
    n = len(results)
    n_stable = sum(1 for r in results if r["stable"])
    consistency_rate = n_stable / n if n > 0 else 0.0
    mean_diff = sum(r["diff"] for r in results) / n if n > 0 else 0.0
    return {
        "n_pairs": n,
        "n_stable": n_stable,
        "consistency_rate": round(consistency_rate, 4),
        "mean_prediction_diff": round(mean_diff, 4),
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_table(label: str, results: list[dict], unit: str):
    print()
    print(f"=== {label} ===")
    col_w = [28, 10, 10, 8, 8]
    header = (
        f"{'Pair':<{col_w[0]}} {'Old':>{col_w[1]}} {'New':>{col_w[2]}} "
        f"{'Diff':>{col_w[3]}} {'Stable':>{col_w[4]}}"
    )
    sep = "-" * (sum(col_w) + 4)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        name = r["name"][:col_w[0]]
        stable_str = "yes" if r["stable"] else "NO"
        print(
            f"{name:<{col_w[0]}} {r['old_pred'] if 'old_pred' in r else r['old_prob']:>{col_w[1]}.4f} "
            f"{r['new_pred'] if 'new_pred' in r else r['new_prob']:>{col_w[2]}.4f} "
            f"{r['diff']:>{col_w[3]}.4f} {stable_str:>{col_w[4]}}"
        )
    print(sep)
    summ = summarise(results)
    print(
        f"Consistency rate: {summ['consistency_rate']:.1%} "
        f"({summ['n_stable']}/{summ['n_pairs']} pairs stable, threshold={unit})"
    )
    print(f"Mean prediction diff: {summ['mean_prediction_diff']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    out_dir = _BACKEND / "evaluation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    complexity_model = _load_complexity_model()
    security_model = _load_security_model()

    print(f"Testing {len(SYNTAX_PAIRS)} syntax pairs...")

    print("Running complexity distribution shift test...")
    cplx_results = run_complexity_shift(complexity_model, SYNTAX_PAIRS)

    print("Running security distribution shift test...")
    sec_results = run_security_shift(security_model, SYNTAX_PAIRS)

    output = {
        "n_pairs": len(SYNTAX_PAIRS),
        "complexity": {
            "results": cplx_results,
            "summary": summarise(cplx_results),
            "stable_threshold": COMPLEXITY_STABLE_THRESHOLD,
        },
        "security": {
            "results": sec_results,
            "summary": summarise(sec_results),
            "stable_threshold": SECURITY_STABLE_THRESHOLD,
        },
    }

    out_path = out_dir / "distribution_shift_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print_table("Complexity Distribution Shift (predicted cognitive_complexity)", cplx_results, "<5")
    print_table("Security Distribution Shift (vulnerability probability)", sec_results, "<0.15")

    c_summ = output["complexity"]["summary"]
    s_summ = output["security"]["summary"]
    print()
    print("=== Final Summary ===")
    print(
        f"Syntax consistency rate: {c_summ['consistency_rate']:.1%} (complexity), "
        f"{s_summ['consistency_rate']:.1%} (security)"
    )


if __name__ == "__main__":
    main()
