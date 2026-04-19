"""
Refactoring Suggester Spot-Check Evaluation
=============================================
Validates the AST-based refactoring suggester against hand-labelled Python
snippets. Each snippet is annotated with the expected refactoring type(s) it
should trigger, or an empty set for clean code.

Thresholds used by the detector (from RefactoringSuggester constants):
  extract_method             -- function body > 40 lines
  introduce_parameter_object -- function has > 4 parameters
  reduce_nesting             -- nesting depth > 3 levels
  magic_number (extract_const)-- >= 3 magic numeric literals in a function
  introduce_explaining_variable -- boolean condition with >= 3 boolean operators

Outputs:
    evaluation/results/refactoring_spotcheck.json
    Printed: per-rule precision/recall/F1 + overall
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Build a 42-line function body for the long-method test
_LONG_BODY = "\n".join(
    ["def process_records(records):"]
    + [f"    item_{i} = records.get('field_{i}', None)" for i in range(41)]
    + ["    return records"]
)

TEST_CASES: list[tuple[str, set[str], str]] = [
    # ── extract_method (long function > 40 lines) ────────────────────────────
    (
        _LONG_BODY,
        {"extract_method"},
        "FLAWED: 42-line function body triggers extract_method"
    ),
    (
        "def short(x):\n    return x + 1\n",
        set(),
        "SAFE: 2-line function is fine"
    ),
    (
        "def medium_function(data):\n    result = []\n    for item in data:\n        result.append(item * 2)\n    return result\n",
        set(),
        "SAFE: 5-line function, well under threshold"
    ),

    # ── introduce_parameter_object (> 4 params) ───────────────────────────────
    (
        "def configure(host, port, timeout, retries, ssl):\n    pass\n",
        {"introduce_parameter_object"},
        "FLAWED: 5 parameters (> 4)"
    ),
    (
        "def connect_server(host, port, timeout, retries, ssl, verify, proxy):\n    pass\n",
        {"introduce_parameter_object"},
        "FLAWED: 7 parameters"
    ),
    (
        "def connect(host, port, timeout):\n    pass\n",
        set(),
        "SAFE: 3 parameters (at or below threshold)"
    ),
    (
        "def create(name, value, flag, mode):\n    pass\n",
        set(),
        "SAFE: exactly 4 parameters (threshold is > 4)"
    ),

    # ── reduce_nesting (depth > 3 levels) ────────────────────────────────────
    (
        """def process(data):
    if data:
        for item in data:
            if item > 0:
                for sub in item:
                    return sub
    return None
""",
        {"reduce_nesting"},
        "FLAWED: 4-level nesting (if/for/if/for)"
    ),
    (
        """def find(data, target):
    if data:
        for item in data:
            if item == target:
                return item
    return None
""",
        set(),
        "SAFE: 3-level nesting (if/for/if), at threshold not above"
    ),

    # ── introduce_explaining_variable (>= 3 boolean ops) ──────────────────────
    (
        """def is_valid(a, b, c, d):
    if a > 0 and b > 0 and c > 0 and d > 0:
        return True
    return False
""",
        {"introduce_explaining_variable"},
        "FLAWED: 3 'and' operators in one condition"
    ),
    (
        """def check(x, y):
    if x > 0 and y > 0:
        return True
    return False
""",
        set(),
        "SAFE: only 1 boolean operator"
    ),

    # ── extract_constant / magic numbers (>= 3 magic numbers) ────────────────
    (
        """def compute_metrics(value):
    score = value * 1.08
    adjusted = score + 42.5
    final = adjusted / 3.14
    return final
""",
        {"extract_constant"},
        "FLAWED: 3 magic numbers (1.08, 42.5, 3.14)"
    ),
    (
        """def price_with_tax(price):
    TAX_RATE = 0.08
    return price * (1 + TAX_RATE)
""",
        set(),
        "SAFE: only one literal, named constant pattern"
    ),
    (
        """def increment(x):
    return x + 1
""",
        set(),
        "SAFE: literal 1 is not counted as magic number"
    ),

    # ── Clean code (no issues expected) ──────────────────────────────────────
    (
        "def add(a, b):\n    return a + b\n",
        set(),
        "SAFE: minimal clean function"
    ),
    (
        "def greet(name: str) -> str:\n    return f'Hello, {name}!'\n",
        set(),
        "SAFE: one-liner with type hints"
    ),
    (
        """class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1

    def reset(self):
        self.count = 0
""",
        set(),
        "SAFE: clean class, short methods"
    ),
]


def _detect(snippet: str) -> set[str]:
    from models.refactoring_suggester import RefactoringSuggester
    result = RefactoringSuggester().analyze(snippet)
    return {s.refactoring_type for s in result.suggestions}


def evaluate() -> dict:
    rule_tp: dict[str, int] = {}
    rule_fp: dict[str, int] = {}
    rule_fn: dict[str, int] = {}
    total_tp = total_fp = total_fn = 0
    case_results = []

    for snippet, expected, desc in TEST_CASES:
        detected = _detect(snippet)
        tp_rules = detected & expected
        fp_rules = detected - expected
        fn_rules = expected - detected

        for r in tp_rules:
            rule_tp[r] = rule_tp.get(r, 0) + 1
            total_tp += 1
        for r in fp_rules:
            rule_fp[r] = rule_fp.get(r, 0) + 1
            total_fp += 1
        for r in fn_rules:
            rule_fn[r] = rule_fn.get(r, 0) + 1
            total_fn += 1

        if fn_rules:
            outcome = "FN"
        elif fp_rules:
            outcome = "FP"
        elif tp_rules:
            outcome = "TP"
        else:
            outcome = "TN"

        case_results.append({
            "desc": desc, "expected": sorted(expected),
            "detected": sorted(detected), "outcome": outcome,
        })

    all_rules = sorted(set(list(rule_tp) + list(rule_fn)))
    per_rule: dict[str, dict] = {}
    for rule in all_rules:
        tp = rule_tp.get(rule, 0)
        fp = rule_fp.get(rule, 0)
        fn = rule_fn.get(rule, 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_rule[rule] = {
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn,
        }

    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1   = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
    macro_f1   = sum(v["f1"] for v in per_rule.values()) / len(per_rule) if per_rule else 0.0
    total_clean = sum(1 for _, exp, _ in TEST_CASES if not exp)
    fp_cases    = sum(1 for r in case_results if r["outcome"] == "FP")
    fp_rate     = fp_cases / total_clean if total_clean > 0 else 0.0

    return {
        "dataset": {
            "total_cases": len(TEST_CASES),
            "flawed": sum(1 for _, exp, _ in TEST_CASES if exp),
            "clean": total_clean,
            "rules_covered": len(all_rules),
        },
        "overall": {
            "micro_precision": round(micro_prec, 4),
            "micro_recall": round(micro_rec, 4),
            "micro_f1": round(micro_f1, 4),
            "macro_f1": round(macro_f1, 4),
            "false_positive_rate": round(fp_rate, 4),
            "tp": total_tp, "fp": total_fp, "fn": total_fn,
        },
        "per_rule": per_rule,
        "case_results": case_results,
    }


def _print_report(ev: dict) -> None:
    ds = ev["dataset"]
    ov = ev["overall"]
    print("=" * 60)
    print("Refactoring Suggester Spot-Check")
    print("=" * 60)
    print(f"Cases: {ds['total_cases']}  ({ds['flawed']} flawed, {ds['clean']} clean)")
    print(f"Rules covered: {ds['rules_covered']}")
    print(f"Micro F1  : {ov['micro_f1']:.4f}  FP rate: {ov['false_positive_rate']:.4f}")
    print(f"Macro F1  : {ov['macro_f1']:.4f}")
    print()
    print(f"{'Rule':<32}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'TP':>3}  {'FP':>3}  {'FN':>3}")
    print("-" * 70)
    for rule, m in sorted(ev["per_rule"].items()):
        print(f"{rule:<32}  {m['precision']:6.4f}  {m['recall']:6.4f}  {m['f1']:6.4f}"
              f"  {m['tp']:3d}  {m['fp']:3d}  {m['fn']:3d}")
    fns = [r for r in ev["case_results"] if r["outcome"] == "FN"]
    fps = [r for r in ev["case_results"] if r["outcome"] == "FP"]
    if fns:
        print(f"\nFalse negatives ({len(fns)}):")
        for r in fns: print(f"  missed {r['expected']} -- {r['desc']}")
    if fps:
        print(f"\nFalse positives ({len(fps)}):")
        for r in fps: print(f"  spurious {r['detected']} -- {r['desc']}")


def main():
    ev = evaluate()
    _print_report(ev)
    out_path = Path(__file__).parent / "results" / "refactoring_spotcheck.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(ev, f, indent=2)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
