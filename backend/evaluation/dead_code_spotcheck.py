"""
Dead Code Detector Spot-Check Evaluation
=========================================
Validates the rule-based dead code detector against a hand-annotated set of
Python snippets.  Each snippet has a known ground-truth set of expected
issue_type strings (or an empty set for clean code).

This script covers the 8 deterministic detection rules:
  1. unreachable_code      -- code after return/raise/break/continue
  2. unused_function       -- function defined but never called in file
  3. unused_class          -- class defined but never instantiated/referenced
  4. unused_import         -- import never referenced
  5. unused_variable       -- variable assigned but never read
  6. unused_parameter      -- function parameter never used
  7. empty_except          -- bare except: pass (swallows errors)
  8. redundant_else        -- else after return

Outputs:
    evaluation/results/dead_code_spotcheck.json
    Printed: per-rule precision/recall/F1 + overall
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Labelled test cases
# Each entry: (python_snippet, set_of_expected_issue_types, description)
# Empty set -> clean (no issues expected)
# ---------------------------------------------------------------------------

TEST_CASES: list[tuple[str, set[str], str]] = [
    # ── unreachable_code ────────────────────────────────────────────────────
    # Functions have callers to prevent unused_function FP
    (
        "def f():\n    return 1\n    x = 2\n\nf()\n",
        {"unreachable_code"},
        "FLAWED: code after return",
    ),
    (
        "def f(x):\n    if x:\n        raise ValueError()\n        do_work()\n\nf(True)\n",
        {"unreachable_code"},
        "FLAWED: code after raise",
    ),
    (
        "def f():\n    for i in range(10):\n        break\n        print(i)\n\nf()\n",
        {"unreachable_code"},
        "FLAWED: code after break",
    ),
    (
        "def f():\n    return 42\n\nf()\n",
        set(),
        "SAFE: clean return, no trailing code",
    ),
    (
        "def f():\n    x = 1\n    return x\n\nf()\n",
        set(),
        "SAFE: variable used before return",
    ),

    # ── unused_function ─────────────────────────────────────────────────────
    (
        "def helper():\n    pass\n\ndef main():\n    pass\n\nmain()\n",
        {"unused_function"},
        "FLAWED: helper defined but never called",
    ),
    (
        "def process():\n    pass\n\ndef main():\n    process()\n\nmain()\n",
        set(),
        "SAFE: process() is called by main",
    ),

    # ── unused_import ───────────────────────────────────────────────────────
    (
        "import os\nimport sys\n\nprint(sys.argv)\n",
        {"unused_import"},
        "FLAWED: os imported but not used",
    ),
    (
        "import os\nimport sys\n\nprint(os.getcwd(), sys.argv)\n",
        set(),
        "SAFE: both imports used",
    ),
    (
        "from pathlib import Path\n\ndef f():\n    return Path('.')\n\nf()\n",
        set(),
        "SAFE: Path used in function",
    ),

    # ── unused_variable ─────────────────────────────────────────────────────
    (
        "def f():\n    x = 10\n    return 42\n\nf()\n",
        {"unused_variable"},
        "FLAWED: x assigned but never read",
    ),
    (
        "def f():\n    x = 10\n    return x\n\nf()\n",
        set(),
        "SAFE: x is returned",
    ),
    (
        "def f():\n    total = 0\n    for i in range(5):\n        total += i\n    return total\n\nf()\n",
        set(),
        "SAFE: total accumulated and returned",
    ),

    # ── unused_parameter ────────────────────────────────────────────────────
    # Must include a caller so the detector has call-graph information
    (
        "def greet(name, greeting):\n    return 'Hello'\n\ngreet('Alice', 'Hi')\n",
        {"unused_parameter"},
        "FLAWED: both parameters unused",
    ),
    (
        "def greet(name):\n    return f'Hello, {name}'\n\ngreet('Alice')\n",
        set(),
        "SAFE: name is used",
    ),
    (
        "def greet(name, greeting='Hi'):\n    return f'{greeting}, {name}'\n\ngreet('Bob')\n",
        set(),
        "SAFE: both params used",
    ),

    # ── empty_except ────────────────────────────────────────────────────────
    # bare_except produces both bare_except and empty_except issue types
    (
        "try:\n    risky()\nexcept:\n    pass\n",
        {"empty_except", "bare_except"},
        "FLAWED: bare except swallows errors",
    ),
    (
        "try:\n    risky()\nexcept Exception:\n    pass\n",
        {"empty_except"},
        "FLAWED: except Exception: pass also empty",
    ),
    (
        "try:\n    risky()\nexcept Exception as e:\n    print(e)\n",
        set(),
        "SAFE: exception handled",
    ),
    (
        "try:\n    x = int(s)\nexcept ValueError:\n    x = 0\n",
        set(),
        "SAFE: except has fallback logic",
    ),

    # ── redundant_else ──────────────────────────────────────────────────────
    (
        "def f(x):\n    if x > 0:\n        return 'pos'\n    else:\n        return 'neg'\n\nf(1)\n",
        {"redundant_else"},
        "FLAWED: else after return is redundant",
    ),
    (
        "def f(x):\n    if x > 0:\n        return 'pos'\n    return 'neg'\n\nf(1)\n",
        set(),
        "SAFE: no else, direct fallthrough",
    ),

    # ── Clean code: no issues expected ──────────────────────────────────────
    (
        "def add(a, b):\n    return a + b\n\nresult = add(1, 2)\n",
        set(),
        "SAFE: simple clean function",
    ),
    (
        "import json\n\ndef load(path):\n    with open(path) as f:\n        return json.load(f)\n\nload('x.json')\n",
        set(),
        "SAFE: import used in function",
    ),
    (
        "class Counter:\n    def __init__(self):\n        self.n = 0\n    def inc(self):\n        self.n += 1\n\nc = Counter()\nc.inc()\n",
        set(),
        "SAFE: class instantiated and used",
    ),
]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _detect(snippet: str) -> set[str]:
    """Run the dead code detector and return the set of detected issue types."""
    from models.dead_code_detector import DeadCodeDetector
    result = DeadCodeDetector().detect(snippet)
    return {issue.issue_type for issue in result.issues}


def evaluate() -> dict:
    """Evaluate all test cases and return precision/recall/F1 per rule."""
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
            "desc": desc,
            "expected": sorted(expected),
            "detected": sorted(detected),
            "tp": sorted(tp_rules),
            "fp": sorted(fp_rules),
            "fn": sorted(fn_rules),
            "outcome": outcome,
        })

    all_rules = sorted(set(list(rule_tp.keys()) + list(rule_fn.keys())))
    per_rule: dict[str, dict] = {}
    for rule in all_rules:
        tp = rule_tp.get(rule, 0)
        fp = rule_fp.get(rule, 0)
        fn = rule_fn.get(rule, 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_rule[rule] = {
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
            "tp": tp, "fp": fp, "fn": fn,
        }

    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1   = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    macro_prec = sum(v["precision"] for v in per_rule.values()) / len(per_rule) if per_rule else 0.0
    macro_rec  = sum(v["recall"]    for v in per_rule.values()) / len(per_rule) if per_rule else 0.0
    macro_f1   = sum(v["f1"]        for v in per_rule.values()) / len(per_rule) if per_rule else 0.0

    total_clean = sum(1 for _, exp, _ in TEST_CASES if not exp)
    total_vuln  = sum(1 for _, exp, _ in TEST_CASES if exp)
    fp_cases    = sum(1 for r in case_results if r["outcome"] == "FP")
    fp_rate     = fp_cases / total_clean if total_clean > 0 else 0.0

    return {
        "dataset": {
            "total_cases": len(TEST_CASES),
            "flawed": total_vuln,
            "clean": total_clean,
            "rules_covered": len(all_rules),
        },
        "overall": {
            "micro_precision": round(micro_prec, 4),
            "micro_recall":    round(micro_rec,  4),
            "micro_f1":        round(micro_f1,   4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall":    round(macro_rec,  4),
            "macro_f1":        round(macro_f1,   4),
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
    print("Dead Code Detector Spot-Check")
    print("=" * 60)
    print(f"Test cases  : {ds['total_cases']}  "
          f"({ds['flawed']} flawed, {ds['clean']} clean)")
    print(f"Rules covered: {ds['rules_covered']}")
    print()
    print("Overall (micro):")
    print(f"  Precision : {ov['micro_precision']:.4f}")
    print(f"  Recall    : {ov['micro_recall']:.4f}")
    print(f"  F1        : {ov['micro_f1']:.4f}")
    print(f"  FP rate   : {ov['false_positive_rate']:.4f}")
    print()
    print("Overall (macro across rules):")
    print(f"  Precision : {ov['macro_precision']:.4f}")
    print(f"  Recall    : {ov['macro_recall']:.4f}")
    print(f"  F1        : {ov['macro_f1']:.4f}")
    print()
    print(f"{'Rule':<22}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'TP':>3}  {'FP':>3}  {'FN':>3}")
    print("-" * 60)
    for rule, m in sorted(ev["per_rule"].items()):
        print(f"{rule:<22}  {m['precision']:6.4f}  {m['recall']:6.4f}  {m['f1']:6.4f}"
              f"  {m['tp']:3d}  {m['fp']:3d}  {m['fn']:3d}")
    print()
    fns = [r for r in ev["case_results"] if r["outcome"] == "FN"]
    fps = [r for r in ev["case_results"] if r["outcome"] == "FP"]
    if fns:
        print(f"False negatives ({len(fns)}):")
        for r in fns:
            print(f"  missed {r['fn']} -- {r['desc']}")
    if fps:
        print(f"\nFalse positives ({len(fps)}):")
        for r in fps:
            print(f"  spurious {r['fp']} -- {r['desc']}")
    print()


def main():
    ev = evaluate()
    _print_report(ev)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "dead_code_spotcheck.json"
    with open(out_path, "w") as f:
        json.dump(ev, f, indent=2)
    print(f"Results saved -> {out_path}")


if __name__ == "__main__":
    main()
