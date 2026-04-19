"""
Readability Scorer Spot-Check Evaluation
==========================================
Validates the rule-based readability scorer against hand-labelled Python snippets.
Each snippet has a known expected grade band (high/medium/low) based on the
Buse & Weimer (2010) readability criteria adapted for Python:
  - Identifier clarity (meaningful names, consistent style)
  - Comment density proportional to complexity
  - Structural clarity (short functions, low nesting, short lines)
  - Cognitive load (branches + loops + function calls per SLOC)

Test design: 25 cases across three grade bands.
  - High readability (expected grade A or B): 10 cases
  - Medium readability (expected grade C): 7 cases
  - Low readability (expected grade D or F): 8 cases

Outputs:
    evaluation/results/readability_spotcheck.json
    Printed: per-band precision/recall/F1 + overall
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Test cases: (snippet, expected_band, description)
# expected_band: "high" | "medium" | "low"
# ---------------------------------------------------------------------------

TEST_CASES: list[tuple[str, str, str]] = [
    # ── High readability (A or B) ────────────────────────────────────────────
    (
        """def calculate_average(numbers: list) -> float:
    \"\"\"Return the arithmetic mean of a non-empty list.\"\"\"
    if not numbers:
        raise ValueError("Cannot average empty list")
    return sum(numbers) / len(numbers)
""",
        "high", "Clean: docstring, guard clause, meaningful names"
    ),
    (
        """def is_valid_email(address: str) -> bool:
    \"\"\"Check basic email format: contains @ and a dot after it.\"\"\"
    at_index = address.find('@')
    if at_index < 1:
        return False
    domain = address[at_index + 1:]
    return '.' in domain and len(domain) > 2
""",
        "high", "Clean: clear naming, simple logic, docstring"
    ),
    (
        """def load_config(path: str) -> dict:
    \"\"\"Load JSON configuration from disk.\"\"\"
    with open(path, encoding='utf-8') as config_file:
        return json.load(config_file)
""",
        "high", "Clean: short, one responsibility, descriptive parameter"
    ),
    (
        """def format_duration(seconds: int) -> str:
    minutes, remaining_seconds = divmod(seconds, 60)
    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours:02d}:{remaining_minutes:02d}:{remaining_seconds:02d}"
""",
        "high", "Clean: meaningful intermediate variables, no magic numbers"
    ),
    (
        """class BankAccount:
    def __init__(self, owner: str, initial_balance: float = 0.0):
        self.owner = owner
        self.balance = initial_balance

    def deposit(self, amount: float) -> None:
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self.balance += amount

    def withdraw(self, amount: float) -> None:
        if amount > self.balance:
            raise ValueError("Insufficient funds")
        self.balance -= amount
""",
        "high", "Clean: consistent naming, guard clauses, docstrings implied"
    ),
    (
        """def find_duplicates(items: list) -> list:
    seen = set()
    duplicates = []
    for item in items:
        if item in seen:
            duplicates.append(item)
        else:
            seen.add(item)
    return duplicates
""",
        "high", "Clean: clear variable names, linear logic"
    ),
    (
        """def celsius_to_fahrenheit(celsius: float) -> float:
    return celsius * 9 / 5 + 32
""",
        "high", "Clean: minimal, self-documenting via naming"
    ),
    (
        """def count_words(text: str) -> dict:
    \"\"\"Count word frequency in text (case-insensitive).\"\"\"
    word_counts: dict[str, int] = {}
    for word in text.lower().split():
        word_counts[word] = word_counts.get(word, 0) + 1
    return word_counts
""",
        "high", "Clean: docstring, typed annotation, clear names"
    ),
    (
        """def get_first_n_primes(n: int) -> list[int]:
    primes = []
    candidate = 2
    while len(primes) < n:
        if all(candidate % p != 0 for p in primes):
            primes.append(candidate)
        candidate += 1
    return primes
""",
        "high", "Clean: meaningful names, no magic numbers"
    ),
    (
        """def safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator
""",
        "high", "Clean: guard clause, explicit None return, descriptive names"
    ),

    # ── Medium readability (C) ────────────────────────────────────────────────
    (
        """def proc(lst):
    r = []
    for i in range(len(lst)):
        if lst[i] > 0:
            r.append(lst[i] * 2)
    return r
""",
        "medium", "Medium: short names, index loop instead of iteration, no docstring"
    ),
    (
        """def check(x, y, z):
    if x > 0:
        if y > 0:
            if z > 0:
                return True
    return False
""",
        "medium", "Medium: nested ifs, vague parameter names, no docstring"
    ),
    (
        """def update_record(d, k, v, flag=False):
    if flag:
        d[k] = v
        return True
    elif k in d:
        d[k] = v
        return True
    return False
""",
        "medium", "Medium: unclear flag parameter, abbreviated names"
    ),
    (
        """def compute(data):
    total = 0
    count = 0
    for item in data:
        if item is not None:
            total += item
            count += 1
    if count == 0:
        return 0
    return total / count
""",
        "medium", "Medium: no docstring, vague function name, acceptable structure"
    ),
    (
        """def parse_line(line):
    parts = line.strip().split(',')
    if len(parts) < 3:
        return None
    name = parts[0].strip()
    age = int(parts[1].strip())
    score = float(parts[2].strip())
    return name, age, score
""",
        "medium", "Medium: no docstring, magic number 3, otherwise clear"
    ),
    (
        """def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result
""",
        "medium", "Medium: no docstring, recursive but readable, vague name"
    ),
    (
        """def match(a, b, thresh=0.8):
    score = 0
    for x, y in zip(a, b):
        if x == y:
            score += 1
    ratio = score / max(len(a), len(b), 1)
    return ratio >= thresh
""",
        "medium", "Medium: magic 0.8, short names, no type hints"
    ),

    # ── Low readability (D or F) ──────────────────────────────────────────────
    (
        """def f(x,y,z,w,v,u):
    if x>0:
        if y>0:
            if z>0:
                if w>0:
                    if v>0:
                        return x+y+z+w+v+u
    return 0
""",
        "low", "Poor: single-letter params, 5-level nesting, no spaces"
    ),
    (
        """def p(d,t=1,f=0,r=None):
    if r is None:r=[]
    for k in d:
        if t>0:
            if isinstance(d[k],dict):p(d[k],t-1,f,r)
            else:
                if f==0:r.append(d[k])
                elif f==1:r.append((k,d[k]))
    return r
""",
        "low", "Poor: compressed layout, single-letter vars, no docstring"
    ),
    (
        """def calc(a,b,c,d,e,f,g,h,i,j):
    return((a+b)*c-(d/e)*f+(g-h)*i)%j
""",
        "low", "Poor: 10 single-letter parameters, no semantics, no docstring"
    ),
    (
        """def x(n):
    if n<=1:return n
    return x(n-1)+x(n-2)
""",
        "low", "Poor: single-letter name for recursive function, compressed"
    ),
    (
        """def do_things(a, b, c, d, e):
    r1 = a * 2 + b - c
    r2 = r1 / d if d != 0 else 999
    r3 = r2 ** e
    r4 = r3 + 42
    r5 = r4 - 7
    r6 = r5 * 3.14
    r7 = r6 / 2
    r8 = r7 + r1
    r9 = r8 - r2
    return r9
""",
        "low", "Poor: magic numbers, meaningless intermediate vars, no docstring"
    ),
    (
        """def proc2(data,m,s,e,t,k,p,q):
    out=[]
    for i in range(0,len(data),s):
        c=data[i:i+m]
        if len(c)<m:break
        if t:c=[x*k for x in c]
        if p:c=[x+q for x in c]
        out.extend(c)
    if e:out=list(reversed(out))
    return out
""",
        "low", "Poor: 8 single-letter params, no docstring, compressed"
    ),
    (
        """def g(x,y):
    z=[]
    for i in x:
        for j in y:
            if i==j:z.append(i)
    return z
""",
        "low", "Poor: single-letter names throughout, compressed layout"
    ),
    (
        """def h(l,n=10,r=False,s=False,d=None):
    if d is None:d={}
    out=l[:n] if not r else l[-n:]
    if s:out=sorted(out)
    for i,v in enumerate(out):
        if v in d:out[i]=d[v]
    return out
""",
        "low", "Poor: abbreviated everything, flag parameters, no docstring"
    ),
]

_BAND_ORDER = {"high": 0, "medium": 1, "low": 2}
_GRADE_TO_BAND = {
    "A": "high", "B": "high",
    "C": "medium",
    "D": "low", "F": "low",
}


def _score(snippet: str) -> str:
    from models.readability_scorer import ReadabilityScorer
    result = ReadabilityScorer().score(snippet)
    return _GRADE_TO_BAND.get(result.grade, "medium")


def evaluate() -> dict:
    band_tp: dict[str, int] = {}
    band_fp: dict[str, int] = {}
    band_fn: dict[str, int] = {}
    total_tp = total_fp = total_fn = 0
    case_results = []

    for snippet, expected_band, desc in TEST_CASES:
        predicted_band = _score(snippet)
        if predicted_band == expected_band:
            band_tp[expected_band] = band_tp.get(expected_band, 0) + 1
            total_tp += 1
            outcome = "TP"
        else:
            band_fn[expected_band] = band_fn.get(expected_band, 0) + 1
            band_fp[predicted_band] = band_fp.get(predicted_band, 0) + 1
            total_fp += 1
            total_fn += 1
            outcome = "WRONG"
        case_results.append({
            "desc": desc,
            "expected": expected_band,
            "predicted": predicted_band,
            "outcome": outcome,
        })

    all_bands = ["high", "medium", "low"]
    per_band: dict[str, dict] = {}
    for band in all_bands:
        tp = band_tp.get(band, 0)
        fp = band_fp.get(band, 0)
        fn = band_fn.get(band, 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_band[band] = {
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn,
        }

    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1   = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0
    macro_f1   = sum(v["f1"] for v in per_band.values()) / len(per_band)
    accuracy   = total_tp / len(TEST_CASES)

    return {
        "dataset": {
            "total_cases": len(TEST_CASES),
            "by_band": {b: sum(1 for _, eb, _ in TEST_CASES if eb == b) for b in all_bands},
        },
        "overall": {
            "accuracy": round(accuracy, 4),
            "micro_precision": round(micro_prec, 4),
            "micro_recall": round(micro_rec, 4),
            "micro_f1": round(micro_f1, 4),
            "macro_f1": round(macro_f1, 4),
        },
        "per_band": per_band,
        "case_results": case_results,
    }


def _print_report(ev: dict) -> None:
    ds = ev["dataset"]
    ov = ev["overall"]
    print("=" * 60)
    print("Readability Scorer Spot-Check")
    print("=" * 60)
    print(f"Cases: {ds['total_cases']}  ({ds['by_band']})")
    print(f"Accuracy  : {ov['accuracy']:.4f}")
    print(f"Micro F1  : {ov['micro_f1']:.4f}")
    print(f"Macro F1  : {ov['macro_f1']:.4f}")
    print()
    print(f"{'Band':<8}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'TP':>3}  {'FP':>3}  {'FN':>3}")
    print("-" * 50)
    for band, m in ev["per_band"].items():
        print(f"{band:<8}  {m['precision']:6.4f}  {m['recall']:6.4f}  {m['f1']:6.4f}"
              f"  {m['tp']:3d}  {m['fp']:3d}  {m['fn']:3d}")
    wrong = [r for r in ev["case_results"] if r["outcome"] == "WRONG"]
    if wrong:
        print(f"\nMisclassified ({len(wrong)}):")
        for r in wrong:
            print(f"  expected={r['expected']} got={r['predicted']} -- {r['desc']}")


def main():
    ev = evaluate()
    _print_report(ev)
    out_path = Path(__file__).parent / "results" / "readability_spotcheck.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(ev, f, indent=2)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
