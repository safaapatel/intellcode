"""
Documentation Quality Analyser Spot-Check Evaluation
======================================================
Validates the rule-based documentation quality analyser against hand-labelled
Python snippets. Each snippet has a known expected grade band.

Grade bands:
  A/B  -- well-documented (complete or near-complete docstrings)
  C/D  -- partially documented (missing params/returns sections)
  F    -- undocumented (missing docstring entirely)

The analyser uses these issue types:
  missing                 -- function/class has no docstring at all
  missing_params_section  -- docstring exists but params not documented
  missing_returns_section -- docstring exists but return not documented
  short_description       -- docstring is too brief

Outputs:
    evaluation/results/doc_quality_spotcheck.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_GRADE_TO_BAND = {
    "A": "documented", "B": "documented",
    "C": "partial", "D": "partial",
    "F": "undocumented",
}

TEST_CASES: list[tuple[str, str, str]] = [
    # ── documented (A or B) ─────────────────────────────────────────────────
    (
        '''def calculate_average(numbers: list) -> float:
    """Calculate the arithmetic mean of a list of numbers.

    Args:
        numbers: A list of numeric values.

    Returns:
        float: The arithmetic mean.
    """
    return sum(numbers) / len(numbers)
''',
        "documented", "Full docstring with summary, Args and Returns"
    ),
    (
        '''def is_valid_email(address: str) -> bool:
    """Check whether an email address has a valid basic format.

    Args:
        address: The email string to validate.

    Returns:
        bool: True if the format is valid, False otherwise.
    """
    return "@" in address and "." in address.split("@")[-1]
''',
        "documented", "Full docstring with all sections"
    ),
    (
        '''def connect(host: str, port: int) -> None:
    """Open a TCP connection to the specified host and port.

    Args:
        host: Hostname or IP address.
        port: Port number (1-65535).
    """
    pass
''',
        "documented", "Docstring with Args (no return for None function)"
    ),
    (
        '''class BankAccount:
    """A simple bank account with deposit and withdraw operations."""

    def __init__(self, owner: str, balance: float = 0.0):
        """Initialise account.

        Args:
            owner: Account holder name.
            balance: Initial balance (default 0.0).
        """
        self.owner = owner
        self.balance = balance
''',
        "documented", "Class and __init__ both documented"
    ),

    # ── partial (C or D) ────────────────────────────────────────────────────
    (
        '''def load_data(path: str) -> list:
    """Load records from a file."""
    with open(path) as f:
        return [line.strip() for line in f]
''',
        "partial", "Docstring exists but missing Args and Returns"
    ),
    (
        '''def process(items, threshold=0.5):
    """Filter items above threshold and return results."""
    return [x for x in items if x > threshold]
''',
        "partial", "Docstring exists but missing Args section"
    ),
    (
        '''def transform(data: list) -> dict:
    """Transform input list into a frequency map."""
    result = {}
    for item in data:
        result[item] = result.get(item, 0) + 1
    return result
''',
        "partial", "Docstring exists but missing Args and Returns sections"
    ),

    # ── undocumented (F) ────────────────────────────────────────────────────
    (
        '''def add(a, b):
    return a + b
''',
        "undocumented", "No docstring at all"
    ),
    (
        '''def find_user(user_id, db):
    result = db.query(user_id)
    if not result:
        return None
    return result[0]
''',
        "undocumented", "No docstring on function with parameters"
    ),
    (
        '''class DataProcessor:
    def __init__(self):
        self.data = []

    def add(self, item):
        self.data.append(item)

    def clear(self):
        self.data = []
''',
        "undocumented", "Class and all methods missing docstrings"
    ),
    (
        '''def configure(host, port, timeout, retries):
    sock = socket.create_connection((host, port), timeout)
    sock.settimeout(timeout)
    return sock
''',
        "undocumented", "4-param function with no docstring"
    ),
    (
        '''def serialize(obj, format="json"):
    if format == "json":
        return json.dumps(obj)
    elif format == "xml":
        return to_xml(obj)
    return str(obj)
''',
        "undocumented", "Multi-branch function with no docstring"
    ),
]


def _classify(snippet: str) -> str:
    from models.doc_quality_analyzer import DocQualityAnalyzer
    result = DocQualityAnalyzer().analyze(snippet)
    return _GRADE_TO_BAND.get(result.grade, "partial")


def evaluate() -> dict:
    band_tp: dict[str, int] = {}
    band_fp: dict[str, int] = {}
    band_fn: dict[str, int] = {}
    total_correct = 0
    case_results = []

    for snippet, expected, desc in TEST_CASES:
        predicted = _classify(snippet)
        correct = predicted == expected
        if correct:
            band_tp[expected] = band_tp.get(expected, 0) + 1
            total_correct += 1
        else:
            band_fn[expected] = band_fn.get(expected, 0) + 1
            band_fp[predicted] = band_fp.get(predicted, 0) + 1
        case_results.append({
            "desc": desc, "expected": expected,
            "predicted": predicted, "correct": correct,
        })

    all_bands = ["documented", "partial", "undocumented"]
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

    accuracy = total_correct / len(TEST_CASES)
    macro_f1 = sum(v["f1"] for v in per_band.values()) / len(per_band)

    return {
        "dataset": {
            "total_cases": len(TEST_CASES),
            "by_band": {b: sum(1 for _, eb, _ in TEST_CASES if eb == b) for b in all_bands},
        },
        "overall": {
            "accuracy": round(accuracy, 4),
            "macro_f1": round(macro_f1, 4),
        },
        "per_band": per_band,
        "case_results": case_results,
    }


def _print_report(ev: dict) -> None:
    ds = ev["dataset"]
    ov = ev["overall"]
    print("=" * 60)
    print("Documentation Quality Spot-Check")
    print("=" * 60)
    print(f"Cases: {ds['total_cases']}  {ds['by_band']}")
    print(f"Accuracy : {ov['accuracy']:.4f}  Macro F1: {ov['macro_f1']:.4f}")
    print()
    print(f"{'Band':<14}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'TP':>3}  {'FP':>3}  {'FN':>3}")
    print("-" * 55)
    for band, m in ev["per_band"].items():
        print(f"{band:<14}  {m['precision']:6.4f}  {m['recall']:6.4f}  {m['f1']:6.4f}"
              f"  {m['tp']:3d}  {m['fp']:3d}  {m['fn']:3d}")
    wrong = [r for r in ev["case_results"] if not r["correct"]]
    if wrong:
        print(f"\nMisclassified ({len(wrong)}):")
        for r in wrong:
            print(f"  expected={r['expected']} got={r['predicted']} -- {r['desc']}")


def main():
    ev = evaluate()
    _print_report(ev)
    out_path = Path(__file__).parent / "results" / "doc_quality_spotcheck.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(ev, f, indent=2)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
