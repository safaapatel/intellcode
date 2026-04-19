"""
Code Clone Detection Spot-Check Evaluation
==========================================
Validates the TF-IDF + FAISS-based code clone detector against hand-labelled
Python snippets. Each case is labelled as either containing a clone pair or
being clone-free, based on the BigCloneBench clone taxonomy:

  Type-1  Exact clones (identical after whitespace normalisation)
  Type-2  Renamed clones (same structure, different identifiers)
  Type-3  Near-miss clones (structurally similar with modifications)

The detector reports clone_rate > 0 when at least one clone pair is found.
A snippet is classified as "cloned" if clone_rate > 0, "clean" otherwise.

Outputs:
    evaluation/results/clone_detection_spotcheck.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_type2(func_a: str, func_b: str) -> str:
    return func_a + "\n\n" + func_b


# ---------------------------------------------------------------------------
# Test cases: (snippet, has_clone: bool, clone_type_or_none, description)
# ---------------------------------------------------------------------------

TEST_CASES: list[tuple[str, bool, str, str]] = [
    # ── Type-2 clones (renamed identifiers, same structure) ──────────────────
    (
        _make_type2(
            """def process_orders(orders):
    results = []
    for order in orders:
        if order.status == 'pending':
            total = 0
            for item in order.items:
                total += item.quantity * item.price
            results.append({'id': order.id, 'total': total})
    return results""",
            """def process_invoices(invoices):
    results = []
    for invoice in invoices:
        if invoice.status == 'pending':
            total = 0
            for item in invoice.items:
                total += item.quantity * item.price
            results.append({'id': invoice.id, 'total': total})
    return results"""
        ),
        True, "type2", "T2: order vs invoice processing — identical structure, renamed vars"
    ),
    (
        _make_type2(
            """def validate_user(user):
    if user is None:
        return False
    if not user.email:
        return False
    if not user.name or len(user.name) < 2:
        return False
    return True""",
            """def validate_product(product):
    if product is None:
        return False
    if not product.sku:
        return False
    if not product.name or len(product.name) < 2:
        return False
    return True"""
        ),
        True, "type2", "T2: validate_user vs validate_product — renamed fields"
    ),
    (
        _make_type2(
            """def calculate_statistics(values):
    if not values:
        return None
    total = sum(values)
    count = len(values)
    mean = total / count
    variance = sum((x - mean) ** 2 for x in values) / count
    return {'mean': mean, 'variance': variance, 'count': count}""",
            """def compute_metrics(measurements):
    if not measurements:
        return None
    total = sum(measurements)
    count = len(measurements)
    average = total / count
    spread = sum((x - average) ** 2 for x in measurements) / count
    return {'mean': average, 'variance': spread, 'count': count}"""
        ),
        True, "type2", "T2: statistics vs metrics — renamed variables throughout"
    ),

    # ── Type-3 clones (near-miss: extra statements or minor structural changes)
    (
        _make_type2(
            """def send_email_notification(user, message):
    if not user.email:
        return False
    subject = f'Notification for {user.name}'
    body = message
    send_email(user.email, subject, body)
    return True""",
            """def send_sms_notification(user, message):
    if not user.phone:
        return False
    subject = f'SMS for {user.name}'
    body = message[:160]
    log_notification(user.id, 'sms')
    send_sms(user.phone, body)
    return True"""
        ),
        True, "type3", "T3: email vs SMS notification — similar flow, added log + truncation"
    ),

    # ── Clean code (no clones) ────────────────────────────────────────────────
    (
        """def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
""",
        False, None, "SAFE: two unrelated trivial functions"
    ),
    (
        """def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def save_config(config: dict, path: str) -> None:
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)
""",
        False, None, "SAFE: load and save are complementary, not clones"
    ),
    (
        """def format_date(dt):
    return dt.strftime('%Y-%m-%d')

def format_time(dt):
    return dt.strftime('%H:%M:%S')

def format_datetime(dt):
    return dt.strftime('%Y-%m-%d %H:%M:%S')
""",
        False, None, "SAFE: related but structurally distinct one-liners"
    ),
    (
        """def get_user_by_id(user_id: int):
    return db.query(User).filter(User.id == user_id).first()

def get_users_by_role(role: str):
    return db.query(User).filter(User.role == role).all()
""",
        False, None, "SAFE: similar ORM queries but different return type and filter"
    ),
    (
        """class FileReader:
    def __init__(self, path):
        self.path = path

    def read(self):
        with open(self.path) as f:
            return f.read()

class DatabaseReader:
    def __init__(self, connection_string):
        self.connection_string = connection_string

    def read(self, query):
        conn = connect(self.connection_string)
        return conn.execute(query).fetchall()
""",
        False, None, "SAFE: structurally dissimilar classes with same method name"
    ),
]


def _has_clone(snippet: str) -> bool:
    from models.code_clone_detection import CodeCloneDetector
    result = CodeCloneDetector().detect(snippet)
    return result.clone_rate > 0


def evaluate() -> dict:
    tp = fp = tn = fn = 0
    case_results = []

    for snippet, expected_clone, clone_type, desc in TEST_CASES:
        detected = _has_clone(snippet)
        if expected_clone and detected:
            tp += 1; outcome = "TP"
        elif expected_clone and not detected:
            fn += 1; outcome = "FN"
        elif not expected_clone and detected:
            fp += 1; outcome = "FP"
        else:
            tn += 1; outcome = "TN"

        case_results.append({
            "desc": desc, "expected_clone": expected_clone,
            "clone_type": clone_type, "detected": detected, "outcome": outcome,
        })

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc  = (tp + tn) / len(TEST_CASES)
    total_clean = sum(1 for _, ec, _, _ in TEST_CASES if not ec)
    fp_rate = fp / total_clean if total_clean > 0 else 0.0

    return {
        "dataset": {
            "total_cases": len(TEST_CASES),
            "cloned": sum(1 for _, ec, _, _ in TEST_CASES if ec),
            "clean": total_clean,
        },
        "overall": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "false_positive_rate": round(fp_rate, 4),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        },
        "case_results": case_results,
    }


def _print_report(ev: dict) -> None:
    ds = ev["dataset"]
    ov = ev["overall"]
    print("=" * 60)
    print("Clone Detection Spot-Check")
    print("=" * 60)
    print(f"Cases: {ds['total_cases']}  ({ds['cloned']} cloned, {ds['clean']} clean)")
    print(f"Accuracy  : {ov['accuracy']:.4f}")
    print(f"Precision : {ov['precision']:.4f}  Recall: {ov['recall']:.4f}  F1: {ov['f1']:.4f}")
    print(f"FP rate   : {ov['false_positive_rate']:.4f}")
    print(f"TP={ov['tp']} FP={ov['fp']} TN={ov['tn']} FN={ov['fn']}")
    wrong = [r for r in ev["case_results"] if r["outcome"] in ("FP", "FN")]
    if wrong:
        print(f"\nErrors ({len(wrong)}):")
        for r in wrong:
            print(f"  [{r['outcome']}] {r['desc']}")


def main():
    ev = evaluate()
    _print_report(ev)
    out_path = Path(__file__).parent / "results" / "clone_detection_spotcheck.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(ev, f, indent=2)
    print(f"\nResults saved -> {out_path}")


if __name__ == "__main__":
    main()
