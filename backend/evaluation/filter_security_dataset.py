"""
Security Dataset Filter
========================
Removes toy-app and educational-example records from the security dataset.

Root cause of Bandit/Semgrep AUC ~0.5 anomaly (documented in _notes):
  Positive labels came from Bandit test cases (origin='bandit/*') which are
  deliberately-vulnerable EDUCATIONAL snippets.  Real SAST tools like Bandit
  and Semgrep detect REAL vulnerabilities in PRODUCTION code -- not the
  hand-crafted patterns in test fixtures.

This filter removes:
  1. Records with origin starting with 'bandit' (Bandit test-suite fixtures)
  2. Records with suspiciously short source (< 30 chars -- synthetic snippets)
  3. Records with origin matching known toy-app patterns

After filtering, retrain security models for honest cross-project evaluation.

Usage:
    python evaluation/filter_security_dataset.py \\
        --input  data/security_dataset.jsonl \\
        --output data/security_dataset_filtered.jsonl \\
        --report evaluation/results/dataset_filter_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Origins known to contain educational / toy-app records
TOY_ORIGINS = {
    "bandit",          # Bandit SAST test fixtures
    "dvwa",            # Damn Vulnerable Web App
    "juice-shop",      # OWASP Juice Shop
    "webgoat",         # WebGoat
    "hacme",           # HacMe Bank / Casino
    "hackthebox",      # HackTheBox challenges
    "vulnhub",         # VulnHub VMs
}

MIN_SOURCE_LENGTH = 30   # chars -- very short = synthetic snippet
MIN_SOURCE_LINES  = 3    # lines -- single-line = unlikely real production code


def _is_toy_origin(origin: str) -> bool:
    if not origin:
        return False
    parts = origin.lower().replace("\\", "/").split("/")
    root = parts[0] if parts else ""
    return root in TOY_ORIGINS or any(t in origin.lower() for t in TOY_ORIGINS)


def _is_too_short(source: str) -> bool:
    if len(source) < MIN_SOURCE_LENGTH:
        return True
    if source.count("\n") < MIN_SOURCE_LINES - 1:
        return True
    return False


def filter_records(
    records: list[dict],
    remove_toy: bool = True,
    remove_short: bool = True,
) -> tuple[list[dict], dict]:
    """
    Filter records and return (kept, report).
    """
    kept, removed_toy, removed_short = [], [], []

    for r in records:
        origin = str(r.get("origin", ""))
        source = str(r.get("source", ""))

        if remove_toy and _is_toy_origin(origin):
            removed_toy.append(r)
            continue
        if remove_short and _is_too_short(source):
            removed_short.append(r)
            continue
        kept.append(r)

    def _label_dist(recs):
        c = Counter(r.get("label", -1) for r in recs)
        return {"total": len(recs), "negative": c[0], "positive": c[1],
                "prevalence": round(c[1] / max(1, len(recs)), 4)}

    def _origin_breakdown(recs):
        origins = Counter()
        for r in recs:
            o = str(r.get("origin", "unknown"))
            key = o.split("/")[0] if "/" in o else o[:40]
            origins[key] += 1
        return dict(origins.most_common(20))

    report = {
        "before": _label_dist(records),
        "after":  _label_dist(kept),
        "removed_toy_app": _label_dist(removed_toy),
        "removed_too_short": _label_dist(removed_short),
        "origin_breakdown_before": _origin_breakdown(records),
        "origin_breakdown_after":  _origin_breakdown(kept),
        "toy_origins_removed": sorted(
            set(str(r.get("origin","")).split("/")[0] for r in removed_toy)
        ),
    }
    return kept, report


def main():
    parser = argparse.ArgumentParser(description="Filter security dataset")
    parser.add_argument("--input",  default="data/security_dataset.jsonl")
    parser.add_argument("--output", default="data/security_dataset_filtered.jsonl")
    parser.add_argument("--report", default="evaluation/results/dataset_filter_report.json")
    parser.add_argument("--keep-toy",   action="store_true", help="Skip toy-app filter")
    parser.add_argument("--keep-short", action="store_true", help="Skip short-source filter")
    args = parser.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)
    rep_path = Path(args.report)

    if not in_path.exists():
        print(f"ERROR: input not found: {in_path}")
        sys.exit(1)

    with open(in_path, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]

    kept, report = filter_records(
        records,
        remove_toy   = not args.keep_toy,
        remove_short = not args.keep_short,
    )

    # Save filtered dataset
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")

    # Save report
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Print summary (ASCII safe)
    b = report["before"]
    a = report["after"]
    rt = report["removed_toy_app"]
    rs = report["removed_too_short"]

    print("=" * 60)
    print("Security Dataset Filter Report")
    print("=" * 60)
    print(f"Before : {b['total']} records  pos={b['positive']}  neg={b['negative']}  "
          f"prev={b['prevalence']:.3f}")
    print(f"After  : {a['total']} records  pos={a['positive']}  neg={a['negative']}  "
          f"prev={a['prevalence']:.3f}")
    print(f"Removed (toy) : {rt['total']}  (pos={rt['positive']}, neg={rt['negative']})")
    print(f"Removed (short): {rs['total']}  (pos={rs['positive']}, neg={rs['negative']})")
    print(f"\nFiltered dataset -> {out_path}")
    print(f"Report           -> {rep_path}")

    if a["total"] < 100:
        print("\nWARNING: fewer than 100 records remain after filtering.")
        print("Consider relaxing filters (--keep-toy or --keep-short).")


if __name__ == "__main__":
    main()
