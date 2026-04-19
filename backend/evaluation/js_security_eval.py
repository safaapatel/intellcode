"""
NIST SARD-Inspired JS/TS Security Rule Evaluation
===================================================
Evaluates the regex-based JS/TS security rules against a manually labelled
test suite of vulnerable and clean code snippets.  Each snippet is annotated
with the expected CWE category (or None for clean code).

The test cases are modelled on the NIST Software Assurance Reference Dataset
(SARD) test-case structure: each entry is either a "flawed" (buggy/vulnerable)
or "fixed" (clean/patched) variant.

Outputs:
    evaluation/results/js_security_eval.json
    Printed: per-CWE precision/recall/F1 + overall macro averages
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Labelled test suite (NIST SARD-style)
# Each entry: (snippet, expected_cwe_or_None, description)
# expected_cwe = None  -> clean (no vulnerability)
# expected_cwe = "CWE-XX" -> vulnerable with that CWE
# ---------------------------------------------------------------------------

TEST_CASES: list[tuple[str, Optional[str], str]] = [
    # ── CWE-79: XSS ──────────────────────────────────────────────────────────
    # FLAWED
    ('element.innerHTML = userInput;',                              "CWE-79",  "XSS: innerHTML with user input"),
    ('div.outerHTML = req.body.html;',                             "CWE-79",  "XSS: outerHTML with request body"),
    ('document.write("<b>" + name + "</b>");',                     "CWE-79",  "XSS: document.write with variable"),
    ('container.innerHTML = "<h1>" + title + "</h1>";',            "CWE-79",  "XSS: innerHTML with concatenation"),
    # FIXED / CLEAN
    ('element.textContent = userInput;',                           None,       "SAFE: textContent avoids XSS"),
    ('element.textContent = DOMPurify.sanitize(userInput);',       None,       "SAFE: DOMPurify sanitized"),
    ('const el = document.createElement("p"); el.textContent = x;', None,     "SAFE: createElement + textContent"),
    ('console.log("innerHTML usage in docs");',                    None,       "SAFE: innerHTML in string literal comment"),

    # ── CWE-95: Code Injection / eval ────────────────────────────────────────
    # FLAWED
    ('eval(userInput);',                                            "CWE-95",  "Injection: eval with user input"),
    ('const fn = new Function(userCode);',                         "CWE-95",  "Injection: Function constructor"),
    ('setTimeout("doSomething()", 100);',                          "CWE-95",  "Injection: setTimeout with string"),
    ('setInterval("refresh()", 5000);',                            "CWE-95",  "Injection: setInterval with string"),
    # FIXED / CLEAN
    ('setTimeout(() => doSomething(), 100);',                      None,       "SAFE: setTimeout with arrow function"),
    ('setInterval(refresh, 5000);',                                None,       "SAFE: setInterval with function ref"),
    ('// eval is mentioned in docs only',                          None,       "SAFE: eval in comment string"),

    # ── CWE-89: SQL Injection ─────────────────────────────────────────────────
    # FLAWED
    ('db.query("SELECT * FROM users WHERE id = " + userId);',      "CWE-89",  "SQLi: query() with concatenation"),
    ('conn.execute("DELETE FROM t WHERE name=\'" + name + "\'");', "CWE-89",  "SQLi: execute() with concatenation"),
    ('const sql = "SELECT * FROM t WHERE x = " + val;',           "CWE-89",  "SQLi: concatenated string variable"),
    # FIXED / CLEAN
    ('db.query("SELECT * FROM users WHERE id = ?", [userId]);',    None,       "SAFE: parameterised query"),
    ('db.query("SELECT * FROM users WHERE id = $1", [userId]);',   None,       "SAFE: postgres parameterised"),
    ('const q = "SELECT * FROM users WHERE active = true";',       None,       "SAFE: static SQL, no concat"),

    # ── CWE-338: Weak Crypto ─────────────────────────────────────────────────
    # FLAWED
    ('const token = Math.random().toString(36);',                  "CWE-338", "Weak crypto: Math.random for token"),
    ('const iv = Math.random();',                                  "CWE-338", "Weak crypto: Math.random for IV"),
    ('const cipher = crypto.createCipher("aes-256-cbc", key);',   "CWE-338", "Weak crypto: deprecated createCipher"),
    ('const hash = md5(password);',                                "CWE-338", "Weak crypto: MD5 hash"),
    ('const digest = sha1(data);',                                 "CWE-338", "Weak crypto: SHA-1 hash"),
    # FIXED / CLEAN
    ('const buf = crypto.randomBytes(16);',                        None,       "SAFE: cryptographic randomBytes"),
    ('const arr = new Uint8Array(16); crypto.getRandomValues(arr);', None,    "SAFE: getRandomValues"),
    ('const cipher = crypto.createCipheriv("aes-256-gcm", key, iv);', None,   "SAFE: createCipheriv"),
    ('const hash = crypto.createHash("sha256").update(data).digest("hex");', None, "SAFE: SHA-256"),

    # ── CWE-1321: Prototype Pollution ────────────────────────────────────────
    # FLAWED
    ('obj.__proto__ = userPayload;',                               "CWE-1321", "Proto pollution: __proto__ assign"),
    ('target[key].__proto__[method] = fn;',                        "CWE-1321", "Proto pollution: nested __proto__"),
    # FIXED / CLEAN
    ('const safe = Object.create(null);',                          None,       "SAFE: null-prototype object"),
    ('const clone = structuredClone(obj);',                        None,       "SAFE: structuredClone"),

    # ── CWE-22: Path Traversal ───────────────────────────────────────────────
    # FLAWED
    ('fs.readFile(baseDir + req.params.file, cb);',                "CWE-22",  "Path traversal: readFile with param"),
    ('fs.readFileSync(uploadDir + filename);',                     "CWE-22",  "Path traversal: readFileSync concat"),
    ('fs.createReadStream(root + userPath);',                      "CWE-22",  "Path traversal: createReadStream"),
    # FIXED / CLEAN
    ('const safe = path.resolve(base, path.basename(req.params.file));', None, "SAFE: path.basename normalization"),
    ('fs.readFile(path.join(__dirname, "static", "index.html"), cb);', None,  "SAFE: static path only"),

    # ── CWE-601: Open Redirect ───────────────────────────────────────────────
    # FLAWED
    ('res.redirect(req.query.next);',                              "CWE-601", "Open redirect: query param"),
    ('res.redirect(req.body.returnUrl);',                          "CWE-601", "Open redirect: body param"),
    ('res.redirect(req.params.url);',                              "CWE-601", "Open redirect: route param"),
    # FIXED / CLEAN
    ('res.redirect("/dashboard");',                                None,       "SAFE: static redirect target"),
    ('if (allowedUrls.includes(url)) res.redirect(url);',         None,       "SAFE: allowlist checked first"),

    # ── CWE-943: NoSQL Injection ─────────────────────────────────────────────
    # FLAWED
    ('db.collection.find({ $where: userInput });',                 "CWE-943", "NoSQL injection: $where with input"),
    ('Model.find({ $where: "this.age > " + age });',              "CWE-943", "NoSQL injection: $where concatenation"),
    # FIXED / CLEAN
    ('db.collection.find({ age: { $gt: parseInt(age, 10) } });',  None,       "SAFE: typed query operator"),
    ('Model.find({ status: "active" });',                          None,       "SAFE: static query"),

    # ── Clean code with no vulnerabilities ───────────────────────────────────
    ('const x = 42;',                                              None,       "SAFE: numeric literal"),
    ('function add(a, b) { return a + b; }',                      None,       "SAFE: pure function"),
    ('import { useState } from "react";',                          None,       "SAFE: module import"),
    ('console.log("Hello, world!");',                              None,       "SAFE: console log"),
]


# ---------------------------------------------------------------------------
# Evaluation logic
# ---------------------------------------------------------------------------

def _scan(snippet: str) -> list[str]:
    """Run the JS security rules and return a list of detected CWE IDs."""
    import re
    from features.security_patterns import _JS_SECURITY_RULES
    hits: list[str] = []
    for pattern, _category, _severity, _name, _desc, _conf, cwe in _JS_SECURITY_RULES:
        if re.search(pattern, snippet):
            if cwe not in hits:
                hits.append(cwe)
    return hits


def evaluate() -> dict:
    """Run all test cases and compute precision/recall/F1 per CWE + overall."""
    # Collect per-CWE TP/FP/FN counts
    cwe_tp: dict[str, int] = {}
    cwe_fp: dict[str, int] = {}
    cwe_fn: dict[str, int] = {}

    total_tp = total_fp = total_fn = 0

    results = []
    for snippet, expected_cwe, desc in TEST_CASES:
        detected_cwes = _scan(snippet)
        is_vuln = expected_cwe is not None

        if is_vuln:
            if expected_cwe in detected_cwes:
                # True positive
                cwe_tp[expected_cwe] = cwe_tp.get(expected_cwe, 0) + 1
                total_tp += 1
                outcome = "TP"
            else:
                # False negative
                cwe_fn[expected_cwe] = cwe_fn.get(expected_cwe, 0) + 1
                total_fn += 1
                outcome = "FN"
        else:
            if detected_cwes:
                # False positive (clean snippet flagged)
                for cwe in detected_cwes:
                    cwe_fp[cwe] = cwe_fp.get(cwe, 0) + 1
                total_fp += 1
                outcome = "FP"
            else:
                # True negative
                outcome = "TN"

        results.append({
            "desc": desc,
            "expected_cwe": expected_cwe,
            "detected_cwes": detected_cwes,
            "outcome": outcome,
        })

    # Per-CWE metrics
    all_cwes = sorted(set(list(cwe_tp.keys()) + list(cwe_fn.keys())))
    per_cwe: dict[str, dict] = {}
    for cwe in all_cwes:
        tp = cwe_tp.get(cwe, 0)
        fp = cwe_fp.get(cwe, 0)
        fn = cwe_fn.get(cwe, 0)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_cwe[cwe] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
                         "tp": tp, "fp": fp, "fn": fn}

    # Overall micro metrics
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_rec  = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1   = 2 * micro_prec * micro_rec / (micro_prec + micro_rec) if (micro_prec + micro_rec) > 0 else 0.0

    # Macro averages (over CWEs with at least one vulnerable test case)
    macro_prec = sum(v["precision"] for v in per_cwe.values()) / len(per_cwe) if per_cwe else 0.0
    macro_rec  = sum(v["recall"]    for v in per_cwe.values()) / len(per_cwe) if per_cwe else 0.0
    macro_f1   = sum(v["f1"]        for v in per_cwe.values()) / len(per_cwe) if per_cwe else 0.0

    total_clean  = sum(1 for _, c, _ in TEST_CASES if c is None)
    total_vuln   = sum(1 for _, c, _ in TEST_CASES if c is not None)
    total_fps    = sum(1 for r in results if r["outcome"] == "FP")
    fp_rate      = total_fps / total_clean if total_clean > 0 else 0.0

    return {
        "dataset": {
            "total_cases": len(TEST_CASES),
            "vulnerable": total_vuln,
            "clean": total_clean,
            "cwe_categories": len(all_cwes),
        },
        "overall": {
            "micro_precision": round(micro_prec, 4),
            "micro_recall":    round(micro_rec,  4),
            "micro_f1":        round(micro_f1,   4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall":    round(macro_rec,  4),
            "macro_f1":        round(macro_f1,   4),
            "false_positive_rate": round(fp_rate, 4),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
        },
        "per_cwe": per_cwe,
        "case_results": results,
    }


def _print_report(ev: dict) -> None:
    ds = ev["dataset"]
    ov = ev["overall"]
    print("=" * 60)
    print("JS/TS Security Rule Evaluation (NIST SARD-style)")
    print("=" * 60)
    print(f"Test cases: {ds['total_cases']}  "
          f"({ds['vulnerable']} vulnerable, {ds['clean']} clean)")
    print(f"CWE categories covered: {ds['cwe_categories']}")
    print()
    print("Overall (micro):")
    print(f"  Precision : {ov['micro_precision']:.4f}")
    print(f"  Recall    : {ov['micro_recall']:.4f}")
    print(f"  F1        : {ov['micro_f1']:.4f}")
    print(f"  FP rate   : {ov['false_positive_rate']:.4f}")
    print()
    print("Overall (macro across CWEs):")
    print(f"  Precision : {ov['macro_precision']:.4f}")
    print(f"  Recall    : {ov['macro_recall']:.4f}")
    print(f"  F1        : {ov['macro_f1']:.4f}")
    print()
    print(f"{'CWE':<12}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'TP':>4}  {'FP':>4}  {'FN':>4}")
    print("-" * 60)
    for cwe, m in sorted(ev["per_cwe"].items()):
        print(f"{cwe:<12}  {m['precision']:6.4f}  {m['recall']:6.4f}  {m['f1']:6.4f}"
              f"  {m['tp']:4d}  {m['fp']:4d}  {m['fn']:4d}")
    print()
    missed = [r for r in ev["case_results"] if r["outcome"] == "FN"]
    fps    = [r for r in ev["case_results"] if r["outcome"] == "FP"]
    if missed:
        print(f"False negatives ({len(missed)}):")
        for r in missed:
            print(f"  [{r['expected_cwe']}] {r['desc']}")
    if fps:
        print(f"\nFalse positives ({len(fps)}):")
        for r in fps:
            print(f"  (detected {r['detected_cwes']}) {r['desc']}")
    print()


def main():
    ev = evaluate()
    _print_report(ev)

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "js_security_eval.json"
    with open(out_path, "w") as f:
        json.dump(ev, f, indent=2)
    print(f"Results saved -> {out_path}")


if __name__ == "__main__":
    main()
