"""
Security LOPO Ablation: Project-Style Feature Exclusion
=======================================================
Tests whether removing project-style features (n_calls, n_imports) alone
can bring security LOPO AUC above 0.55.

Three conditions:
  1. Full 16-dim vector (baseline, matches existing lopo_security.json)
  2. Remove project-style: zero out n_calls (idx 0) + n_imports (idx 1)
  3. Remove structural size proxies: zero out n_bytes (idx 12) + n_lines (idx 13)
  4. Remove all project-style + size proxies: zero out [0,1,12,13]

Saves result to results/security_project_style_ablation.json
"""

from __future__ import annotations
import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.cross_project_benchmark import _prepare_security, _build_model, _score

RESULTS_DIR = Path(__file__).resolve().parent / "results"
DATA_PATH   = Path("D:/projexts/intellcode/backend/data/security_dataset.jsonl")

# Project-style feature indices in the 16-dim security vector:
#  0  n_calls       <- varies by project coding style (number of function calls)
#  1  n_imports     <- varies by project dependency profile
#  12 n_bytes       <- raw file size proxy, project-level variation
#  13 n_lines       <- line count, correlated with project conventions

PROJECT_STYLE_INDICES  = [0, 1]          # call + import counts only
SIZE_PROXY_INDICES     = [12, 13]        # byte + line counts
ALL_STYLE_INDICES      = [0, 1, 12, 13]  # combined


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def run_lopo(X: np.ndarray, y: np.ndarray, repos: list[str]) -> dict:
    """Run LOPO on the given feature matrix; return per-fold and summary AUC."""
    unique_repos = [r for r in dict.fromkeys(repos) if r != "unknown"]
    fold_aucs = []
    fold_results = []

    for held_out in unique_repos:
        mask_test  = np.array([r == held_out for r in repos])
        mask_train = ~mask_test

        X_tr, y_tr = X[mask_train], y[mask_train]
        X_te, y_te = X[mask_test],  y[mask_test]

        if len(X_tr) < 20 or len(X_te) < 5:
            continue
        if len(set(y_te)) < 2:
            continue

        model = _build_model("security")
        try:
            model.fit(X_tr, y_tr)
            scores = _score(model, X_te, y_te, "security")
            auc = scores.get("auc")
            if auc is not None:
                fold_aucs.append(auc)
                fold_results.append({"repo": held_out, "n_test": int(mask_test.sum()), "auc": round(auc, 4)})
        except Exception as e:
            fold_results.append({"repo": held_out, "n_test": int(mask_test.sum()), "auc": None, "error": str(e)})

    mean_auc = float(np.mean(fold_aucs)) if fold_aucs else None
    std_auc  = float(np.std(fold_aucs))  if len(fold_aucs) > 1 else 0.0
    return {"folds": fold_results, "mean_auc": mean_auc, "std_auc": std_auc, "n_folds": len(fold_aucs)}


def ablate(X: np.ndarray, indices: list[int]) -> np.ndarray:
    Xa = X.copy()
    for i in indices:
        Xa[:, i] = 0.0
    return Xa


def main():
    if not DATA_PATH.exists():
        alt = Path(__file__).resolve().parent.parent.parent / "data" / "security_dataset.jsonl"
        if alt.exists():
            dp = alt
        else:
            print(f"ERROR: security_dataset.jsonl not found at {DATA_PATH}")
            sys.exit(1)
    else:
        dp = DATA_PATH

    print(f"Loading security records from {dp}")
    records = load_records(dp)
    print(f"  Loaded {len(records)} records")

    X, y, repos = _prepare_security(records)
    print(f"  Prepared: X={X.shape}, positives={y.sum()}/{len(y)}, repos={len(set(repos))}")
    print()

    conditions = {
        "full_16dim": (X, "No ablation (baseline)"),
        "remove_calls_imports": (ablate(X, PROJECT_STYLE_INDICES),
                                 "Zero out n_calls + n_imports (project-style)"),
        "remove_size_proxies": (ablate(X, SIZE_PROXY_INDICES),
                                "Zero out n_bytes + n_lines (size proxies)"),
        "remove_all_style": (ablate(X, ALL_STYLE_INDICES),
                             "Zero out n_calls + n_imports + n_bytes + n_lines"),
    }

    results = {}
    for name, (Xa, desc) in conditions.items():
        print(f"Running LOPO: {desc}")
        r = run_lopo(Xa, y, repos)
        results[name] = {"description": desc, **r}
        if r["mean_auc"] is not None:
            print(f"  mean AUC = {r['mean_auc']:.4f} +/- {r['std_auc']:.4f}  ({r['n_folds']} folds)")
        else:
            print("  mean AUC = N/A (insufficient data)")
        print()

    out_path = RESULTS_DIR / "security_project_style_ablation.json"
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")

    # Summary table
    print("\n=== SUMMARY ===")
    print(f"{'Condition':<35} {'Mean LOPO AUC':>14} {'Std':>8} {'> 0.55?':>8}")
    print("-" * 70)
    for name, r in results.items():
        auc = r.get("mean_auc")
        std = r.get("std_auc", 0.0)
        above = ("YES" if auc and auc > 0.55 else "no") if auc else "N/A"
        auc_str = f"{auc:.4f}" if auc else "N/A"
        std_str = f"{std:.4f}" if std else "N/A"
        print(f"  {name:<33} {auc_str:>14} {std_str:>8} {above:>8}")


if __name__ == "__main__":
    main()
