"""
Train Pattern Recognition Model (Random Forest on code metrics + AST features)

Lightweight alternative to CodeBERT — no GPU or HuggingFace download required.
Achieves good accuracy using static code analysis features extracted with the
existing code_metrics and ast_extractor modules.

Labels:
  0  clean
  1  code_smell
  2  anti_pattern
  3  style_violation

Usage:
    cd backend
    python training/train_pattern.py --data data/pattern_dataset.jsonl

Outputs:
    checkpoints/pattern/rf_model.pkl
    checkpoints/pattern/metrics.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LABEL_NAMES = ["clean", "code_smell", "anti_pattern", "style_violation"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_NAMES)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_NAMES)}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_features(source: str) -> np.ndarray:
    """Extract a fixed-length feature vector from source code."""
    from features.code_metrics import compute_all_metrics
    from features.ast_extractor import ASTExtractor

    try:
        m = compute_all_metrics(source)
        ast_feats = ASTExtractor().extract(source)

        sloc = max(m.lines.sloc, 1)
        comment_ratio = m.lines.comments / sloc

        return np.array([
            float(m.cyclomatic_complexity),
            float(m.cognitive_complexity),
            float(m.max_function_complexity),
            float(m.avg_function_complexity),
            float(m.lines.sloc),
            float(m.lines.comments),
            float(comment_ratio),
            float(m.halstead.volume),
            float(m.halstead.difficulty),
            float(m.halstead.bugs_delivered),
            float(m.maintainability_index),
            float(m.n_long_functions),
            float(m.n_complex_functions),
            float(m.n_lines_over_80),
            float(ast_feats.get("n_functions", 0)),
            float(ast_feats.get("n_classes", 0)),
            float(ast_feats.get("n_try_blocks", 0)),
            float(ast_feats.get("n_raises", 0)),
            float(ast_feats.get("n_with_blocks", 0)),
            float(ast_feats.get("max_nesting_depth", 0)),
            float(ast_feats.get("max_params", 0)),
            float(ast_feats.get("avg_params", 0.0)),
            float(ast_feats.get("n_decorated_functions", 0)),
            float(ast_feats.get("n_imports", 0)),
            float(ast_feats.get("max_function_body_lines", 0)),
            float(ast_feats.get("avg_function_body_lines", 0.0)),
        ], dtype=np.float32)
    except Exception:
        return np.zeros(26, dtype=np.float32)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> tuple[list[str], list[int]]:
    """Load pattern_dataset.jsonl and return texts and integer labels."""
    texts, labels = [], []
    skipped = 0
    with open(path) as f:
        for line in f:
            rec = json.loads(line.strip())
            code = rec.get("code", "")
            label_str = rec.get("label", "clean")
            if not code or label_str not in LABEL2ID:
                skipped += 1
                continue
            texts.append(code)
            labels.append(LABEL2ID[label_str])

    from collections import Counter
    counts = Counter(labels)
    print(f"Loaded {len(texts)} samples ({skipped} skipped) from {path}")
    for lid, count in sorted(counts.items()):
        print(f"  {ID2LABEL[lid]:20s}: {count}")
    return texts, labels


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    data_path: str,
    output_dir: str = "checkpoints/pattern",
    test_split: float = 0.15,
    n_estimators: int = 300,
    **kwargs,
) -> dict:
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (
        classification_report, accuracy_score, f1_score, roc_auc_score
    )
    from sklearn.preprocessing import label_binarize

    texts, labels = load_dataset(data_path)

    if len(texts) == 0:
        print("[WARN] No samples loaded — skipping pattern training.")
        return {}

    # Extract features
    print("\nExtracting code features...")
    X = np.array([_extract_features(t) for t in texts], dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    print(f"Feature matrix: {X.shape}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_split, stratify=y, random_state=42
    )
    print(f"Train: {len(X_tr)}, Test: {len(X_te)}")

    print("\n-- Random Forest ---------------------------------------------")
    base = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    clf = CalibratedClassifierCV(base, cv=5, method="sigmoid")

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1 = cross_val_score(clf, X_tr, y_tr, cv=skf, scoring="f1_macro", n_jobs=1)
    cv_acc = cross_val_score(clf, X_tr, y_tr, cv=skf, scoring="accuracy", n_jobs=1)
    print(f"CV F1 (macro):   {np.mean(cv_f1):.4f} ± {np.std(cv_f1):.4f}")
    print(f"CV Accuracy:     {np.mean(cv_acc):.4f} ± {np.std(cv_acc):.4f}")

    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    test_acc = accuracy_score(y_te, y_pred)
    test_f1 = f1_score(y_te, y_pred, average="macro")
    print(f"\nTest Accuracy:   {test_acc:.4f}")
    print(f"Test F1 (macro): {test_f1:.4f}")
    print(classification_report(y_te, y_pred, target_names=LABEL_NAMES))

    # Multi-class AUC (one-vs-rest)
    y_te_bin = label_binarize(y_te, classes=list(range(4)))
    y_prob = clf.predict_proba(X_te)
    try:
        test_auc = float(roc_auc_score(y_te_bin, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        test_auc = 0.0
    print(f"Test AUC (macro OvR): {test_auc:.4f}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / "rf_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"\nRF model saved -> {model_path}")

    metrics = {
        "cv_f1_mean": float(np.mean(cv_f1)),
        "cv_f1_std": float(np.std(cv_f1)),
        "cv_acc_mean": float(np.mean(cv_acc)),
        "cv_acc_std": float(np.std(cv_acc)),
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(test_f1),
        "test_auc_macro": float(test_auc),
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "n_features": int(X.shape[1]),
        "label_names": LABEL_NAMES,
    }
    metrics_path = out / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved -> {metrics_path}")
    print("\n[OK] Pattern classifier training complete.")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train RF pattern classifier")
    parser.add_argument("--data", required=True, help="Path to pattern_dataset.jsonl")
    parser.add_argument("--out", default="checkpoints/pattern")
    parser.add_argument("--trees", type=int, default=300)
    args = parser.parse_args()

    train(data_path=args.data, output_dir=args.out, n_estimators=args.trees)


if __name__ == "__main__":
    main()
