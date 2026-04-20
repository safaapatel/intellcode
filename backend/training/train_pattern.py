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

"""
Debiased feature set: structural AST features ONLY.

Excluded (leaky — directly used in label assignment rules):
  CC, cog, maxCC, avgCC, sloc, comments, comment_ratio,
  halstead_volume, halstead_difficulty, halstead_bugs, MI,
  n_long_functions, n_complex_functions, n_lines_over_80

These 14 features encode the label rules themselves (e.g., CC > 20 -> anti_pattern,
SLOC <= 40 -> clean).  A model that sees them during training partially memorises
the labeling function — not genuine pattern generalisation.

The 12 clean features below are derived from AST node counts and structural
properties, not from the metric thresholds used to construct labels.
"""
FEATURE_NAMES = [
    # Structural counts (original 12)
    "n_functions", "n_classes", "n_try_blocks", "n_raises", "n_with_blocks",
    "max_nesting_depth", "max_params", "avg_params",
    "n_decorated_functions", "n_imports",
    "max_function_body_lines", "avg_function_body_lines",
    # Additional non-leaky AST node counts (10 new)
    "n_lambdas", "n_comprehensions", "n_yields",
    "n_assertions", "n_global_stmts", "n_type_annotations",
    "n_ternary_exprs", "n_bool_ops", "n_calls", "avg_class_methods",
]
N_FEATURES = len(FEATURE_NAMES)  # 22


def _extract_features(source: str) -> np.ndarray:
    """Extract debiased structural AST feature vector (22-dim)."""
    from features.ast_extractor import ASTExtractor

    try:
        ast_feats = ASTExtractor().extract(source)
        nc = ast_feats.get("node_counts", {})
        n_comp = (nc.get("ListComp", 0) + nc.get("DictComp", 0)
                  + nc.get("SetComp", 0) + nc.get("GeneratorExp", 0))
        n_yields = nc.get("Yield", 0) + nc.get("YieldFrom", 0)
        return np.array([
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
            float(nc.get("Lambda", 0)),
            float(n_comp),
            float(n_yields),
            float(nc.get("Assert", 0)),
            float(nc.get("Global", 0)),
            float(nc.get("AnnAssign", 0)),
            float(nc.get("IfExp", 0)),
            float(nc.get("BoolOp", 0)),
            float(ast_feats.get("n_calls", 0)),
            float(ast_feats.get("avg_class_methods", 0.0)),
        ], dtype=np.float32)
    except Exception:
        return np.zeros(N_FEATURES, dtype=np.float32)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(path: str, binary: bool = False) -> tuple[list[str], list[int]]:
    """
    Load pattern_dataset.jsonl and return texts and integer labels.

    Args:
        binary: if True, collapse all non-clean labels to 1 (pattern present)
            and clean to 0. Produces a binary classifier that avoids the uncertain
            four-class label boundaries (kappa=0.168 inter-rater agreement).
    """
    texts, labels = [], []
    skipped = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            code = rec.get("code", "")
            label_str = rec.get("label", "clean")
            if not code or label_str not in LABEL2ID:
                skipped += 1
                continue
            texts.append(code)
            if binary:
                labels.append(0 if label_str == "clean" else 1)
            else:
                labels.append(LABEL2ID[label_str])

    from collections import Counter
    counts = Counter(labels)
    mode = "binary" if binary else "4-class"
    print(f"Loaded {len(texts)} samples ({skipped} skipped) [{mode}] from {path}")
    if binary:
        print(f"  clean=0: {counts[0]}  pattern=1: {counts[1]}")
    else:
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
    binary: bool = False,
    **kwargs,
) -> dict:
    """
    Train the pattern classifier.

    Args:
        binary: if True, train as binary (clean vs pattern_present) instead of
            4-class. Recommended after the kappa=0.168 finding — the four-class
            boundary labels have low inter-rater agreement, so collapsing to
            binary reduces circular label confusion while keeping the AUC signal.
    """
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (
        classification_report, accuracy_score, f1_score, roc_auc_score
    )
    from sklearn.preprocessing import label_binarize

    texts, labels = load_dataset(data_path, binary=binary)
    n_classes = 2 if binary else len(LABEL_NAMES)

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

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    target_names = ["clean", "pattern_present"] if binary else LABEL_NAMES

    def _eval(model, name):
        cv_f1 = cross_val_score(model, X_tr, y_tr, cv=skf, scoring="f1_macro", n_jobs=1)
        cv_acc = cross_val_score(model, X_tr, y_tr, cv=skf, scoring="accuracy", n_jobs=1)
        print(f"CV F1 (macro):   {np.mean(cv_f1):.4f} +/- {np.std(cv_f1):.4f}")
        print(f"CV Accuracy:     {np.mean(cv_acc):.4f} +/- {np.std(cv_acc):.4f}")
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        t_acc = accuracy_score(y_te, y_pred)
        t_f1  = f1_score(y_te, y_pred, average="macro")
        print(f"Test Accuracy:   {t_acc:.4f}")
        print(f"Test F1 (macro): {t_f1:.4f}")
        print(classification_report(y_te, y_pred, target_names=target_names))
        y_prob = model.predict_proba(X_te)
        try:
            if binary:
                t_auc = float(roc_auc_score(y_te, y_prob[:, 1]))
            else:
                y_te_bin = label_binarize(y_te, classes=list(range(4)))
                t_auc = float(roc_auc_score(y_te_bin, y_prob, multi_class="ovr", average="macro"))
        except Exception:
            t_auc = 0.0
        print(f"Test AUC: {t_auc:.4f}")
        return model, np.mean(cv_f1), np.std(cv_f1), np.mean(cv_acc), np.std(cv_acc), t_acc, t_f1, t_auc

    print("\n-- Random Forest ---------------------------------------------")
    rf_base = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_split=5,
        class_weight="balanced",
        n_jobs=1,
        random_state=42,
    )
    rf_clf = CalibratedClassifierCV(rf_base, cv=5, method="sigmoid")
    rf_clf, rf_cv_f1, rf_cv_f1_std, rf_cv_acc, rf_cv_acc_std, rf_acc, rf_f1, rf_auc = _eval(rf_clf, "RF")

    print("\n-- XGBoost ---------------------------------------------------")
    try:
        from xgboost import XGBClassifier
        n_cls = 2 if binary else len(LABEL_NAMES)
        xgb_raw = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            num_class=n_cls if n_cls > 2 else None,
            objective="multi:softprob" if n_cls > 2 else "binary:logistic",
            eval_metric="mlogloss" if n_cls > 2 else "logloss",
            random_state=42,
            n_jobs=1,
            use_label_encoder=False,
        )
        xgb_clf = CalibratedClassifierCV(xgb_raw, cv=5, method="sigmoid")
        xgb_clf, xgb_cv_f1, xgb_cv_f1_std, xgb_cv_acc, xgb_cv_acc_std, xgb_acc, xgb_f1, xgb_auc = _eval(xgb_clf, "XGB")
        use_xgb = xgb_f1 > rf_f1
    except ImportError:
        print("[WARN] xgboost not installed — RF only.")
        use_xgb = False

    # Pick the better model as primary; save both
    clf       = xgb_clf  if use_xgb else rf_clf
    cv_f1     = xgb_cv_f1     if use_xgb else rf_cv_f1
    cv_f1_std = xgb_cv_f1_std if use_xgb else rf_cv_f1_std
    cv_acc    = xgb_cv_acc    if use_xgb else rf_cv_acc
    cv_acc_std= xgb_cv_acc_std if use_xgb else rf_cv_acc_std
    test_acc  = xgb_acc  if use_xgb else rf_acc
    test_f1   = xgb_f1   if use_xgb else rf_f1
    test_auc  = xgb_auc  if use_xgb else rf_auc
    winner    = "XGBoost" if use_xgb else "RandomForest"
    print(f"\n=> Best model: {winner}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Always save RF for backward compat; save XGB separately if trained
    with open(out / "rf_model.pkl", "wb") as f:
        pickle.dump(rf_clf, f)
    if use_xgb:
        with open(out / "xgb_model.pkl", "wb") as f:
            pickle.dump(xgb_clf, f)
    print(f"\nModels saved -> {out}/")

    model_path = out / "rf_model.pkl"

    metrics = {
        "cv_f1_mean": float(cv_f1),
        "cv_f1_std": float(cv_f1_std),
        "cv_acc_mean": float(cv_acc),
        "cv_acc_std": float(cv_acc_std),
        "test_accuracy": float(test_acc),
        "test_f1_macro": float(test_f1),
        "test_auc_macro": float(test_auc),
        "best_model": winner,
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "n_features": int(X.shape[1]),
        "feature_names": FEATURE_NAMES,
        "label_names": ["clean", "pattern_present"] if binary else LABEL_NAMES,
        "binary_mode": binary,
        "debiased": True,
        "leaky_features_excluded": [
            "CC", "cog", "maxCC", "avgCC", "sloc", "comments", "comment_ratio",
            "halstead_volume", "halstead_difficulty", "halstead_bugs", "MI",
            "n_long_functions", "n_complex_functions", "n_lines_over_80",
        ],
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
    parser.add_argument(
        "--binary", action="store_true",
        help=(
            "Binary mode: collapse code_smell/anti_pattern/style_violation -> 1, "
            "clean -> 0. Avoids kappa=0.168 four-class label uncertainty."
        ),
    )
    args = parser.parse_args()

    train(data_path=args.data, output_dir=args.out, n_estimators=args.trees, binary=args.binary)


if __name__ == "__main__":
    main()
