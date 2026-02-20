"""
Train Security Detection Ensemble (Random Forest + 1D CNN)

Usage:
    cd backend
    python training/train_security.py --data data/security_dataset.jsonl

Outputs:
    checkpoints/security/rf_model.pkl
    checkpoints/security/cnn_model.pt
    checkpoints/security/cnn_vocab.json
    checkpoints/security/metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_dataset(path: str):
    """
    Load JSONL security dataset.
    Returns X_rf (structured features), X_tokens (token lists), y (labels).
    """
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Loaded {len(records)} samples from {path}")
    return records


def build_rf_features_from_records(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Re-extract RF features from raw source in records (if 'source' is available),
    or use pre-computed 'features' field.
    """
    from features.security_detection import _build_rf_feature_vector

    X, y = [], []
    for rec in records:
        try:
            if "source" in rec:
                feat = _build_rf_feature_vector(rec["source"])
            elif "features" in rec:
                feat = np.array(rec["features"], dtype=np.float32)
            else:
                continue
            X.append(feat)
            y.append(int(rec["label"]))
        except Exception:
            continue
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def build_token_ids(
    records: list[dict],
    vocab: dict[str, int],
    max_len: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    from features.ast_extractor import tokens_to_ids

    X, y = [], []
    for rec in records:
        tokens = rec.get("tokens", [])
        ids = tokens_to_ids(tokens, vocab, max_len)
        X.append(ids)
        y.append(int(rec["label"]))
    return np.array(X, dtype=np.int32), np.array(y, dtype=np.int32)


def train(
    data_path: str,
    output_dir: str = "checkpoints/security",
    test_split: float = 0.15,
    cnn_epochs: int = 15,
    rf_estimators: int = 200,
):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        classification_report, roc_auc_score, precision_score, recall_score, f1_score
    )
    from features.ast_extractor import build_token_vocab
    from models.security_detection import (
        RandomForestSecurityModel, CNNSecurityModel
    )

    records = load_dataset(data_path)

    # Split first to prevent data leakage
    train_records, test_records = train_test_split(
        records, test_size=test_split, random_state=42,
        stratify=[r["label"] for r in records]
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Train Random Forest
    # ----------------------------------------------------------------
    print("\n=== Training Random Forest ===")
    X_rf_train, y_rf_train = build_rf_features_from_records(train_records)
    X_rf_test, y_rf_test = build_rf_features_from_records(test_records)

    if len(X_rf_train) > 0:
        rf_model = RandomForestSecurityModel()
        rf_model.fit(X_rf_train, y_rf_train, n_estimators=rf_estimators)
        rf_model.save(str(out_dir / "rf_model.pkl"))

        rf_preds = np.array([rf_model.predict_proba(x) for x in X_rf_test])
        rf_labels = (rf_preds > 0.5).astype(int)

        print(f"\nRF Test Results:")
        print(classification_report(y_rf_test, rf_labels,
                                    target_names=["clean", "vulnerable"]))
        rf_auc = roc_auc_score(y_rf_test, rf_preds)
        print(f"RF AUC: {rf_auc:.4f}")
    else:
        print("[WARN] No RF features available, skipping RF training")
        rf_auc = 0.0

    # ----------------------------------------------------------------
    # Build vocabulary and train CNN
    # ----------------------------------------------------------------
    print("\n=== Building Token Vocabulary ===")
    all_token_lists = [rec.get("tokens", []) for rec in records]

    # Build vocab from raw token lists (flatten back to strings)
    token_corpus = [" ".join(tl) for tl in all_token_lists]
    vocab = build_token_vocab(token_corpus, max_vocab=10_000)
    print(f"Vocabulary size: {len(vocab)}")

    # Save vocab
    vocab_path = str(out_dir / "cnn_vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)
    print(f"Vocab saved to {vocab_path}")

    print("\n=== Training 1D CNN ===")
    X_cnn_train, y_cnn_train = build_token_ids(train_records, vocab)
    X_cnn_test, y_cnn_test = build_token_ids(test_records, vocab)

    cnn_model = CNNSecurityModel()
    cnn_model.build(vocab)
    cnn_model.train_loop(
        X_cnn_train, y_cnn_train,
        epochs=cnn_epochs,
        batch_size=64,
        lr=1e-3,
    )

    cnn_model_path = str(out_dir / "cnn_model.pt")
    cnn_model.save(cnn_model_path, vocab_path)
    print(f"CNN saved to {cnn_model_path}")

    # Evaluate CNN
    cnn_probs = np.array([cnn_model.predict_proba(
        " ".join(rec.get("tokens", []))
    ) for rec in test_records])
    cnn_labels = (cnn_probs > 0.5).astype(int)
    cnn_auc = roc_auc_score(y_cnn_test, cnn_probs)
    print(f"\nCNN Test Results:")
    print(classification_report(y_cnn_test, cnn_labels,
                                target_names=["clean", "vulnerable"]))
    print(f"CNN AUC: {cnn_auc:.4f}")

    # ----------------------------------------------------------------
    # Ensemble evaluation
    # ----------------------------------------------------------------
    print("\n=== Ensemble Evaluation ===")
    if len(X_rf_test) > 0:
        rf_w, cnn_w = 0.55, 0.45
        ensemble_probs = rf_w * rf_preds + cnn_w * cnn_probs
        ensemble_labels = (ensemble_probs > 0.5).astype(int)
        ensemble_auc = roc_auc_score(y_cnn_test, ensemble_probs)
        print(classification_report(y_cnn_test, ensemble_labels,
                                    target_names=["clean", "vulnerable"]))
        print(f"Ensemble AUC: {ensemble_auc:.4f}")
    else:
        ensemble_auc = cnn_auc

    # Save metrics
    metrics = {
        "rf_auc": float(rf_auc),
        "cnn_auc": float(cnn_auc),
        "ensemble_auc": float(ensemble_auc),
        "n_train": len(train_records),
        "n_test": len(test_records),
        "vocab_size": len(vocab),
        "cnn_epochs": cnn_epochs,
        "rf_estimators": rf_estimators,
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train security detection ensemble")
    parser.add_argument("--data", required=True, help="Path to JSONL dataset")
    parser.add_argument("--out", default="checkpoints/security")
    parser.add_argument("--cnn-epochs", type=int, default=15)
    parser.add_argument("--rf-trees", type=int, default=200)
    args = parser.parse_args()

    train(
        data_path=args.data,
        output_dir=args.out,
        cnn_epochs=args.cnn_epochs,
        rf_estimators=args.rf_trees,
    )


if __name__ == "__main__":
    main()
