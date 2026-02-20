"""
Train Pattern Recognition Model (CodeBERT Fine-Tune)

Usage:
    cd backend
    python training/train_pattern.py --data data/pattern_dataset.jsonl

Outputs:
    checkpoints/pattern/    (HuggingFace Trainer checkpoint directory)
    checkpoints/pattern/metrics.json

Requirements:
    pip install transformers torch datasets evaluate scikit-learn
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LABEL_NAMES = ["clean", "code_smell", "anti_pattern", "style_violation"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_NAMES)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_NAMES)}
MODEL_NAME = "microsoft/codebert-base"
MAX_LENGTH = 512


def load_dataset(path: str) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("code") and rec.get("label") in LABEL2ID:
                texts.append(rec["code"])
                labels.append(LABEL2ID[rec["label"]])

    # Print class balance
    from collections import Counter
    counts = Counter(labels)
    print(f"Loaded {len(texts)} samples")
    for lid, count in sorted(counts.items()):
        print(f"  {ID2LABEL[lid]:20s}: {count}")
    return texts, labels


def compute_metrics(eval_pred):
    """Callback for HuggingFace Trainer."""
    import evaluate
    import numpy as np

    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = metric.compute(predictions=predictions, references=labels)

    # Per-class F1
    from sklearn.metrics import f1_score, precision_score, recall_score
    f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    precision = precision_score(labels, predictions, average="macro", zero_division=0)
    recall = recall_score(labels, predictions, average="macro", zero_division=0)

    return {
        "accuracy": acc["accuracy"],
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall,
    }


def train(
    data_path: str,
    output_dir: str = "checkpoints/pattern",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    test_split: float = 0.15,
):
    try:
        import torch
        from transformers import (
            RobertaTokenizer,
            RobertaForSequenceClassification,
            TrainingArguments,
            Trainer,
            EarlyStoppingCallback,
        )
        from torch.utils.data import Dataset
        from sklearn.model_selection import train_test_split
    except ImportError as e:
        print(f"ERROR: {e}")
        print("Install: pip install transformers torch evaluate scikit-learn")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Dataset class
    # ------------------------------------------------------------------
    class CodeDataset(Dataset):
        def __init__(self, texts, labels, tokenizer):
            self.encodings = tokenizer(
                texts,
                truncation=True,
                max_length=MAX_LENGTH,
                padding="max_length",
                return_tensors="pt",
            )
            self.labels = torch.tensor(labels, dtype=torch.long)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings["input_ids"][idx],
                "attention_mask": self.encodings["attention_mask"][idx],
                "labels": self.labels[idx],
            }

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    texts, labels = load_dataset(data_path)
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=test_split, stratify=labels, random_state=42
    )
    print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}")

    # ------------------------------------------------------------------
    # Load tokenizer and model
    # ------------------------------------------------------------------
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_NAMES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    train_dataset = CodeDataset(X_train, y_train, tokenizer)
    val_dataset = CodeDataset(X_val, y_val, tokenizer)

    # ------------------------------------------------------------------
    # Training arguments
    # ------------------------------------------------------------------
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=str(out_dir / "logs"),
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to="none",          # disable wandb/mlflow
    )

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\nStarting fine-tuning...")
    train_result = trainer.train()

    # ------------------------------------------------------------------
    # Save model + tokenizer
    # ------------------------------------------------------------------
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"\nModel saved to {out_dir}")

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    eval_results = trainer.evaluate()
    print("\nFinal Evaluation:")
    for k, v in eval_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    metrics = {
        "train_loss": train_result.training_loss,
        "train_steps": train_result.global_step,
        **{k: float(v) if isinstance(v, float) else v for k, v in eval_results.items()},
        "n_train": len(X_train),
        "n_val": len(X_val),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "base_model": MODEL_NAME,
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CodeBERT for pattern recognition")
    parser.add_argument("--data", required=True, help="Path to JSONL dataset")
    parser.add_argument("--out", default="checkpoints/pattern")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    train(
        data_path=args.data,
        output_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
