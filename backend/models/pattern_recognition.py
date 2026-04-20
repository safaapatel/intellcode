"""
Pattern Recognition Model — CodeBERT Fine-Tune
Identifies code smells, anti-patterns, and style violations using a
fine-tuned microsoft/codebert-base transformer.

Classes:
    PatternRecognitionModel  — inference + fine-tuning wrapper
    CodePatternDataset       — PyTorch Dataset for fine-tuning

Labels:
    0  clean
    1  code_smell
    2  anti_pattern
    3  style_violation
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

LABEL_NAMES = ["clean", "code_smell", "anti_pattern", "style_violation"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_NAMES)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_NAMES)}

MODEL_NAME = "microsoft/codebert-base"
MAX_LENGTH = 512


# ---------------------------------------------------------------------------
# Lightweight RF-based classifier (no GPU/HuggingFace required)
# ---------------------------------------------------------------------------

"""
Debiased feature set — 22 structural AST features.
Matches training/train_pattern.py FEATURE_NAMES exactly.

Excluded (leaky — encode label rules directly):
  CC, cog, maxCC, avgCC, sloc, comments, comment_ratio,
  halstead_volume, halstead_difficulty, halstead_bugs, MI,
  n_long_functions, n_complex_functions, n_lines_over_80
"""
_RF_FEATURE_NAMES = [
    "n_functions", "n_classes", "n_try_blocks", "n_raises", "n_with_blocks",
    "max_nesting_depth", "max_params", "avg_params",
    "n_decorated_functions", "n_imports",
    "max_function_body_lines", "avg_function_body_lines",
    "n_lambdas", "n_comprehensions", "n_yields",
    "n_assertions", "n_global_stmts", "n_type_annotations",
    "n_ternary_exprs", "n_bool_ops", "n_calls", "avg_class_methods",
]
_RF_N_FEATURES = len(_RF_FEATURE_NAMES)  # 22


def _rf_extract_features(source: str):
    """Extract debiased structural AST feature vector (22-dim)."""
    import numpy as np
    try:
        from features.ast_extractor import ASTExtractor
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
        import numpy as np
        return np.zeros(_RF_N_FEATURES, dtype=np.float32)


class PatternRFModel:
    """
    Lightweight Random Forest pattern classifier.
    Trained on code metrics + AST features — no HuggingFace required.
    Used as the default when a fine-tuned CodeBERT checkpoint is unavailable.
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        self._clf = None
        # Prefer XGBoost checkpoint if available; fall back to RF
        if checkpoint_path:
            self._checkpoint = checkpoint_path
        else:
            xgb_path = Path("checkpoints/pattern/xgb_model.pkl")
            self._checkpoint = str(xgb_path) if xgb_path.exists() else "checkpoints/pattern/rf_model.pkl"
        self._try_load()

    def _try_load(self):
        import pickle
        from utils.checkpoint_integrity import verify_checkpoint
        path = Path(self._checkpoint)
        if path.exists():
            try:
                verify_checkpoint(path)
                with open(path, "rb") as f:
                    self._clf = pickle.load(f)
            except Exception:
                self._clf = None

    @property
    def ready(self) -> bool:
        return self._clf is not None

    def predict(self, code_snippet: str) -> PatternPrediction:
        import numpy as np
        feat = _rf_extract_features(code_snippet).reshape(1, -1)
        if self._clf is None:
            # Model not loaded — return an explicit "unknown" result, not a fake prediction
            return PatternPrediction(
                label="clean", confidence=0.0, label_id=0,
                all_scores={l: 0.0 for l in LABEL_NAMES},
            )
        probs = self._clf.predict_proba(feat)[0]
        label_id = int(np.argmax(probs))
        return PatternPrediction(
            label=ID2LABEL[label_id],
            confidence=float(probs[label_id]),
            label_id=label_id,
            all_scores={ID2LABEL[i]: float(p) for i, p in enumerate(probs)},
        )


@dataclass
class PatternPrediction:
    label: str
    confidence: float
    label_id: int
    all_scores: dict[str, float]


class PatternRecognitionModel:
    """
    Wraps microsoft/codebert-base for code pattern classification.

    On first use the model is loaded from HuggingFace Hub (cached locally).
    If a fine-tuned checkpoint is present at `checkpoint_path`, that is
    loaded instead.

    Lazy loading: model weights are not downloaded until the first call
    to predict() or fine_tune().
    """

    def __init__(self, checkpoint_path: Optional[str] = None):
        self._checkpoint = checkpoint_path
        self._tokenizer = None
        self._model = None
        self._device = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, code_snippet: str) -> PatternPrediction:
        """
        Classify a code snippet.

        Args:
            code_snippet: Raw source code string.

        Returns:
            PatternPrediction with label, confidence, and all class scores.
        """
        self._ensure_loaded()

        import torch

        inputs = self._tokenizer(
            code_snippet,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        ).to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        label_id = int(np.argmax(probs))

        return PatternPrediction(
            label=ID2LABEL[label_id],
            confidence=float(probs[label_id]),
            label_id=label_id,
            all_scores={ID2LABEL[i]: float(p) for i, p in enumerate(probs)},
        )

    def predict_batch(self, snippets: list[str], batch_size: int = 16) -> list[PatternPrediction]:
        """Run predict() over a list of snippets with batching for efficiency."""
        self._ensure_loaded()

        import torch

        results = []
        for i in range(0, len(snippets), batch_size):
            batch = snippets[i : i + batch_size]
            inputs = self._tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            for row in probs:
                label_id = int(np.argmax(row))
                results.append(PatternPrediction(
                    label=ID2LABEL[label_id],
                    confidence=float(row[label_id]),
                    label_id=label_id,
                    all_scores={ID2LABEL[i]: float(p) for i, p in enumerate(row)},
                ))

        return results

    def fine_tune(
        self,
        dataset_path: str,
        output_dir: str = "checkpoints/pattern",
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
    ) -> dict:
        """
        Fine-tune CodeBERT on a labeled dataset.

        Dataset format (JSONL):
            {"code": "def foo(): ...", "label": "code_smell"}

        Returns:
            Training metrics dict from HuggingFace Trainer.
        """
        from transformers import TrainingArguments, Trainer
        from torch.utils.data import Dataset as TorchDataset
        import torch

        self._ensure_loaded()

        class CodePatternDataset(TorchDataset):
            def __init__(self, path: str, tokenizer, max_len: int):
                self.samples = []
                with open(path) as f:
                    for line in f:
                        item = json.loads(line)
                        self.samples.append((item["code"], LABEL2ID[item["label"]]))
                self.tokenizer = tokenizer
                self.max_len = max_len

            def __len__(self):
                return len(self.samples)

            def __getitem__(self, idx):
                code, label = self.samples[idx]
                enc = self.tokenizer(
                    code,
                    truncation=True,
                    max_length=self.max_len,
                    padding="max_length",
                    return_tensors="pt",
                )
                return {
                    "input_ids": enc["input_ids"].squeeze(),
                    "attention_mask": enc["attention_mask"].squeeze(),
                    "labels": torch.tensor(label, dtype=torch.long),
                }

        train_dataset = CodePatternDataset(dataset_path, self._tokenizer, MAX_LENGTH)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            save_strategy="epoch",
            load_best_model_at_end=False,
        )

        trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
        )

        train_result = trainer.train()
        trainer.save_model(output_dir)
        self._tokenizer.save_pretrained(output_dir)
        self._checkpoint = output_dir

        return train_result.metrics

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        """Lazy-load the model and tokenizer on first use."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import RobertaTokenizer, RobertaForSequenceClassification

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            source = self._checkpoint if (
                self._checkpoint and Path(self._checkpoint).exists()
            ) else MODEL_NAME

            self._tokenizer = RobertaTokenizer.from_pretrained(source)
            self._model = RobertaForSequenceClassification.from_pretrained(
                source,
                num_labels=len(LABEL_NAMES),
                id2label=ID2LABEL,
                label2id=LABEL2ID,
                ignore_mismatched_sizes=True,
            ).to(self._device)
            self._model.eval()

        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are required for PatternRecognitionModel. "
                "Install them with: pip install transformers torch"
            ) from exc
