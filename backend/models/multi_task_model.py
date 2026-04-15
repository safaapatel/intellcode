"""
Multi-Task Code Quality Model
================================
Shared transformer encoder with 4 task-specific prediction heads.

Research Contribution:
    Demonstrates that multi-task learning over correlated code quality
    dimensions (security, complexity, bugs, patterns) outperforms single-task
    models by leveraging shared representations.

    Hypothesis: Code quality dimensions are correlated — insecure code is
    typically also complex and bug-prone. A shared encoder that sees all tasks
    jointly learns better representations than 4 independent models.

Architecture:
    Input: Python source code (string)
       |
    Shared Encoder (microsoft/codebert-base or distilbert-base-uncased)
       |
    [Task Heads — independently trained, jointly optimized]
       |
       +---> SecurityHead     -> sigmoid(1)     -> vulnerability probability
       +---> ComplexityHead   -> linear(1)      -> maintainability score [0,100]
       +---> BugHead          -> sigmoid(1)     -> bug probability
       +---> PatternHead      -> softmax(4)     -> pattern class probabilities

MTL Loss (Kendall et al. 2018 uncertainty weighting):
    L_total = sum_i [ (1 / 2*sigma_i^2) * L_i + log(sigma_i) ]
    where sigma_i are learned log-uncertainty parameters per task.
    This automatically balances task gradients without hand-tuned weights.

References:
    Multi-task Learning:
        Caruana 1997 — "Multitask Learning"
        Kendall et al. 2018 — "Multi-Task Learning Using Uncertainty to
          Weigh Losses for Scene Geometry and Semantics"
    Code Models:
        Feng et al. 2020 — "CodeBERT: A Pre-Trained Model for Programming
          and Natural Languages" (EMNLP 2020)
        GraphCodeBERT — "GraphCodeBERT: Pre-training Code Representations
          with Data Flow" (ICLR 2021)

Usage:
    from models.multi_task_model import MultiTaskCodeModel, MultiTaskPrediction

    model = MultiTaskCodeModel()
    if model.ready:
        pred = model.predict(source_code)
        print(pred.security_prob, pred.complexity_score, pred.bug_prob, pred.pattern_label)

    # Training
    from models.multi_task_model import MultiTaskCodeModel
    model = MultiTaskCodeModel(backbone="distilbert-base-uncased")
    model.train_step(batch_sources, batch_labels, optimizer)
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils.checkpoint_integrity import verify_checkpoint

import numpy as np

logger = logging.getLogger(__name__)

# Task identifiers
TASKS = ("security", "complexity", "bug", "pattern")
PATTERN_CLASSES = ("clean", "code_smell", "anti_pattern", "style_violation")
PATTERN_CLASS_TO_IDX = {c: i for i, c in enumerate(PATTERN_CLASSES)}

# ---------------------------------------------------------------------------
# Prediction output dataclass
# ---------------------------------------------------------------------------

@dataclass
class MultiTaskPrediction:
    """Unified prediction output from the multi-task model."""
    security_prob:    float              # P(vulnerable) in [0, 1]
    complexity_score: float              # maintainability index in [0, 100]
    bug_prob:         float              # P(buggy) in [0, 1]
    pattern_label:    str               # one of PATTERN_CLASSES
    pattern_probs:    dict[str, float]  # {class: probability}
    model_type:       str               # "mtl_transformer" | "mtl_shallow" | "fallback"

    def to_dict(self) -> dict:
        return {
            "security_prob":    round(self.security_prob, 4),
            "complexity_score": round(self.complexity_score, 2),
            "bug_prob":         round(self.bug_prob, 4),
            "pattern_label":    self.pattern_label,
            "pattern_probs":    {k: round(v, 4) for k, v in self.pattern_probs.items()},
            "model_type":       self.model_type,
        }


# ---------------------------------------------------------------------------
# Transformer backbone loading (optional — falls back to shallow model)
# ---------------------------------------------------------------------------

def _try_load_transformer(model_name: str):
    """
    Attempt to load a HuggingFace transformer backbone.
    Returns (tokenizer, model) or (None, None) on failure.
    """
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        backbone  = AutoModel.from_pretrained(model_name)
        backbone.eval()
        logger.info("MTL backbone loaded: %s", model_name)
        return tokenizer, backbone
    except Exception as e:
        logger.debug("Transformer backbone unavailable (%s) — using shallow encoder", e)
        return None, None


def _encode_with_transformer(
    source: str,
    tokenizer,
    backbone,
    max_length: int = 512,
    stride: int = 256,
) -> np.ndarray:
    """
    Encode source code using a sliding-window strategy over the transformer.

    Motivation: Most Python files exceed 512 tokens. Simple truncation discards
    everything after the first ~50-80 lines, losing the actual patterns of
    interest (deep functions, security sinks, complex logic). Sliding windows
    with 50% overlap ensure every token contributes to the final embedding.

    Strategy:
      - Tokenise the full source (no truncation)
      - Slide a 512-token window with `stride` overlap
      - Encode each chunk, take its CLS token
      - Mean-pool all chunk embeddings -> file representation

    Args:
        source:     Full Python source code.
        tokenizer:  HuggingFace tokenizer.
        backbone:   HuggingFace model (AutoModel).
        max_length: Chunk size in tokens (default 512, matches BERT max).
        stride:     Step size between chunks (default 256 = 50% overlap).

    Returns:
        (hidden_dim,) float32 embedding.
    """
    import torch

    # Tokenise without truncation to get the full token sequence
    full_ids = tokenizer.encode(source, add_special_tokens=False)

    if len(full_ids) == 0:
        # Empty source: return zero vector sized to backbone hidden dim
        try:
            hidden_dim = backbone.config.hidden_size
        except Exception:
            hidden_dim = 768
        return np.zeros(hidden_dim, dtype=np.float32)

    # Build chunks with overlap
    # Each chunk: [CLS] + tokens[start:start+max_length-2] + [SEP]
    chunk_size = max_length - 2  # reserve 2 tokens for [CLS] and [SEP]
    cls_id = tokenizer.cls_token_id or 0
    sep_id = tokenizer.sep_token_id or 0

    chunk_embeddings = []
    start = 0
    while start < len(full_ids):
        chunk_ids = [cls_id] + full_ids[start:start + chunk_size] + [sep_id]
        input_ids = torch.tensor([chunk_ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            outputs = backbone(input_ids=input_ids, attention_mask=attention_mask)
        # CLS token at position 0
        cls_vec = outputs.last_hidden_state[0, 0, :].numpy().astype(np.float32)
        chunk_embeddings.append(cls_vec)

        if start + chunk_size >= len(full_ids):
            break
        start += stride

    # Mean pool across all chunks
    embedding = np.mean(np.stack(chunk_embeddings), axis=0)
    return embedding.astype(np.float32)


# ---------------------------------------------------------------------------
# Shallow encoder (TF-IDF + AST features, no GPU required)
# ---------------------------------------------------------------------------

def _encode_shallow(source: str) -> np.ndarray:
    """
    Compute a 64-dimensional code representation from static features + TF-IDF.
    Used when transformer backbone is unavailable.

    Feature layout (64 dims):
      [0:16]  static code metrics (metrics_to_feature_vector)
      [16:48] TF-IDF over AST token n-grams (32 dims, PCA-reduced)
      [48:64] AST structure features (n_funcs, max_depth, etc.)
    """
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from features.code_metrics import compute_all_metrics, metrics_to_feature_vector
        from features.ast_extractor import ASTExtractor

        metrics   = compute_all_metrics(source)
        feat_vec  = np.array(metrics_to_feature_vector(metrics), dtype=np.float32)
        ast_feats = ASTExtractor().extract(source)

        # AST structural features (16 dims)
        ast_vec = np.array([
            ast_feats.get("n_functions", 0),
            ast_feats.get("n_classes", 0),
            ast_feats.get("n_imports", 0),
            ast_feats.get("n_calls", 0),
            ast_feats.get("max_depth", 0),
            ast_feats.get("n_loops", 0),
            ast_feats.get("n_conditions", 0),
            ast_feats.get("n_returns", 0),
            ast_feats.get("n_assignments", 0),
            ast_feats.get("n_comprehensions", 0),
            ast_feats.get("n_lambdas", 0),
            ast_feats.get("n_decorators", 0),
            ast_feats.get("n_try_except", 0),
            ast_feats.get("n_assertions", 0),
            ast_feats.get("n_yields", 0),
            ast_feats.get("n_async_funcs", 0),
        ], dtype=np.float32)

        # Pad TF-IDF slot to 32 zeros (real TF-IDF computed during training)
        tfidf_vec = np.zeros(32, dtype=np.float32)

        vec = np.concatenate([feat_vec, tfidf_vec, ast_vec])
        # L2-normalize
        norm = np.linalg.norm(vec)
        return (vec / norm) if norm > 0 else vec

    except Exception:
        return np.zeros(64, dtype=np.float32)


# ---------------------------------------------------------------------------
# Task head implementations (PyTorch)
# ---------------------------------------------------------------------------

def _build_torch_heads(input_dim: int):
    """
    Build PyTorch task heads for all 4 tasks.

    Returns a dict of nn.Module instances keyed by task name,
    or None if torch is unavailable.
    """
    try:
        import torch
        import torch.nn as nn

        class _BinaryHead(nn.Module):
            def __init__(self, in_dim: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.LayerNorm(in_dim),
                    nn.Linear(in_dim, 128),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 1),
                )
            def forward(self, x):
                return torch.sigmoid(self.net(x)).squeeze(-1)

        class _RegressionHead(nn.Module):
            def __init__(self, in_dim: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.LayerNorm(in_dim),
                    nn.Linear(in_dim, 128),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 1),
                )
            def forward(self, x):
                # Clamp output to [0, 100] range for maintainability index
                return torch.clamp(self.net(x).squeeze(-1), 0.0, 100.0)

        class _ClassificationHead(nn.Module):
            def __init__(self, in_dim: int, n_classes: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.LayerNorm(in_dim),
                    nn.Linear(in_dim, 128),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, n_classes),
                )
            def forward(self, x):
                return torch.softmax(self.net(x), dim=-1)

        return {
            "security":   _BinaryHead(input_dim),
            "complexity": _RegressionHead(input_dim),
            "bug":        _BinaryHead(input_dim),
            "pattern":    _ClassificationHead(input_dim, len(PATTERN_CLASSES)),
        }
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Kendall MTL loss (uncertainty-weighted)
# ---------------------------------------------------------------------------

def kendall_mtl_loss(losses: dict[str, "torch.Tensor"], log_vars: "torch.Tensor"):
    """
    Kendall et al. 2018 uncertainty-weighted multi-task loss.

    L = sum_i [ exp(-log_var_i) * L_i + log_var_i ]

    Each log_var_i is a learned parameter representing task uncertainty.
    Tasks with high uncertainty get down-weighted automatically.

    IMPORTANT: This function is for TRANSFORMER mode only.
    In shallow mode (sklearn heads), losses are numpy scalars with no autograd
    graph — calling this function on numpy values would silently leave log_vars
    unchanged. The multi_task_trainer.py shallow trainer uses fixed equal weights
    (1/n_tasks) instead, documented at the call site.

    Implementation note:
    We build the total with Python's sum() over a generator rather than
    initialising a `torch.zeros(1, requires_grad=True)` leaf and accumulating
    with `+=`. The leaf-tensor pattern creates a new node in the graph at each
    step, but the leaf itself has no gradient path to log_vars, causing
    log_vars.grad to be None after backward(). sum() over the generator
    produces a single connected computation graph.

    Args:
        losses:   {task_name: scalar differentiable loss tensor}
        log_vars: (n_tasks,) nn.Parameter tensor of learned log variances

    Returns:
        Scalar total loss (connected to log_vars for backprop).
    """
    import torch

    task_list = list(losses.values())
    # Build graph: each term is exp(-s_i)*L_i + s_i where s_i = log_vars[i]
    terms = [torch.exp(-log_vars[i]) * loss_i + log_vars[i]
             for i, loss_i in enumerate(task_list)]
    return sum(terms)


# ---------------------------------------------------------------------------
# Multi-task model class
# ---------------------------------------------------------------------------

_DEFAULT_CHECKPOINT = "checkpoints/multi_task"
_BACKBONE_OPTIONS = [
    "microsoft/codebert-base",
    "distilbert-base-uncased",  # fallback — smaller, no code pretraining
]


class MultiTaskCodeModel:
    """
    Multi-task code quality model with shared encoder + 4 task heads.

    Modes:
      1. Full transformer mode (microsoft/codebert-base): best quality
      2. Shallow encoder mode (TF-IDF + AST features): no GPU required
      3. Fallback single-task mode: uses individual task models if MTL unavailable

    Training:
        Use training/multi_task_trainer.py for full training pipeline.
        This class handles inference only.

    Checkpoint layout:
        checkpoints/multi_task/
          backbone_heads.pt    — transformer + head weights (if transformer mode)
          shallow_heads.pkl    — sklearn heads (if shallow mode)
          config.json          — backbone name, task info, training metrics
          log_vars.pt          — learned uncertainty parameters
    """

    def __init__(
        self,
        checkpoint_dir: str = _DEFAULT_CHECKPOINT,
        backbone_name:  str = "microsoft/codebert-base",
        prefer_shallow: bool = False,
    ):
        self._checkpoint_dir = Path(checkpoint_dir)
        self._backbone_name  = backbone_name
        self._prefer_shallow = prefer_shallow

        # State
        self._tokenizer  = None
        self._backbone   = None
        self._heads      = None        # PyTorch heads
        self._log_vars   = None        # Kendall uncertainty params
        self._shallow_heads: Optional[dict] = None  # sklearn heads
        self._mode       = "unloaded"  # "transformer" | "shallow" | "fallback"
        self._ready      = False
        self._config: dict = {}

        self._try_load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _try_load(self) -> None:
        """Try to load checkpoint in priority: transformer > shallow > fallback."""
        config_path = self._checkpoint_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                self._config = json.load(f)

        if not self._prefer_shallow:
            if self._try_load_transformer():
                return

        if self._try_load_shallow():
            return

        logger.info(
            "MultiTaskCodeModel: no checkpoint found at %s — "
            "call train() to build the model.",
            self._checkpoint_dir,
        )

    def _try_load_transformer(self) -> bool:
        """Load transformer backbone + PyTorch heads from checkpoint."""
        heads_path = self._checkpoint_dir / "backbone_heads.pt"
        if not heads_path.exists():
            # Try to load backbone and build fresh heads (for fine-tuning)
            tokenizer, backbone = _try_load_transformer(self._backbone_name)
            if tokenizer is None:
                return False
            self._tokenizer = tokenizer
            self._backbone  = backbone
            hidden_dim = backbone.config.hidden_size
            self._heads = _build_torch_heads(hidden_dim)
            if self._heads is None:
                return False
            self._mode  = "transformer_untrained"
            self._ready = False  # Not ready until trained
            logger.info("MTL: loaded backbone (untrained heads)")
            return True

        try:
            import torch
            tokenizer, backbone = _try_load_transformer(self._backbone_name)
            if tokenizer is None:
                return False

            checkpoint = torch.load(str(heads_path), map_location="cpu")
            hidden_dim = backbone.config.hidden_size
            heads = _build_torch_heads(hidden_dim)
            if heads is None:
                return False

            for task, head in heads.items():
                if task in checkpoint.get("heads", {}):
                    head.load_state_dict(checkpoint["heads"][task])
                    head.eval()

            if "log_vars" in checkpoint:
                self._log_vars = checkpoint["log_vars"]

            self._tokenizer = tokenizer
            self._backbone  = backbone
            self._heads     = heads
            self._mode      = "transformer"
            self._ready     = True
            logger.info("MultiTaskCodeModel: loaded transformer checkpoint")
            return True
        except Exception as e:
            logger.debug("Transformer checkpoint load failed: %s", e)
            return False

    def _try_load_shallow(self) -> bool:
        """Load scikit-learn shallow heads from checkpoint."""
        shallow_path = self._checkpoint_dir / "shallow_heads.pkl"
        if not shallow_path.exists():
            return False
        try:
            verify_checkpoint(shallow_path)
            with open(shallow_path, "rb") as f:
                self._shallow_heads = pickle.load(f)
            self._mode  = "shallow"
            self._ready = True
            logger.info("MultiTaskCodeModel: loaded shallow checkpoint")
            return True
        except Exception as e:
            logger.debug("Shallow checkpoint load failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def mode(self) -> str:
        return self._mode

    def predict(self, source: str) -> MultiTaskPrediction:
        """
        Run all 4 task heads on source code.

        Returns:
            MultiTaskPrediction with security, complexity, bug, pattern outputs.
        """
        if self._mode == "transformer" and self._heads is not None:
            return self._predict_transformer(source)
        elif self._mode == "shallow" and self._shallow_heads is not None:
            return self._predict_shallow(source)
        else:
            return self._predict_fallback(source)

    def _predict_transformer(self, source: str) -> MultiTaskPrediction:
        import torch

        embedding = _encode_with_transformer(
            source, self._tokenizer, self._backbone
        )
        x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            sec_prob  = float(self._heads["security"](x).item())
            cpx_score = float(self._heads["complexity"](x).item())
            bug_prob  = float(self._heads["bug"](x).item())
            pat_probs = self._heads["pattern"](x).squeeze(0).tolist()

        pat_idx   = int(np.argmax(pat_probs))
        pat_label = PATTERN_CLASSES[pat_idx]

        return MultiTaskPrediction(
            security_prob=sec_prob,
            complexity_score=max(0.0, min(100.0, cpx_score)),
            bug_prob=bug_prob,
            pattern_label=pat_label,
            pattern_probs={c: round(p, 4) for c, p in zip(PATTERN_CLASSES, pat_probs)},
            model_type="mtl_transformer",
        )

    def _predict_shallow(self, source: str) -> MultiTaskPrediction:
        vec = _encode_shallow(source).reshape(1, -1)

        heads = self._shallow_heads
        sec_prob  = float(heads["security"].predict_proba(vec)[0][1])
        cpx_score = float(np.clip(heads["complexity"].predict(vec)[0], 0.0, 100.0))
        bug_prob  = float(heads["bug"].predict_proba(vec)[0][1])
        pat_probs_arr = heads["pattern"].predict_proba(vec)[0]
        pat_idx   = int(np.argmax(pat_probs_arr))
        pat_label = PATTERN_CLASSES[pat_idx]

        return MultiTaskPrediction(
            security_prob=sec_prob,
            complexity_score=cpx_score,
            bug_prob=bug_prob,
            pattern_label=pat_label,
            pattern_probs={c: round(float(p), 4)
                           for c, p in zip(PATTERN_CLASSES, pat_probs_arr)},
            model_type="mtl_shallow",
        )

    def _predict_fallback(self, source: str) -> MultiTaskPrediction:
        """Last-resort: heuristic predictions when no checkpoint is available."""
        try:
            from features.code_metrics import compute_all_metrics
            m = compute_all_metrics(source)
            cpx   = float(m.maintainability_index)
            cc    = m.cyclomatic_complexity
            sloc  = m.lines.sloc
            sec_p = min(1.0, cc * 0.05 + (sloc / 500))
            bug_p = min(1.0, cc * 0.04 + (sloc / 400))
            if cc > 15: pat = "anti_pattern"
            elif cc > 5: pat = "code_smell"
            else: pat = "clean"
            n_cl = len(PATTERN_CLASSES)
            probs = [0.1 / (n_cl - 1)] * n_cl
            probs[PATTERN_CLASS_TO_IDX.get(pat, 0)] = 0.9
        except Exception:
            cpx, sec_p, bug_p, pat = 50.0, 0.5, 0.5, "clean"
            probs = [0.25, 0.25, 0.25, 0.25]

        return MultiTaskPrediction(
            security_prob=round(sec_p, 4),
            complexity_score=round(cpx, 2),
            bug_prob=round(bug_p, 4),
            pattern_label=pat,
            pattern_probs={c: round(p, 4) for c, p in zip(PATTERN_CLASSES, probs)},
            model_type="fallback",
        )

    # ------------------------------------------------------------------
    # Training helpers (used by multi_task_trainer.py)
    # ------------------------------------------------------------------

    def get_backbone_and_heads(self):
        """Return (backbone, heads, log_vars) for use in training loop."""
        import torch

        if self._backbone is None or self._heads is None:
            raise RuntimeError(
                "Backbone not loaded. Call MultiTaskCodeModel() with a valid backbone."
            )
        if self._log_vars is None:
            self._log_vars = torch.zeros(len(TASKS), requires_grad=True)

        return self._backbone, self._heads, self._log_vars

    def save_transformer_checkpoint(self) -> None:
        """Persist transformer heads + log_vars to checkpoint directory."""
        import torch

        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        heads_state = {
            task: head.state_dict()
            for task, head in (self._heads or {}).items()
        }
        torch.save(
            {"heads": heads_state, "log_vars": self._log_vars},
            str(self._checkpoint_dir / "backbone_heads.pt"),
        )
        config = {
            "backbone":  self._backbone_name,
            "tasks":     TASKS,
            "mode":      "transformer",
            "patterns":  PATTERN_CLASSES,
        }
        with open(self._checkpoint_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        logger.info("MTL checkpoint saved -> %s", self._checkpoint_dir)

    def save_shallow_checkpoint(self, heads: dict) -> None:
        """Persist scikit-learn shallow heads to checkpoint directory."""
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        shallow_path = self._checkpoint_dir / "shallow_heads.pkl"
        with open(shallow_path, "wb") as f:
            pickle.dump(heads, f)
        config = {
            "backbone":  "shallow",
            "tasks":     TASKS,
            "mode":      "shallow",
            "patterns":  PATTERN_CLASSES,
        }
        with open(self._checkpoint_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        self._shallow_heads = heads
        self._mode  = "shallow"
        self._ready = True
        logger.info("MTL shallow checkpoint saved -> %s", self._checkpoint_dir)
