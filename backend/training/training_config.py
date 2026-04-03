"""
Training Configuration and Reproducibility
============================================
Centralised configuration for all IntelliCode training scripts.

Provides:
  1. Deterministic seed setting (Python, NumPy, PyTorch, CUDA)
  2. Hydra-style dataclass configs for every model
  3. Config validation with clear error messages
  4. Config serialisation to JSON for experiment provenance

Every training run saves its config alongside model checkpoints so any result
can be reproduced exactly from the stored config alone.

Usage:
    from training.training_config import (
        ComplexityConfig, SecurityConfig, BugConfig, PatternConfig,
        MultiTaskConfig, set_global_seed, load_config,
    )

    cfg = ComplexityConfig(n_estimators=500, max_depth=6)
    cfg.validate()
    set_global_seed(cfg.seed)
    cfg.save("checkpoints/complexity/train_config.json")
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global seed setter — deterministic training across all libraries
# ---------------------------------------------------------------------------

def set_global_seed(seed: int = 42) -> None:
    """
    Set all random seeds for fully deterministic training.

    Covers: Python random, NumPy, PyTorch CPU+CUDA, PYTHONHASHSEED.
    Also disables PyTorch cuDNN auto-tuning (trades speed for determinism).

    NOTE: Even with all seeds set, non-determinism can arise from:
      - Floating-point reordering in multi-threaded BLAS ops
      - CUDA atomic operations (non-deterministic by spec)
    For exact reproducibility, set n_jobs=1 and CUDA_VISIBLE_DEVICES=''.
    """
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    except ImportError:
        pass

    logger.debug("Global seed set to %d", seed)


# ---------------------------------------------------------------------------
# Base config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Base configuration shared across all training scripts."""
    seed:            int   = 42
    test_split:      float = 0.20
    # CRITICAL: n_jobs=1 on Windows — multiprocessing with n_jobs>1 causes
    # OOM/pickle errors (see training memory note in MEMORY.md)
    n_jobs:          int   = 1
    output_dir:      str   = "checkpoints"
    experiment_name: str   = "experiment"
    log_level:       str   = "INFO"

    def validate(self) -> None:
        if not (0.05 <= self.test_split <= 0.40):
            raise ValueError(f"test_split={self.test_split} must be in [0.05, 0.40]")
        if self.n_jobs != 1 and os.name == "nt":
            raise ValueError(
                "n_jobs > 1 causes OOM/pickle errors on Windows. "
                "Set n_jobs=1. See MEMORY.md."
            )
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        logger.info("Config saved -> %s", path)

    @classmethod
    def load(cls, path: str) -> "TrainingConfig":
        with open(path) as f:
            data = json.load(f)
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def summary(self) -> str:
        w = 50
        lines = [f"{'='*w}", f"  Config: {self.__class__.__name__}", f"{'='*w}"]
        for k, v in asdict(self).items():
            lines.append(f"  {k:<28} {v}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-model configurations
# ---------------------------------------------------------------------------

@dataclass
class ComplexityConfig(TrainingConfig):
    """XGBoost complexity prediction configuration."""
    n_estimators:          int   = 500
    max_depth:             int   = 6
    learning_rate:         float = 0.05
    subsample:             float = 0.8
    colsample_bytree:      float = 0.8
    min_child_weight:      int   = 3
    gamma:                 float = 0.1
    reg_alpha:             float = 0.1
    reg_lambda:            float = 1.0
    early_stopping_rounds: int   = 30
    cv_folds:              int   = 5
    # Cross-project evaluation — 5 diverse repos spanning different domains,
    # sizes, and styles (web framework, CLI, data science, networking, ML).
    # Zimmermann 2009 recommends ≥5 projects for honest cross-project claims.
    use_cross_project:     bool  = True
    test_repos:            list  = field(default_factory=lambda: [
        "scikit-learn/scikit-learn",   # data science / ML library
        "pallets/flask",               # web micro-framework
        "psf/requests",                # HTTP / networking utility
        "pytest-dev/pytest",           # testing framework / CLI tool
        "sqlalchemy/sqlalchemy",       # ORM / large codebase
    ])
    # Conformal prediction (MAPIE)
    fit_conformal:         bool  = True
    conformal_alpha:       float = 0.10   # target coverage = 90%
    output_dir:            str   = "checkpoints/complexity"

    def validate(self) -> None:
        super().validate()
        if not (0.0 < self.learning_rate <= 0.3):
            raise ValueError(f"learning_rate={self.learning_rate} out of range (0, 0.3]")
        if not (1 <= self.max_depth <= 12):
            raise ValueError(f"max_depth={self.max_depth} out of range [1, 12]")
        if not (0.0 < self.conformal_alpha < 0.5):
            raise ValueError(f"conformal_alpha={self.conformal_alpha} must be in (0, 0.5)")


@dataclass
class SecurityConfig(TrainingConfig):
    """Security detection RF+CNN ensemble configuration."""
    rf_n_estimators:     int   = 200
    rf_max_depth:        Optional[int] = None
    rf_class_weight:     str   = "balanced"
    cnn_epochs:          int   = 15
    cnn_lr:              float = 1e-3
    cnn_batch_size:      int   = 64
    max_tokens:          int   = 512
    max_vocab:           int   = 10_000
    ensemble_rf_weight:  float = 0.55
    ensemble_cnn_weight: float = 0.45
    # Real CVE labels (replaces circular heuristic labeling)
    use_cvefixes:        bool  = True
    cvefixes_limit:      int   = 1500
    output_dir:          str   = "checkpoints/security"

    def validate(self) -> None:
        super().validate()
        total = self.ensemble_rf_weight + self.ensemble_cnn_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Ensemble weights must sum to 1.0, got {total}")
        if self.max_tokens > 512:
            raise ValueError("max_tokens > 512 exceeds standard BERT context window")


@dataclass
class BugConfig(TrainingConfig):
    """Bug prediction LR+XGBoost+MLP ensemble configuration."""
    # XGBoost
    xgb_n_estimators:     int   = 200
    xgb_max_depth:        int   = 5
    xgb_lr:               float = 0.05
    xgb_subsample:        float = 0.8
    # Set to (n_negatives / n_positives) for class imbalance
    xgb_scale_pos_weight: float = 1.0
    # LR
    lr_C:                 float = 1.0
    lr_max_iter:          int   = 500
    # MLP
    mlp_hidden:           tuple = (64, 32)
    mlp_epochs:           int   = 30
    mlp_lr:               float = 1e-3
    mlp_batch_size:       int   = 64
    # Ensemble weights (must sum to 1.0)
    ensemble_weights:     dict  = field(default_factory=lambda: {
        "lr": 0.30, "xgb": 0.50, "mlp": 0.20
    })
    # JIT features (Kamei et al. 2013, all 14 features)
    use_jit_features:     bool  = True
    # SZZ-based bug-introducing commit labels (precision-oriented)
    use_szz_labels:       bool  = False
    output_dir:           str   = "checkpoints/bug_predictor"

    def validate(self) -> None:
        super().validate()
        total = sum(self.ensemble_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Bug ensemble weights sum to {total:.4f}, must be 1.0. "
                f"Weights: {self.ensemble_weights}"
            )


@dataclass
class PatternConfig(TrainingConfig):
    """Pattern recognition RF configuration."""
    n_estimators:        int   = 300
    max_depth:           Optional[int] = None
    class_weight:        str   = "balanced"
    calibration_method:  str   = "sigmoid"   # Platt scaling
    cv_folds:            int   = 5
    output_dir:          str   = "checkpoints/pattern"

    def validate(self) -> None:
        super().validate()
        if self.calibration_method not in ("sigmoid", "isotonic"):
            raise ValueError(
                f"calibration_method must be 'sigmoid' or 'isotonic', "
                f"got '{self.calibration_method}'"
            )


@dataclass
class MultiTaskConfig(TrainingConfig):
    """Multi-task learning configuration (shared encoder + 4 heads)."""
    backbone:             str   = "microsoft/codebert-base"
    mode:                 str   = "shallow"   # "shallow" | "transformer"
    shallow_n_estimators: int   = 200
    transformer_epochs:   int   = 10
    transformer_lr:       float = 2e-5
    warmup_epochs:        int   = 2
    batch_size:           int   = 32
    tasks:                list  = field(default_factory=lambda: [
        "security", "complexity", "bug", "pattern"
    ])
    output_dir:           str   = "checkpoints/multi_task"

    def validate(self) -> None:
        super().validate()
        if self.mode not in ("shallow", "transformer"):
            raise ValueError(f"mode must be 'shallow' or 'transformer', got '{self.mode}'")
        valid_tasks = {"security", "complexity", "bug", "pattern"}
        unknown = set(self.tasks) - valid_tasks
        if unknown:
            raise ValueError(f"Unknown tasks: {unknown}. Valid: {valid_tasks}")


# ---------------------------------------------------------------------------
# Config registry
# ---------------------------------------------------------------------------

CONFIG_REGISTRY: dict[str, type] = {
    "complexity": ComplexityConfig,
    "security":   SecurityConfig,
    "bug":        BugConfig,
    "pattern":    PatternConfig,
    "multi_task": MultiTaskConfig,
}


def load_config(task: str, config_path: Optional[str] = None) -> TrainingConfig:
    """
    Load a validated config for the given task.

    Priority:
      1. config_path JSON (if provided and exists)
      2. Default dataclass values

    Args:
        task:        One of complexity | security | bug | pattern | multi_task
        config_path: Optional path to a saved config JSON

    Returns:
        Populated and validated config dataclass instance.
    """
    cls = CONFIG_REGISTRY.get(task)
    if cls is None:
        raise ValueError(
            f"Unknown task: '{task}'. "
            f"Valid options: {list(CONFIG_REGISTRY.keys())}"
        )

    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            data = json.load(f)
        valid_keys = set(cls.__dataclass_fields__.keys())
        cfg = cls(**{k: v for k, v in data.items() if k in valid_keys})
        logger.info("Loaded config from %s", config_path)
    else:
        cfg = cls()

    cfg.validate()
    return cfg
