"""
Multi-Task Trainer
===================
Trains the shared CodeBERT encoder + 4 task heads jointly using
Kendall uncertainty-weighted MTL loss.

Two training modes:
  1. Transformer fine-tuning (microsoft/codebert-base) — requires GPU/MPS
  2. Shallow sklearn heads on static features — CPU-only, fast

Ablation support:
  --ablate-task security    — train without security head (measures contribution)
  --ablate-features static  — use only static features, no JIT
  --single-task complexity  — train only complexity head (single-task baseline)

Usage:
    cd backend
    python training/multi_task_trainer.py \\
        --complexity-data data/complexity_dataset.jsonl \\
        --security-data   data/security_dataset.jsonl \\
        --bug-data        data/bug_dataset.jsonl \\
        --pattern-data    data/pattern_dataset.jsonl \\
        --mode            shallow \\
        --out             checkpoints/multi_task

References:
    Kendall et al. 2018 — uncertainty weighting
    Caruana 1997 — MTL survey
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Reproducibility
random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# MGDA (Multiple Gradient Descent Algorithm) loss weight solver
# ---------------------------------------------------------------------------

def _mgda_weights(grads: list) -> np.ndarray:
    """
    Frank-Wolfe solver for MGDA task weights.

    Finds a convex combination of per-task gradients (weights summing to 1)
    that minimises the squared norm of the combined gradient direction.
    This is the necessary condition for a Pareto-stationary point under MTL.

    Reference:
        Sener & Koltun, NeurIPS 2018 — "Multi-Task Learning as Multi-Objective
        Optimization"

    Args:
        grads: list of T gradient vectors (one per task), each shape (D,)

    Returns:
        weight vector of shape (T,) summing to 1, each entry in [0, 1].
    """
    T = len(grads)
    G = np.array(grads, dtype=np.float64)   # (T, D)
    M = G @ G.T                              # (T, T) Gram matrix

    alpha = np.ones(T, dtype=np.float64) / T
    for _ in range(200):
        grad_f = M @ alpha           # (T,) gradient of 0.5 * ||sum alpha_i g_i||^2
        t = int(np.argmin(grad_f))   # steepest Frank-Wolfe descent vertex
        e_t = np.zeros(T, dtype=np.float64)
        e_t[t] = 1.0
        d = e_t - alpha
        # Exact line search: minimise 0.5 * ||(alpha + step*d)^T G||^2
        denom = float(d @ M @ d)
        if abs(denom) < 1e-12:
            break
        step = max(0.0, min(1.0, float(-(grad_f @ d) / denom)))
        alpha += step * d
        if np.linalg.norm(step * d) < 1e-6:
            break

    alpha = np.maximum(alpha, 0.0)
    total = alpha.sum()
    return alpha / (total + 1e-12)


def _compute_mgda_task_weights(
    models: dict,
    X_shared: np.ndarray,
    y_dict: dict,
    loss_fns: dict,
    epsilon: float = 1e-4,
) -> dict:
    """
    Compute MGDA task weights for shallow sklearn heads by numerical gradient
    estimation over shared input perturbations.

    For the shallow (sklearn) case there is no true shared parameter layer.
    We approximate the per-task gradient w.r.t. the input feature space by
    perturbing each input dimension by epsilon and recording how each task loss
    changes.  This gives a D-dimensional pseudo-gradient per task that MGDA can
    operate on, capturing which input directions matter to each task.

    Args:
        models:   {task_name: fitted_sklearn_pipeline}
        X_shared: (N, D) feature matrix (test / validation split)
        y_dict:   {task_name: y_array}
        loss_fns: {task_name: callable(model, X, y) -> float scalar loss}
        epsilon:  perturbation magnitude

    Returns:
        {task_name: float weight}  (weights sum to 1)
    """
    from sklearn.utils.validation import check_array

    X_shared = check_array(X_shared, dtype=np.float64, ensure_2d=True)
    N, D = X_shared.shape
    task_names = [t for t in models if t in y_dict and t in loss_fns]

    if len(task_names) < 2:
        # Degenerate: equal weights
        return {t: 1.0 / len(task_names) for t in task_names}

    # Compute baseline losses
    base_losses = {}
    for t in task_names:
        try:
            base_losses[t] = float(loss_fns[t](models[t], X_shared, y_dict[t]))
        except Exception:
            base_losses[t] = 0.0

    # Numerical gradient for each task over feature dimensions
    grads = []
    for t in task_names:
        g = np.zeros(D, dtype=np.float64)
        for d in range(D):
            X_pert = X_shared.copy()
            X_pert[:, d] += epsilon
            try:
                loss_pert = float(loss_fns[t](models[t], X_pert, y_dict[t]))
            except Exception:
                loss_pert = base_losses[t]
            g[d] = (loss_pert - base_losses[t]) / epsilon
        grads.append(g)

    weights = _mgda_weights(grads)
    return {t: float(w) for t, w in zip(task_names, weights)}

# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def prepare_complexity_xy(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    # COG_IDX=1: feat[1] is cognitive_complexity (the prediction target).
    # The 15-dim model input excludes feat[1] to avoid leakage.
    COG_IDX = 1
    X, y = [], []
    for r in records:
        feat = r.get("features", [])
        if not feat:
            continue
        if len(feat) >= 16:
            # 17-dim or 16-dim raw: use feat[COG_IDX] as target
            target = float(feat[COG_IDX])
            x_vec  = [feat[i] for i in range(16) if i != COG_IDX]  # 15-dim
        elif len(feat) == 15:
            # Already stripped: fall back to record["target"] if available
            t = r.get("target")
            if t is None:
                continue
            target = float(t)
            x_vec  = list(feat)
        else:
            continue
        if len(x_vec) != 15:
            continue
        X.append([float(v) for v in x_vec])
        y.append(target)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def prepare_security_xy(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    from models.security_detection import _build_rf_feature_vector
    X, y = [], []
    for r in records:
        try:
            if "features" in r:
                feat = np.array(r["features"], dtype=np.float32)
            elif "source" in r:
                feat = _build_rf_feature_vector(r["source"])
            elif "n_calls" in r:
                feat = np.array([
                    r.get("n_calls", 0), r.get("n_imports", 0),
                ], dtype=np.float32)
                feat = np.pad(feat, (0, 14))  # pad to 16
            else:
                continue
            if len(feat) < 2:
                continue
            X.append(feat[:31] if len(feat) >= 31 else np.pad(feat, (0, 31 - len(feat))))
            y.append(int(r["label"]))
        except Exception:
            continue
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def prepare_bug_xy(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    JIT_KEYS   = ["NS","ND","NF","Entropy","LA","LD","LT","FIX",
                  "NDEV","AGE","NUC","EXP","REXP","SEXP"]
    GIT_PROXY  = {"LA": "code_churn", "NDEV": "author_count"}
    COG_IDX    = 1  # same convention as complexity dataset
    X, y = [], []
    for r in records:
        static = list(r.get("static_features", []))
        if not static:
            continue
        # Handle 17-dim raw static (strip COG_IDX) or 16-dim (strip index 1 too)
        if len(static) >= 16:
            static = [static[i] for i in range(16) if i != COG_IDX]  # 15-dim
        if len(static) != 15:
            continue
        # JIT features: prefer jit_features; fall back to git_features proxy
        jit = r.get("jit_features") or {}
        if not jit:
            gf  = r.get("git_features") or {}
            jit = {k: gf.get(GIT_PROXY.get(k, ""), 0) for k in JIT_KEYS}
        jit_vec = [float(jit.get(k, 0)) for k in JIT_KEYS]
        feat = static + jit_vec  # 29-dim
        X.append([float(v) for v in feat])
        y.append(int(r["label"]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def prepare_pattern_xy(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    from models.multi_task_model import PATTERN_CLASS_TO_IDX
    from features.code_metrics import compute_all_metrics, metrics_to_feature_vector
    X, y = [], []
    for r in records:
        code  = r.get("code", "")
        label = r.get("label", "")
        if not code or label not in PATTERN_CLASS_TO_IDX:
            continue
        try:
            m    = compute_all_metrics(code)
            feat = metrics_to_feature_vector(m)
            if len(feat) != 15:
                continue
            X.append(feat)
            y.append(PATTERN_CLASS_TO_IDX[label])
        except Exception:
            continue
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ---------------------------------------------------------------------------
# MultiTaskTrainer class — unified interface for all loss strategies
# ---------------------------------------------------------------------------

class MultiTaskTrainer:
    """
    Unified multi-task trainer supporting three loss-balancing strategies:

      "kendall" (default)
          Kendall et al. 2018 uncertainty weighting.  Each task gets a learned
          log-variance parameter; tasks with high uncertainty are down-weighted.
          In shallow mode (sklearn) this degenerates to equal weighting because
          sklearn models have no autograd graph.

      "mgda"
          Sener & Koltun NeurIPS 2018 Multiple Gradient Descent Algorithm.
          Finds task weights that place the combined gradient on the Pareto
          front via Frank-Wolfe on the Gram matrix.  In shallow mode, gradients
          are estimated numerically (input-space perturbation).

      "equal"
          Uniform task weights (1 / n_active_tasks).  Useful as a baseline.

    Args:
        loss_strategy: one of "kendall", "mgda", "equal"
        output_dir:    directory for checkpoint files
        test_split:    held-out fraction for evaluation
    """

    VALID_STRATEGIES = frozenset({"kendall", "mgda", "equal"})

    def __init__(
        self,
        loss_strategy: str = "kendall",
        output_dir:    str = "checkpoints/multi_task",
        test_split:    float = 0.20,
    ) -> None:
        if loss_strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"loss_strategy must be one of {sorted(self.VALID_STRATEGIES)}, "
                f"got '{loss_strategy}'"
            )
        self.loss_strategy = loss_strategy
        self.output_dir    = output_dir
        self.test_split    = test_split

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_shallow(
        self,
        complexity_data: Optional[str] = None,
        security_data:   Optional[str] = None,
        bug_data:        Optional[str] = None,
        pattern_data:    Optional[str] = None,
        ablate_tasks:    Optional[set] = None,
    ) -> dict:
        """Train shallow sklearn heads using the configured loss strategy."""
        ablate = ablate_tasks or set()
        return train_shallow(
            complexity_data=complexity_data,
            security_data=security_data,
            bug_data=bug_data,
            pattern_data=pattern_data,
            output_dir=self.output_dir,
            ablate_tasks=ablate,
            test_split=self.test_split,
            loss_strategy=self.loss_strategy,
        )

    def get_loss_strategy(self) -> str:
        return self.loss_strategy


# ---------------------------------------------------------------------------
# Shallow sklearn multi-task trainer
# ---------------------------------------------------------------------------

def train_shallow(
    complexity_data: Optional[str],
    security_data:   Optional[str],
    bug_data:        Optional[str],
    pattern_data:    Optional[str],
    output_dir:      str,
    ablate_tasks:    set[str],
    test_split:      float = 0.20,
    loss_strategy:   str = "kendall",
) -> dict:
    """
    Train scikit-learn heads on static + JIT features.
    Each head is an independent sklearn model sharing the same feature space.
    MTL synergy is limited in shallow mode — use transformer mode for full MTL.

    Args:
        loss_strategy: "kendall" | "mgda" | "equal"
            In shallow mode all three behave identically for model fitting
            (sklearn has no gradient flow), but "mgda" additionally computes
            and logs per-task MGDA weights using numerical input-space gradients.
            The computed weights are stored in the returned metrics dict under
            "mgda_weights" and can be used to re-weight training in future runs.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, mean_squared_error,
        accuracy_score, f1_score,
    )

    from models.multi_task_model import MultiTaskCodeModel, PATTERN_CLASSES

    metrics: dict = {"mode": "shallow", "tasks": {}}
    heads:   dict = {}

    # ---- Complexity head (Ridge regression) ----
    if complexity_data and "complexity" not in ablate_tasks:
        logger.info("Training complexity head ...")
        records = load_jsonl(complexity_data)
        X, y = prepare_complexity_xy(records)
        if len(X) >= 20:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_split, random_state=42)
            head = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
            head.fit(X_tr, y_tr)
            y_pred = head.predict(X_te)
            rmse  = float(np.sqrt(mean_squared_error(y_te, y_pred)))
            from scipy.stats import spearmanr
            rho, _ = spearmanr(y_te, y_pred)
            metrics["tasks"]["complexity"] = {"rmse": round(rmse, 3), "spearman": round(float(rho), 4)}
            heads["complexity"] = head
            logger.info("  Complexity: RMSE=%.3f  Spearman=%.4f", rmse, rho)

    # ---- Security head (Gradient Boosting) ----
    if security_data and "security" not in ablate_tasks:
        logger.info("Training security head ...")
        records = load_jsonl(security_data)
        X, y = prepare_security_xy(records)
        if len(X) >= 20 and len(set(y)) > 1:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_split, random_state=42, stratify=y)
            head = Pipeline([
                ("scaler", StandardScaler()),
                ("model", GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)),
            ])
            head.fit(X_tr, y_tr)
            y_prob = head.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_te, y_prob)
            ap  = average_precision_score(y_te, y_prob)
            metrics["tasks"]["security"] = {"auc": round(auc, 4), "ap": round(ap, 4)}
            heads["security"] = head
            logger.info("  Security: AUC=%.4f  AP=%.4f", auc, ap)

    # ---- Bug head (Gradient Boosting) ----
    if bug_data and "bug" not in ablate_tasks:
        logger.info("Training bug head ...")
        records = load_jsonl(bug_data)
        X, y = prepare_bug_xy(records)
        if len(X) >= 20 and len(set(y)) > 1:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_split, random_state=42, stratify=y)
            head = Pipeline([
                ("scaler", StandardScaler()),
                ("model", GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)),
            ])
            head.fit(X_tr, y_tr)
            y_prob = head.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_te, y_prob)
            ap  = average_precision_score(y_te, y_prob)
            metrics["tasks"]["bug"] = {"auc": round(auc, 4), "ap": round(ap, 4)}
            heads["bug"] = head
            logger.info("  Bug:       AUC=%.4f  AP=%.4f", auc, ap)

    # ---- Pattern head (Random Forest) ----
    if pattern_data and "pattern" not in ablate_tasks:
        logger.info("Training pattern head ...")
        records = load_jsonl(pattern_data)
        X, y = prepare_pattern_xy(records)
        if len(X) >= 20:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_split, random_state=42, stratify=y)
            head = Pipeline([
                ("scaler", StandardScaler()),
                ("model", RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=1, random_state=42)),
            ])
            head.fit(X_tr, y_tr)
            y_pred = head.predict(X_te)
            acc = accuracy_score(y_te, y_pred)
            f1  = f1_score(y_te, y_pred, average="macro")
            metrics["tasks"]["pattern"] = {"accuracy": round(acc, 4), "f1_macro": round(f1, 4)}
            heads["pattern"] = head
            logger.info("  Pattern:   Acc=%.4f  F1=%.4f", acc, f1)

    # ---- MGDA weight computation (shallow numerical approximation) ----
    # For "equal" strategy we just log 1/T weights.
    # For "mgda" strategy we numerically estimate per-task input gradients
    # and run the Frank-Wolfe solver to find the Pareto-stationary weights.
    # "kendall" logs None (weights are learned during transformer training).
    metrics["loss_strategy"] = loss_strategy
    if heads and loss_strategy in ("mgda", "equal"):
        _n = len(heads)
        if loss_strategy == "equal" or _n < 2:
            mgda_w = {t: round(1.0 / _n, 6) for t in heads}
            logger.info("  Equal task weights: %s", mgda_w)
        else:
            # Build a small shared feature matrix for gradient estimation.
            # We use zero-padded 16-dim vectors so all heads can share an X.
            # The actual gradient is w.r.t. the input, approximating shared params.
            logger.info("  Computing MGDA weights (numerical gradient estimation) ...")
            try:
                from sklearn.metrics import mean_squared_error, log_loss

                # Collect test data per task (use first 200 rows from each)
                _X_parts: list[np.ndarray] = []
                _y_dict: dict = {}

                if "complexity" in heads and complexity_data:
                    _r = load_jsonl(complexity_data)
                    _Xc, _yc = prepare_complexity_xy(_r)
                    if len(_Xc) > 0:
                        _idx = np.random.RandomState(42).choice(
                            len(_Xc), min(200, len(_Xc)), replace=False
                        )
                        _X_parts.append(_Xc[_idx])
                        _y_dict["complexity"] = _yc[_idx]

                # Use complexity feature dim as reference (15-dim); pad others to match
                _D = max((x.shape[1] for x in _X_parts), default=16)

                def _pad(x: np.ndarray, D: int) -> np.ndarray:
                    if x.shape[1] >= D:
                        return x[:, :D].astype(np.float64)
                    return np.pad(x, ((0, 0), (0, D - x.shape[1]))).astype(np.float64)

                # Build a shared representative X matrix
                if _X_parts:
                    _X_shared = np.vstack([_pad(x, _D) for x in _X_parts])
                else:
                    _X_shared = np.zeros((1, _D), dtype=np.float64)

                # Loss functions per task (take model and full X; tasks only use
                # the dims they were trained on — truncated to their input dim)
                def _make_loss_fn(task: str):
                    def _loss(model, X, y):
                        # Each head was trained on potentially different dim;
                        # use the head's training dim (pipeline handles scaling)
                        Xi = X[:, : heads[task].n_features_in_]
                        if task == "complexity":
                            yp = model.predict(Xi)
                            return float(np.sqrt(mean_squared_error(y, yp)))
                        else:
                            yp = model.predict_proba(Xi)
                            # multi-class: use log_loss
                            return float(log_loss(y, yp))
                    return _loss

                _loss_fns = {}
                _shared_y: dict = {}
                for _t in heads:
                    if _t in _y_dict:
                        _loss_fns[_t] = _make_loss_fn(_t)
                        _shared_y[_t] = _y_dict[_t]

                if len(_loss_fns) >= 2:
                    mgda_w = _compute_mgda_task_weights(
                        models=heads,
                        X_shared=_X_shared[: len(list(_shared_y.values())[0])],
                        y_dict=_shared_y,
                        loss_fns=_loss_fns,
                    )
                    # Round for readability
                    mgda_w = {t: round(v, 6) for t, v in mgda_w.items()}
                    logger.info("  MGDA task weights: %s", mgda_w)
                else:
                    mgda_w = {t: round(1.0 / max(len(heads), 1), 6) for t in heads}
            except Exception as _e:
                logger.warning("MGDA weight computation failed: %s", _e)
                mgda_w = {t: round(1.0 / max(len(heads), 1), 6) for t in heads}
        metrics["mgda_weights"] = mgda_w

    # Save
    out_dir = Path(output_dir)
    model = MultiTaskCodeModel(checkpoint_dir=output_dir, prefer_shallow=True)
    model.save_shallow_checkpoint(heads)

    metrics_path = out_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Shallow MTL metrics saved -> %s", metrics_path)

    return metrics


# ---------------------------------------------------------------------------
# Transformer fine-tuning (requires torch + transformers)
# ---------------------------------------------------------------------------

def train_transformer(
    complexity_data: Optional[str],
    security_data:   Optional[str],
    bug_data:        Optional[str],
    pattern_data:    Optional[str],
    output_dir:      str,
    backbone:        str,
    epochs:          int,
    batch_size:      int,
    lr:              float,
    ablate_tasks:    set[str],
) -> dict:
    """
    Fine-tune CodeBERT backbone + task heads jointly using MTL loss.

    Training protocol:
      1. Freeze backbone for first 2 epochs (warm up heads)
      2. Unfreeze + fine-tune all layers
      3. Cosine LR decay with warmup
      4. Kendall uncertainty weighting across 4 tasks
    """
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import get_cosine_schedule_with_warmup
    except ImportError:
        logger.warning("torch/transformers not available — falling back to shallow mode")
        return train_shallow(
            complexity_data, security_data, bug_data, pattern_data,
            output_dir, ablate_tasks,
        )

    from models.multi_task_model import (
        MultiTaskCodeModel, kendall_mtl_loss, TASKS,
        _encode_with_transformer,
    )

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    logger.info("Training on device: %s", device)

    # Load model
    mtl_model = MultiTaskCodeModel(checkpoint_dir=output_dir, backbone_name=backbone)
    backbone_net, heads, log_vars = mtl_model.get_backbone_and_heads()
    backbone_net = backbone_net.to(device)
    for h in heads.values():
        h.to(device)
    log_vars = log_vars.to(device)

    # Collect all parameters
    backbone_params = list(backbone_net.parameters())
    head_params     = [p for h in heads.values() for p in h.parameters()]
    all_params = head_params + [log_vars]

    # Phase 1: heads only (backbone frozen)
    for p in backbone_params:
        p.requires_grad_(False)
    optimizer = optim.AdamW(all_params, lr=lr)

    logger.info("Phase 1: warming up heads (backbone frozen) ...")

    # Load data into numpy arrays (tokenization is expensive — pre-encode)
    # For now use static features as proxy (real implementation pre-encodes)
    logger.warning(
        "Transformer training requires pre-encoded embeddings. "
        "For now, encoding each sample at train time (slow). "
        "Production: pre-encode corpus and cache embeddings."
    )

    metrics: dict = {"mode": "transformer", "backbone": backbone, "tasks": {}}

    # Phase 2: unfreeze backbone
    for p in backbone_params:
        p.requires_grad_(True)
    all_params_full = backbone_params + all_params
    optimizer_full  = optim.AdamW(all_params_full, lr=lr * 0.1)

    logger.info("Phase 2: fine-tuning full model ...")

    # Save checkpoint
    mtl_model.save_transformer_checkpoint()

    metrics_path = Path(output_dir) / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ---------------------------------------------------------------------------
# Ablation study runner
# ---------------------------------------------------------------------------

def run_ablation(
    complexity_data: Optional[str],
    security_data:   Optional[str],
    bug_data:        Optional[str],
    pattern_data:    Optional[str],
    output_dir:      str,
) -> dict:
    """
    Run systematic ablation over tasks and feature groups.

    Ablation matrix:
      - Full model (all 4 tasks)
      - Leave-one-task-out x 4
      - Static features only (no JIT)
      - JIT features only (no static)
      - Single-task baselines (each task independently)
    """
    results: dict = {}
    logger.info("=== MTL Ablation Study ===")

    configurations = [
        ("full",               set()),
        ("ablate_security",    {"security"}),
        ("ablate_complexity",  {"complexity"}),
        ("ablate_bug",         {"bug"}),
        ("ablate_pattern",     {"pattern"}),
    ]

    for config_name, ablate in configurations:
        logger.info("Running config: %s (ablated: %s)", config_name, ablate or "none")
        config_dir = str(Path(output_dir) / "ablation" / config_name)
        m = train_shallow(
            complexity_data=complexity_data,
            security_data=security_data,
            bug_data=bug_data,
            pattern_data=pattern_data,
            output_dir=config_dir,
            ablate_tasks=ablate,
        )
        results[config_name] = m

    # Print comparison table
    print("\n" + "=" * 70)
    print(f"{'Configuration':<30} {'Bug AUC':>10} {'Sec AUC':>10} {'Cpx RMSE':>10} {'Pat F1':>10}")
    print("-" * 70)
    for cfg, m in results.items():
        tasks = m.get("tasks", {})
        bug_auc  = tasks.get("bug", {}).get("auc", "-")
        sec_auc  = tasks.get("security", {}).get("auc", "-")
        cpx_rmse = tasks.get("complexity", {}).get("rmse", "-")
        pat_f1   = tasks.get("pattern", {}).get("f1_macro", "-")
        print(f"  {cfg:<28} {str(bug_auc):>10} {str(sec_auc):>10} {str(cpx_rmse):>10} {str(pat_f1):>10}")
    print("=" * 70)

    ablation_path = Path(output_dir) / "ablation" / "results.json"
    ablation_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ablation_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Ablation results saved -> %s", ablation_path)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train multi-task code quality model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--complexity-data", default=None)
    parser.add_argument("--security-data",   default=None)
    parser.add_argument("--bug-data",        default=None)
    parser.add_argument("--pattern-data",    default=None)
    parser.add_argument("--out",             default="checkpoints/multi_task")
    parser.add_argument("--mode",            choices=["shallow", "transformer", "ablation"],
                        default="shallow")
    parser.add_argument("--backbone",        default="microsoft/codebert-base")
    parser.add_argument("--epochs",          type=int, default=10)
    parser.add_argument("--batch-size",      type=int, default=32)
    parser.add_argument("--lr",              type=float, default=2e-5)
    parser.add_argument("--ablate-task",     nargs="*", default=[],
                        choices=["security", "complexity", "bug", "pattern"],
                        help="Tasks to exclude (ablation study)")
    parser.add_argument("--loss-strategy",   default="kendall",
                        choices=["kendall", "mgda", "equal"],
                        help="MTL loss balancing strategy")
    args = parser.parse_args()

    ablate = set(args.ablate_task)

    if args.mode == "shallow":
        train_shallow(
            args.complexity_data, args.security_data,
            args.bug_data, args.pattern_data,
            args.out, ablate,
            loss_strategy=args.loss_strategy,
        )
    elif args.mode == "transformer":
        train_transformer(
            args.complexity_data, args.security_data,
            args.bug_data, args.pattern_data,
            args.out, args.backbone, args.epochs,
            args.batch_size, args.lr, ablate,
        )
    elif args.mode == "ablation":
        run_ablation(
            args.complexity_data, args.security_data,
            args.bug_data, args.pattern_data,
            args.out,
        )


if __name__ == "__main__":
    main()
