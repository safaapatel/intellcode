"""
Differential Risk Encoder (DRE)
=================================
Predicts bug risk from feature DIFFERENCES between consecutive commit
snapshots, rather than from the absolute feature values at a single point.

Novel contribution
------------------
Every JIT-SDP model in the literature (Kamei 2013, Ni 2022, Zeng 2021,
McIntosh 2016) trains on absolute feature vectors X_t = [ns, nd, nf, ...].
DRE trains on the DELTA vector:

    DeltaX_t = [X_t - X_{t-1},          (signed change)
                X_t,                      (current absolute value)
                |X_t - X_{t-1}|,          (magnitude of change)
                sign(X_t - X_{t-1})]      (direction of change)

This gives 4 × D input features where D is the number of raw features.

Key insight: the SAME absolute complexity (say, cog=14) carries a very
different risk signal depending on whether:
  a) cog was 14 last commit too (stable) → lower risk
  b) cog jumped from 6 to 14 (+8 in one commit) → higher risk
  c) cog fell from 22 to 14 (improving) → lower risk still

No existing JIT-SDP paper encodes this explicitly as input features.

Architecture: purely feed-forward, no recurrent layers.  DRE uses a
two-layer MLP trained with PyTorch (the only deep-learning component in
IntelliCode, and it only requires a forward pass at inference time).

Gradient / loss: standard binary cross-entropy — the novelty is in the
feature construction, not the loss function.

Fallback: if PyTorch is unavailable or the checkpoint is missing, DRE
falls back to a NumPy logistic regression computed from scratch using
Newton's method (no sklearn).

Outputs
-------
DREResult(
    risk_score:       float   — bug probability from delta-encoded features
    delta_contribution: float — |risk_score_delta - risk_score_static|
                                how much the delta features changed the prediction
    top_delta_features: list[dict] — features with largest |delta| (most changed)
    confidence:       float   — model confidence
    static_score:     float   — score using absolute features only (for comparison)
)
"""

from __future__ import annotations

import json
import logging
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("checkpoints/dre")
MODEL_PATH     = CHECKPOINT_DIR / "dre.pt"
METRICS_PATH   = CHECKPOINT_DIR / "metrics.json"

_HIDDEN_DIM   = 64
_DROPOUT      = 0.2
_LR           = 1e-3
_EPOCHS       = 50
_BATCH_SIZE   = 256
_SEED         = 42

_FEATURE_NAMES = [
    "cyclomatic_complexity", "cognitive_complexity", "sloc",
    "comment_ratio", "halstead_volume", "halstead_difficulty",
    "halstead_effort", "n_bugs_heuristic", "n_long_methods",
    "n_complex_methods", "max_line_length", "avg_line_length",
    "lines_over_80", "ns", "nd",  # number of subsystems/dirs (JIT)
]


# ---------------------------------------------------------------------------
# Delta feature construction
# ---------------------------------------------------------------------------

def build_delta_features(
    x_curr: np.ndarray,
    x_prev: np.ndarray,
) -> np.ndarray:
    """
    Concatenate [delta, absolute, magnitude, sign] for the DRE input.

    Args:
        x_curr: current commit feature vector (D-dim)
        x_prev: previous commit feature vector (D-dim)

    Returns:
        (4*D)-dim delta-encoded feature vector
    """
    x_curr = np.asarray(x_curr, dtype=np.float64)
    x_prev = np.asarray(x_prev, dtype=np.float64)

    # Pad to same length if needed
    D = max(len(x_curr), len(x_prev))
    if len(x_curr) < D:
        x_curr = np.pad(x_curr, (0, D - len(x_curr)))
    if len(x_prev) < D:
        x_prev = np.pad(x_prev, (0, D - len(x_prev)))

    delta     = x_curr - x_prev
    magnitude = np.abs(delta)
    direction = np.sign(delta)

    return np.concatenate([delta, x_curr, magnitude, direction]).astype(np.float32)


# ---------------------------------------------------------------------------
# NumPy logistic regression — Newton's method (pure numpy, no sklearn)
# ---------------------------------------------------------------------------

class _NumpyLogisticRegression:
    """
    Binary logistic regression trained with Newton's method.
    No sklearn or scipy dependency — purely NumPy.
    """

    def __init__(self, lam: float = 0.01, max_iter: int = 50):
        self.lam = lam
        self.max_iter = max_iter
        self.w: Optional[np.ndarray] = None
        self.b: float = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_NumpyLogisticRegression":
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0

        for _ in range(self.max_iter):
            z  = X @ w + b
            p  = self._sigmoid(z)
            r  = p * (1 - p) + 1e-8

            # Gradient
            g_w = X.T @ (p - y) / n + self.lam * w
            g_b = float(np.mean(p - y))

            # Hessian diagonal (Fisher information)
            H_w_diag = (X ** 2).T @ r / n + self.lam
            H_b      = float(np.mean(r))

            # Newton step (diagonal approximation for speed)
            w = w - g_w / H_w_diag
            b = b - g_b / H_b

        self.w = w
        self.b = b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = X @ self.w + self.b
        return self._sigmoid(z)


# ---------------------------------------------------------------------------
# PyTorch MLP (optional, loaded lazily)
# ---------------------------------------------------------------------------

try:
    import torch
    import torch.nn as nn

    class _MLP(nn.Module):
        """2-layer MLP for DRE — defined at module level so pickle works."""
        def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout / 2),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

except ImportError:
    _MLP = None  # type: ignore


def _build_mlp(input_dim: int, hidden_dim: int = _HIDDEN_DIM, dropout: float = _DROPOUT):
    """Instantiate the MLP for the given input dimension."""
    if _MLP is None:
        raise ImportError("PyTorch is required for DRE MLP backend")
    return _MLP(input_dim, hidden_dim, dropout)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class DREResult:
    risk_score: float
    delta_contribution: float
    top_delta_features: list = field(default_factory=list)
    confidence: float = 0.5
    static_score: float = 0.0
    model_type: str = "mlp"    # "mlp" | "logistic" | "heuristic"

    def to_dict(self) -> dict:
        return {
            "risk_score":          round(self.risk_score, 4),
            "delta_contribution":  round(self.delta_contribution, 4),
            "top_delta_features":  self.top_delta_features,
            "confidence":          round(self.confidence, 3),
            "static_score":        round(self.static_score, 4),
            "model_type":          self.model_type,
        }


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class DifferentialRiskEncoder:
    """
    Bug-risk predictor trained on feature deltas between consecutive commits.

    Primary backend: PyTorch MLP.
    Fallback: NumPy logistic regression trained with Newton's method.
    Emergency fallback: complexity-delta heuristic (no training needed).
    """

    def __init__(
        self,
        hidden_dim: int = _HIDDEN_DIM,
        epochs: int = _EPOCHS,
        lr: float = _LR,
        batch_size: int = _BATCH_SIZE,
    ):
        self.hidden_dim  = hidden_dim
        self.epochs      = epochs
        self.lr          = lr
        self.batch_size  = batch_size

        self._mlp        = None   # PyTorch model
        self._logistic   = None   # NumPy fallback
        self._input_dim: int = 0
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std:  Optional[np.ndarray] = None
        self._model_type: str = "heuristic"
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_pairs: list[tuple[np.ndarray, np.ndarray]],
        y: np.ndarray,
        output_dir: str | None = None,
    ) -> dict:
        """
        Train DRE on consecutive commit feature pairs.

        Args:
            X_pairs:    List of (x_curr, x_prev) tuples.
            y:          Binary labels (1=buggy, 0=clean).
            output_dir: Checkpoint directory.

        Returns:
            Training metrics dict.
        """
        logger.info("DRE: building delta features from %d pairs ...", len(X_pairs))

        X_delta = np.array([
            build_delta_features(xc, xp) for xc, xp in X_pairs
        ], dtype=np.float32)

        y_arr = np.asarray(y, dtype=np.float32)

        # Standardise
        self._scaler_mean = X_delta.mean(axis=0)
        self._scaler_std  = X_delta.std(axis=0) + 1e-8
        X_scaled = (X_delta - self._scaler_mean) / self._scaler_std

        self._input_dim = X_scaled.shape[1]

        # Static baseline (for delta_contribution at inference)
        D = len(X_pairs[0][0]) if X_pairs else 15
        X_static = X_scaled[:, D:2*D]   # absolute features are columns [D:2D]

        metrics: dict = {}

        # Try PyTorch MLP
        torch_ok = self._train_mlp(X_scaled, y_arr)
        if torch_ok:
            self._model_type = "mlp"
            # Train a logistic as static baseline for delta_contribution
            self._train_logistic_static(X_static, y_arr)
            metrics["model"] = "mlp"
        else:
            # Fallback: Newton logistic regression on delta features
            logger.info("DRE: PyTorch unavailable, using Newton logistic regression")
            lr_model = _NumpyLogisticRegression()
            lr_model.fit(X_scaled, y_arr)
            self._logistic    = lr_model
            self._model_type  = "logistic"
            metrics["model"]  = "logistic"

        # Training metrics
        preds = self._predict_scaled(X_scaled)
        auc   = self._roc_auc(y_arr, preds)
        metrics["n_samples"] = len(y_arr)
        metrics["train_auc"] = round(auc, 4)
        metrics["input_dim"] = self._input_dim

        self._ready = True

        if output_dir:
            cp = Path(output_dir)
            cp.mkdir(parents=True, exist_ok=True)
            self._save(cp)
            with open(cp / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info("DRE saved to %s  (model=%s  auc=%.4f)", cp, self._model_type, auc)

        logger.info("DRE trained: %s  auc=%.4f  input_dim=%d", self._model_type, auc, self._input_dim)
        return metrics

    def _train_mlp(self, X: np.ndarray, y: np.ndarray) -> bool:
        """Returns True if PyTorch MLP training succeeded."""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset

            torch.manual_seed(_SEED)
            model = _build_mlp(X.shape[1], self.hidden_dim)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=1e-4)

            pos_weight = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)])
            criterion  = nn.BCELoss()

            Xt = torch.FloatTensor(X)
            yt = torch.FloatTensor(y)
            ds = TensorDataset(Xt, yt)
            dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

            model.train()
            for ep in range(self.epochs):
                for xb, yb in dl:
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()

            self._mlp = model
            self._mlp.eval()
            return True
        except Exception as e:
            logger.warning("DRE MLP training failed: %s", e)
            return False

    def _train_logistic_static(self, X_static: np.ndarray, y: np.ndarray):
        """Logistic regression on static-only features (for delta_contribution)."""
        try:
            lr = _NumpyLogisticRegression()
            lr.fit(X_static, y)
            self._logistic = lr
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        x_curr: np.ndarray,
        x_prev: Optional[np.ndarray] = None,
    ) -> DREResult:
        """
        Predict bug risk from current and (optionally) previous commit features.

        If x_prev is None, we treat x_prev = zeros (first commit or no history).

        Args:
            x_curr: Current commit static feature vector (D-dim).
            x_prev: Previous commit feature vector (D-dim), or None.

        Returns:
            DREResult with risk_score, delta_contribution, and attribution.
        """
        if not self.ready:
            return self._heuristic(x_curr, x_prev)

        if x_prev is None:
            x_prev = np.zeros_like(x_curr)

        delta_vec = build_delta_features(x_curr, x_prev)
        D = len(x_curr)

        # Scale
        if self._scaler_mean is not None:
            pad = len(delta_vec) - len(self._scaler_mean)
            mean = self._scaler_mean
            std  = self._scaler_std
            if pad > 0:
                mean = np.pad(mean, (0, pad))
                std  = np.pad(std, (0, pad), constant_values=1.0)
            elif pad < 0:
                delta_vec = delta_vec[:len(mean)]
            delta_scaled = (delta_vec - mean) / std
        else:
            delta_scaled = delta_vec

        risk = float(self._predict_scaled(delta_scaled.reshape(1, -1))[0])

        # Static-only score for delta_contribution
        static_score = 0.0
        try:
            x_static = delta_scaled[D:2*D]
            if self._logistic is not None:
                static_score = float(self._logistic.predict_proba(x_static.reshape(1, -1))[0])
        except Exception:
            pass

        delta_contribution = abs(risk - static_score)

        # Attribution: top changed features (highest magnitude delta in first D cols)
        delta_raw = delta_vec[:D]
        top_idx   = np.argsort(np.abs(delta_raw))[::-1][:5]
        top_feats = []
        for i in top_idx:
            name = _FEATURE_NAMES[i] if i < len(_FEATURE_NAMES) else f"feat_{i}"
            top_feats.append({
                "feature": name,
                "delta":   round(float(delta_raw[i]), 4),
                "current": round(float(x_curr[i]) if i < len(x_curr) else 0.0, 4),
            })

        # Confidence: entropy-based
        p = max(min(risk, 1 - 1e-6), 1e-6)
        entropy = -(p * math.log(p) + (1 - p) * math.log(1 - p))
        confidence = float(1.0 - entropy / math.log(2))

        return DREResult(
            risk_score=round(risk, 4),
            delta_contribution=round(delta_contribution, 4),
            top_delta_features=top_feats,
            confidence=round(confidence, 3),
            static_score=round(static_score, 4),
            model_type=self._model_type,
        )

    def _predict_scaled(self, X_scaled: np.ndarray) -> np.ndarray:
        """Run inference on already-scaled feature matrix."""
        if self._mlp is not None:
            try:
                import torch
                with torch.no_grad():
                    Xt = torch.FloatTensor(X_scaled)
                    return self._mlp(Xt).numpy()
            except Exception:
                pass
        if self._logistic is not None:
            return self._logistic.predict_proba(X_scaled)
        # Ultimate fallback
        return np.full(len(X_scaled), 0.3, dtype=np.float32)

    def _heuristic(self, x_curr: np.ndarray, x_prev: Optional[np.ndarray]) -> DREResult:
        """Emergency fallback: complexity-delta heuristic, no training needed."""
        cog_curr = float(x_curr[1]) if len(x_curr) > 1 else 0.0
        cog_prev = float(x_prev[1]) if (x_prev is not None and len(x_prev) > 1) else cog_curr

        delta_cog = cog_curr - cog_prev
        # Positive delta (increasing complexity) → higher risk
        risk = 0.3 + min(delta_cog / 30.0, 0.5) if delta_cog > 0 else max(0.1, 0.3 + delta_cog / 30.0)
        risk = float(np.clip(risk, 0.05, 0.95))

        return DREResult(
            risk_score=round(risk, 4),
            delta_contribution=round(abs(delta_cog) / 30.0, 4),
            top_delta_features=[{"feature": "cognitive_complexity",
                                  "delta": round(delta_cog, 2),
                                  "current": round(cog_curr, 2)}],
            confidence=0.25,
            static_score=round(0.3 + cog_curr / 50.0, 4),
            model_type="heuristic",
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, cp: Path):
        if self._mlp is not None:
            try:
                import torch
                torch.save(self._mlp.state_dict(), cp / "dre_mlp.pt")
                meta = {
                    "input_dim":    self._input_dim,
                    "hidden_dim":   self.hidden_dim,
                    "scaler_mean":  self._scaler_mean.tolist() if self._scaler_mean is not None else None,
                    "scaler_std":   self._scaler_std.tolist()  if self._scaler_std  is not None else None,
                    "model_type":   "mlp",
                }
                with open(cp / "dre_meta.json", "w") as f:
                    json.dump(meta, f)
            except Exception as e:
                logger.warning("DRE MLP save failed: %s", e)

        if self._logistic is not None:
            with open(cp / "dre_logistic.pkl", "wb") as f:
                pickle.dump(self._logistic, f)

        # Save full object as fallback
        with open(cp / "dre.pkl", "wb") as f:
            pickle.dump(self, f)

    def load(self, checkpoint_dir: str | None = None) -> bool:
        """Load DRE from disk."""
        cp = Path(checkpoint_dir) if checkpoint_dir else CHECKPOINT_DIR
        pkl = cp / "dre.pkl"
        if not pkl.exists():
            return False
        try:
            with open(pkl, "rb") as f:
                loaded: DifferentialRiskEncoder = pickle.load(f)
            self._mlp          = loaded._mlp
            self._logistic     = loaded._logistic
            self._input_dim    = loaded._input_dim
            self._scaler_mean  = loaded._scaler_mean
            self._scaler_std   = loaded._scaler_std
            self._model_type   = loaded._model_type
            self._ready        = True
            logger.info("DRE loaded from %s (type=%s)", cp, self._model_type)
            return True
        except Exception as e:
            logger.warning("DRE load failed: %s", e)
            return False

    @staticmethod
    def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Trapezoid-rule AUC, no sklearn."""
        order = np.argsort(-y_score)
        yt, ys = y_true[order], y_score[order]
        npos = yt.sum()
        nneg = len(yt) - npos
        if npos == 0 or nneg == 0:
            return 0.5
        tp = np.cumsum(yt)
        fp = np.cumsum(1 - yt)
        tpr = tp / npos
        fpr = fp / nneg
        trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
        return float(trapz(tpr, fpr))
