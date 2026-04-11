"""
conformal_predictor.py
======================
Lightweight runtime conformal prediction wrapper for IntelliCode endpoints.

Split-conformal prediction intervals (Vovk et al. 2005, Angelopoulos & Bates 2021).
The calibration quantile is computed once during training on a held-out calibration
set and saved to a JSON file.  At inference, the saved quantile is loaded and
applied to produce a prediction interval around each point estimate.

For regression (complexity):
    interval = [y_hat - q, y_hat + q]  where q = (1-alpha) quantile of |y_cal - y_hat_cal|

For binary classification (security, bugs):
    interval = [max(0, p - q), min(1, p + q)]  where q = (1-alpha) quantile of |y_cal - p_cal|

Coverage guarantee (Barber et al. 2021):
    P(y in interval) >= 1 - alpha  for exchangeable (y, x) pairs.

Default fallback quantiles (derived from thesis evaluation results, alpha=0.10):
    security   : q = 0.18  (calibrated on CVEFixes held-out set)
    bug        : q = 0.22  (calibrated on held-out JIT dataset)
    complexity : q = 52.0  (residual quantile on cognitive complexity, RMSE=49.5)

References:
    Vovk et al. 2005   -- "Algorithmic Learning in a Random World"
    Barber et al. 2021 -- "Predictive inference with the jackknife+"
    Angelopoulos & Bates 2021 -- "A Gentle Introduction to Conformal Prediction"
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default conservative quantiles (alpha=0.10) derived from thesis evaluation.
# These are used when no saved calibration file is available.
_DEFAULTS: dict[str, float] = {
    "security":   0.18,
    "bug":        0.22,
    "complexity": 52.0,
}

# Default target coverage levels per task.
_ALPHA: dict[str, float] = {
    "security":   0.10,
    "bug":        0.10,
    "complexity": 0.10,
}


@dataclass
class ConformalResult:
    lower: float
    upper: float
    point_estimate: float
    coverage_level: float   # 1 - alpha
    quantile: float         # q used
    n_cal: int              # calibration set size (0 = default fallback)
    is_fallback: bool       # True when using hardcoded default


class ResidualConformalPredictor:
    """
    Split-conformal predictor using absolute residuals.

    Designed for both regression and classification outputs (probabilities).
    For classification, clamps [lower, upper] to [0, 1].

    Usage (training time)::

        predictor = ResidualConformalPredictor(task="bug")
        predictor.calibrate(y_cal, y_pred_cal)
        predictor.save("checkpoints/bug_predictor/conformal.json")

    Usage (inference)::

        predictor = ResidualConformalPredictor.load("checkpoints/bug_predictor/conformal.json")
        result = predictor.predict_interval(0.72)
        # -> ConformalResult(lower=0.50, upper=0.94, ...)
    """

    def __init__(self, task: str = "bug", alpha: float | None = None):
        self.task = task
        self.alpha = alpha if alpha is not None else _ALPHA.get(task, 0.10)
        self._quantile: float | None = None
        self._n_cal: int = 0
        self._is_fallback: bool = True

    # ------------------------------------------------------------------
    # Calibration (run once after training)
    # ------------------------------------------------------------------

    def calibrate(self, y_cal: np.ndarray, y_pred_cal: np.ndarray) -> "ResidualConformalPredictor":
        """
        Compute the (1-alpha) quantile of absolute residuals on a calibration set.

        Args:
            y_cal:      True labels / targets (n,).
            y_pred_cal: Model predictions on calibration set (n,).
        """
        residuals = np.abs(np.asarray(y_cal, dtype=float) - np.asarray(y_pred_cal, dtype=float))
        n = len(residuals)
        level = min(1.0, (1.0 - self.alpha) * (1.0 + 1.0 / n))
        self._quantile = float(np.quantile(residuals, level))
        self._n_cal = n
        self._is_fallback = False
        logger.info(
            "Conformal calibrated: task=%s  q=%.4f  n_cal=%d  alpha=%.2f",
            self.task, self._quantile, n, self.alpha,
        )
        return self

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "task":        self.task,
            "alpha":       self.alpha,
            "quantile":    self._quantile,
            "n_cal":       self._n_cal,
            "is_fallback": self._is_fallback,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Conformal quantile saved to %s", path)

    @classmethod
    def load(cls, path: str | Path, task: str | None = None) -> "ResidualConformalPredictor":
        """
        Load a saved conformal predictor.  Falls back to defaults if file missing.
        """
        p = Path(path)
        obj = cls(task=task or "bug")
        if not p.exists():
            logger.warning(
                "Conformal quantile file not found at %s — using default q=%.2f for task=%s",
                p, _DEFAULTS.get(obj.task, 0.20), obj.task,
            )
            obj._is_fallback = True
            return obj

        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            obj.task        = data.get("task", obj.task)
            obj.alpha       = data.get("alpha", obj.alpha)
            obj._quantile   = data.get("quantile")
            obj._n_cal      = data.get("n_cal", 0)
            obj._is_fallback = data.get("is_fallback", False)
            logger.info(
                "Conformal quantile loaded: task=%s  q=%.4f  n_cal=%d",
                obj.task, obj._quantile or 0.0, obj._n_cal,
            )
        except Exception as exc:
            logger.warning("Failed to load conformal quantile from %s: %s", p, exc)
            obj._is_fallback = True
        return obj

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @property
    def _effective_quantile(self) -> float:
        if self._quantile is not None:
            return self._quantile
        return _DEFAULTS.get(self.task, 0.20)

    def predict_interval(self, point_estimate: float) -> ConformalResult:
        """
        Return a conformal prediction interval around *point_estimate*.

        For classification tasks the interval is clamped to [0, 1].
        For regression tasks (complexity) the interval may be negative at the lower end
        (cognitive complexity cannot be negative — caller should clamp if desired).
        """
        q = self._effective_quantile
        lower = point_estimate - q
        upper = point_estimate + q

        is_classification = self.task in ("security", "bug")
        if is_classification:
            lower = max(0.0, lower)
            upper = min(1.0, upper)
        else:
            lower = max(0.0, lower)   # complexity is non-negative

        return ConformalResult(
            lower=round(lower, 4),
            upper=round(upper, 4),
            point_estimate=round(point_estimate, 4),
            coverage_level=round(1.0 - self.alpha, 2),
            quantile=round(q, 4),
            n_cal=self._n_cal,
            is_fallback=self._is_fallback,
        )

    def to_dict(self, point_estimate: float) -> dict:
        """Convenience wrapper: predict_interval -> dict."""
        return asdict(self.predict_interval(point_estimate))
