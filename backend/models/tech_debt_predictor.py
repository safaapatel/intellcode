"""
Technical Debt Interest Rate Predictor
========================================
Predicts the RATE OF COMPLEXITY GROWTH (interest rate) for a Python file
based on its current static metrics.

Growth rate interpretation:
  positive  => complexity is growing per commit (accumulating debt)
  negative  => complexity is declining per commit (debt paydown)
  near-zero => stable complexity (low-interest debt)

Features: same 15-dim vector as complexity model (cognitive_complexity excluded).
Model: XGBoost regressor trained by training/train_tech_debt.py.

Novel prediction target — prior work (Zazworka et al. 2011, Kruchten et al. 2012)
studies TD descriptively; this model predicts the rate prospectively.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class TechDebtResult:
    """
    Result of the technical debt interest rate prediction.

    Attributes:
        growth_rate:     Predicted complexity growth rate (units/commit).
                         Positive = debt accumulating, negative = improving.
        risk_level:      "low" | "medium" | "high" | "critical"
        interpretation:  Human-readable explanation of the prediction.
        features_used:   15-dim feature vector (for diagnostics).
    """
    growth_rate:    float
    risk_level:     str
    interpretation: str
    features_used:  Optional[list[float]] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Risk thresholds
# ---------------------------------------------------------------------------

# These thresholds are calibrated to the growth-rate distribution observed during
# training. A growth rate near 0 is neutral; positive rates indicate debt accrual.
_THRESHOLDS = {
    "low":      0.10,   # < 0.10 units/commit: stable or improving
    "medium":   0.50,   # 0.10 - 0.50: moderate debt accumulation
    "high":     1.50,   # 0.50 - 1.50: significant debt accumulation
    # above 1.50: critical
}


def _classify_risk(growth_rate: float) -> str:
    if growth_rate < _THRESHOLDS["low"]:
        return "low"
    elif growth_rate < _THRESHOLDS["medium"]:
        return "medium"
    elif growth_rate < _THRESHOLDS["high"]:
        return "high"
    else:
        return "critical"


def _build_interpretation(growth_rate: float, risk: str) -> str:
    if risk == "low":
        if growth_rate < 0:
            return (
                f"Complexity is declining ({growth_rate:+.2f} units/commit). "
                "Technical debt is actively being paid down."
            )
        return (
            f"Complexity growth is minimal ({growth_rate:+.2f} units/commit). "
            "File is stable with low technical debt interest."
        )
    elif risk == "medium":
        return (
            f"Moderate complexity growth detected ({growth_rate:+.2f} units/commit). "
            "Technical debt is accumulating at a manageable rate. "
            "Consider refactoring within the next few sprints."
        )
    elif risk == "high":
        return (
            f"High complexity growth rate ({growth_rate:+.2f} units/commit). "
            "This file is accumulating technical debt rapidly. "
            "Prioritise refactoring to prevent compounding interest."
        )
    else:
        return (
            f"Critical debt growth rate ({growth_rate:+.2f} units/commit). "
            "This file shows signs of runaway complexity accumulation. "
            "Immediate architectural intervention recommended."
        )


# ---------------------------------------------------------------------------
# Predictor class
# ---------------------------------------------------------------------------

class TechDebtPredictor:
    """
    XGBoost-based technical debt interest rate predictor.

    Falls back to a rule-based estimate when no trained model is available.

    Args:
        checkpoint_path: Path to the saved XGBoost model pickle file.
                         Default: checkpoints/tech_debt/xgb_model.pkl
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = "checkpoints/tech_debt/xgb_model.pkl",
    ):
        self._model = None
        self._checkpoint = checkpoint_path
        self._try_load()

    def _try_load(self) -> None:
        if not self._checkpoint:
            return
        path = Path(self._checkpoint)
        if not path.exists():
            return
        try:
            with open(path, "rb") as f:
                self._model = pickle.load(f)
        except Exception:
            self._model = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, source: str) -> TechDebtResult:
        """
        Predict technical debt interest rate from Python source code.

        Args:
            source: Python source code as a string.

        Returns:
            TechDebtResult with growth_rate, risk_level, and interpretation.
        """
        try:
            from features.code_metrics import compute_all_metrics, metrics_to_feature_vector
            metrics = compute_all_metrics(source)
            feat = metrics_to_feature_vector(metrics)
        except Exception:
            feat = [0.0] * 15

        growth_rate = self.predict_from_features(feat)
        risk = _classify_risk(growth_rate)
        return TechDebtResult(
            growth_rate=round(growth_rate, 4),
            risk_level=risk,
            interpretation=_build_interpretation(growth_rate, risk),
            features_used=feat,
        )

    def predict_from_features(self, feat: list) -> float:
        """
        Predict growth rate from a pre-computed 15-dim feature vector.

        Args:
            feat: 15-element list produced by metrics_to_feature_vector().

        Returns:
            Predicted growth rate as a float.
        """
        x = np.array(feat, dtype=np.float32).reshape(1, -1)

        if self._model is not None:
            try:
                return float(self._model.predict(x)[0])
            except Exception:
                pass

        # -- Rule-based fallback when no model is loaded -------------------
        return self._rule_based_rate(feat)

    def predict_batch(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Predict growth rates for a batch of feature vectors.

        Args:
            feature_matrix: (N, 15) numpy array.

        Returns:
            (N,) array of predicted growth rates.
        """
        if self._model is not None:
            try:
                return self._model.predict(feature_matrix.astype(np.float32))
            except Exception:
                pass
        return np.array(
            [self._rule_based_rate(row.tolist()) for row in feature_matrix],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------

    def _rule_based_rate(self, feat: list) -> float:
        """
        Rule-based growth rate estimate when no trained model is available.

        Intuition: files with high cyclomatic complexity, large Halstead effort,
        and many long/complex functions grow in complexity faster than simple files.

        Feature indices (matching metrics_to_feature_vector):
            0  cyclomatic_complexity
            1  max_func_cc
            2  avg_func_cc
            3  sloc
            6  halstead_volume
            7  halstead_difficulty
            8  halstead_effort
            10 n_long_functions
            11 n_complex_functions
        """
        cc      = feat[0] if len(feat) > 0 else 1.0
        max_cc  = feat[1] if len(feat) > 1 else 1.0
        sloc    = feat[3] if len(feat) > 3 else 1.0
        effort  = feat[8] if len(feat) > 8 else 0.0
        n_long  = feat[10] if len(feat) > 10 else 0.0
        n_cplx  = feat[11] if len(feat) > 11 else 0.0

        # Normalised signals
        cc_norm    = min(1.0, cc / 20.0)
        effort_norm = min(1.0, effort / 50000.0)
        long_pen   = min(0.5, n_long * 0.1)
        cplx_pen   = min(0.5, n_cplx * 0.1)

        # Estimate: positive = growing debt
        rate = 0.4 * cc_norm + 0.3 * effort_norm + 0.2 * long_pen + 0.1 * cplx_pen
        # Scale to plausible range [0, 2.0]
        return float(rate * 2.0)


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def load_predictor(
    checkpoint_path: str = "checkpoints/tech_debt/xgb_model.pkl",
) -> TechDebtPredictor:
    """Load and return a TechDebtPredictor instance."""
    return TechDebtPredictor(checkpoint_path=checkpoint_path)
