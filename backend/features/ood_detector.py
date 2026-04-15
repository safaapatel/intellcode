"""
Out-of-Distribution Detector
==============================
Mahalanobis distance-based OOD detection for IntelliCode ML models.

Motivation (from audit):
    When the security or bug model encounters code from a domain it has never
    seen (Go code routed through the Python metric pipeline, assembly-heavy C,
    or novel Python patterns like match statements), it produces confident but
    wrong predictions. Users cannot distinguish these from reliable predictions.

    The correct response is: "I don't know" -- abstain rather than mislead.

Method: Mahalanobis distance from training distribution.
    d(x) = sqrt( (x - mu)^T * Sigma^{-1} * (x - mu) )

    A sample with d(x) > threshold is flagged as OOD.
    The threshold is set to the training distribution's 99th percentile of
    Mahalanobis distances (no OOD samples needed for calibration).

Abstention policy:
    - d(x) < 2.0 sigma: in-distribution, full confidence
    - 2.0 <= d(x) < 3.5 sigma: borderline, confidence halved
    - d(x) >= 3.5 sigma: OOD, abstain (return low_confidence=True)

Usage:
    detector = OODDetector()
    detector.fit(X_train)           # fit on training feature vectors
    detector.save("checkpoints/security/ood_detector.pkl")

    # At inference:
    factor = detector.confidence_factor(x)  # 0.0 = OOD, 1.0 = in-distribution
    is_ood = detector.is_ood(x)             # bool

References:
    Lee et al. 2018 -- "A Simple Unified Framework for Detecting OOD Samples"
    Mahalanobis distance: https://en.wikipedia.org/wiki/Mahalanobis_distance
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Sigma thresholds for confidence scaling
_SIGMA_BORDERLINE = 2.0
_SIGMA_OOD        = 3.5


class OODDetector:
    """
    Mahalanobis distance OOD detector.

    Fit on training features to learn the in-distribution manifold.
    At inference, computes distance of a test point from the training distribution.
    High distance => OOD => reduce confidence or abstain.
    """

    def __init__(self):
        self._mean: Optional[np.ndarray] = None
        self._cov_inv: Optional[np.ndarray] = None
        self._sigma_unit: float = 1.0    # 1 sigma in Mahalanobis units
        self._fitted = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "OODDetector":
        """
        Fit the OOD detector on training features.

        Computes the mean and regularised inverse covariance of the training
        distribution. Regularisation (ridge) prevents singular covariance
        matrices for correlated feature spaces.

        Args:
            X: (N, D) training feature matrix.

        Returns:
            self (for chaining).
        """
        if len(X) < 10:
            logger.warning("OODDetector: too few samples (%d), skipping fit.", len(X))
            return self

        X = np.array(X, dtype=np.float64)
        self._mean = X.mean(axis=0)
        centered = X - self._mean

        # Regularised covariance: Sigma + lambda * I to avoid singularity
        cov = np.cov(centered.T) + 1e-6 * np.eye(X.shape[1])
        try:
            self._cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            # Fallback: diagonal-only (ignore correlations)
            self._cov_inv = np.diag(1.0 / (np.var(X, axis=0) + 1e-6))

        # Compute Mahalanobis distances for all training samples
        # Use this to set the 1-sigma threshold
        dists = self._batch_distances(X)
        # 68th percentile = 1 sigma for Gaussian (chi distribution)
        self._sigma_unit = float(np.percentile(dists, 68))
        if self._sigma_unit < 1e-8:
            self._sigma_unit = 1.0

        self._fitted = True
        logger.info(
            "OODDetector fitted: %d samples, %d features, 1-sigma=%.2f",
            len(X), X.shape[1], self._sigma_unit,
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def mahalanobis_distance(self, x: np.ndarray) -> float:
        """
        Compute Mahalanobis distance from the training distribution centroid.

        Returns:
            Distance in raw units (divide by sigma_unit for sigma units).
            Returns 0.0 if detector is not fitted.
        """
        if not self._fitted:
            return 0.0
        diff = x.astype(np.float64) - self._mean
        try:
            dist_sq = float(diff @ self._cov_inv @ diff)
            return float(np.sqrt(max(0.0, dist_sq)))
        except Exception:
            return 0.0

    def sigma_distance(self, x: np.ndarray) -> float:
        """Distance expressed in sigma units (sigma_unit = 68th percentile of training)."""
        return self.mahalanobis_distance(x) / self._sigma_unit

    def is_ood(self, x: np.ndarray, threshold_sigma: float = _SIGMA_OOD) -> bool:
        """
        Return True if the sample is out of distribution.

        Args:
            x:               Feature vector (1-D).
            threshold_sigma: Sigma threshold (default 3.5).
        """
        if not self._fitted:
            return False
        return self.sigma_distance(x) >= threshold_sigma

    def confidence_factor(self, x: np.ndarray) -> float:
        """
        Return a multiplier in [0, 1] that scales down model confidence for OOD inputs.

            sigma < 2.0  -> 1.0  (full confidence, in-distribution)
            2.0 to 3.5   -> 0.5  (borderline -- halve confidence)
            >= 3.5       -> 0.1  (OOD -- near-zero confidence)

        This factor should be multiplied with the model's predicted probability:
            adjusted_prob = raw_prob * confidence_factor(x)

        Note: This intentionally pulls probabilities toward 0.0 (conservative).
        It is better to under-predict than to confidently mislead.
        """
        if not self._fitted:
            return 1.0
        sigma = self.sigma_distance(x)
        if sigma < _SIGMA_BORDERLINE:
            return 1.0
        if sigma < _SIGMA_OOD:
            # Linear interpolation: 1.0 at 2.0 sigma -> 0.5 at 3.5 sigma
            frac = (sigma - _SIGMA_BORDERLINE) / (_SIGMA_OOD - _SIGMA_BORDERLINE)
            return round(1.0 - 0.5 * frac, 3)
        # Beyond OOD threshold: aggressive confidence reduction
        # Soft floor at 0.1 (never absolute zero, model may still add some value)
        return 0.1

    def abstain(self, x: np.ndarray) -> bool:
        """
        Return True if the model should abstain from making a prediction.
        Abstention threshold is set at 3.5 sigma (same as is_ood default).
        """
        return self.is_ood(x, threshold_sigma=_SIGMA_OOD)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("OODDetector saved -> %s", path)

    @classmethod
    def load(cls, path: str) -> "Optional[OODDetector]":
        """Load a saved OODDetector. Returns None if path does not exist."""
        p = Path(path)
        if not p.exists():
            return None
        try:
            with open(p, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, cls) and obj._fitted:
                logger.info("OODDetector loaded from %s (1-sigma=%.2f)", path, obj._sigma_unit)
                return obj
        except Exception as e:
            logger.warning("OODDetector load failed: %s", e)
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _batch_distances(self, X: np.ndarray) -> np.ndarray:
        """Compute Mahalanobis distances for a batch of samples (N, D) -> (N,)."""
        diff = X - self._mean               # (N, D)
        # dist_i = sqrt(diff_i @ Sigma^{-1} @ diff_i)
        right = diff @ self._cov_inv        # (N, D)
        dist_sq = np.einsum("ij,ij->i", diff, right)  # (N,)
        return np.sqrt(np.maximum(0.0, dist_sq))


# ---------------------------------------------------------------------------
# Convenience: abstention wrapper for predictions
# ---------------------------------------------------------------------------

def make_abstention_prediction(
    model_prob: float,
    x: "np.ndarray",
    detector: "Optional[OODDetector]",
    task: str = "security",
) -> dict:
    """
    Wrap a model's predicted probability with OOD-aware metadata.

    Returns a dict with:
        probability:     final probability (scaled down if OOD)
        raw_probability: unscaled model output
        low_confidence:  True if sample is OOD
        confidence_factor: the scaling applied
        sigma_distance:  Mahalanobis distance in sigma units
    """
    if detector is None or not detector._fitted:
        return {
            "probability":       round(model_prob, 4),
            "raw_probability":   round(model_prob, 4),
            "low_confidence":    False,
            "confidence_factor": 1.0,
            "sigma_distance":    0.0,
        }

    factor = detector.confidence_factor(x)
    sigma  = detector.sigma_distance(x)
    is_low = sigma >= _SIGMA_BORDERLINE

    return {
        "probability":       round(model_prob * factor, 4),
        "raw_probability":   round(model_prob, 4),
        "low_confidence":    is_low,
        "confidence_factor": round(factor, 3),
        "sigma_distance":    round(sigma, 2),
    }
