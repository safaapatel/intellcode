"""
Complexity Trajectory Predictor (CTP)
=======================================
Predicts whether a file's cognitive complexity is on an INCREASING trajectory
by fitting a weighted linear regression over the last N static feature snapshots
from consecutive commits.

Novel contribution
------------------
Every complexity model in IntelliCode (and in the broader literature) evaluates
complexity at a single point in time.  CTP models complexity as a time series
and predicts its direction of change.

Key insight: a file with cognitive complexity 18 today is more dangerous if it
was 10 two months ago (slope = +4/month) than if it was 22 two months ago
(slope = -2/month).  The same snapshot produces opposite risk signals depending
on trajectory.

This distinction is invisible to point-in-time models.  CTP makes it explicit.

Method
------
Given a sequence of (timestamp, complexity_value) pairs from git history:

  1. Normalise timestamps to [0, 1] (oldest = 0, newest = 1)
  2. Fit weighted linear regression: weight = exp(-lambda * (1 - t))
     (recent measurements count more — same decay as TDCP)
  3. Compute:
       slope          : change in complexity per normalised time unit
       trajectory     : "increasing" | "stable" | "decreasing"
       velocity       : |slope| / current_complexity (relative rate of change)
       acceleration   : second difference of last three points (curvature)
       risk_multiplier: scalar in [0.7, 1.5] that adjusts the point-in-time
                        bug probability based on trajectory

Outputs
-------
TrajectoryResult(
    slope: float             — complexity change per unit time (positive = worsening)
    trajectory: str          — "increasing" | "stable" | "decreasing"
    velocity: float          — |slope| / current_complexity
    acceleration: float      — curvature of last 3 points (positive = accelerating)
    risk_multiplier: float   — multiply point-in-time bug probability by this
    n_snapshots: int         — number of data points used
    confidence: float        — fit quality (R^2 of the regression)
    forecast_30d: float      — predicted complexity in 30 days (linear extrapolation)
)

Usage
-----
    from models.complexity_trajectory import ComplexityTrajectoryPredictor
    ctp = ComplexityTrajectoryPredictor()

    # snapshots: list of (unix_timestamp, cognitive_complexity) tuples
    result = ctp.predict(snapshots=[
        (1_700_000_000, 8),
        (1_702_000_000, 11),
        (1_704_000_000, 14),
        (1_706_000_000, 19),
    ])
    print(result.trajectory)     # "increasing"
    print(result.risk_multiplier)# e.g. 1.35
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Trajectory classification thresholds (slope in normalised time units)
_SLOPE_INCREASING  =  3.0   # > +3 complexity units per normalised time unit
_SLOPE_DECREASING  = -3.0   # < -3 complexity units

# Time-decay lambda for regression weights (same as TDCP)
_DECAY_LAMBDA = 2.0

# Risk multiplier bounds
_MULTIPLIER_MAX =  1.50   # strongly increasing trajectory
_MULTIPLIER_MIN =  0.70   # strongly decreasing trajectory

# Minimum snapshots required for a meaningful fit
_MIN_SNAPSHOTS = 3


@dataclass
class TrajectoryResult:
    slope: float
    trajectory: str            # "increasing" | "stable" | "decreasing"
    velocity: float            # |slope| / current_complexity (relative rate)
    acceleration: float        # curvature (positive = accelerating)
    risk_multiplier: float     # adjusts point-in-time bug probability
    n_snapshots: int
    confidence: float          # R^2 of the linear fit [0, 1]
    forecast_30d: float        # predicted complexity in 30 days
    insufficient_data: bool = False

    def to_dict(self) -> dict:
        return {
            "slope":             round(self.slope, 3),
            "trajectory":        self.trajectory,
            "velocity":          round(self.velocity, 4),
            "acceleration":      round(self.acceleration, 3),
            "risk_multiplier":   round(self.risk_multiplier, 3),
            "n_snapshots":       self.n_snapshots,
            "confidence":        round(self.confidence, 3),
            "forecast_30d":      round(self.forecast_30d, 1),
            "insufficient_data": self.insufficient_data,
        }


class ComplexityTrajectoryPredictor:
    """
    Predicts complexity trajectory from a sequence of historical snapshots.
    No checkpoint required — purely analytical.
    """

    def __init__(self, decay_lambda: float = _DECAY_LAMBDA):
        self.decay_lambda = decay_lambda

    def predict(
        self,
        snapshots: list[tuple[float, float]],
        current_complexity: Optional[float] = None,
        days_per_unit: float = 30.0,
    ) -> TrajectoryResult:
        """
        Predict complexity trajectory from historical snapshots.

        Args:
            snapshots:          List of (unix_timestamp, cognitive_complexity) tuples,
                                sorted oldest-first.  At least 2 required; 3+ recommended.
            current_complexity: Override for the most recent complexity value.
                                If None, uses the last snapshot's complexity.
            days_per_unit:      How many real-world days correspond to 1 normalised
                                time unit for forecast_30d computation.

        Returns:
            TrajectoryResult.  If fewer than MIN_SNAPSHOTS provided,
            returns a result with insufficient_data=True and neutral values.
        """
        if len(snapshots) < 2:
            return self._insufficient(
                current_complexity or (snapshots[-1][1] if snapshots else 0.0)
            )

        # Sort by timestamp ascending
        snaps = sorted(snapshots, key=lambda x: x[0])
        timestamps = np.array([s[0] for s in snaps], dtype=np.float64)
        complexities = np.array([s[1] for s in snaps], dtype=np.float64)

        current = float(current_complexity if current_complexity is not None
                        else complexities[-1])

        if len(snaps) < _MIN_SNAPSHOTS:
            # Can still compute slope but mark low confidence
            slope, r2 = self._weighted_slope(timestamps, complexities)
            return self._package(slope, r2, current, len(snaps),
                                 complexities, timestamps, days_per_unit,
                                 insufficient_data=True)

        slope, r2 = self._weighted_slope(timestamps, complexities)
        return self._package(slope, r2, current, len(snaps),
                             complexities, timestamps, days_per_unit)

    def predict_from_commits(
        self,
        commit_records: list[dict],
        current_complexity: Optional[float] = None,
    ) -> TrajectoryResult:
        """
        Convenience wrapper that accepts commit records from the bug dataset format.

        Each record should have:
            "author_date":    ISO8601 date string (e.g. "2023-11-15T12:00:00")
            "static_features": list[float] where index 1 = cognitive_complexity

        Args:
            commit_records:     List of commit record dicts (any ordering).
            current_complexity: Override for current complexity (latest commit).

        Returns:
            TrajectoryResult.
        """
        import dateutil.parser

        snapshots = []
        for rec in commit_records:
            try:
                ts = dateutil.parser.parse(rec["author_date"]).timestamp()
                feats = rec.get("static_features", [])
                cog = float(feats[1]) if len(feats) > 1 else 0.0
                snapshots.append((ts, cog))
            except Exception:
                continue

        return self.predict(snapshots, current_complexity)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _weighted_slope(
        self,
        timestamps: np.ndarray,
        complexities: np.ndarray,
    ) -> tuple[float, float]:
        """
        Fit weighted linear regression y = a*t + b using time-decay weights.

        w_i = exp(-lambda * (1 - t_norm_i))
        where t_norm is timestamps normalised to [0, 1].

        Returns (slope, R^2).
        """
        if timestamps.max() == timestamps.min():
            return 0.0, 0.0

        t_norm = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
        weights = np.exp(-self.decay_lambda * (1.0 - t_norm))
        weights /= weights.sum()

        # Weighted least squares: minimise sum(w_i * (y_i - a*t_i - b)^2)
        W = np.diag(weights)
        A = np.column_stack([t_norm, np.ones_like(t_norm)])
        try:
            AW = A.T @ W
            params = np.linalg.solve(AW @ A, AW @ complexities)
            slope = float(params[0])
        except np.linalg.LinAlgError:
            slope = 0.0

        # R^2 of the weighted fit
        y_pred = slope * t_norm + (np.average(complexities, weights=weights) - slope * 0.5)
        ss_res = float(np.sum(weights * (complexities - y_pred) ** 2))
        ss_tot = float(np.sum(weights * (complexities - np.average(complexities, weights=weights)) ** 2))
        r2 = max(0.0, 1.0 - ss_res / (ss_tot + 1e-9))

        return slope, r2

    def _package(
        self,
        slope: float,
        r2: float,
        current: float,
        n: int,
        complexities: np.ndarray,
        timestamps: np.ndarray,
        days_per_unit: float,
        insufficient_data: bool = False,
    ) -> TrajectoryResult:
        if slope > _SLOPE_INCREASING:
            trajectory = "increasing"
        elif slope < _SLOPE_DECREASING:
            trajectory = "decreasing"
        else:
            trajectory = "stable"

        velocity = abs(slope) / max(current, 1.0)

        # Acceleration: second difference of last 3 points (if available)
        acceleration = 0.0
        if len(complexities) >= 3:
            last3 = complexities[-3:]
            d1 = last3[1] - last3[0]
            d2 = last3[2] - last3[1]
            acceleration = float(d2 - d1)

        # Risk multiplier: linearly interpolated based on slope and trajectory
        if trajectory == "increasing":
            excess = min((slope - _SLOPE_INCREASING) / 10.0, 1.0)
            multiplier = 1.0 + excess * (_MULTIPLIER_MAX - 1.0)
        elif trajectory == "decreasing":
            deficit = min((_SLOPE_DECREASING - slope) / 10.0, 1.0)
            multiplier = 1.0 - deficit * (1.0 - _MULTIPLIER_MIN)
        else:
            multiplier = 1.0

        # Forecast: extrapolate 30 days forward
        t_span = timestamps[-1] - timestamps[0]
        t_range_days = t_span / 86400.0 if t_span > 0 else days_per_unit
        slope_per_day = slope / max(t_range_days, 1.0)
        forecast_30d = float(np.clip(current + slope_per_day * 30.0, 0.0, 1000.0))

        return TrajectoryResult(
            slope=round(slope, 3),
            trajectory=trajectory,
            velocity=round(velocity, 4),
            acceleration=round(acceleration, 3),
            risk_multiplier=round(float(np.clip(multiplier, _MULTIPLIER_MIN, _MULTIPLIER_MAX)), 3),
            n_snapshots=n,
            confidence=round(r2, 3),
            forecast_30d=round(forecast_30d, 1),
            insufficient_data=insufficient_data,
        )

    def _insufficient(self, current: float) -> TrajectoryResult:
        return TrajectoryResult(
            slope=0.0,
            trajectory="stable",
            velocity=0.0,
            acceleration=0.0,
            risk_multiplier=1.0,
            n_snapshots=0,
            confidence=0.0,
            forecast_30d=float(current),
            insufficient_data=True,
        )
