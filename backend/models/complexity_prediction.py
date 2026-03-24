"""
Complexity Prediction Model — XGBoost
Predicts a maintainability score (0–100) and flags problematic functions.

Features (17-dim vector from code_metrics.py):
  cyclomatic_complexity, cognitive_complexity, max_function_complexity,
  avg_function_complexity, sloc, comments, blank_lines,
  halstead_volume, halstead_difficulty, halstead_effort, bugs_delivered,
  maintainability_index (raw), n_long_functions, n_complex_functions,
  max_line_length, avg_line_length, n_lines_over_80

Target: maintainability score in [0, 100] (100 = perfectly clean)
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np

from features.code_metrics import (
    compute_all_metrics,
    metrics_to_feature_vector,
    CodeMetricsResult,
)
from features.ast_extractor import ASTExtractor


FEATURE_NAMES = [
    "cyclomatic_complexity",
    "cognitive_complexity",
    "max_function_complexity",
    "avg_function_complexity",
    "sloc",
    "comments",
    "blank_lines",
    "halstead_volume",
    "halstead_difficulty",
    "halstead_effort",
    "bugs_delivered",
    "maintainability_index_raw",
    "n_long_functions",
    "n_complex_functions",
    "max_line_length",
    "avg_line_length",
    "n_lines_over_80",
]


@dataclass
class FunctionIssue:
    name: str
    lineno: int
    cyclomatic: int
    body_lines: int
    n_params: int
    issue: str      # "high_complexity" | "too_long" | "too_many_params"


@dataclass
class ComplexityResult:
    score: float                            # 0–100 (higher = better)
    grade: str                              # A / B / C / D / F
    cyclomatic: int = 0
    cognitive: int = 0
    halstead_bugs: float = 0.0
    maintainability_index: float = 0.0
    sloc: int = 0
    n_long_functions: int = 0
    n_complex_functions: int = 0
    n_lines_over_80: int = 0
    function_issues: list[FunctionIssue] = field(default_factory=list)
    breakdown: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["function_issues"] = [asdict(fi) for fi in self.function_issues]
        return d


def _score_to_grade(score: float) -> str:
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    if score >= 40:
        return "D"
    return "F"


def _rule_based_score(metrics: CodeMetricsResult) -> float:
    """
    Fallback scoring when no trained XGBoost model is available.
    Linearly penalises each quality metric from a perfect 100.
    """
    score = 100.0

    # Cyclomatic complexity penalty (threshold: 10)
    cc = metrics.cyclomatic_complexity
    if cc > 10:
        score -= min(30, (cc - 10) * 1.5)

    # Cognitive complexity penalty (threshold: 15)
    cog = metrics.cognitive_complexity
    if cog > 15:
        score -= min(20, (cog - 15) * 0.8)

    # Long functions (> 50 lines)
    score -= min(15, metrics.n_long_functions * 5)

    # Complex functions (CC > 10)
    score -= min(15, metrics.n_complex_functions * 5)

    # Lines over 80 chars
    score -= min(10, metrics.n_lines_over_80 * 0.5)

    # No try/except (checked via ast below — skip here)

    # Halstead bugs estimate
    bugs = metrics.halstead.bugs_delivered
    if bugs > 1.0:
        score -= min(10, bugs * 2)

    return max(0.0, score)


class ComplexityPredictionModel:
    """
    XGBoost regressor that predicts a maintainability score for a code file.

    Falls back to rule-based scoring if no trained model is available.
    """

    def __init__(self, checkpoint_path: Optional[str] = "checkpoints/complexity/model.pkl"):
        self._regressor = None
        self._checkpoint = checkpoint_path
        self._try_load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, source: str) -> ComplexityResult:
        """
        Analyse *source* and return a ComplexityResult.
        """
        metrics = compute_all_metrics(source)
        ast_feats = ASTExtractor().extract(source)

        rule_score = _rule_based_score(metrics)
        if self._regressor is not None:
            feat_vec = np.array([metrics_to_feature_vector(metrics)], dtype=np.float32)
            raw_score = float(self._regressor.predict(feat_vec)[0])
            ml_score = max(0.0, min(100.0, raw_score))
            # Blend 40% ML + 60% rule-based: prevents out-of-distribution extremes
            # The XGBoost was trained on synthetic data and can extrapolate wildly on real code
            score = ml_score * 0.4 + rule_score * 0.6
        else:
            score = rule_score

        # Build per-function issues
        function_issues: list[FunctionIssue] = []
        for func in ast_feats.get("functions", []):
            issues = []
            if func["body_lines"] > 50:
                issues.append("too_long")
            if func["n_params"] > 5:
                issues.append("too_many_params")
            # estimate per-function CC (rough: use module CC if only 1 function)
            for issue in issues:
                function_issues.append(FunctionIssue(
                    name=func["name"],
                    lineno=func["lineno"],
                    cyclomatic=0,  # filled in training; estimation here
                    body_lines=func["body_lines"],
                    n_params=func["n_params"],
                    issue=issue,
                ))

        result = ComplexityResult(
            score=round(score, 1),
            grade=_score_to_grade(score),
            cyclomatic=metrics.cyclomatic_complexity,
            cognitive=metrics.cognitive_complexity,
            halstead_bugs=round(metrics.halstead.bugs_delivered, 3),
            maintainability_index=round(metrics.maintainability_index, 1),
            sloc=metrics.lines.sloc,
            n_long_functions=metrics.n_long_functions,
            n_complex_functions=metrics.n_complex_functions,
            n_lines_over_80=metrics.n_lines_over_80,
            function_issues=function_issues,
            breakdown={
                "cyclomatic_complexity": metrics.cyclomatic_complexity,
                "cognitive_complexity": metrics.cognitive_complexity,
                "halstead_volume": round(metrics.halstead.volume, 1),
                "halstead_difficulty": round(metrics.halstead.difficulty, 2),
                "maintainability_index": round(metrics.maintainability_index, 1),
                "sloc": metrics.lines.sloc,
                "lines_over_80": metrics.n_lines_over_80,
                "long_functions": metrics.n_long_functions,
                "complex_functions": metrics.n_complex_functions,
            },
        )
        return result

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        output_path: str = "checkpoints/complexity/model.pkl",
        **xgb_kwargs,
    ) -> dict:
        """
        Train an XGBoost regressor.

        Args:
            X: Feature matrix of shape (N, 17) from metrics_to_feature_vector().
            y: Target scores in [0, 100].
            output_path: Where to save the trained model.

        Returns:
            dict of evaluation metrics (RMSE, R²).
        """
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise RuntimeError("xgboost is required. pip install xgboost") from exc

        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import mean_squared_error, r2_score
        import math

        params = {
            "n_estimators": xgb_kwargs.get("n_estimators", 500),
            "max_depth": xgb_kwargs.get("max_depth", 6),
            "learning_rate": xgb_kwargs.get("learning_rate", 0.05),
            "subsample": xgb_kwargs.get("subsample", 0.8),
            "colsample_bytree": xgb_kwargs.get("colsample_bytree", 0.8),
            "reg_lambda": xgb_kwargs.get("reg_lambda", 1.0),
            "random_state": 42,
            "n_jobs": -1,
        }

        self._regressor = xgb.XGBRegressor(**params)
        self._regressor.fit(
            X, y,
            eval_set=[(X, y)],
            verbose=False,
        )

        y_pred = self._regressor.predict(X)
        rmse = math.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(self._regressor, f)
        self._checkpoint = output_path

        print(f"Training complete — RMSE: {rmse:.2f}, R²: {r2:.4f}")
        return {"rmse": rmse, "r2": r2}

    def feature_importance(self) -> list[tuple[str, float]]:
        """Return (feature_name, importance) pairs sorted by importance."""
        if self._regressor is None:
            return []
        importances = self._regressor.feature_importances_
        pairs = sorted(
            zip(FEATURE_NAMES, importances),
            key=lambda x: x[1],
            reverse=True,
        )
        return [(name, float(imp)) for name, imp in pairs]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _try_load(self):
        if self._checkpoint and Path(self._checkpoint).exists():
            try:
                with open(self._checkpoint, "rb") as f:
                    self._regressor = pickle.load(f)
            except Exception:
                self._regressor = None
