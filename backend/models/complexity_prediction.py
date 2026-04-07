"""
Complexity Prediction Model — XGBoost
Predicts the cognitive complexity of a Python file.

Features (15-dim vector — cognitive_complexity EXCLUDED to prevent leakage):
  cyclomatic_complexity, max_function_complexity, avg_function_complexity,
  sloc, comments, blank_lines,
  halstead_volume, halstead_difficulty, halstead_effort, bugs_delivered,
  n_long_functions, n_complex_functions,
  max_line_length, avg_line_length, n_lines_over_80

  NOTE: cognitive_complexity is the training TARGET so it is excluded from
  features.  maintainability_index is also excluded — it is a closed-form
  formula of halstead_volume * cyclomatic * sloc (all already in features),
  which previously caused trivial R²≈1.0 target leakage.

Target: cognitive_complexity (non-negative integer; lower = simpler)
"""

from __future__ import annotations

import ast
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
    _function_cyclomatic,
)
from features.ast_extractor import ASTExtractor
from utils.checkpoint_integrity import verify_checkpoint


FEATURE_NAMES = [
    # cognitive_complexity is EXCLUDED — it is the prediction target
    "cyclomatic_complexity",
    "max_function_complexity",
    "avg_function_complexity",
    "sloc",
    "comments",
    "blank_lines",
    "halstead_volume",
    "halstead_difficulty",
    "halstead_effort",
    "bugs_delivered",
    "n_long_functions",
    "n_complex_functions",
    "max_line_length",
    "avg_line_length",
    "n_lines_over_80",
]  # 15 features


@dataclass
class FunctionIssue:
    name: str
    lineno: int
    cyclomatic: int
    body_lines: int
    n_params: int
    issue: str      # "high_complexity" | "too_long" | "too_many_params"


@dataclass
class FunctionComplexityResult:
    """Per-function complexity prediction produced by predict_functions()."""
    name: str
    lineno: int
    end_lineno: int
    cognitive_complexity: float    # predicted by XGBoost (or rule-based fallback)
    cyclomatic_complexity: float   # computed directly via McCabe
    grade: str                     # A / B / C / D / F
    is_problematic: bool           # predicted cognitive_complexity > 20
    recommendation: str            # actionable guidance string

    def to_dict(self) -> dict:
        return asdict(self)


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
        self._mapie = None            # MAPIE conformal wrapper (optional)
        self._shap_explainer = None   # SHAP TreeExplainer (lazy-init)
        self._try_load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_cognitive_complexity(self, source: str) -> float:
        """Return the XGBoost's direct prediction of cognitive_complexity (same scale as training target)."""
        from features.code_metrics import compute_all_metrics, metrics_to_feature_vector
        metrics = compute_all_metrics(source)
        if self._regressor is not None:
            feat_vec = np.array([metrics_to_feature_vector(metrics)], dtype=np.float32)
            return float(self._regressor.predict(feat_vec)[0])
        return float(metrics.cognitive_complexity)

    def predict_from_features(self, feat: list) -> float:
        """Predict cognitive_complexity from a raw dataset feature vector.

        Mirrors train_complexity.py: COG_IDX=1 is excluded from X,
        so the model input is [feat[i] for i in range(16) if i != 1].
        """
        if self._regressor is None:
            raise RuntimeError("Model not loaded.")
        COG_IDX = 1
        x_vec = [feat[i] for i in range(16) if i != COG_IDX]  # 15-dim
        feat_vec = np.array([x_vec], dtype=np.float32)
        return float(self._regressor.predict(feat_vec)[0])

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
        # Calibration correction: SonarQube validation showed +1.067 positive bias
        # in cognitive complexity prediction (Mar 2026 audit). Subtract to remove
        # systematic overestimation. Clamp to [0, 100] after correction.
        score = max(0.0, min(100.0, score - 1.067))

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

        # Build function-level complexity summary (top-3 most complex)
        func_level_summary: list[dict] = []
        try:
            func_results = self.predict_functions(source)
            func_level_summary = [
                {
                    "name": fr.name,
                    "lineno": fr.lineno,
                    "end_lineno": fr.end_lineno,
                    "cognitive_complexity": fr.cognitive_complexity,
                    "cyclomatic_complexity": fr.cyclomatic_complexity,
                    "grade": fr.grade,
                    "is_problematic": fr.is_problematic,
                    "recommendation": fr.recommendation,
                }
                for fr in func_results[:3]
            ]
        except Exception:
            func_level_summary = []

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
                "function_level": func_level_summary,
            },
        )
        return result

    # ------------------------------------------------------------------
    # Function-level complexity
    # ------------------------------------------------------------------

    def predict_functions(self, source: str) -> list[FunctionComplexityResult]:
        """
        Analyse each top-level and nested function in *source* individually.

        For each function:
          - Extracts the function's source text with ast.get_source_segment()
          - Computes per-function metrics and runs the XGBoost regressor (or
            rule-based fallback) on that function in isolation
          - Returns a list sorted by predicted cognitive_complexity descending

        Args:
            source: Full Python source code string.

        Returns:
            List of FunctionComplexityResult, most complex first.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        results: list[FunctionComplexityResult] = []

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            lineno = node.lineno
            end_lineno = node.end_lineno or node.lineno

            # Extract this function's source text and dedent so ast.parse succeeds
            # on methods (which are indented relative to their class body).
            import textwrap
            func_src = ast.get_source_segment(source, node)
            if not func_src:
                # Fallback: extract by line numbers
                lines = source.splitlines()
                func_lines = lines[lineno - 1 : end_lineno]
                func_src = "\n".join(func_lines)
            func_src = textwrap.dedent(func_src)

            # Cyclomatic complexity directly from AST node (no re-parse needed)
            cc = float(_function_cyclomatic(node))

            # Predicted cognitive complexity via model (on isolated function source)
            if self._regressor is not None:
                try:
                    func_metrics = compute_all_metrics(func_src)
                    feat_vec = np.array(
                        [metrics_to_feature_vector(func_metrics)], dtype=np.float32
                    )
                    pred_cog = float(self._regressor.predict(feat_vec)[0])
                    pred_cog = max(0.0, pred_cog)
                except Exception:
                    from features.code_metrics import cognitive_complexity as _cog
                    pred_cog = float(_cog(func_src))
            else:
                from features.code_metrics import cognitive_complexity as _cog
                pred_cog = float(_cog(func_src))

            grade = _score_to_grade(max(0.0, min(100.0, 100.0 - pred_cog * 2.0)))
            is_problematic = pred_cog > 20.0

            # Build recommendation
            if is_problematic:
                if cc > 10:
                    recommendation = (
                        "Refactor '{name}': high cyclomatic ({cc:.0f}) and cognitive "
                        "complexity ({cog:.1f}). Extract sub-functions to reduce nesting."
                    ).format(name=node.name, cc=cc, cog=pred_cog)
                else:
                    recommendation = (
                        "Simplify '{name}': cognitive complexity {cog:.1f} is high. "
                        "Consider flattening nested conditionals or using early returns."
                    ).format(name=node.name, cog=pred_cog)
            elif pred_cog > 10:
                recommendation = (
                    "Review '{name}': moderate complexity ({cog:.1f}). "
                    "Add docstrings and unit tests for all branches."
                ).format(name=node.name, cog=pred_cog)
            else:
                recommendation = (
                    "'{name}' is acceptably simple (cognitive complexity {cog:.1f})."
                ).format(name=node.name, cog=pred_cog)

            results.append(
                FunctionComplexityResult(
                    name=node.name,
                    lineno=lineno,
                    end_lineno=end_lineno,
                    cognitive_complexity=round(pred_cog, 2),
                    cyclomatic_complexity=round(cc, 2),
                    grade=grade,
                    is_problematic=is_problematic,
                    recommendation=recommendation,
                )
            )

        # Sort most complex first
        results.sort(key=lambda r: r.cognitive_complexity, reverse=True)
        return results

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
            X: Feature matrix of shape (N, 15) from metrics_to_feature_vector().
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
            "n_jobs": 1,  # n_jobs=-1 causes OOM/pickle errors on Windows
        }

        self._regressor = xgb.XGBRegressor(**params)
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
        self._regressor.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        y_pred = self._regressor.predict(X_val)
        rmse = math.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(self._regressor, f)
        self._checkpoint = output_path

        print(f"Training complete — RMSE: {rmse:.2f}, R²: {r2:.4f}")
        return {"rmse": rmse, "r2": r2}

    def predict_with_interval(
        self,
        source: str,
        alpha: float = 0.1,
    ) -> dict:
        """
        Return prediction with a conformal prediction interval (90% CI by default).

        Uses MAPIE (Model-Agnostic Prediction Interval Estimator) if available,
        otherwise returns the point prediction with None intervals.

        Args:
            source: Python source code.
            alpha:  Error rate — 0.1 gives a 90% prediction interval.

        Returns:
            {
              "score": float,          # point prediction
              "lower": float | None,   # lower bound of PI
              "upper": float | None,   # upper bound of PI
              "coverage": float,       # nominal coverage (1 - alpha)
              "grade": str,
            }
        """
        result = self.predict(source)
        point = result.score

        if self._mapie is not None:
            try:
                metrics = compute_all_metrics(source)
                feat_vec = np.array([metrics_to_feature_vector(metrics)], dtype=np.float32)
                _, pi = self._mapie.predict(feat_vec, alpha=alpha)
                lower = float(max(0.0, pi[0, 0, 0]))
                upper = float(min(100.0, pi[0, 1, 0]))
            except Exception:
                lower, upper = None, None
        else:
            lower, upper = None, None

        return {
            "score": point,
            "lower": lower,
            "upper": upper,
            "coverage": 1.0 - alpha,
            "grade": result.grade,
        }

    def explain(self, source: str) -> list[dict]:
        """
        Return SHAP-based feature contributions for a prediction.

        Each entry:
            {
              "feature":   str,    # feature name
              "value":     float,  # raw feature value
              "shap":      float,  # SHAP contribution (+ve raises score, -ve lowers)
              "direction": str,    # "positive" | "negative"
            }

        Returns [] if SHAP is unavailable or no model is loaded.
        """
        if self._regressor is None:
            return []
        try:
            import shap
        except ImportError:
            return []

        metrics = compute_all_metrics(source)
        feat_vec = np.array([metrics_to_feature_vector(metrics)], dtype=np.float32)

        if self._shap_explainer is None:
            self._shap_explainer = shap.TreeExplainer(self._regressor)

        shap_values = self._shap_explainer.shap_values(feat_vec)  # shape (1, 15)
        contributions = shap_values[0]

        results = []
        for i, name in enumerate(FEATURE_NAMES):
            sv = float(contributions[i])
            results.append({
                "feature": name,
                "value": float(feat_vec[0][i]),
                "shap": round(sv, 4),
                "direction": "positive" if sv >= 0 else "negative",
            })
        # Sort by absolute contribution descending
        results.sort(key=lambda x: abs(x["shap"]), reverse=True)
        return results

    def fit_conformal(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> None:
        """
        Fit a MAPIE conformal wrapper around the already-trained regressor.

        Call this after train() with a held-out calibration set to enable
        predict_with_interval(). Saves the fitted MAPIE wrapper alongside the
        base model checkpoint.

        Args:
            X_cal: Calibration features (N, 16).
            y_cal: Calibration targets in [0, 100].
        """
        try:
            from mapie.regression import MapieRegressor
        except ImportError:
            print("[WARN] mapie not installed — conformal intervals unavailable. pip install mapie")
            return

        if self._regressor is None:
            raise RuntimeError("Train the model before fitting conformal intervals.")

        self._mapie = MapieRegressor(
            self._regressor,
            method="plus",
            cv="prefit",   # regressor already fitted; use calibration set directly
        )
        self._mapie.fit(X_cal, y_cal)

        # Persist alongside the base checkpoint
        if self._checkpoint:
            mapie_path = Path(self._checkpoint).parent / "mapie_wrapper.pkl"
            with open(mapie_path, "wb") as f:
                pickle.dump(self._mapie, f)
            print(f"MAPIE conformal wrapper saved → {mapie_path}")

    def feature_importance(self) -> list[tuple[str, float]]:
        """Return (feature_name, importance) pairs sorted by importance."""
        if self._regressor is None or not hasattr(self._regressor, "feature_importances_"):
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
                verify_checkpoint(self._checkpoint)
                with open(self._checkpoint, "rb") as f:
                    self._regressor = pickle.load(f)
            except Exception:
                self._regressor = None

        # Load optional MAPIE conformal wrapper
        if self._checkpoint:
            mapie_path = Path(self._checkpoint).parent / "mapie_wrapper.pkl"
            if mapie_path.exists():
                try:
                    verify_checkpoint(mapie_path)
                    with open(mapie_path, "rb") as f:
                        self._mapie = pickle.load(f)
                except Exception:
                    self._mapie = None
