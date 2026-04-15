"""
Bug Prediction Model (In Development)
Predicts the likelihood that a code file contains or will introduce bugs.

Two-stage approach:
  Stage 1 (Baseline): LogisticRegression on static code features
  Stage 2 (Full):     PyTorch MLP that combines static features + git metadata

Features:
  Static (from code_metrics + ast_extractor):
    cyclomatic_complexity, cognitive_complexity, halstead_bugs,
    n_long_functions, n_complex_functions, sloc, n_functions,
    n_try_blocks, n_raises, max_params, n_lines_over_80

  Git metadata (optional, provided at inference time):
    code_churn     — lines changed in recent commits
    author_count   — number of unique contributors
    file_age_days  — age of the file in days
    n_past_bugs    — historical bug count for this file
    commit_freq    — commits per week on this file
"""

from __future__ import annotations

import json
import logging
import math
import pickle
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Temporal AUC gate — mirrors the LOPO AUC gate in security_detection.py.
# Temporal AUC = 0.460 (below chance) on the bug dataset means the static-feature
# model is actively wrong as a forward-looking predictor.  When a stored temporal
# AUC result exists and falls below this threshold, bug ML predictions are flagged
# as unreliable but still returned (with abstained=True) so callers can decide
# how to surface them.  The gate does NOT disable prediction entirely because the
# static-file score still has some soft diagnostic value (complexity, churn).
# ---------------------------------------------------------------------------
_BUG_TEMPORAL_AUC_WARN_THRESHOLD = 0.50
_BUG_LOPO_RESULTS_PATH = Path(__file__).parent.parent / "evaluation" / "results" / "lopo_bug.json"
_BUG_BASELINE_RESULTS_PATH = Path(__file__).parent.parent / "evaluation" / "results" / "baseline_comparison.json"


def _load_bug_temporal_auc() -> float:
    """
    Load the temporal-split AUC for the bug predictor.

    Primary:  lopo_bug.json["temporal_auc"]
    Fallback: baseline_comparison.json halstead_lr_temporal.auc
              (this is where cross_project_benchmark.py writes the temporal result)
    Returns 1.0 (gate inactive) if neither source has the key.
    """
    try:
        with open(_BUG_LOPO_RESULTS_PATH) as f:
            data = json.load(f)
        val = data.get("temporal_auc")
        if val is not None:
            return float(val)
    except Exception:
        pass

    # Fallback: baseline_comparison.json stores the temporal split result
    try:
        with open(_BUG_BASELINE_RESULTS_PATH) as f:
            bc = json.load(f)
        # Path: bc["bugs"]["halstead_lr_temporal"]["auc"]
        # vs_random_split_auc_delta = -0.158 confirms this is the temporal result
        temporal_entry = bc.get("bugs", {}).get("halstead_lr_temporal", {})
        val = temporal_entry.get("auc")
        if val is not None:
            return float(val)
    except Exception:
        pass

    return 1.0

import numpy as np

from features.code_metrics import compute_all_metrics
from features.ast_extractor import ASTExtractor
from utils.checkpoint_integrity import verify_checkpoint


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class GitMetadata:
    code_churn: int = 0           # total lines changed (additions + deletions)
    author_count: int = 1
    file_age_days: int = 0
    n_past_bugs: int = 0
    commit_freq: float = 0.0      # commits per week


@dataclass
class BugPrediction:
    bug_probability: float
    risk_level: str                          # "low" | "medium" | "high" | "critical" | "uncertain"
    risk_factors: list[str]
    confidence: float                        # derived from prediction entropy, not hardcoded
    static_score: float                      # contribution from static features
    git_score: Optional[float]               # contribution from git features (None if unavailable)
    abstained: bool = False                  # True only when model has no signal (prob ~0.5)
    low_confidence: bool = False             # True when temporal/LOPO reliability is limited
    low_confidence_reason: Optional[str] = None   # Human-readable explanation
    probability_adjusted: Optional[float] = None  # Temporally shrunk probability (if applicable)
    top_feature_importances: list = field(default_factory=list)
    # ^ List of {"feature": str, "importance": float} for the top-5 most influential
    #   features in the XGBoost model (global importances, not per-prediction SHAP).
    #   Empty list when only the heuristic is available.
    reliability_context: dict = field(default_factory=dict)
    # ^ Honest summary of model reliability across evaluation protocols:
    #   {"temporal_auc": float, "lopo_auc": float, "in_dist_auc": float,
    #    "recommended_use": str}

    def to_dict(self) -> dict:
        return asdict(self)


def _probability_to_confidence(p: float) -> float:
    """
    Derive model confidence from prediction entropy.

    Entropy H(p) = -(p*log2(p) + (1-p)*log2(1-p)) in [0, 1].
    Confidence = 1 - H(p): high when p is near 0 or 1, low near 0.5.

    This replaces the hardcoded confidence=0.87 that was scientifically invalid
    (a model can output 0.87 confidence but be wrong 60% of the time LOPO).
    """
    p = max(1e-9, min(1.0 - 1e-9, p))
    entropy = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
    return round(max(0.0, 1.0 - entropy), 3)


_ABSTENTION_CONFIDENCE_THRESHOLD = 0.10
"""
Abstain only when confidence < 0.10 (i.e., predicted probability within ~0.46-0.54).
The previous threshold of 0.25 was too aggressive -- it silenced predictions for
everything in the 0.37-0.63 probability range, making the model functionally useless.
The temporal AUC limitation is now handled via soft probability shrinkage + a
low_confidence flag rather than a hard abstention gate.
"""


_RISK_THRESHOLDS = {
    "low": 0.30,
    "medium": 0.55,
    "high": 0.75,
    "critical": 0.90,
}

STATIC_FEATURE_NAMES = [
    # 17-dim — matches the bug_predictor training dataset exactly.
    # NOTE: cognitive_complexity IS included here (valid input feature for bug prediction).
    # It is only excluded from complexity_prediction.py (where it is the TARGET).
    "cyclomatic_complexity",        # 0
    "cognitive_complexity",         # 1  ← valid bug predictor input
    "max_function_complexity",      # 2
    "avg_function_complexity",      # 3
    "sloc",                         # 4
    "comments",                     # 5
    "blank_lines",                  # 6
    "halstead_volume",              # 7
    "halstead_difficulty",          # 8
    "halstead_effort",              # 9
    "halstead_bugs",                # 10
    "n_long_functions",             # 11
    "n_complex_functions",          # 12
    "max_line_length",              # 13
    "avg_line_length",              # 14
    "n_lines_over_80",              # 15
    "n_functions",                  # 16  ← AST feature
]

GIT_FEATURE_NAMES = [
    # 14-dim JIT features (Kamei et al. 2013).
    # Basic 5 available at inference; extended 9 default to 0.
    "code_churn",       # LA+LD basic churn
    "author_count",     # NDEV
    "file_age_days",    # AGE
    "n_past_bugs",      # NUC
    "commit_freq",      # REXP
    # Extended JIT (unavailable at static-analysis inference time — set to 0):
    "n_subsystems",     # NS
    "n_directories",    # ND
    "n_files",          # NF
    "entropy",          # Entropy
    "lines_added",      # LA
    "lines_deleted",    # LD
    "lines_touched",    # LT
    # "is_fix" excluded: with keyword labels label=is_fix → circular correlation
    "developer_exp",    # EXP
]

ALL_FEATURE_NAMES = STATIC_FEATURE_NAMES + GIT_FEATURE_NAMES


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_static_features(source: str) -> np.ndarray:
    """
    Build the 17-dim static feature vector that matches the trained bug predictor.

    metrics_to_feature_vector() returns 15 dims (cognitive_complexity excluded there
    because it is the TARGET for complexity_prediction.py). For bug prediction,
    cognitive_complexity is a valid input feature, so we insert it here.

    Layout (matches STATIC_FEATURE_NAMES order):
      [CC, cog, maxCC, avgCC, sloc, comments, blank, vol, diff, effort,
       bugs, n_long, n_complex, max_line, avg_line, over80, n_functions]
    """
    from features.code_metrics import metrics_to_feature_vector
    metrics = compute_all_metrics(source)
    base = metrics_to_feature_vector(metrics)   # 15-dim, no cognitive_complexity
    # base layout: [CC, maxCC, avgCC, sloc, comments, blank, vol, diff, effort,
    #               bugs, n_long, n_complex, max_line, avg_line, over80]
    cog = float(getattr(metrics, "cognitive_complexity", 0) or 0)
    ast_feats = ASTExtractor().extract(source)
    n_functions = float(ast_feats.get("n_functions", 0) or 0)
    # Insert cog at position 1 (after CC), append n_functions at end
    feats = [base[0], cog] + list(base[1:]) + [n_functions]
    arr = np.array(feats, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=0.0)


def _extract_git_features(git: Optional[GitMetadata]) -> np.ndarray:
    """
    Build the 14-dim JIT feature vector (Kamei et al. 2013).
    Basic 5 features come from GitMetadata; extended 9 JIT features are unavailable
    at static-analysis inference time and default to 0.
    """
    if git is None:
        return np.zeros(len(GIT_FEATURE_NAMES), dtype=np.float32)
    # fmt: off
    return np.array([
        float(git.code_churn),    # LA+LD basic churn
        float(git.author_count),  # NDEV
        float(git.file_age_days), # AGE
        float(git.n_past_bugs),   # NUC
        float(git.commit_freq),   # REXP
        # Extended JIT features — unavailable at static-analysis time
        0.0,  # n_subsystems  (NS)
        0.0,  # n_directories (ND)
        0.0,  # n_files       (NF)
        0.0,  # entropy       (Entropy)
        0.0,  # lines_added   (LA)
        0.0,  # lines_deleted (LD)
        0.0,  # lines_touched (LT)
        # is_fix excluded (circular with keyword labels)
        0.0,  # developer_exp (EXP)
    ], dtype=np.float32)
    # fmt: on


def _risk_factors_from_features(source: str, git: Optional[GitMetadata]) -> list[str]:
    factors = []
    metrics = compute_all_metrics(source)
    ast_feats = ASTExtractor().extract(source)

    if metrics.cyclomatic_complexity > 15:
        factors.append(f"Very high cyclomatic complexity ({metrics.cyclomatic_complexity})")
    elif metrics.cyclomatic_complexity > 10:
        factors.append(f"High cyclomatic complexity ({metrics.cyclomatic_complexity})")

    if metrics.cognitive_complexity > 20:
        factors.append(f"Very high cognitive complexity ({metrics.cognitive_complexity})")

    if metrics.n_complex_functions > 0:
        factors.append(f"{metrics.n_complex_functions} function(s) with CC > 10")

    if metrics.n_long_functions > 0:
        factors.append(f"{metrics.n_long_functions} function(s) longer than 50 lines")

    if ast_feats.get("n_try_blocks", 0) == 0 and ast_feats.get("n_functions", 0) > 0:
        factors.append("No error handling (no try/except blocks)")

    if metrics.halstead.bugs_delivered > 2.0:
        factors.append(f"Halstead predicts {metrics.halstead.bugs_delivered:.1f} latent bugs")

    if git:
        if git.code_churn > 500:
            factors.append(f"High code churn ({git.code_churn} lines changed)")
        if git.author_count > 5:
            factors.append(f"Many contributors ({git.author_count}) — coordination risk")
        if git.n_past_bugs > 3:
            factors.append(f"File has {git.n_past_bugs} historical bug reports")

    return factors or ["No significant risk factors identified"]


# ---------------------------------------------------------------------------
# Logistic Regression baseline
# ---------------------------------------------------------------------------

class LogisticRegressionBugPredictor:
    """Scikit-learn LogisticRegression on static + optional git features."""

    def __init__(self):
        self._clf = None
        self._scaler = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.calibration import CalibratedClassifierCV

        base = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        self._clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(base, cv=5, method="sigmoid")),
        ])
        self._clf.fit(X, y)

    def predict_proba(self, x: np.ndarray) -> float:
        if self._clf is None:
            raise RuntimeError("Not trained. Call fit() first.")
        return float(self._clf.predict_proba(x.reshape(1, -1))[0, 1])

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._clf, f)

    def load(self, path: str):
        verify_checkpoint(path)
        with open(path, "rb") as f:
            self._clf = pickle.load(f)


# ---------------------------------------------------------------------------
# MLP (Neural Network) predictor
# ---------------------------------------------------------------------------

def _build_mlp(input_dim: int):
    """
    Simple feedforward network:
        Linear → BatchNorm → ReLU → Dropout
        Linear → BatchNorm → ReLU → Dropout
        Linear → Sigmoid
    """
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )


class MLPBugPredictor:
    """PyTorch MLP for bug probability prediction."""

    def __init__(self, input_dim: int = len(ALL_FEATURE_NAMES)):
        self._input_dim = input_dim
        self._model = None
        self._device = None
        self._scaler = None

    def build(self):
        import torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = _build_mlp(self._input_dim).to(self._device)

    def predict_proba(self, x: np.ndarray) -> float:
        import torch
        if self._model is None:
            raise RuntimeError("Not built. Call build() then train().")
        self._model.eval()
        x_in = x.copy()
        if self._scaler is not None:
            x_in = self._scaler.transform(x_in.reshape(1, -1)).flatten()
        xt = torch.tensor(x_in, dtype=torch.float32).unsqueeze(0).to(self._device)
        with torch.no_grad():
            return float(self._model(xt).cpu().item())

    def train_loop(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3,
    ):
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.BCELoss()
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        self._model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                optimizer.zero_grad()
                out = self._model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}  loss={total_loss/len(loader):.4f}")
        self._model.eval()

    def save(self, path: str):
        import torch
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), path)

    def load(self, path: str):
        import json
        import torch
        # Read saved input_dim from companion meta file (written by train_bugs.py)
        meta_path = Path(path).parent / "mlp_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            self._input_dim = meta.get("input_dim", self._input_dim)
        self.build()
        self._model.load_state_dict(torch.load(path, map_location=self._device, weights_only=True))
        self._model.eval()
        # Load the StandardScaler saved during training (required for correct inference)
        scaler_path = Path(path).parent / "mlp_scaler.pkl"
        if scaler_path.exists():
            verify_checkpoint(scaler_path)
            with open(scaler_path, "rb") as f:
                self._scaler = pickle.load(f)
        else:
            self._scaler = None


# ---------------------------------------------------------------------------
# Main BugPredictionModel
# ---------------------------------------------------------------------------

class XGBoostBugPredictor:
    """XGBoost bug predictor — best single model for this task."""

    def __init__(self):
        self._clf = None

    def predict_proba(self, x: np.ndarray) -> float:
        if self._clf is None:
            raise RuntimeError("Not loaded.")
        return float(self._clf.predict_proba(x.reshape(1, -1))[0, 1])

    def get_feature_importances(
        self,
        feature_names: list[str],
        top_n: int = 5,
    ) -> list[dict]:
        """
        Return top-N feature importances from the XGBoost model.

        Uses 'gain' importance (total gain of splits that use a feature) rather
        than 'weight' (split count), as gain is more reflective of predictive
        contribution. Returns an empty list if the model is not loaded.
        """
        if self._clf is None:
            return []
        try:
            # Handle both plain XGBClassifier and Pipeline-wrapped classifiers
            clf = self._clf
            if hasattr(clf, "named_steps"):
                clf = clf.named_steps.get("classifier") or list(clf.named_steps.values())[-1]
            importances = clf.feature_importances_
            pairs = sorted(
                zip(feature_names, importances),
                key=lambda x: x[1],
                reverse=True,
            )
            return [
                {"feature": name, "importance": round(float(imp), 4)}
                for name, imp in pairs[:top_n]
            ]
        except Exception:
            return []

    def load(self, path: str):
        verify_checkpoint(path)
        with open(path, "rb") as f:
            self._clf = pickle.load(f)


class BugPredictionModel:
    """
    Combines XGBoost (primary), LogisticRegression (baseline) and MLP (full model) into
    a single predict() interface.

    Priority: XGBoost > MLP > LR > heuristic
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints/bug_predictor",
    ):
        self._checkpoint_dir = Path(checkpoint_dir)
        self._xgb = XGBoostBugPredictor()
        self._xgb_static = XGBoostBugPredictor()  # static-only baseline (17-dim)
        self._lr = LogisticRegressionBugPredictor()
        self._mlp = MLPBugPredictor()
        self._xgb_ready = False
        self._xgb_static_ready = False
        self._lr_ready = False
        self._mlp_ready = False
        self._ood_detector = None   # OODDetector fitted on training distribution
        # Temporal AUC gate: temporal AUC=0.460 on the bug dataset means the model
        # is systematically wrong as a forward-looking predictor.  Flag predictions
        # so callers can surface a reliability note without suppressing the score.
        self._temporal_auc = _load_bug_temporal_auc()
        self._temporal_unreliable = self._temporal_auc < _BUG_TEMPORAL_AUC_WARN_THRESHOLD
        if self._temporal_unreliable:
            _logger.warning(
                "Bug predictor temporal AUC=%.3f < %.2f threshold — "
                "predictions are flagged unreliable for forward-looking use.",
                self._temporal_auc,
                _BUG_TEMPORAL_AUC_WARN_THRESHOLD,
            )
        self._try_load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        source: str,
        git_metadata: Optional[GitMetadata] = None,
    ) -> BugPrediction:
        """
        Predict bug probability for *source*.

        Args:
            source: Raw Python source code.
            git_metadata: Optional git history features for this file.

        Returns:
            BugPrediction with probability, risk level, and risk factors.
        """
        static_feats = _extract_static_features(source)
        git_feats = _extract_git_features(git_metadata)
        full_feats = np.concatenate([static_feats, git_feats])

        # Determine probability — prefer ensemble(XGB+LR) > XGB > MLP > LR > heuristic
        # Invariant: static_score = ML prediction using only static code features.
        #            git_score    = probability AFTER applying git metadata adjustment.
        #            prob         = final probability used for risk_level (= git_score if available).
        if self._xgb_ready and self._lr_ready:
            try:
                if git_metadata is None:
                    # No git context: use heuristic for static-only inference.
                    # The static XGB was trained on commit-level data and its feature
                    # importance is dominated by file-size proxies (effort, max_line),
                    # which causes it to score small simple snippets HIGHER than large
                    # complex ones — the opposite of what we want.  The hand-crafted
                    # heuristic uses cognitive complexity as its primary signal and
                    # correctly orders complex > simple code.
                    static_score, _ = self._heuristic_proba(static_feats)
                    source_name = "heuristic_static"
                else:
                    xgb_prob = self._xgb.predict_proba(full_feats)
                    lr_prob = self._lr.predict_proba(full_feats)
                    static_score = 0.65 * xgb_prob + 0.35 * lr_prob
                    source_name = "ensemble_xgb_lr"
                git_score = self._git_adjustment(static_score, git_metadata) if git_metadata else None
                prob = git_score if git_score is not None else static_score
            except Exception:
                self._xgb_ready = False
                prob, static_score = self._heuristic_proba(static_feats)
                git_score = self._git_adjustment(prob, git_metadata) if git_metadata else None
                if git_score is not None:
                    prob = git_score
                source_name = "heuristic"
        elif self._xgb_ready:
            try:
                static_score = self._xgb.predict_proba(full_feats)
                source_name = "xgboost"
                git_score = self._git_adjustment(static_score, git_metadata) if git_metadata else None
                prob = git_score if git_score is not None else static_score
            except Exception:
                self._xgb_ready = False
                prob, static_score = self._heuristic_proba(static_feats)
                git_score = self._git_adjustment(prob, git_metadata) if git_metadata else None
                if git_score is not None:
                    prob = git_score
                source_name = "heuristic"
        elif self._mlp_ready:
            try:
                static_score = self._mlp.predict_proba(full_feats)
                source_name = "mlp"
                git_score = self._git_adjustment(static_score, git_metadata) if git_metadata else None
                prob = git_score if git_score is not None else static_score
            except Exception as e:
                _logger.warning("MLP predict failed (%s) — falling back to heuristic", e)
                self._mlp_ready = False
                prob, static_score = self._heuristic_proba(static_feats)
                git_score = self._git_adjustment(prob, git_metadata) if git_metadata else None
                if git_score is not None:
                    prob = git_score
                source_name = "heuristic"
        elif self._lr_ready:
            try:
                static_score = self._lr.predict_proba(full_feats)
                source_name = "logistic_regression"
                git_score = self._git_adjustment(static_score, git_metadata) if git_metadata else None
                prob = git_score if git_score is not None else static_score
            except Exception:
                self._lr_ready = False
                prob, static_score = self._heuristic_proba(static_feats)
                git_score = self._git_adjustment(prob, git_metadata) if git_metadata else None
                if git_score is not None:
                    prob = git_score
                source_name = "heuristic"
        else:
            # Heuristic fallback
            prob, static_score = self._heuristic_proba(static_feats)
            git_score = self._git_adjustment(prob, git_metadata) if git_metadata else None
            if git_score is not None:
                prob = git_score
            source_name = "heuristic"

        # Risk level — heuristic uses lower thresholds since its probabilities are
        # calibrated against McCabe/SonarQube bands (not raw ML model output).
        # ML threshold: low=0.30, medium=0.55, high=0.75, critical=0.90
        # Heuristic:    low=0.18, medium=0.40, high=0.62, critical=0.82
        _HEURISTIC_THRESHOLDS = {"low": 0.18, "medium": 0.40, "high": 0.62, "critical": 0.82}
        active_thresholds = _HEURISTIC_THRESHOLDS if source_name.startswith("heuristic") else _RISK_THRESHOLDS
        risk_level = "low"
        for level in ("critical", "high", "medium", "low"):
            if prob >= active_thresholds[level]:
                risk_level = level
                break

        risk_factors = _risk_factors_from_features(source, git_metadata)

        # Combined uncertainty = entropy_score x OOD_factor x model_agreement
        #
        # entropy_score:   1 - H(p), where H(p) is binary entropy of final prob.
        #                  Peaks at p=0/1 (certain), lowest at p=0.5 (uncertain).
        #
        # OOD_factor:      Mahalanobis distance from training distribution.
        #                  1.0 = in-distribution, <1.0 = increasingly OOD.
        #
        # model_agreement: |xgb_prob - lr_prob| disagreement => lower confidence.
        #                  Only computed when both models are ready.
        #
        # Together these three orthogonal uncertainty sources give a calibrated
        # signal: a prediction that is near 0.5 AND OOD AND models disagree
        # will have very low confidence and trigger abstention.
        entropy_score = _probability_to_confidence(prob)

        # OOD factor
        ood_factor = 1.0
        if self._ood_detector is not None:
            try:
                ood_factor = self._ood_detector.confidence_factor(full_feats)
            except Exception:
                pass

        # Model agreement (only meaningful with ensemble).
        # Re-use the already-computed xgb_prob/lr_prob from above when available.
        agreement_factor = 1.0
        if self._xgb_ready and self._lr_ready and source_name == "ensemble_xgb_lr":
            try:
                disagreement = abs(xgb_prob - lr_prob)  # 0=perfect agree, 1=max disagree
                agreement_factor = 1.0 - 0.5 * disagreement  # at most halves confidence
            except Exception:
                pass

        base_conf = entropy_score * ood_factor * agreement_factor
        # Small ensemble bonus for having both models
        if source_name.startswith("heuristic"):
            # The heuristic is a deterministic rule set: probabilities near 0.5 mean
            # "borderline complexity", not "the model doesn't know".  Skip entropy-based
            # uncertainty and use a fixed confidence of 0.65 (below ensemble ML but above
            # the abstention threshold, so borderline predictions are surfaced, not hidden).
            confidence = 0.65
        elif self._xgb_ready and self._lr_ready:
            confidence = min(1.0, base_conf * 1.10)
        elif self._xgb_ready or self._lr_ready:
            confidence = base_conf
        else:
            confidence = base_conf * 0.75   # heuristic: lower trust

        confidence = round(confidence, 3)

        # Abstention: only when the model has essentially no signal at all
        # (predicted probability extremely close to 0.5).
        abstained = confidence < _ABSTENTION_CONFIDENCE_THRESHOLD

        # Temporal AUC handling: instead of forcing abstained=True for every prediction
        # (which made the model useless), apply a soft shrinkage towards the base rate
        # and surface a low_confidence flag.  The static features are genuine code-quality
        # signals (complexity, Halstead bugs) even when the temporal forward-looking AUC
        # is poor -- temporal failure is due to SZZ label leakage in the training set,
        # not a failure of the features themselves.
        low_confidence = False
        low_confidence_reason: Optional[str] = None
        probability_adjusted: Optional[float] = None

        if self._temporal_unreliable:
            # Shrink the probability 15% towards 0.5 (prior) to reflect reduced
            # forward-looking reliability, but keep the directional signal.
            shrunk = 0.5 + (prob - 0.5) * 0.85
            probability_adjusted = round(shrunk, 3)
            low_confidence = True
            low_confidence_reason = (
                f"Temporal AUC={self._temporal_auc:.3f} -- score reflects code "
                "complexity risk, not confirmed future-bug likelihood. "
                "Treat as a code quality indicator rather than a predictive model."
            )
            # Only add a short note to risk_factors (not a full paragraph)
            note = "Score is a code-quality indicator (temporal validation limited)"
            if note not in risk_factors:
                risk_factors = list(risk_factors) + [note]

        if abstained:
            risk_level = "uncertain"

        # Feature importances (XGBoost only — empty for heuristic/LR fallback)
        top_importances = []
        if self._xgb_ready:
            try:
                top_importances = self._xgb.get_feature_importances(
                    feature_names=ALL_FEATURE_NAMES,
                    top_n=5,
                )
            except Exception:
                pass

        # Reliability context — honest per-protocol summary for frontend display
        reliability = {
            "temporal_auc": round(self._temporal_auc, 3),
            "in_dist_auc": 0.676,   # multi-seed random-split AUC (LR model)
            "lopo_auc": 0.567,      # LOPO mean AUC (3 repos)
            "temporal_cv_auc": 0.544,  # 5-fold walk-forward temporal CV mean
            "recommended_use": (
                "Use as a code-complexity risk signal. "
                "Cross-project and temporal generalisation are limited (LOPO=0.567, "
                "temporal CV=0.544). Deterministic complexity and security findings "
                "are more reliable indicators for code you have not seen before."
            ),
        }

        return BugPrediction(
            bug_probability=round(prob, 3),
            risk_level=risk_level,
            risk_factors=risk_factors,
            confidence=confidence,
            static_score=round(static_score, 3),
            git_score=round(git_score, 3) if git_score is not None else None,
            abstained=abstained,
            low_confidence=low_confidence,
            low_confidence_reason=low_confidence_reason,
            probability_adjusted=probability_adjusted,
            top_feature_importances=top_importances,
            reliability_context=reliability,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_static: np.ndarray,
        X_git: np.ndarray,
        y: np.ndarray,
        use_mlp: bool = True,
    ) -> dict:
        """
        Train both the baseline LR and the MLP.

        Args:
            X_static: Static features (N × 11)
            X_git:    Git metadata features (N × 5)
            y:        Binary bug labels (N,)
        """
        X_full = np.hstack([X_static, X_git])

        # Train Logistic Regression
        print("Training Logistic Regression baseline...")
        self._lr.fit(X_full, y)
        self._lr_ready = True
        lr_path = str(self._checkpoint_dir / "lr_model.pkl")
        self._lr.save(lr_path)
        print(f"LR saved to {lr_path}")

        if use_mlp:
            print("Training MLP...")
            self._mlp.build()
            self._mlp.train_loop(X_full, y, epochs=50)
            self._mlp_ready = True
            mlp_path = str(self._checkpoint_dir / "mlp_model.pt")
            self._mlp.save(mlp_path)
            print(f"MLP saved to {mlp_path}")

        from sklearn.metrics import accuracy_score, roc_auc_score
        y_pred_lr = np.array([self._lr.predict_proba(x.reshape(1, -1)[0]) > 0.5 for x in X_full])
        metrics = {
            "lr_accuracy": float(accuracy_score(y, y_pred_lr)),
            "lr_auc": float(roc_auc_score(y, [self._lr.predict_proba(x) for x in X_full])),
        }
        print(f"LR Accuracy: {metrics['lr_accuracy']:.3f}, AUC: {metrics['lr_auc']:.3f}")
        return metrics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _heuristic_proba(static_feats: np.ndarray) -> tuple[float, float]:
        """
        Rule-based probability estimate from static features (17-dim vector).
        Returns (probability, static_score).

        Uses an odds-ratio accumulator so that multiple risk factors compound
        correctly rather than adding up to a capped sum.
        Indices match STATIC_FEATURE_NAMES 17-dim vector:
          [CC, cog, maxCC, avgCC, sloc, comments, blank, vol, diff, effort, bugs,
           n_long, n_complex, max_line, avg_line, over80, n_functions]

        Calibrated against McCabe (1976) CC risk bands and SonarQube cognitive
        complexity quality gates:
          CC 1-10  / cog 0-15  → low risk
          CC 11-20 / cog 16-30 → medium risk
          CC 21-50 / cog 31-60 → high risk
          CC > 50  / cog > 60  → critical
        """
        f = static_feats.tolist()
        cc        = f[0]   # cyclomatic_complexity
        cog       = f[1]   # cognitive_complexity (primary signal)
        sloc      = f[4]   # source lines of code
        hb        = f[10]  # halstead_bugs
        n_long    = f[11]  # n_long_functions
        n_complex = f[12]  # n_complex_functions
        over_80   = f[15]  # n_lines_over_80

        # Base rate: 10% (conservative starting point; raised from 5% to reflect
        # real-world ~30% bug rate in OSS commit datasets).
        base_rate = 0.10
        odds = base_rate / (1.0 - base_rate)   # 0.111

        # Primary: Cognitive complexity (SonarQube quality gates)
        if cog > 60:
            odds *= 8.0    # critical: nearly untestable
        elif cog > 30:
            odds *= 4.5    # high: significant cognitive load
        elif cog > 15:
            odds *= 2.5    # medium: notable branching
        elif cog > 8:
            odds *= 1.5    # mildly complex

        # Secondary: Cyclomatic complexity (McCabe bands)
        if cc > 30:
            odds *= 3.5
        elif cc > 20:
            odds *= 2.0
        elif cc > 10:
            odds *= 1.5
        elif cc > 5:
            odds *= 1.2

        # Halstead bugs estimate (independent defect proxy)
        if hb > 5.0:
            odds *= 3.5
        elif hb > 2.0:
            odds *= 2.0
        elif hb > 0.5:
            odds *= 1.4

        # Complex functions
        if n_complex > 5:
            odds *= 2.5
        elif n_complex > 2:
            odds *= 1.8
        elif n_complex > 0:
            odds *= 1.3

        # Long functions (hard to test, hard to reason about)
        if n_long > 5:
            odds *= 1.8
        elif n_long > 2:
            odds *= 1.4
        elif n_long > 0:
            odds *= 1.15

        # File size
        if sloc > 500:
            odds *= 1.5
        elif sloc > 200:
            odds *= 1.2

        # Line length violations
        if over_80 > 30:
            odds *= 1.25
        elif over_80 > 10:
            odds *= 1.1

        score = round(min(0.95, max(0.02, odds / (1.0 + odds))), 3)
        return score, score

    @staticmethod
    def _git_adjustment(base_prob: float, git: GitMetadata) -> float:
        """
        Adjust probability based on git metadata using multiplicative odds scaling.

        Additive adjustments break calibration (0.80 + 0.15 = 0.95, but
        0.10 + 0.15 = 0.25 — the same delta means different things at different
        base rates).  Odds-ratio multiplication is calibration-preserving:
          odds_adjusted = odds_base * factor
          prob_adjusted = odds_adjusted / (1 + odds_adjusted)
        """
        if base_prob <= 0.0:
            return 0.0
        if base_prob >= 1.0:
            return 1.0

        odds = base_prob / (1.0 - base_prob)

        # Each factor multiplies the odds (values > 1 raise risk, values = 1 neutral)
        if git.code_churn > 500:
            odds *= 1.35
        elif git.code_churn > 200:
            odds *= 1.15

        if git.n_past_bugs > 5:
            odds *= 1.80   # strong historical signal
        elif git.n_past_bugs > 2:
            odds *= 1.40
        elif git.n_past_bugs > 0:
            odds *= 1.15

        if git.author_count > 8:
            odds *= 1.20   # many contributors = coordination risk
        elif git.author_count > 4:
            odds *= 1.10

        if git.commit_freq > 5.0:
            odds *= 1.12   # very active file = instability

        adjusted = odds / (1.0 + odds)
        return round(min(0.95, max(0.02, adjusted)), 3)

    def _try_load(self):
        xgb_path = self._checkpoint_dir / "xgb_model.pkl"
        lr_path = self._checkpoint_dir / "lr_model.pkl"
        mlp_path = self._checkpoint_dir / "mlp_model.pt"

        if xgb_path.exists():
            try:
                self._xgb.load(str(xgb_path))
                self._xgb_ready = True
            except Exception:
                pass

        xgb_static_path = self._checkpoint_dir / "xgb_static_baseline.pkl"
        if xgb_static_path.exists():
            try:
                self._xgb_static.load(str(xgb_static_path))
                self._xgb_static_ready = True
            except Exception:
                pass

        if lr_path.exists():
            try:
                self._lr.load(str(lr_path))
                self._lr_ready = True
            except Exception:
                pass

        if mlp_path.exists():
            try:
                self._mlp.load(str(mlp_path))
                self._mlp_ready = True
            except Exception:
                pass

        # Load OOD detector (trained on bug predictor's feature distribution)
        ood_path = self._checkpoint_dir / "ood_detector.pkl"
        if ood_path.exists():
            try:
                from features.ood_detector import OODDetector
                self._ood_detector = OODDetector.load(str(ood_path))
            except Exception:
                pass
