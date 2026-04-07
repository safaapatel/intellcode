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


def _load_bug_temporal_auc() -> float:
    """Load stored temporal AUC for bug predictor. Returns 1.0 if file absent."""
    try:
        with open(_BUG_LOPO_RESULTS_PATH) as f:
            data = json.load(f)
        return float(data.get("temporal_auc", 1.0))
    except Exception:
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
    risk_level: str               # "low" | "medium" | "high" | "critical" | "uncertain"
    risk_factors: list[str]
    confidence: float             # derived from prediction entropy, not hardcoded
    static_score: float           # contribution from static features
    git_score: Optional[float]    # contribution from git features (None if unavailable)
    abstained: bool = False       # True if model chose not to predict (low confidence)

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


_ABSTENTION_CONFIDENCE_THRESHOLD = 0.25
"""
Abstain when confidence < 0.25 (i.e., predicted probability between ~0.37 and ~0.63).
In this range the model has no confident view on the outcome.
This is especially important under LOPO where temporal AUC=0.460 shows the model
can be systematically wrong -- abstaining beats a wrong confident prediction.
"""


_RISK_THRESHOLDS = {
    "low": 0.30,
    "medium": 0.55,
    "high": 0.75,
    "critical": 0.90,
}

STATIC_FEATURE_NAMES = [
    # 15-dim — matches metrics_to_feature_vector() exactly.
    # cognitive_complexity is EXCLUDED (it is the complexity prediction target).
    "cyclomatic_complexity",        # 0
    "max_function_complexity",      # 1
    "avg_function_complexity",      # 2
    "sloc",                         # 3
    "comments",                     # 4
    "blank_lines",                  # 5
    "halstead_volume",              # 6
    "halstead_difficulty",          # 7
    "halstead_effort",              # 8
    "halstead_bugs",                # 9
    "n_long_functions",             # 10
    "n_complex_functions",          # 11
    "max_line_length",              # 12
    "avg_line_length",              # 13
    "n_lines_over_80",              # 14
]

GIT_FEATURE_NAMES = [
    "code_churn",
    "author_count",
    "file_age_days",
    "n_past_bugs",
    "commit_freq",
]

ALL_FEATURE_NAMES = STATIC_FEATURE_NAMES + GIT_FEATURE_NAMES


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_static_features(source: str) -> np.ndarray:
    from features.code_metrics import metrics_to_feature_vector
    metrics = compute_all_metrics(source)
    # Use metrics_to_feature_vector to match the 15-feature vector used during training
    return np.array(metrics_to_feature_vector(metrics), dtype=np.float32)


def _extract_git_features(git: Optional[GitMetadata]) -> np.ndarray:
    if git is None:
        return np.zeros(len(GIT_FEATURE_NAMES), dtype=np.float32)
    return np.array([
        float(git.code_churn),
        float(git.author_count),
        float(git.file_age_days),
        float(git.n_past_bugs),
        float(git.commit_freq),
    ], dtype=np.float32)


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
        self._lr = LogisticRegressionBugPredictor()
        self._mlp = MLPBugPredictor()
        self._xgb_ready = False
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
                xgb_prob = self._xgb.predict_proba(full_feats)
                lr_prob = self._lr.predict_proba(full_feats)
                static_score = 0.5 * xgb_prob + 0.5 * lr_prob
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
            static_score = self._mlp.predict_proba(full_feats)
            source_name = "mlp"
            git_score = self._git_adjustment(static_score, git_metadata) if git_metadata else None
            prob = git_score if git_score is not None else static_score
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

        # Risk level
        risk_level = "low"
        for level in ("critical", "high", "medium", "low"):
            if prob >= _RISK_THRESHOLDS[level]:
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

        # Model agreement (only meaningful with ensemble)
        agreement_factor = 1.0
        if self._xgb_ready and self._lr_ready:
            try:
                xgb_p = self._xgb.predict_proba(full_feats)
                lr_p  = self._lr.predict_proba(full_feats)
                disagreement = abs(xgb_p - lr_p)   # 0=perfect agree, 1=max disagree
                agreement_factor = 1.0 - 0.5 * disagreement  # at most halves confidence
            except Exception:
                pass

        base_conf = entropy_score * ood_factor * agreement_factor
        # Small ensemble bonus for having both models
        if self._xgb_ready and self._lr_ready:
            confidence = min(1.0, base_conf * 1.10)
        elif self._xgb_ready or self._lr_ready:
            confidence = base_conf
        else:
            confidence = base_conf * 0.75   # heuristic: lower trust

        confidence = round(confidence, 3)

        # Abstention: if confidence is below threshold, decline to predict.
        # This avoids misleading triage when the model has no signal
        # (e.g., on projects unlike the training distribution).
        abstained = confidence < _ABSTENTION_CONFIDENCE_THRESHOLD

        # Temporal AUC gate: if stored temporal AUC < 0.50 the static-feature model
        # is anti-predictive for future bugs.  Mark abstained and add a note so the
        # caller can surface this to the user rather than suppressing output entirely.
        if self._temporal_unreliable:
            abstained = True
            temporal_note = (
                f"Reliability note: temporal AUC={self._temporal_auc:.3f} "
                f"(< {_BUG_TEMPORAL_AUC_WARN_THRESHOLD:.2f}) -- score reflects code "
                "complexity, not confirmed future-bug likelihood."
            )
            if temporal_note not in risk_factors:
                risk_factors = list(risk_factors) + [temporal_note]

        if abstained:
            risk_level = "uncertain"

        return BugPrediction(
            bug_probability=round(prob, 3),
            risk_level=risk_level,
            risk_factors=risk_factors,
            confidence=confidence,
            static_score=round(static_score, 3),
            git_score=round(git_score, 3) if git_score is not None else None,
            abstained=abstained,
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
        Rule-based probability estimate from static features (15-feature vector).
        Returns (probability, static_score).
        """
        f = static_feats.tolist()
        # indices match metrics_to_feature_vector() 15-dim vector:
        # [CC, maxCC, avgCC, sloc, comments, blank, vol, diff, effort, bugs,
        #  n_long, n_complex, max_line, avg_line, over80]
        cc        = f[0]   # cyclomatic_complexity
        hb        = f[9]   # halstead_bugs (bugs_delivered)
        n_long    = f[10]  # n_long_functions
        n_complex = f[11]  # n_complex_functions
        over_80   = f[14]  # n_lines_over_80

        score = 0.0
        score += min(0.3, (cc - 5) * 0.02) if cc > 5 else 0
        score += min(0.15, hb * 0.05)
        score += min(0.15, n_complex * 0.05)
        score += min(0.10, n_long * 0.03)
        score += min(0.05, over_80 * 0.001)

        return min(0.95, max(0.02, score)), min(0.95, max(0.02, score))

    @staticmethod
    def _git_adjustment(base_prob: float, git: GitMetadata) -> float:
        """Adjust probability based on git metadata."""
        adjustment = 0.0
        if git.code_churn > 500:
            adjustment += 0.10
        elif git.code_churn > 200:
            adjustment += 0.05
        if git.n_past_bugs > 3:
            adjustment += 0.15
        elif git.n_past_bugs > 0:
            adjustment += 0.05
        if git.author_count > 5:
            adjustment += 0.05
        return min(0.95, base_prob + adjustment)

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
