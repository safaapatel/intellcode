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

import pickle
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np

from features.code_metrics import compute_all_metrics
from features.ast_extractor import ASTExtractor


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
    risk_level: str               # "low" | "medium" | "high" | "critical"
    risk_factors: list[str]
    confidence: float
    static_score: float           # contribution from static features
    git_score: Optional[float]    # contribution from git features (None if unavailable)

    def to_dict(self) -> dict:
        return asdict(self)


_RISK_THRESHOLDS = {
    "low": 0.30,
    "medium": 0.55,
    "high": 0.75,
    "critical": 0.90,
}

STATIC_FEATURE_NAMES = [
    "cyclomatic_complexity",
    "cognitive_complexity",
    "halstead_bugs",
    "n_long_functions",
    "n_complex_functions",
    "sloc",
    "n_functions",
    "n_try_blocks",
    "n_raises",
    "max_params",
    "n_lines_over_80",
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
    metrics = compute_all_metrics(source)
    ast_feats = ASTExtractor().extract(source)

    return np.array([
        float(metrics.cyclomatic_complexity),
        float(metrics.cognitive_complexity),
        float(metrics.halstead.bugs_delivered),
        float(metrics.n_long_functions),
        float(metrics.n_complex_functions),
        float(metrics.lines.sloc),
        float(ast_feats.get("n_functions", 0)),
        float(ast_feats.get("n_try_blocks", 0)),
        float(ast_feats.get("n_raises", 0)),
        float(ast_feats.get("max_params", 0)),
        float(metrics.n_lines_over_80),
    ], dtype=np.float32)


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

    def build(self):
        import torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = _build_mlp(self._input_dim).to(self._device)

    def predict_proba(self, x: np.ndarray) -> float:
        import torch
        if self._model is None:
            raise RuntimeError("Not built. Call build() then train().")
        self._model.eval()
        xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self._device)
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
        import torch
        self.build()
        self._model.load_state_dict(torch.load(path, map_location=self._device))
        self._model.eval()


# ---------------------------------------------------------------------------
# Main BugPredictionModel
# ---------------------------------------------------------------------------

class BugPredictionModel:
    """
    Combines LogisticRegression (baseline) and MLP (full model) into
    a single predict() interface.

    Status: In Development — falls back to heuristic scoring when no
    trained model is available.
    """

    def __init__(
        self,
        checkpoint_dir: str = "checkpoints/bug_predictor",
    ):
        self._checkpoint_dir = Path(checkpoint_dir)
        self._lr = LogisticRegressionBugPredictor()
        self._mlp = MLPBugPredictor()
        self._lr_ready = False
        self._mlp_ready = False
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

        # Determine probability
        if self._mlp_ready:
            prob = self._mlp.predict_proba(full_feats)
            source_name = "mlp"
            static_score = prob  # approximation
            git_score = prob if git_metadata else None
        elif self._lr_ready:
            try:
                prob = self._lr.predict_proba(full_feats)
                source_name = "logistic_regression"
                static_score = prob
                git_score = prob if git_metadata else None
            except Exception:
                # Checkpoint version mismatch — fall back to heuristic
                self._lr_ready = False
                prob, static_score = self._heuristic_proba(static_feats)
                git_score = self._git_adjustment(prob, git_metadata) if git_metadata else None
                if git_score is not None:
                    prob = min(1.0, (prob + git_score) / 2 * 1.2)
                source_name = "heuristic"
        else:
            # Heuristic fallback
            prob, static_score = self._heuristic_proba(static_feats)
            git_score = self._git_adjustment(prob, git_metadata) if git_metadata else None
            if git_score is not None:
                prob = min(1.0, (prob + git_score) / 2 * 1.2)
            source_name = "heuristic"

        # Risk level
        risk_level = "low"
        for level in ("critical", "high", "medium", "low"):
            if prob >= _RISK_THRESHOLDS[level]:
                risk_level = level
                break

        risk_factors = _risk_factors_from_features(source, git_metadata)

        return BugPrediction(
            bug_probability=round(prob, 3),
            risk_level=risk_level,
            risk_factors=risk_factors,
            confidence=0.78 if self._lr_ready else 0.60,
            static_score=round(static_score, 3),
            git_score=round(git_score, 3) if git_score is not None else None,
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
        Rule-based probability estimate from static features.
        Returns (probability, static_score).
        """
        (cc, cog, hb, n_long, n_complex, sloc,
         n_funcs, n_try, n_raises, max_params, over_80) = static_feats.tolist()

        score = 0.0
        # Each metric pushes toward 1.0
        score += min(0.3, (cc - 5) * 0.02) if cc > 5 else 0
        score += min(0.2, cog * 0.008)
        score += min(0.15, hb * 0.05)
        score += min(0.15, n_complex * 0.05)
        score += min(0.10, n_long * 0.03)
        if n_try == 0 and n_funcs > 0:
            score += 0.05   # no error handling
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
        lr_path = self._checkpoint_dir / "lr_model.pkl"
        mlp_path = self._checkpoint_dir / "mlp_model.pt"

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
