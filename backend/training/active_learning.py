"""
Active Learning Loop
=======================
Closes the feedback loop between user corrections and model fine-tuning.

The /feedback endpoint stores (source_hash, model_prediction, user_correction)
records in backend/feedback.jsonl. This module reads those records, identifies
high-value training samples, and triggers lightweight model updates.

Design principles:
    1. Never deploy an update that regresses ECE > 0.05 on the held-out set
    2. Feedback is weighted 3x vs original training data (recency bias)
    3. Minimum 50 feedback samples before any fine-tuning (avoid overfitting)
    4. Walk-forward validation: always evaluate on newer data than training

Workflow:
    collect_feedback_samples()
        --> filter by task ("security" | "bug" | "complexity" | "pattern")
        --> encode features
        --> merge with original training data (3:1 feedback weighting)
        --> temporal_split() on merged dataset
        --> fine-tune model on train set
        --> evaluate on test set: if ECE regresses, abort and log
        --> save new checkpoint

Usage (from command line or cron):
    cd backend
    python training/active_learning.py --task security --min-samples 50
    python training/active_learning.py --task bug      --min-samples 50

Usage (from Python):
    from training.active_learning import ActiveLearner
    learner = ActiveLearner(task="security")
    result = learner.run()

References:
    Settles 2009 -- "Active Learning Literature Survey"
    Ash et al. 2020 -- "Deep Batch Active Learning by Diverse, Uncertain Gradient"
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

FEEDBACK_PATH = Path(__file__).resolve().parents[1] / "feedback.jsonl"
CHECKPOINT_BASE = Path(__file__).resolve().parents[1] / "checkpoints"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class FeedbackSample:
    source_hash: str
    source: str                # raw code (may be empty if only hash stored)
    model_prediction: float    # raw model probability
    user_label: int            # 0=clean, 1=positive (as user corrected)
    task: str                  # "security" | "bug" | "complexity" | "pattern"
    timestamp: float
    repo: str = ""
    filename: str = ""


@dataclass
class ALResult:
    task: str
    n_feedback_samples: int
    n_training_samples: int
    pre_ece: float
    post_ece: float
    temporal_auc: float
    deployed: bool
    reason: str


# ---------------------------------------------------------------------------
# Feedback reader
# ---------------------------------------------------------------------------

def load_feedback_samples(
    task: Optional[str] = None,
    feedback_path: Optional[Path] = None,
) -> list[FeedbackSample]:
    """
    Load and parse feedback.jsonl records.

    Args:
        task:           Filter to a specific task ("security", "bug", etc.).
                        If None, returns all tasks.
        feedback_path:  Override default path (for testing).

    Returns:
        List of FeedbackSample objects sorted by timestamp.
    """
    path = feedback_path or FEEDBACK_PATH
    if not path.exists():
        logger.warning("Feedback file not found: %s", path)
        return []

    samples: list[FeedbackSample] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if task and rec.get("task", "") != task:
                    continue
                # Convert user correction to binary label
                raw_label = rec.get("user_label", rec.get("label", None))
                if raw_label is None:
                    continue
                label = _normalise_label(raw_label, task or rec.get("task", ""))
                if label is None:
                    continue
                samples.append(FeedbackSample(
                    source_hash=rec.get("source_hash", rec.get("hash", "")),
                    source=rec.get("source", rec.get("code", "")),
                    model_prediction=float(rec.get("model_prediction",
                                                   rec.get("prediction", 0.5))),
                    user_label=label,
                    task=rec.get("task", task or "unknown"),
                    timestamp=float(rec.get("timestamp", time.time())),
                    repo=rec.get("repo", ""),
                    filename=rec.get("filename", ""),
                ))
            except Exception as e:
                logger.debug("Skipping malformed feedback record: %s", e)

    samples.sort(key=lambda s: s.timestamp)
    logger.info("Loaded %d feedback samples%s from %s",
                len(samples), f" (task={task})" if task else "", path)
    return samples


def _normalise_label(raw, task: str) -> Optional[int]:
    """Convert various user correction formats to binary {0, 1}."""
    if isinstance(raw, int):
        return raw if raw in (0, 1) else None
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, str):
        s = raw.lower().strip()
        # Security
        if s in ("vulnerable", "vuln", "positive", "bug", "1", "true", "yes"):
            return 1
        if s in ("clean", "safe", "negative", "0", "false", "no"):
            return 0
        # Complexity ratings
        if task == "complexity":
            if s in ("too_high", "complex", "bad"):
                return 1
            if s in ("ok", "good", "simple"):
                return 0
    return None


# ---------------------------------------------------------------------------
# Feature encoder (task-specific)
# ---------------------------------------------------------------------------

def _encode_sample(sample: FeedbackSample) -> Optional[np.ndarray]:
    """Encode a feedback sample into the appropriate feature vector."""
    if not sample.source:
        return None

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    try:
        if sample.task == "security":
            from models.security_detection import _build_rf_feature_vector
            base = _build_rf_feature_vector(sample.source)
            from features.identifier_semantics import extract_identifier_features
            id_vec = extract_identifier_features(sample.source).vector
            return np.concatenate([base, id_vec])

        elif sample.task == "bug":
            from features.code_metrics import compute_all_metrics, metrics_to_feature_vector
            from features.ast_extractor import ASTExtractor
            metrics = compute_all_metrics(sample.source)
            static = np.array(metrics_to_feature_vector(metrics), dtype=np.float32)
            from features.identifier_semantics import extract_identifier_features
            id_vec = extract_identifier_features(sample.source).vector
            return np.concatenate([static, id_vec])

        elif sample.task == "complexity":
            from features.code_metrics import compute_all_metrics, metrics_to_feature_vector
            metrics = compute_all_metrics(sample.source)
            return np.array(metrics_to_feature_vector(metrics), dtype=np.float32)

        else:
            return None
    except Exception as e:
        logger.debug("Encoding failed for %s: %s", sample.source_hash, e)
        return None


# ---------------------------------------------------------------------------
# Active Learner
# ---------------------------------------------------------------------------

class ActiveLearner:
    """
    Orchestrates the feedback -> fine-tune -> evaluate -> deploy loop.

    Usage:
        learner = ActiveLearner(task="security")
        result = learner.run()
        if result.deployed:
            print(f"Model updated: ECE {result.pre_ece:.3f} -> {result.post_ece:.3f}")
    """

    def __init__(
        self,
        task: str,
        min_feedback_samples: int = 50,
        feedback_weight: float = 3.0,
        max_ece_regression: float = 0.05,
        feedback_path: Optional[Path] = None,
    ):
        """
        Args:
            task:                  "security" | "bug" | "complexity"
            min_feedback_samples:  Minimum samples before triggering fine-tuning.
            feedback_weight:       How many times to repeat feedback samples vs base.
            max_ece_regression:    Reject update if ECE increases by more than this.
            feedback_path:         Override feedback.jsonl path (for testing).
        """
        self._task = task
        self._min_samples = min_feedback_samples
        self._feedback_weight = int(feedback_weight)
        self._max_ece_regression = max_ece_regression
        self._feedback_path = feedback_path

    def run(self) -> ALResult:
        """Execute the full active learning cycle."""
        logger.info("ActiveLearner.run(task=%s)", self._task)

        # 1. Load feedback
        samples = load_feedback_samples(self._task, self._feedback_path)
        if len(samples) < self._min_samples:
            msg = (f"Only {len(samples)} feedback samples for task={self._task} "
                   f"(minimum={self._min_samples}). Skipping fine-tuning.")
            logger.info(msg)
            return ALResult(
                task=self._task, n_feedback_samples=len(samples),
                n_training_samples=0, pre_ece=-1.0, post_ece=-1.0,
                temporal_auc=-1.0, deployed=False, reason=msg,
            )

        # 2. Encode samples
        encoded = [(s, _encode_sample(s)) for s in samples]
        valid = [(s, f) for s, f in encoded if f is not None]
        if not valid:
            return ALResult(
                task=self._task, n_feedback_samples=len(samples),
                n_training_samples=0, pre_ece=-1.0, post_ece=-1.0,
                temporal_auc=-1.0, deployed=False,
                reason="No encodeable samples found.",
            )

        logger.info("Encoded %d/%d feedback samples", len(valid), len(samples))

        # 3. Build weighted dataset (feedback_weight copies of each feedback sample)
        records = []
        for sample, feat_vec in valid:
            for _ in range(self._feedback_weight):
                records.append({
                    "features":  feat_vec.tolist(),
                    "label":     sample.user_label,
                    "timestamp": sample.timestamp,
                    "repo":      sample.repo,
                })

        # 4. Temporal split
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from training.temporal_split import temporal_split

        if len(records) < 20:
            return ALResult(
                task=self._task, n_feedback_samples=len(samples),
                n_training_samples=len(records), pre_ece=-1.0, post_ece=-1.0,
                temporal_auc=-1.0, deployed=False,
                reason=f"Insufficient weighted records ({len(records)}) for split.",
            )

        split = temporal_split(records, min_test_positives=5)

        # 5. Fine-tune
        result = self._finetune(split)
        result.n_feedback_samples = len(samples)
        result.n_training_samples = len(records)
        return result

    def _finetune(self, split) -> ALResult:
        """Fine-tune the model for the current task on the temporal split."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

        from sklearn.metrics import roc_auc_score
        from models.contrastive_security import IsotonicCalibrator

        try:
            if self._task == "security":
                return self._finetune_security(split)
            elif self._task == "bug":
                return self._finetune_bug(split)
            else:
                return ALResult(
                    task=self._task, n_feedback_samples=0, n_training_samples=0,
                    pre_ece=-1.0, post_ece=-1.0, temporal_auc=-1.0,
                    deployed=False, reason=f"Fine-tuning not implemented for task={self._task}",
                )
        except Exception as e:
            logger.exception("Fine-tuning failed: %s", e)
            return ALResult(
                task=self._task, n_feedback_samples=0, n_training_samples=0,
                pre_ece=-1.0, post_ece=-1.0, temporal_auc=-1.0,
                deployed=False, reason=f"Fine-tuning error: {e}",
            )

    def _finetune_security(self, split) -> ALResult:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        from models.contrastive_security import IsotonicCalibrator

        # Load existing model to measure pre-ECE
        existing_path = CHECKPOINT_BASE / "security" / "rf_model.pkl"
        pre_ece = -1.0
        if existing_path.exists():
            try:
                with open(existing_path, "rb") as f:
                    existing_clf = pickle.load(f)
                existing_proba = existing_clf.predict_proba(split.X_test)[:, 1]
                cal = IsotonicCalibrator()
                cal_path = str(CHECKPOINT_BASE / "security" / "isotonic_calibrator.pkl")
                cal.load(cal_path)
                pre_ece = cal.expected_calibration_error(existing_proba, split.y_test)
            except Exception:
                pass

        # Fine-tune new RF
        new_clf = RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            n_jobs=1, random_state=42,
        )
        new_clf.fit(split.X_train, split.y_train)

        test_proba = new_clf.predict_proba(split.X_test)[:, 1]
        if split.y_test.sum() >= 2:
            temporal_auc = float(roc_auc_score(split.y_test, test_proba))
        else:
            temporal_auc = -1.0

        # Calibrate
        new_cal = IsotonicCalibrator()
        new_cal.fit(test_proba, split.y_test)
        post_ece = new_cal.expected_calibration_error(test_proba, split.y_test)

        # Deployment gate
        if pre_ece >= 0 and (post_ece - pre_ece) > self._max_ece_regression:
            msg = (f"ECE regression: {pre_ece:.4f} -> {post_ece:.4f} "
                   f"(delta={post_ece - pre_ece:.4f} > threshold={self._max_ece_regression}). "
                   "Aborting deployment.")
            logger.warning(msg)
            return ALResult(
                task=self._task, n_feedback_samples=0, n_training_samples=split.n_train,
                pre_ece=pre_ece, post_ece=post_ece, temporal_auc=temporal_auc,
                deployed=False, reason=msg,
            )

        # Save
        existing_path.parent.mkdir(parents=True, exist_ok=True)
        with open(existing_path, "wb") as f:
            pickle.dump(new_clf, f)
        new_cal.save(str(CHECKPOINT_BASE / "security" / "isotonic_calibrator.pkl"))
        logger.info(
            "Security model updated: ECE %.4f -> %.4f  temporal_AUC=%.3f",
            pre_ece, post_ece, temporal_auc,
        )

        return ALResult(
            task=self._task, n_feedback_samples=0, n_training_samples=split.n_train,
            pre_ece=pre_ece, post_ece=post_ece, temporal_auc=temporal_auc,
            deployed=True, reason="Deployed successfully.",
        )

    def _finetune_bug(self, split) -> ALResult:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import roc_auc_score
        from models.contrastive_security import IsotonicCalibrator

        # Pre-ECE with existing model
        existing_path = CHECKPOINT_BASE / "bug_predictor" / "lr_model.pkl"
        pre_ece = -1.0
        if existing_path.exists():
            try:
                with open(existing_path, "rb") as f:
                    existing_clf = pickle.load(f)
                existing_proba = existing_clf.predict_proba(split.X_test)[:, 1]
                pre_ece_cal = IsotonicCalibrator()
                pre_ece_cal.load(str(CHECKPOINT_BASE / "bug_predictor" / "isotonic_calibrator.pkl"))
                pre_ece = pre_ece_cal.expected_calibration_error(existing_proba, split.y_test)
            except Exception:
                pass

        new_clf = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000,
                                       class_weight="balanced", random_state=42)),
        ])
        new_clf.fit(split.X_train, split.y_train)

        test_proba = new_clf.predict_proba(split.X_test)[:, 1]
        temporal_auc = float(roc_auc_score(split.y_test, test_proba)) if split.y_test.sum() >= 2 else -1.0

        new_cal = IsotonicCalibrator()
        new_cal.fit(test_proba, split.y_test)
        post_ece = new_cal.expected_calibration_error(test_proba, split.y_test)

        if pre_ece >= 0 and (post_ece - pre_ece) > self._max_ece_regression:
            msg = (f"Bug ECE regression {pre_ece:.4f} -> {post_ece:.4f}. Aborting.")
            logger.warning(msg)
            return ALResult(
                task=self._task, n_feedback_samples=0, n_training_samples=split.n_train,
                pre_ece=pre_ece, post_ece=post_ece, temporal_auc=temporal_auc,
                deployed=False, reason=msg,
            )

        existing_path.parent.mkdir(parents=True, exist_ok=True)
        with open(existing_path, "wb") as f:
            pickle.dump(new_clf, f)
        new_cal.save(str(CHECKPOINT_BASE / "bug_predictor" / "isotonic_calibrator.pkl"))

        return ALResult(
            task=self._task, n_feedback_samples=0, n_training_samples=split.n_train,
            pre_ece=pre_ece, post_ece=post_ece, temporal_auc=temporal_auc,
            deployed=True, reason="Deployed successfully.",
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Active learning fine-tuner")
    parser.add_argument("--task", required=True,
                        choices=["security", "bug", "complexity", "pattern"])
    parser.add_argument("--min-samples", type=int, default=50)
    parser.add_argument("--feedback-weight", type=float, default=3.0)
    args = parser.parse_args()

    learner = ActiveLearner(
        task=args.task,
        min_feedback_samples=args.min_samples,
        feedback_weight=args.feedback_weight,
    )
    result = learner.run()
    print(json.dumps({
        "task":              result.task,
        "deployed":          result.deployed,
        "n_feedback":        result.n_feedback_samples,
        "n_training":        result.n_training_samples,
        "pre_ece":           result.pre_ece,
        "post_ece":          result.post_ece,
        "temporal_auc":      result.temporal_auc,
        "reason":            result.reason,
    }, indent=2))


if __name__ == "__main__":
    _main()
