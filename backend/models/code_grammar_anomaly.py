"""
Code Grammar Anomaly Model (CGAM)
===================================
An unsupervised anomaly detector that models Python code as a Variable-Order
Markov chain over AST node-type sequences and flags anomalous files by their
perplexity under the learned grammar.

Invention
---------
Standard code quality models are discriminative: they learn P(buggy | features)
from labeled data. CGAM is generative: it learns P(code | "normal code grammar")
from UNLABELED clean code, then detects anomalies as code with low probability
under that grammar.

The novel insight: bugs, anti-patterns, and security vulnerabilities are ALL
structural anomalies — they break the patterns that "normal" code follows.
Rather than training separate models for each defect type, CGAM provides a
single unified signal: does this code look like normal Python?

This is different from:
  - OOD detection (Mahalanobis on feature vectors): operates on scalar metrics,
    not on structural grammar.
  - CodeBERT perplexity: requires a 125M-parameter transformer; CGAM runs in
    <1ms on CPU using pure Python counts.
  - AST edit distance: compares to a specific reference; CGAM compares to a
    learned distribution over all training code.

Algorithm: Variable-Order Markov Model (VOMM)
---------------------------------------------
1. Extract the DFS-order sequence of AST node type names from each training file.
2. Count n-gram frequencies for n = 1, 2, 3 (context window up to 2 nodes).
3. At inference:
     For each trigram (t_{i-2}, t_{i-1}, t_i) in the file:
       P(t_i | t_{i-2}, t_{i-1}) = count(t_{i-2}, t_{i-1}, t_i) / count(t_{i-2}, t_{i-1})
       Fall back to bigram if trigram not seen; fall back to unigram if bigram not seen.
       Laplace smoothing: add alpha=0.1 to all counts.
4. Perplexity = exp(-mean(log P(t_i | context)))
5. Anomaly score = sigmoid((perplexity - threshold) / scale) in [0, 1]

Perplexity interpretation:
  - Low perplexity  → code follows the learned grammar → likely clean
  - High perplexity → code violates expected structure → anomalous

Outputs
-------
GrammarAnomalyResult(
    perplexity: float       — raw perplexity of the code under the learned grammar
    anomaly_score: float    — calibrated [0, 1] score (1 = most anomalous)
    is_anomalous: bool      — True if anomaly_score > threshold (default 0.6)
    top_anomalous_ngrams: list[dict]  — trigrams with lowest P (most surprising)
    n_tokens: int           — number of AST node types in sequence
)
"""

from __future__ import annotations

import ast
import json
import logging
import math
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("checkpoints/cgam")
MODEL_PATH     = CHECKPOINT_DIR / "vomm.pkl"
METRICS_PATH   = CHECKPOINT_DIR / "metrics.json"

_LAPLACE_ALPHA = 0.1
_ANOMALY_THRESHOLD = 0.60    # anomaly_score > this → is_anomalous = True


# ---------------------------------------------------------------------------
# AST sequence extraction
# ---------------------------------------------------------------------------

def extract_node_sequence(source: str) -> list[str]:
    """
    DFS-order sequence of AST node type names from Python source.

    Example: "def f(): return 1" →
        ["Module", "FunctionDef", "arguments", "Return", "Constant"]

    Terminal nodes (Load, Store, Del, etc.) are stripped since they carry
    no structural information — they are always children of Name/Attribute
    and add noise without signal.
    """
    _SKIP = frozenset({
        "Load", "Store", "Del", "AugLoad", "AugStore",
        "Param", "Suite", "Expr",  # Expr is a statement wrapper, not informative
    })

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    sequence: list[str] = []

    def _walk(node):
        name = type(node).__name__
        if name not in _SKIP:
            sequence.append(name)
        for child in ast.iter_child_nodes(node):
            _walk(child)

    _walk(tree)
    return sequence


# ---------------------------------------------------------------------------
# Variable-Order Markov Model
# ---------------------------------------------------------------------------

class VOMM:
    """
    Variable-Order Markov Model over token sequences.

    Stores unigram, bigram, and trigram counts and computes
    smoothed conditional probabilities using back-off.
    """

    def __init__(self, alpha: float = _LAPLACE_ALPHA):
        self.alpha = alpha
        self._uni: dict[str, int] = defaultdict(int)     # count(t)
        self._bi:  dict[tuple, int] = defaultdict(int)   # count(t1, t2)
        self._tri: dict[tuple, int] = defaultdict(int)   # count(t1, t2, t3)
        self._vocab: set[str] = set()
        self._total_tokens: int = 0
        self._fitted = False

    def fit(self, sequences: list[list[str]]) -> "VOMM":
        """Count n-grams from a list of token sequences."""
        for seq in sequences:
            self._vocab.update(seq)
            self._total_tokens += len(seq)
            for i, tok in enumerate(seq):
                self._uni[tok] += 1
                if i >= 1:
                    self._bi[(seq[i-1], tok)] += 1
                if i >= 2:
                    self._tri[(seq[i-2], seq[i-1], tok)] += 1
        self._fitted = True
        return self

    def log_prob(self, sequence: list[str]) -> tuple[float, list[tuple]]:
        """
        Compute mean log-probability of a sequence and return the
        individual trigram probabilities for anomaly attribution.

        Returns (mean_log_prob, [(trigram_str, log_prob), ...])
        """
        if not self._fitted or len(sequence) < 2:
            return 0.0, []

        V = max(len(self._vocab), 1)
        log_probs: list[float] = []
        trigram_details: list[tuple] = []

        for i in range(1, len(sequence)):
            t = sequence[i]
            t1 = sequence[i-1]
            t2 = sequence[i-2] if i >= 2 else None

            # Try trigram first
            if t2 is not None:
                ctx = (t2, t1)
                ctx_count = self._bi.get(ctx, 0)
                tri_count = self._tri.get((t2, t1, t), 0)
                if ctx_count > 0:
                    p = (tri_count + self.alpha) / (ctx_count + self.alpha * V)
                    lp = math.log(p)
                    log_probs.append(lp)
                    trigram_details.append((f"{t2}→{t1}→{t}", lp))
                    continue

            # Bigram fallback
            uni_ctx = self._uni.get(t1, 0)
            bi_count = self._bi.get((t1, t), 0)
            if uni_ctx > 0:
                p = (bi_count + self.alpha) / (uni_ctx + self.alpha * V)
                lp = math.log(p)
                log_probs.append(lp)
                trigram_details.append((f"{t1}→{t}", lp))
                continue

            # Unigram fallback
            uni_count = self._uni.get(t, 0)
            p = (uni_count + self.alpha) / (self._total_tokens + self.alpha * V)
            lp = math.log(p)
            log_probs.append(lp)
            trigram_details.append((f"{t}", lp))

        mean_lp = float(np.mean(log_probs)) if log_probs else 0.0
        return mean_lp, trigram_details

    def perplexity(self, sequence: list[str]) -> float:
        mean_lp, _ = self.log_prob(sequence)
        return float(math.exp(-mean_lp)) if mean_lp < 0 else 1.0

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> Optional["VOMM"]:
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Calibration: map perplexity → anomaly score
# ---------------------------------------------------------------------------

@dataclass
class CalibrationParams:
    """Sigmoid calibration: score = sigmoid((perplexity - mu) / sigma)."""
    mu: float     # median perplexity of training set
    sigma: float  # IQR / 1.35 (robust scale estimate)

    def score(self, perplexity: float) -> float:
        z = (perplexity - self.mu) / max(self.sigma, 1.0)
        return float(1.0 / (1.0 + math.exp(-z)))


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class GrammarAnomalyResult:
    perplexity: float
    anomaly_score: float
    is_anomalous: bool
    top_anomalous_ngrams: list = field(default_factory=list)
    n_tokens: int = 0
    grammar_coverage: float = 0.0   # fraction of trigrams seen in training

    def to_dict(self) -> dict:
        return {
            "perplexity":           round(self.perplexity, 3),
            "anomaly_score":        round(self.anomaly_score, 4),
            "is_anomalous":         self.is_anomalous,
            "top_anomalous_ngrams": self.top_anomalous_ngrams,
            "n_tokens":             self.n_tokens,
            "grammar_coverage":     round(self.grammar_coverage, 3),
        }


# ---------------------------------------------------------------------------
# Main model class
# ---------------------------------------------------------------------------

class CodeGrammarAnomalyModel:
    """
    CGAM: learns normal Python grammar from unlabeled code and detects
    anomalies by perplexity under the learned model.
    """

    def __init__(self, alpha: float = _LAPLACE_ALPHA,
                 anomaly_threshold: float = _ANOMALY_THRESHOLD):
        self._vomm: Optional[VOMM] = None
        self._calib: Optional[CalibrationParams] = None
        self._alpha = alpha
        self._threshold = anomaly_threshold
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready and self._vomm is not None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, sources: list[str], output_dir: str | None = None) -> dict:
        """
        Train CGAM on a list of Python source code strings.

        No labels needed — all training sources are treated as "normal" code.
        Use clean, well-reviewed repos (e.g., the SECURITY_NEGATIVE_REPOS list).

        Args:
            sources:    List of Python source strings (clean code only).
            output_dir: Directory for saving checkpoint.

        Returns:
            Training metrics dict.
        """
        logger.info("CGAM: extracting AST sequences from %d files ...", len(sources))
        sequences = []
        for src in sources:
            seq = extract_node_sequence(src)
            if len(seq) >= 5:
                sequences.append(seq)

        if len(sequences) < 10:
            raise ValueError(f"Need at least 10 valid sources; got {len(sequences)}")

        self._vomm = VOMM(alpha=self._alpha)
        self._vomm.fit(sequences)

        # Compute perplexity on training set for calibration
        train_perps = [self._vomm.perplexity(seq) for seq in sequences]
        train_perps_arr = np.array(train_perps)

        mu    = float(np.median(train_perps_arr))
        q25, q75 = float(np.percentile(train_perps_arr, 25)), float(np.percentile(train_perps_arr, 75))
        sigma = (q75 - q25) / 1.35   # normalised IQR = robust std estimate

        self._calib = CalibrationParams(mu=mu, sigma=max(sigma, 1.0))
        self._ready = True

        metrics = {
            "n_training_files":  len(sequences),
            "vocab_size":        len(self._vomm._vocab),
            "n_unigrams":        len(self._vomm._uni),
            "n_bigrams":         len(self._vomm._bi),
            "n_trigrams":        len(self._vomm._tri),
            "total_tokens":      self._vomm._total_tokens,
            "train_perplexity_median": round(mu, 3),
            "train_perplexity_p25":   round(q25, 3),
            "train_perplexity_p75":   round(q75, 3),
            "calibration_mu":    round(mu, 3),
            "calibration_sigma": round(sigma, 3),
        }

        if output_dir:
            cp = Path(output_dir)
            cp.mkdir(parents=True, exist_ok=True)
            self._vomm.save(cp / "vomm.pkl")
            with open(cp / "calibration.json", "w") as f:
                json.dump({"mu": mu, "sigma": sigma}, f)
            with open(cp / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info("CGAM saved to %s", output_dir)

        logger.info(
            "CGAM trained: vocab=%d  trigrams=%d  median_perplexity=%.1f",
            len(self._vomm._vocab), len(self._vomm._tri), mu,
        )
        return metrics

    def load(self, checkpoint_dir: str | None = None) -> bool:
        """Load trained CGAM from disk. Returns True on success."""
        cp = Path(checkpoint_dir) if checkpoint_dir else CHECKPOINT_DIR
        vomm = VOMM.load(cp / "vomm.pkl")
        if vomm is None:
            return False
        cal_path = cp / "calibration.json"
        if not cal_path.exists():
            return False
        try:
            with open(cal_path) as f:
                cal = json.load(f)
            self._vomm  = vomm
            self._calib = CalibrationParams(mu=cal["mu"], sigma=cal["sigma"])
            self._ready = True
            logger.info("CGAM loaded from %s (vocab=%d)", cp, len(vomm._vocab))
            return True
        except Exception as e:
            logger.warning("CGAM load failed: %s", e)
            return False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, source: str) -> GrammarAnomalyResult:
        """
        Compute grammar anomaly score for a Python source file.

        Args:
            source: Python source code string.

        Returns:
            GrammarAnomalyResult with perplexity, anomaly_score, and attribution.
        """
        if not self.ready:
            return GrammarAnomalyResult(
                perplexity=0.0, anomaly_score=0.0,
                is_anomalous=False, n_tokens=0,
            )

        seq = extract_node_sequence(source)
        if len(seq) < 3:
            return GrammarAnomalyResult(
                perplexity=0.0, anomaly_score=0.0,
                is_anomalous=False, n_tokens=len(seq),
            )

        perp = self._vomm.perplexity(seq)
        score = self._calib.score(perp) if self._calib else 0.5
        score = float(np.clip(score, 0.0, 1.0))

        # Identify most surprising n-grams (lowest individual log-probs)
        _, ngram_details = self._vomm.log_prob(seq)
        ngram_details.sort(key=lambda x: x[1])   # lowest log-prob first
        top_anomalous = [
            {"ngram": ng, "log_prob": round(lp, 4)}
            for ng, lp in ngram_details[:5]
        ]

        # Grammar coverage: fraction of seen trigrams in VOMM
        n_tri = max(len(seq) - 2, 1)
        seen = sum(
            1 for i in range(2, len(seq))
            if (seq[i-2], seq[i-1], seq[i]) in self._vomm._tri
        )
        coverage = round(seen / n_tri, 4)

        return GrammarAnomalyResult(
            perplexity=round(perp, 3),
            anomaly_score=round(score, 4),
            is_anomalous=score > self._threshold,
            top_anomalous_ngrams=top_anomalous,
            n_tokens=len(seq),
            grammar_coverage=coverage,
        )
