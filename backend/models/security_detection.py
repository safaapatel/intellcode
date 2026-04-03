"""
Security Detection Model
Ensemble of:
  1. Random Forest — trained on structured AST/metric features
  2. 1D CNN       — trained on token ID sequences

The ensemble combines both via a learned weighted average.

Training data: labeled vulnerable/clean code from open-source projects
  and dedicated security datasets (e.g. Devign, D2A, CodeXGLUE).
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from features.ast_extractor import ASTExtractor, tokenize_code, tokens_to_ids
from features.code_metrics import compute_all_metrics, metrics_to_feature_vector
from features.security_patterns import scan_security_patterns, SecurityFinding


# ---------------------------------------------------------------------------
# Vulnerability categories the model can detect
# ---------------------------------------------------------------------------

VULN_TYPES = [
    "clean",
    "sql_injection",
    "command_injection",
    "path_traversal",
    "hardcoded_secret",
    "weak_cryptography",
    "insecure_deserialization",
    "code_injection",
    "xxe",
]

SEVERITY_MAP = {
    "sql_injection": "critical",
    "command_injection": "critical",
    "hardcoded_secret": "critical",
    "insecure_deserialization": "critical",
    "path_traversal": "high",
    "weak_cryptography": "high",
    "code_injection": "high",
    "xxe": "medium",
}


@dataclass
class VulnerabilityPrediction:
    vuln_type: str
    severity: str
    confidence: float
    lineno: int
    title: str
    description: str
    snippet: str
    cwe: str = ""
    source: str = "ensemble"     # "pattern_scan" | "rf" | "cnn" | "ensemble"


# ---------------------------------------------------------------------------
# Feature extraction for the structured (RF) branch
# ---------------------------------------------------------------------------

_DANGEROUS_IMPORTS = {
    "os", "subprocess", "pickle", "marshal", "shelve", "yaml",
    "xml", "lxml", "minidom", "exec", "eval",
}

_DANGEROUS_CALLS = {
    "execute", "executemany", "system", "popen", "loads", "load",
    "eval", "exec", "unsafe_load",
}

# Extended Python-specific security signals (v2 feature set)
_INJECTION_APIS = {
    # SQL
    "execute", "executemany", "executescript", "raw", "extra", "RawSQL",
    # OS command
    "system", "popen", "Popen", "call", "run", "check_output", "check_call",
    "spawn", "spawnl", "spawnle", "spawnlp", "spawnv", "spawnve",
    # Code execution
    "eval", "exec", "compile", "execfile", "__import__",
    # Deserialization
    "loads", "load", "unsafe_load", "Unpickler",
    # Template injection
    "render_string", "render", "Template",
}

_CRYPTO_WEAK = {"md5", "sha1", "des", "rc4", "blowfish", "arcfour"}

_SECRET_PATTERNS = [
    "password", "passwd", "secret", "api_key", "apikey", "token",
    "private_key", "access_key", "auth_token", "credential",
]

_NETWORK_APIS = {"urllib", "requests", "httpx", "aiohttp", "socket", "ssl"}

_TAINT_SOURCES = {
    "input", "request", "environ", "argv", "stdin",
    "GET", "POST", "FILES", "COOKIES", "META",
}


def _count_format_string_injections(source: str) -> int:
    """Count f-strings and %-format patterns that may feed user input into queries."""
    import re
    # f-string with variable: f"... {var} ..."
    fstring_count = len(re.findall(r'f["\'].*?\{[^}]+\}.*?["\']', source))
    # % formatting: "..." % var
    percent_count = len(re.findall(r'["\'].*?%[sd].*?["\'].*?%', source))
    return fstring_count + percent_count


def _count_taint_sources(ast_feats: dict) -> int:
    """Count calls to known taint sources (user-controlled input)."""
    calls = ast_feats.get("calls", [])
    return sum(1 for c in calls if c.get("name") in _TAINT_SOURCES)


def _count_weak_crypto(source: str) -> int:
    import re
    count = 0
    src_lower = source.lower()
    for weak in _CRYPTO_WEAK:
        if weak in src_lower:
            count += 1
    return count


def _count_hardcoded_secrets(ast_feats: dict) -> int:
    """Count assignments where LHS name suggests a secret and RHS is a string literal."""
    count = 0
    for lit in ast_feats.get("string_literals", []):
        val = lit.get("value", "")
        if len(val) >= 8 and any(p in val.lower() for p in _SECRET_PATTERNS):
            count += 1
    return count


def _count_network_usage(ast_feats: dict) -> int:
    imports = [imp.split(".")[0] for imp in ast_feats.get("imports", [])]
    return sum(1 for imp in imports if imp in _NETWORK_APIS)


def _build_rf_feature_vector(source: str) -> np.ndarray:
    """
    Build a fixed-length numeric feature vector for the Random Forest.
    v2: Extended with Python-specific security signals.

    Feature groups (24 total):
      [0-14]  Static code metrics (15-dim from metrics_to_feature_vector)
      [15-23] Security-specific features (9 original + 6 new = extended)

    New features added in v2:
      - format string injection count (f-strings / % format feeding queries)
      - taint source count (calls to input/request/environ/etc.)
      - weak crypto usage count
      - hardcoded secret string count
      - network API usage count
      - injection API call density (injection_calls / total_calls)
    """
    ast_feats = ASTExtractor().extract(source)
    metric_vec = metrics_to_feature_vector(compute_all_metrics(source))

    # Original security features
    imports = [imp.split(".")[0] for imp in ast_feats.get("imports", [])]
    dangerous_import_count = sum(1 for imp in imports if imp in _DANGEROUS_IMPORTS)

    calls = ast_feats.get("calls", [])
    dangerous_call_count = sum(1 for c in calls if c.get("name") in _DANGEROUS_CALLS)

    string_literals = ast_feats.get("string_literals", [])
    long_string_count = sum(1 for s in string_literals if len(s.get("value", "")) > 20)

    scan = scan_security_patterns(source)
    scan_critical = len(scan.critical)
    scan_high = len(scan.high)
    scan_total = len(scan.findings)

    n_try_blocks = float(ast_feats.get("n_try_blocks", 0))
    n_string_literals = float(ast_feats.get("n_string_literals", 0))
    n_calls = float(ast_feats.get("n_calls", 0))

    # New v2 features
    injection_api_count = sum(1 for c in calls if c.get("name") in _INJECTION_APIS)
    injection_density = injection_api_count / max(1.0, n_calls)
    format_inject_count = float(_count_format_string_injections(source))
    taint_source_count = float(_count_taint_sources(ast_feats))
    weak_crypto_count = float(_count_weak_crypto(source))
    hardcoded_secret_count = float(_count_hardcoded_secrets(ast_feats))
    network_usage_count = float(_count_network_usage(ast_feats))

    security_feats = [
        float(dangerous_import_count),
        float(dangerous_call_count),
        float(long_string_count),
        float(scan_critical),
        float(scan_high),
        float(scan_total),
        n_try_blocks,
        n_string_literals,
        n_calls,
        # v2 additions
        float(injection_api_count),
        injection_density,
        format_inject_count,
        taint_source_count,
        weak_crypto_count,
        hardcoded_secret_count,
        network_usage_count,
    ]

    return np.array(metric_vec + security_feats, dtype=np.float32)


# Feature names for interpretability (24-dim)
RF_FEATURE_NAMES = [
    "cc", "max_func_cc", "avg_func_cc", "sloc", "comments", "blank",
    "halstead_vol", "halstead_diff", "halstead_effort", "halstead_bugs",
    "n_long_funcs", "n_complex_funcs", "max_line", "avg_line", "n_over80",
    "dangerous_imports", "dangerous_calls", "long_strings",
    "scan_critical", "scan_high", "scan_total",
    "n_try_blocks", "n_string_literals", "n_calls",
    "injection_api_count", "injection_density", "format_inject",
    "taint_sources", "weak_crypto", "hardcoded_secrets", "network_usage",
]


# ---------------------------------------------------------------------------
# Random Forest model wrapper
# ---------------------------------------------------------------------------

class RandomForestSecurityModel:
    """
    Scikit-learn RandomForestClassifier wrapper.
    Labels: 0 = clean, 1 = vulnerable (binary)
    """

    def __init__(self):
        self._clf = None

    def fit(self, X: np.ndarray, y: np.ndarray, **rf_kwargs):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.calibration import CalibratedClassifierCV

        base = RandomForestClassifier(
            n_estimators=rf_kwargs.get("n_estimators", 200),
            max_depth=rf_kwargs.get("max_depth", None),
            min_samples_split=rf_kwargs.get("min_samples_split", 5),
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        # Calibrate probabilities with Platt scaling
        self._clf = CalibratedClassifierCV(base, cv=5, method="sigmoid")
        self._clf.fit(X, y)

    def predict_proba(self, x: np.ndarray) -> float:
        """Return probability of class 1 (vulnerable)."""
        if self._clf is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        return float(self._clf.predict_proba(x.reshape(1, -1))[0, 1])

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._clf, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self._clf = pickle.load(f)


# ---------------------------------------------------------------------------
# 1D CNN model
# ---------------------------------------------------------------------------

CNN_VOCAB_SIZE = 10_000
CNN_EMBED_DIM = 128
CNN_SEQ_LEN = 512
CNN_NUM_FILTERS = 256
CNN_KERNEL_SIZES = [3, 5, 7]


def _build_cnn_model(num_classes: int = 2):
    """
    Construct a 1D CNN for token sequence classification.
    Architecture:
        Embedding → [Conv1d → ReLU → MaxPool] × 3 → Concat → Dropout → Linear
    """
    import torch.nn as nn

    class TextCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(CNN_VOCAB_SIZE, CNN_EMBED_DIM, padding_idx=0)
            self.convs = nn.ModuleList([
                nn.Conv1d(CNN_EMBED_DIM, CNN_NUM_FILTERS, k)
                for k in CNN_KERNEL_SIZES
            ])
            self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(CNN_NUM_FILTERS * len(CNN_KERNEL_SIZES), num_classes)

        def forward(self, x):
            import torch
            # x: (batch, seq_len)
            emb = self.embedding(x).transpose(1, 2)  # (batch, embed_dim, seq_len)
            pooled = []
            for conv in self.convs:
                c = torch.relu(conv(emb))             # (batch, filters, seq_len-k+1)
                p = c.max(dim=2).values               # (batch, filters)
                pooled.append(p)
            out = torch.cat(pooled, dim=1)            # (batch, filters*n_kernels)
            out = self.dropout(out)
            return self.fc(out)

    return TextCNN()


class CNNSecurityModel:
    """1D CNN for token-sequence based vulnerability detection."""

    def __init__(self):
        self._model = None
        self._vocab: dict[str, int] = {}
        self._device = None

    def build(self, vocab: dict[str, int]):
        import torch
        self._vocab = vocab
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = _build_cnn_model(num_classes=2).to(self._device)

    def predict_proba(self, source: str) -> float:
        """Return probability of vulnerable (class 1)."""
        import torch

        if self._model is None:
            raise RuntimeError("Model not built. Call build() then train().")

        tokens = tokenize_code(source)
        ids = tokens_to_ids(tokens, self._vocab, CNN_SEQ_LEN)
        x = torch.tensor([ids], dtype=torch.long).to(self._device)

        self._model.eval()
        with torch.no_grad():
            logits = self._model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        return float(probs[1])

    def save(self, model_path: str, vocab_path: str):
        import torch
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), model_path)
        with open(vocab_path, "w") as f:
            json.dump(self._vocab, f)

    def load(self, model_path: str, vocab_path: str):
        import torch
        with open(vocab_path) as f:
            self._vocab = json.load(f)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = _build_cnn_model(num_classes=2).to(self._device)
        self._model.load_state_dict(
            torch.load(model_path, map_location=self._device, weights_only=True)
        )
        self._model.eval()

    def train_loop(
        self,
        X_ids: np.ndarray,     # (N, seq_len) int arrays
        y: np.ndarray,         # (N,) binary labels
        epochs: int = 10,
        batch_size: int = 64,
        lr: float = 1e-3,
    ):
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader

        X_t = torch.tensor(X_ids, dtype=torch.long)
        y_t = torch.tensor(y, dtype=torch.long)
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

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
            print(f"Epoch {epoch+1}/{epochs}  loss={total_loss/len(loader):.4f}")
        self._model.eval()


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class EnsembleSecurityModel:
    """
    Combines RandomForestSecurityModel and CNNSecurityModel via
    a weighted average of their vulnerability probabilities.

    rf_weight + cnn_weight should sum to 1.0.
    If only one component is available, it falls back gracefully.
    """

    def __init__(
        self,
        rf_weight: float = 0.55,
        cnn_weight: float = 0.45,
        checkpoint_dir: str = "checkpoints/security",
    ):
        self._rf_weight = rf_weight
        self._cnn_weight = cnn_weight
        self._checkpoint_dir = Path(checkpoint_dir)
        self._rf = RandomForestSecurityModel()
        self._cnn = CNNSecurityModel()
        self._rf_ready = False
        self._cnn_ready = False
        self._rf_threshold: float = 0.5   # loaded from metrics.json if available

        self._try_load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, source: str) -> list[VulnerabilityPrediction]:
        """
        Full prediction pipeline:
          1. Run the deterministic pattern scanner (high precision)
          2. Run ML models to catch anything the scanner missed
          3. Merge and deduplicate results
        """
        results: list[VulnerabilityPrediction] = []

        # --- Deterministic scanner (always runs) ---
        scan = scan_security_patterns(source)
        for finding in scan.findings:
            results.append(VulnerabilityPrediction(
                vuln_type=finding.vuln_type,
                severity=finding.severity,
                confidence=finding.confidence,
                lineno=finding.lineno,
                title=finding.title,
                description=finding.description,
                snippet=finding.snippet,
                cwe=finding.cwe,
                source="pattern_scan",
            ))

        # --- ML ensemble (adds extra confidence or new findings) ---
        if self._rf_ready or self._cnn_ready:
            vuln_prob = self._ensemble_proba(source)
            # Only report an additional ML finding if high confidence AND
            # no pattern-scan finding was already found
            if vuln_prob > self._rf_threshold and not results:
                results.append(VulnerabilityPrediction(
                    vuln_type="unknown_vulnerability",
                    severity="high",
                    confidence=vuln_prob,
                    lineno=0,
                    title="ML-detected Vulnerability Pattern",
                    description=(
                        f"Ensemble model detected a vulnerability pattern with "
                        f"{vuln_prob:.0%} confidence. Manual review recommended."
                    ),
                    snippet="",
                    source="ensemble",
                ))

        return sorted(results, key=lambda r: (
            {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(r.severity, 4),
            r.lineno,
        ))

    def vulnerability_score(self, source: str) -> float:
        """Return a single 0–1 probability of any vulnerability."""
        scan = scan_security_patterns(source)
        if scan.critical:
            return 0.95
        if scan.high:
            return 0.80
        if self._rf_ready or self._cnn_ready:
            return self._ensemble_proba(source)
        if scan.findings:
            return 0.50
        return 0.05

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensemble_proba(self, source: str) -> float:
        rf_p, cnn_p = 0.0, 0.0
        weight_sum = 0.0

        if self._rf_ready:
            try:
                feat = _build_rf_feature_vector(source)
                rf_p = self._rf.predict_proba(feat)
                weight_sum += self._rf_weight
            except Exception:
                pass

        if self._cnn_ready:
            try:
                cnn_p = self._cnn.predict_proba(source)
                weight_sum += self._cnn_weight
            except Exception:
                pass

        if weight_sum == 0:
            return 0.0
        return (rf_p * self._rf_weight + cnn_p * self._cnn_weight) / weight_sum

    def _try_load(self):
        """Silently load saved checkpoints if available."""
        rf_path = self._checkpoint_dir / "rf_model.pkl"
        cnn_model_path = self._checkpoint_dir / "cnn_model.pt"
        cnn_vocab_path = self._checkpoint_dir / "cnn_vocab.json"
        metrics_path = self._checkpoint_dir / "metrics.json"

        if rf_path.exists():
            try:
                self._rf.load(str(rf_path))
                self._rf_ready = True
            except Exception:
                pass

        if cnn_model_path.exists() and cnn_vocab_path.exists():
            try:
                self._cnn.load(str(cnn_model_path), str(cnn_vocab_path))
                self._cnn_ready = True
            except Exception:
                pass

        # Load optimal threshold saved during training
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    saved_metrics = json.load(f)
                self._rf_threshold = float(saved_metrics.get("rf_threshold", 0.5))
            except Exception:
                pass

    def save(self):
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if self._rf_ready:
            self._rf.save(str(self._checkpoint_dir / "rf_model.pkl"))
        if self._cnn_ready:
            self._cnn.save(
                str(self._checkpoint_dir / "cnn_model.pt"),
                str(self._checkpoint_dir / "cnn_vocab.json"),
            )
