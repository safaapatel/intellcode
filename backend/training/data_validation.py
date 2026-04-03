"""
Data Validation & Manifest
============================
Validates JSONL datasets before training and maintains a data manifest
(SHA-256 hashes + stats) for reproducibility.

Usage:
    from training.data_validation import validate_dataset, update_manifest, load_manifest

    # Validate before training (raises DataValidationError on schema violation)
    stats = validate_dataset("data/complexity_dataset.jsonl", schema="complexity")
    stats = validate_dataset("data/security_dataset.jsonl",  schema="security")
    stats = validate_dataset("data/bug_dataset.jsonl",       schema="bug")

    # Update manifest after successful training run
    update_manifest("data/complexity_dataset.jsonl", schema="complexity", metrics=train_metrics)

    # Check that a dataset hasn't changed since last training
    manifest = load_manifest()
    is_fresh = manifest.check_integrity("data/complexity_dataset.jsonl")
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

# ── Schema definitions ────────────────────────────────────────────────────────

Schema = Literal["complexity", "security", "bug", "pattern"]

# Required keys and their expected types per dataset schema
_REQUIRED_FIELDS: dict[Schema, dict[str, type]] = {
    "complexity": {
        "features": list,
        "target":   float,
    },
    "security": {
        # CVEFixes records carry "source" (raw code); older records carry "tokens".
        # Only "label" is strictly required; source/tokens checked separately below.
        "label": int,
    },
    "bug": {
        "label":           int,
        "static_features": list,
    },
    "pattern": {
        "code":  str,
        "label": str,
    },
}

# Security records must have EITHER "tokens" (list) OR "source" (str).
_SECURITY_SOURCE_FIELDS = {"tokens": list, "source": str}

_EXPECTED_FEATURE_DIMS: dict[Schema, int] = {
    "complexity": 16,
    "bug":        16,   # static features only; JIT appended at train time
}

_VALID_LABELS: dict[Schema, set] = {
    "security": {0, 1},
    "bug":      {0, 1},
    "pattern":  {"clean", "code_smell", "anti_pattern", "style_violation"},
}

_TARGET_RANGE: dict[Schema, tuple[float, float]] = {
    "complexity": (0.0, 100.0),
}


# ── Exception ─────────────────────────────────────────────────────────────────

class DataValidationError(Exception):
    pass


# ── Validation ────────────────────────────────────────────────────────────────

@dataclass
class ValidationStats:
    schema:         str
    path:           str
    n_samples:      int
    n_invalid:      int
    n_skipped:      int
    sha256:         str
    class_balance:  Optional[dict]  # for classification tasks
    target_range:   Optional[dict]  # for regression tasks
    feature_shape:  Optional[tuple]
    validated_at:   str


def validate_dataset(
    path: str,
    schema: Schema,
    strict: bool = False,
    max_invalid_fraction: float = 0.05,
) -> ValidationStats:
    """
    Validate a JSONL dataset against the expected schema.

    Args:
        path:                  Path to the .jsonl file.
        schema:                One of "complexity", "security", "bug", "pattern".
        strict:                If True, raise on any validation error.
        max_invalid_fraction:  Max fraction of malformed records before raising.

    Returns:
        ValidationStats dataclass with dataset statistics.

    Raises:
        DataValidationError: If more than max_invalid_fraction records are invalid
                             or required fields are entirely missing.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    required = _REQUIRED_FIELDS.get(schema, {})
    expected_dim = _EXPECTED_FEATURE_DIMS.get(schema)
    valid_labels = _VALID_LABELS.get(schema)
    target_range = _TARGET_RANGE.get(schema)

    n_total = n_invalid = n_skipped = 0
    labels: list = []
    targets: list = []
    feature_dims: set = set()
    errors: list[str] = []

    with open(p, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                n_skipped += 1
                continue
            n_total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                msg = f"line {lineno}: JSON parse error — {e}"
                errors.append(msg)
                n_invalid += 1
                if strict:
                    raise DataValidationError(msg)
                continue

            # ── Required fields ───────────────────────────────────────────────
            for field, expected_type in required.items():
                if field not in rec:
                    msg = f"line {lineno}: missing required field '{field}'"
                    errors.append(msg)
                    n_invalid += 1
                    if strict:
                        raise DataValidationError(msg)
                    break
                val = rec[field]
                # Coerce numeric types (int/float are interchangeable in JSON)
                if expected_type is float and isinstance(val, int):
                    val = float(val)
                if not isinstance(val, expected_type):
                    msg = (f"line {lineno}: field '{field}' expected {expected_type.__name__}, "
                           f"got {type(val).__name__}")
                    errors.append(msg)
                    n_invalid += 1
                    if strict:
                        raise DataValidationError(msg)
                    break
            else:
                # Security: check that at least one of "tokens" or "source" is present
                if schema == "security":
                    has_tokens = isinstance(rec.get("tokens"), list)
                    has_source = isinstance(rec.get("source"), str)
                    if not has_tokens and not has_source:
                        msg = (f"line {lineno}: security record must have either "
                               f"'tokens' (list) or 'source' (str)")
                        errors.append(msg)
                        n_invalid += 1
                        if strict:
                            raise DataValidationError(msg)
                        continue

                # All required fields present — run semantic checks
                if valid_labels and "label" in rec:
                    lbl = rec["label"]
                    if schema == "pattern":
                        lbl = rec.get("label", "")
                    if lbl not in valid_labels:
                        msg = f"line {lineno}: invalid label '{lbl}' (expected {valid_labels})"
                        errors.append(msg)
                        n_invalid += 1
                        if strict:
                            raise DataValidationError(msg)
                        continue
                    labels.append(lbl)

                if target_range and "target" in rec:
                    t = float(rec["target"])
                    lo, hi = target_range
                    if not (lo <= t <= hi):
                        msg = f"line {lineno}: target {t} outside [{lo}, {hi}]"
                        errors.append(msg)
                        n_invalid += 1
                        if strict:
                            raise DataValidationError(msg)
                        continue
                    targets.append(t)

                feat_key = "features" if schema == "complexity" else "static_features"
                if feat_key in rec:
                    dim = len(rec[feat_key])
                    feature_dims.add(dim)
                    if expected_dim and dim != expected_dim:
                        msg = (f"line {lineno}: feature vector length {dim} "
                               f"≠ expected {expected_dim}")
                        errors.append(msg)
                        n_invalid += 1
                        if strict:
                            raise DataValidationError(msg)
                        continue

    # ── Summary checks ────────────────────────────────────────────────────────
    if n_total == 0:
        raise DataValidationError(f"Dataset is empty: {path}")

    invalid_frac = n_invalid / n_total
    if invalid_frac > max_invalid_fraction:
        raise DataValidationError(
            f"Too many invalid records: {n_invalid}/{n_total} ({invalid_frac:.1%}). "
            f"Threshold is {max_invalid_fraction:.0%}.\n"
            f"First errors:\n" + "\n".join(errors[:5])
        )

    if errors:
        print(f"[validate] {path}: {n_invalid} invalid records "
              f"({invalid_frac:.1%}) — proceeding (below threshold).")

    # ── Class balance / target stats ──────────────────────────────────────────
    class_balance = None
    if labels:
        from collections import Counter
        counts = Counter(labels)
        class_balance = {str(k): v for k, v in counts.items()}
        n_samples = n_total - n_invalid
        print(f"[validate] {Path(path).name}: {n_samples} valid samples | "
              f"class balance: {class_balance}")

    target_stats = None
    if targets:
        import statistics
        target_stats = {
            "min":    min(targets),
            "max":    max(targets),
            "mean":   statistics.mean(targets),
            "median": statistics.median(targets),
            "stdev":  statistics.stdev(targets) if len(targets) > 1 else 0.0,
        }
        print(f"[validate] {Path(path).name}: targets  min={target_stats['min']:.2f}  "
              f"max={target_stats['max']:.2f}  mean={target_stats['mean']:.2f}")

    sha = _sha256(p)
    stats = ValidationStats(
        schema=schema,
        path=str(p.resolve()),
        n_samples=n_total - n_invalid,
        n_invalid=n_invalid,
        n_skipped=n_skipped,
        sha256=sha,
        class_balance=class_balance,
        target_range=target_stats,
        feature_shape=(n_total - n_invalid, list(feature_dims)[0]) if feature_dims else None,
        validated_at=datetime.now().isoformat(),
    )
    return stats


# ── Manifest ──────────────────────────────────────────────────────────────────

_MANIFEST_PATH = Path("data/data_manifest.json")


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class ManifestEntry:
    path:         str
    schema:       str
    sha256:       str
    n_samples:    int
    feature_shape: Optional[tuple]
    class_balance: Optional[dict]
    target_range:  Optional[dict]
    last_trained:  Optional[str]   # ISO datetime of last training run
    train_metrics: Optional[dict]  # metrics from last training run
    updated_at:    str


class DataManifest:
    """
    Tracks dataset integrity and training provenance.
    Stored as a JSON file at data/data_manifest.json.
    """

    def __init__(self, path: Path = _MANIFEST_PATH):
        self._path    = path
        self._entries: dict[str, ManifestEntry] = {}
        if path.exists():
            self._load()

    def _load(self) -> None:
        with open(self._path) as f:
            raw = json.load(f)
        for k, v in raw.items():
            self._entries[k] = ManifestEntry(**v)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(
                {k: asdict(v) for k, v in self._entries.items()},
                f, indent=2, default=str,
            )

    def update(self, stats: ValidationStats, train_metrics: Optional[dict] = None) -> None:
        """Register or update a dataset entry from ValidationStats."""
        key = str(Path(stats.path).resolve())
        existing = self._entries.get(key)
        self._entries[key] = ManifestEntry(
            path=stats.path,
            schema=stats.schema,
            sha256=stats.sha256,
            n_samples=stats.n_samples,
            feature_shape=stats.feature_shape,
            class_balance=stats.class_balance,
            target_range=stats.target_range,
            last_trained=datetime.now().isoformat() if train_metrics else (
                existing.last_trained if existing else None
            ),
            train_metrics=train_metrics or (existing.train_metrics if existing else None),
            updated_at=datetime.now().isoformat(),
        )
        self._save()
        print(f"[manifest] Updated entry for {Path(stats.path).name}")

    def check_integrity(self, path: str) -> bool:
        """
        Return True if the dataset SHA matches the manifest.
        False means the dataset was modified after the last recorded training run.
        """
        key = str(Path(path).resolve())
        entry = self._entries.get(key)
        if entry is None:
            print(f"[manifest] No entry for {path} — run validate_dataset() first.")
            return False
        current_sha = _sha256(Path(path))
        if current_sha != entry.sha256:
            print(f"[manifest] INTEGRITY MISMATCH: {path}\n"
                  f"  Recorded: {entry.sha256}\n"
                  f"  Current:  {current_sha}")
            return False
        return True

    def print_summary(self) -> None:
        print(f"\nData Manifest — {len(self._entries)} dataset(s):")
        print("-" * 70)
        for entry in self._entries.values():
            print(f"  {Path(entry.path).name}")
            print(f"    Schema:    {entry.schema}")
            print(f"    Samples:   {entry.n_samples}")
            print(f"    SHA-256:   {entry.sha256[:16]}...")
            print(f"    Trained:   {entry.last_trained or 'never'}")
            if entry.train_metrics:
                print(f"    Metrics:   {entry.train_metrics}")
            print()


def load_manifest(path: Path = _MANIFEST_PATH) -> DataManifest:
    return DataManifest(path)


def update_manifest(
    dataset_path: str,
    schema: Schema,
    train_metrics: Optional[dict] = None,
    manifest_path: Path = _MANIFEST_PATH,
) -> ValidationStats:
    """
    Convenience function: validate dataset, update manifest, return stats.
    Intended to be called at the end of each training script.
    """
    stats = validate_dataset(dataset_path, schema)
    manifest = DataManifest(manifest_path)
    manifest.update(stats, train_metrics=train_metrics)
    return stats


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate IntelliCode datasets")
    parser.add_argument("--data", required=True, help="Path to JSONL dataset")
    parser.add_argument(
        "--schema", required=True,
        choices=["complexity", "security", "bug", "pattern"],
    )
    parser.add_argument("--strict", action="store_true",
                        help="Fail on first invalid record")
    parser.add_argument("--update-manifest", action="store_true",
                        help="Write results to data/data_manifest.json")
    args = parser.parse_args()

    try:
        stats = validate_dataset(args.data, args.schema, strict=args.strict)
        print(f"\nValidation passed: {stats.n_samples} valid samples "
              f"({stats.n_invalid} invalid, {stats.n_skipped} skipped)")

        if args.update_manifest:
            manifest = DataManifest()
            manifest.update(stats)

    except DataValidationError as e:
        print(f"Validation FAILED: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
