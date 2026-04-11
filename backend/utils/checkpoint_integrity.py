"""
checkpoint_integrity.py
=======================
SHA-256 integrity verification for pickle model checkpoints.

Usage
-----
1. Generate a manifest from known-good checkpoints (run once after training):

       python -m backend.utils.checkpoint_integrity generate

   This writes backend/checkpoints/manifest.json.

2. Every model loader calls verify_checkpoint(path) before pickle.load().
   - If no manifest exists: logs a WARNING (first-run or CI environment).
   - If the manifest exists but the file is missing: raises FileNotFoundError.
   - If the hash does not match: raises ValueError (tampered/corrupt file).
   - If the hash matches: returns silently.

Design note: pickle.load() on an attacker-controlled file is arbitrary code
execution. The manifest provides a last line of defence assuming the manifest
itself is committed to the repo (not generated in the same pipeline step that
trains the model).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Manifest lives alongside the checkpoints directory, one level up from any
# individual model folder, so a single file covers all models.
_DEFAULT_MANIFEST = Path(__file__).parent.parent / "checkpoints" / "manifest.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checkpoint(
    path: str | Path,
    manifest_path: str | Path | None = None,
) -> None:
    """
    Verify the SHA-256 hash of *path* against the stored manifest.

    Parameters
    ----------
    path:
        Absolute or relative path to the checkpoint file (.pkl / .pt / .json).
    manifest_path:
        Override the default manifest location (useful in tests).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the manifest records a hash for *path* that does not match the
        actual file content — indicates tampering or corruption.
    """
    path = Path(path).resolve()
    manifest_path = Path(manifest_path or _DEFAULT_MANIFEST)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if not manifest_path.exists():
        logger.warning(
            "No checkpoint manifest found at %s — skipping integrity check. "
            "Run `python -m backend.utils.checkpoint_integrity generate` after "
            "training to create the manifest.",
            manifest_path,
        )
        return

    try:
        manifest: dict[str, str] = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not read checkpoint manifest (%s) — skipping integrity check.", exc)
        return

    # Keys are stored as relative paths from the manifest's parent directory.
    manifest_dir = manifest_path.parent
    try:
        rel_key = str(path.relative_to(manifest_dir.resolve()))
    except ValueError:
        # path is outside the manifest directory — use the full resolved path as key
        rel_key = str(path)

    if rel_key not in manifest:
        logger.warning(
            "Checkpoint %s is not listed in the integrity manifest — "
            "add it by re-running `python -m backend.utils.checkpoint_integrity generate`.",
            rel_key,
        )
        return

    expected = manifest[rel_key]
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(
            f"Checkpoint integrity check FAILED for {path}.\n"
            f"  Expected SHA-256: {expected}\n"
            f"  Actual   SHA-256: {actual}\n"
            "The file may be corrupted or tampered with. "
            "Re-train or restore the checkpoint from a trusted source."
        )

    logger.debug("Checkpoint OK: %s (%s...)", path.name, actual[:12])


def generate_manifest(
    checkpoints_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
    extensions: tuple[str, ...] = (".pkl", ".pt", ".json"),
) -> dict[str, str]:
    """
    Scan *checkpoints_dir* for model files and write a manifest.json with their
    SHA-256 hashes.  Returns the manifest dict.
    """
    checkpoints_dir = Path(checkpoints_dir or _DEFAULT_MANIFEST.parent).resolve()
    manifest_path = Path(manifest_path or _DEFAULT_MANIFEST)

    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {checkpoints_dir}")

    manifest: dict[str, str] = {}
    for fpath in sorted(checkpoints_dir.rglob("*")):
        if fpath.suffix in extensions and fpath.is_file():
            # Skip the manifest itself if it already exists
            if fpath.resolve() == manifest_path.resolve():
                continue
            rel = str(fpath.relative_to(checkpoints_dir))
            manifest[rel] = _sha256(fpath)
            logger.info("  Hashed: %s", rel)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    logger.info("Manifest written to %s (%d entries)", manifest_path, len(manifest))
    return manifest


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    command = sys.argv[1] if len(sys.argv) > 1 else "help"

    if command == "generate":
        checkpoints_dir = sys.argv[2] if len(sys.argv) > 2 else None
        result = generate_manifest(checkpoints_dir)
        print(f"Manifest generated with {len(result)} entries.")
    elif command == "verify":
        if len(sys.argv) < 3:
            print("Usage: checkpoint_integrity.py verify <path>")
            sys.exit(1)
        try:
            verify_checkpoint(sys.argv[2])
            print("OK")
        except (FileNotFoundError, ValueError) as e:
            print(f"FAIL: {e}")
            sys.exit(1)
    else:
        print(__doc__)
