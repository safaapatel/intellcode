"""
Experiment Tracker
===================
Lightweight experiment tracking with two backends:

  1. Weights & Biases (wandb) — if installed and WANDB_API_KEY is set.
  2. JSON file store       — always available as fallback.

Usage in training scripts:
    from training.experiment_tracker import ExperimentTracker

    tracker = ExperimentTracker(
        project="intellicode",
        run_name="complexity-xgb-v2",
        config={"n_estimators": 500, "max_depth": 6, ...},
        tags=["complexity", "xgboost"],
    )
    tracker.log({"fold": 1, "cv_rmse": 8.3, "cv_r2": 0.88})
    tracker.log({"test_rmse": 7.9, "test_r2": 0.91})
    tracker.log_artifact("checkpoints/complexity/model.pkl", artifact_type="model")
    tracker.finish()

All runs are also written to:
    experiments/<project>/<run_name>_<timestamp>.json
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class ExperimentTracker:
    """
    Unified experiment tracking interface.

    Transparently uses W&B when available, otherwise writes to a local
    JSON file under experiments/<project>/.

    All logged metrics are accumulated in-memory and flushed on finish(),
    so individual log() calls are cheap.
    """

    def __init__(
        self,
        project: str = "intellicode",
        run_name: Optional[str] = None,
        config: Optional[dict] = None,
        tags: Optional[list[str]] = None,
        log_dir: str = "experiments",
        use_wandb: bool = True,
    ):
        self._project   = project
        self._run_name  = run_name or f"run_{datetime.now():%Y%m%d_%H%M%S}"
        self._config    = config or {}
        self._tags      = tags or []
        self._log_dir   = Path(log_dir)
        self._history:  list[dict] = []
        self._artifacts:list[dict] = []
        self._start_ts  = time.time()
        self._wandb_run = None
        self._git_sha   = self._get_git_sha()

        # ── Try W&B ──────────────────────────────────────────────────────────
        self._use_wandb = False
        if use_wandb:
            try:
                import wandb
                if os.environ.get("WANDB_API_KEY") or wandb.api.api_key:
                    self._wandb_run = wandb.init(
                        project=project,
                        name=self._run_name,
                        config=config or {},
                        tags=tags or [],
                        reinit=True,
                    )
                    self._use_wandb = True
                    print(f"[tracker] W&B run: {self._wandb_run.url}")
                else:
                    print("[tracker] WANDB_API_KEY not set — using file-based tracking.")
            except Exception as e:
                print(f"[tracker] W&B unavailable ({e}) — using file-based tracking.")

        if not self._use_wandb:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            print(f"[tracker] Logging to {self._log_dir / self._project}/")

    @staticmethod
    def _get_git_sha() -> str:
        """Return the current git commit SHA (short, 8 chars), or 'unknown'."""
        import subprocess
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short=8", "HEAD"],
                capture_output=True, text=True, timeout=5,
            )
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"

    # ── Public API ────────────────────────────────────────────────────────────

    def log(self, metrics: dict[str, Any], step: Optional[int] = None) -> None:
        """
        Log a dict of scalar metrics.

        Args:
            metrics: {metric_name: value, ...}
            step:    Optional global step (auto-incremented if None).
        """
        entry = {"_step": step if step is not None else len(self._history), **metrics}
        self._history.append(entry)

        if self._use_wandb and self._wandb_run is not None:
            try:
                self._wandb_run.log(metrics, step=step)
            except Exception:
                pass

    def log_summary(self, summary: dict[str, Any]) -> None:
        """
        Log final summary metrics (e.g., best test AUC).
        These appear as run-level summary in W&B, and under "summary" in JSON.
        """
        self._summary = summary
        if self._use_wandb and self._wandb_run is not None:
            try:
                for k, v in summary.items():
                    self._wandb_run.summary[k] = v
            except Exception:
                pass

    def log_artifact(
        self,
        path: str,
        artifact_type: str = "model",
        name: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Register a file artifact (model checkpoint, dataset, etc.).

        Args:
            path:          Local path to the file.
            artifact_type: "model" | "dataset" | "evaluation"
            name:          Artifact name (defaults to filename stem).
            metadata:      Extra key-value metadata.
        """
        p = Path(path)
        entry = {
            "path":          str(p.resolve()),
            "name":          name or p.stem,
            "type":          artifact_type,
            "size_bytes":    p.stat().st_size if p.exists() else None,
            "metadata":      metadata or {},
        }
        self._artifacts.append(entry)

        if self._use_wandb and self._wandb_run is not None:
            try:
                import wandb
                artifact = wandb.Artifact(name or p.stem, type=artifact_type,
                                          metadata=metadata or {})
                if p.exists():
                    artifact.add_file(str(p))
                self._wandb_run.log_artifact(artifact)
            except Exception:
                pass

    def finish(self) -> str:
        """
        Flush all data to disk and close W&B run (if active).

        Returns:
            Path to the saved JSON run file.
        """
        elapsed = time.time() - self._start_ts

        run_data = {
            "project":       self._project,
            "run_name":      self._run_name,
            "git_sha":       self._git_sha,
            "config":        self._config,
            "tags":          self._tags,
            "summary":       getattr(self, "_summary", {}),
            "history":       self._history,
            "artifacts":     self._artifacts,
            "elapsed_s":     round(elapsed, 2),
            "finished_at":   datetime.now().isoformat(),
        }

        # ── Write JSON ────────────────────────────────────────────────────────
        proj_dir = self._log_dir / self._project
        proj_dir.mkdir(parents=True, exist_ok=True)
        json_path = proj_dir / f"{self._run_name}.json"
        with open(json_path, "w") as f:
            json.dump(run_data, f, indent=2, default=str)
        print(f"[tracker] Run saved -> {json_path}  (elapsed: {elapsed:.1f}s)")

        # ── Update runs index ─────────────────────────────────────────────────
        _update_runs_index(proj_dir, run_data)

        if self._use_wandb and self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except Exception:
                pass

        return str(json_path)

    # ── Context manager support ───────────────────────────────────────────────

    def __enter__(self) -> "ExperimentTracker":
        return self

    def __exit__(self, *_) -> None:
        self.finish()


# ── Runs index ────────────────────────────────────────────────────────────────

def _update_runs_index(proj_dir: Path, run_data: dict) -> None:
    """Maintain a lightweight runs.jsonl index for quick inspection."""
    index_path = proj_dir / "runs.jsonl"
    summary_entry = {
        "run_name":    run_data["run_name"],
        "git_sha":     run_data.get("git_sha", "unknown"),
        "finished_at": run_data["finished_at"],
        "elapsed_s":   run_data["elapsed_s"],
        "tags":        run_data["tags"],
        "summary":     run_data["summary"],
        # Top-level config (first 10 keys only for brevity)
        "config":      dict(list(run_data["config"].items())[:10]),
    }
    with open(index_path, "a") as f:
        f.write(json.dumps(summary_entry, default=str) + "\n")


# ── Convenience: load and compare runs ───────────────────────────────────────

def load_runs(project: str, log_dir: str = "experiments") -> list[dict]:
    """Load all run summaries for a project from the local index."""
    index_path = Path(log_dir) / project / "runs.jsonl"
    if not index_path.exists():
        return []
    runs = []
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    return runs


def compare_runs(
    project: str,
    metric: str,
    log_dir: str = "experiments",
    higher_is_better: bool = True,
) -> None:
    """Print a sorted leaderboard of runs for a given metric."""
    runs = load_runs(project, log_dir)
    if not runs:
        print(f"No runs found for project '{project}'")
        return

    valid = [(r["run_name"], r["summary"].get(metric)) for r in runs
             if metric in r.get("summary", {})]
    if not valid:
        print(f"Metric '{metric}' not found in any run summaries.")
        return

    valid.sort(key=lambda x: x[1], reverse=higher_is_better)
    direction = "(higher better)" if higher_is_better else "(lower better)"
    print(f"\nLeaderboard -- {project} | {metric} {direction}")
    print("-" * 60)
    for i, (name, val) in enumerate(valid, 1):
        print(f"  {i:2d}. {name:40s}  {val:.5f}")
