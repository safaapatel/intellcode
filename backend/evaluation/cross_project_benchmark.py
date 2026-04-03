"""
Cross-Project Benchmark
========================
Leave-one-project-out (LOPO) evaluation protocol for all IntelliCode tasks.

WHY THIS MATTERS:
    Random file-level splits allow training and test sets to share files from
    the same repository. Files from the same codebase share naming conventions,
    complexity patterns, and style — inflating all metrics by 10-20 AUC points
    vs. honest cross-project evaluation (Zimmermann et al. 2009).

    This module implements the correct evaluation:
      For each project P in the dataset:
        Train on all projects EXCEPT P
        Test on P
        Report: mean and std over all held-out projects

Usage:
    from evaluation.cross_project_benchmark import CrossProjectBenchmark
    bench = CrossProjectBenchmark(task="bug")
    report = bench.run(records)
    bench.print_report(report)
    bench.save_report(report, "results/cross_project_bug.json")

References:
    Zimmermann et al. 2009 — "Cross-project defect prediction"
    Herbold et al. 2018 — "Systematic mapping on automated vulnerability detection"
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ProjectResult:
    """Evaluation result for one held-out project."""
    held_out_repo:  str
    n_train:        int
    n_test:         int
    n_train_repos:  int
    auc:            Optional[float]
    ap:             Optional[float]
    f1:             Optional[float]
    rmse:           Optional[float]
    spearman:       Optional[float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BenchmarkReport:
    """Full cross-project benchmark report."""
    task:            str
    protocol:        str
    n_projects:      int
    project_results: list[ProjectResult]
    mean_auc:        Optional[float]
    std_auc:         Optional[float]
    mean_ap:         Optional[float]
    std_ap:          Optional[float]
    mean_f1:         Optional[float]
    std_f1:          Optional[float]
    mean_rmse:       Optional[float]
    std_rmse:        Optional[float]
    mean_spearman:   Optional[float]
    std_spearman:    Optional[float]
    random_split_auc: Optional[float] = None
    degradation_auc:  Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "task":              self.task,
            "protocol":          self.protocol,
            "n_projects":        self.n_projects,
            "mean_auc":          self.mean_auc,
            "std_auc":           self.std_auc,
            "mean_ap":           self.mean_ap,
            "std_ap":            self.std_ap,
            "mean_f1":           self.mean_f1,
            "std_f1":            self.std_f1,
            "mean_rmse":         self.mean_rmse,
            "std_rmse":          self.std_rmse,
            "mean_spearman":     self.mean_spearman,
            "std_spearman":      self.std_spearman,
            "random_split_auc":  self.random_split_auc,
            "degradation_auc":   self.degradation_auc,
            "project_results":   [r.to_dict() for r in self.project_results],
        }


# ---------------------------------------------------------------------------
# Feature preparation per task
# ---------------------------------------------------------------------------

def _repo_from_record(r: dict) -> str:
    """
    Extract a repository-level group label for LOPO splits.

    Priority:
      1. Explicit "repo" field (ideal — set by dataset_builder when cloning)
      2. Parsed from "origin" field using the following heuristics:
           'cve_mining/flask/conf.py'    -> 'flask'        (CVE prefix stripped)
           'psf/requests/src/file.py'   -> 'psf/requests'  (owner/repo)
           'django/build.py'            -> 'django'         (single-level)
      3. Fallback "unknown"
    """
    if r.get("repo"):
        return r["repo"]
    origin = r.get("origin", "")
    if not origin:
        return "unknown"
    parts = origin.split("/")
    # Strip known data-collection prefixes so each real repo is its own group
    if parts[0] in ("cve_mining", "osv_mining", "github_advisory", "nvd_mining"):
        return parts[1] if len(parts) > 1 else "unknown"
    # owner/repo/path  -> 'owner/repo'
    if len(parts) >= 3:
        return f"{parts[0]}/{parts[1]}"
    # repo/file.py  -> 'repo'
    return parts[0]


def _prepare_complexity(records):
    """Use cognitive_complexity (features[1]) as target; exclude it from X.

    Mirrors the leakage-fix in training/train_complexity.py:
    the original MI target was a closed-form formula of features in X,
    giving trivial R²≈1.0.  Cognitive complexity is NOT algebraically
    derivable from the other 15 features.
    """
    COG_IDX = 1   # position of cognitive_complexity in the 16-dim vector
    X, y, repos = [], [], []
    for r in records:
        feat = r.get("features", [])
        if not feat or len(feat) < 16:
            continue
        cog = feat[COG_IDX]
        if cog < 0:
            continue
        x_vec = [feat[i] for i in range(16) if i != COG_IDX]  # 15 dims
        X.append(x_vec)
        y.append(float(cog))
        repos.append(_repo_from_record(r))
    return np.array(X, np.float32), np.array(y, np.float32), repos


def _prepare_bug(records):
    # Kamei JIT feature names — used when records store jit_features with these keys
    JIT_KAMEI = ["NS","ND","NF","Entropy","LA","LD","LT","FIX","NDEV","AGE","NUC","EXP","REXP","SEXP"]
    # Mapping from dataset_builder git_features keys → JIT proxy values
    GIT_MAP = {
        "code_churn":    "LA",   # lines added proxy
        "author_count":  "NDEV",
        "file_age_days": "AGE",
        "n_past_bugs":   "NUC",
        "commit_freq":   "EXP",
    }
    X, y, repos = [], [], []
    for r in records:
        s = r.get("static_features", [])
        if not s or len(s) not in (16, 17):
            continue
        s16 = s[:16]  # standardise to 16 dims

        # Prefer jit_features (Kamei keys); fall back to git_features (dataset_builder keys)
        jit = r.get("jit_features", {})
        if not jit:
            git = r.get("git_features", {})
            # Build a pseudo-JIT dict from available git fields
            jit = {v: float(git.get(k, 0)) for k, v in GIT_MAP.items()}

        X.append(s16 + [jit.get(k, 0.0) for k in JIT_KAMEI])
        y.append(int(r["label"]))
        repos.append(_repo_from_record(r))
    return np.array(X, np.float32), np.array(y, np.int32), repos


def _prepare_security(records):
    """
    Build security feature vector from all available record fields.

    Feature order (16-dim, matching train_security.py build_rf_features):
      0  n_calls          1  n_imports       2  n_assignments
      3  n_comparisons    4  n_augassigns    5  n_exceptions_raised
      6  n_try_blocks     7  n_returns       8  n_loops
      9  n_conditionals   10 max_depth       11 n_string_literals
      12 n_bytes          13 n_lines         14 has_eval (bool)
      15 has_exec (bool)

    Falls back gracefully when fields are absent (uses 0).
    """
    X, y, repos = [], [], []
    for r in records:
        if "label" not in r:
            continue
        feat = np.array([
            r.get("n_calls",              0),
            r.get("n_imports",            0),
            r.get("n_assignments",        0),
            r.get("n_comparisons",        0),
            r.get("n_augassigns",         0),
            r.get("n_exceptions_raised",  0),
            r.get("n_try_blocks",         0),
            r.get("n_returns",            0),
            r.get("n_loops",              0),
            r.get("n_conditionals",       0),
            r.get("max_depth",            0),
            r.get("n_string_literals",    0),
            r.get("n_bytes",              0),
            r.get("n_lines",              0),
            float(r.get("has_eval",       False)),
            float(r.get("has_exec",       False)),
        ], np.float32)
        X.append(feat)
        y.append(int(r["label"]))
        repos.append(_repo_from_record(r))
    return np.array(X, np.float32), np.array(y, np.int32), repos


def _prepare_pattern(records):
    from features.code_metrics import compute_all_metrics, metrics_to_feature_vector
    MAP = {"clean": 0, "code_smell": 1, "anti_pattern": 2, "style_violation": 3}
    X, y, repos = [], [], []
    for r in records:
        code  = r.get("code", "")
        label = r.get("label", "")
        if not code or label not in MAP:
            continue
        try:
            feat = metrics_to_feature_vector(compute_all_metrics(code))
            if len(feat) not in (15, 16, 17):
                continue
            X.append(feat[:15])
            y.append(MAP[label])
            repos.append(_repo_from_record(r))
        except Exception:
            continue
    return np.array(X, np.float32), np.array(y, np.int32), repos


_PREPARERS = {
    "complexity": _prepare_complexity,
    "bug":        _prepare_bug,
    "security":   _prepare_security,
    "pattern":    _prepare_pattern,
}


# ---------------------------------------------------------------------------
# Model factory and scoring
# ---------------------------------------------------------------------------

def _build_model(task: str):
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if task in ("bug", "security", "pattern"):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)),
        ])
    return Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=42)),
    ])


def _score(model, X_te, y_te, task: str) -> dict:
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                  f1_score, mean_squared_error)
    out = {k: None for k in ("auc", "ap", "f1", "rmse", "spearman")}
    try:
        if task in ("bug", "security"):
            prob = model.predict_proba(X_te)[:, 1]
            pred = (prob > 0.5).astype(int)
            if len(set(y_te)) > 1:
                out["auc"] = float(roc_auc_score(y_te, prob))
                out["ap"]  = float(average_precision_score(y_te, prob))
            out["f1"] = float(f1_score(y_te, pred, zero_division=0))
        elif task == "pattern":
            pred = model.predict(X_te)
            prob = model.predict_proba(X_te)
            out["f1"] = float(f1_score(y_te, pred, average="macro", zero_division=0))
            try:
                out["auc"] = float(roc_auc_score(y_te, prob, multi_class="ovr", average="macro"))
            except Exception:
                pass
        else:
            pred = model.predict(X_te)
            out["rmse"] = float(np.sqrt(mean_squared_error(y_te, pred)))
            try:
                from scipy.stats import spearmanr
                rho, _ = spearmanr(y_te, pred)
                out["spearman"] = float(rho)
            except Exception:
                pass
    except Exception as e:
        logger.debug("Scoring error: %s", e)
    return out


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CrossProjectBenchmark:
    """
    Leave-one-project-out evaluation for IntelliCode tasks.

    For each unique repository:
      1. Hold out all records from that repo as the test set
      2. Train on all remaining repositories
      3. Evaluate on the held-out repo
      4. Report per-project and aggregate statistics

    Demonstrates the performance degradation vs. random file-level splits —
    a key result required for any cross-project defect prediction paper.
    """

    def __init__(self, task: str, min_test_samples: int = 5):
        assert task in _PREPARERS, f"Unknown task: {task}"
        self._task     = task
        self._min_test = min_test_samples

    def run(self, records: list[dict]) -> BenchmarkReport:
        """Run full LOPO benchmark on dataset records."""
        X, y, repos = _PREPARERS[self._task](records)
        repos_arr    = np.array(repos)
        unique_repos = sorted(set(repos))

        logger.info(
            "LOPO benchmark: task=%s | %d samples | %d repos",
            self._task, len(X), len(unique_repos),
        )
        if len(unique_repos) < 2:
            raise ValueError(f"Need >= 2 repos, got: {unique_repos}")

        results: list[ProjectResult] = []
        for held_out in unique_repos:
            te_mask = repos_arr == held_out
            tr_mask = ~te_mask
            X_te, y_te = X[te_mask], y[te_mask]
            X_tr, y_tr = X[tr_mask], y[tr_mask]

            if len(X_te) < self._min_test or len(X_tr) < 10:
                continue

            n_train_repos = len(set(repos_arr[tr_mask]))
            logger.info(
                "  held-out=%-40s train=%d (%d repos) test=%d",
                held_out[:40], len(X_tr), n_train_repos, len(X_te),
            )

            model = _build_model(self._task)
            try:
                model.fit(X_tr, y_tr)
                s = _score(model, X_te, y_te, self._task)
            except Exception as e:
                logger.warning("  Training failed: %s", e)
                s = {}

            results.append(ProjectResult(
                held_out_repo=held_out,
                n_train=len(X_tr), n_test=len(X_te),
                n_train_repos=n_train_repos,
                auc=s.get("auc"), ap=s.get("ap"), f1=s.get("f1"),
                rmse=s.get("rmse"), spearman=s.get("spearman"),
            ))

        return self._build_report(results)

    def run_random_split_baseline(
        self,
        records: list[dict],
        n_trials: int = 5,
    ) -> dict:
        """
        Random file-level split baseline for comparison.
        Inflated metrics here vs. LOPO demonstrates the importance of
        cross-project evaluation — key result for paper Table 1.
        """
        from sklearn.model_selection import StratifiedKFold, KFold

        X, y, _ = _PREPARERS[self._task](records)
        aucs: list[float] = []

        if self._task in ("bug", "security"):
            kf = StratifiedKFold(n_splits=n_trials, shuffle=True, random_state=42)
            splits = list(kf.split(X, y))
        else:
            kf = KFold(n_splits=n_trials, shuffle=True, random_state=42)
            splits = list(kf.split(X))

        for tr_idx, te_idx in splits:
            model = _build_model(self._task)
            model.fit(X[tr_idx], y[tr_idx])
            s = _score(model, X[te_idx], y[te_idx], self._task)
            if s.get("auc") is not None:
                aucs.append(s["auc"])

        return {
            "random_split_auc_mean": round(float(np.mean(aucs)), 4) if aucs else None,
            "random_split_auc_std":  round(float(np.std(aucs)),  4) if aucs else None,
        }

    def _build_report(self, results: list[ProjectResult]) -> BenchmarkReport:
        def _agg(attr):
            vals = [getattr(r, attr) for r in results if getattr(r, attr) is not None]
            if not vals:
                return None, None
            return round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4)

        m_auc,  s_auc  = _agg("auc")
        m_ap,   s_ap   = _agg("ap")
        m_f1,   s_f1   = _agg("f1")
        m_rmse, s_rmse = _agg("rmse")
        m_spr,  s_spr  = _agg("spearman")

        return BenchmarkReport(
            task=self._task, protocol="lopo",
            n_projects=len(results),
            project_results=results,
            mean_auc=m_auc,   std_auc=s_auc,
            mean_ap=m_ap,     std_ap=s_ap,
            mean_f1=m_f1,     std_f1=s_f1,
            mean_rmse=m_rmse, std_rmse=s_rmse,
            mean_spearman=m_spr, std_spearman=s_spr,
        )

    def print_report(self, report: BenchmarkReport) -> None:
        print(f"\n{'='*72}")
        print(f"Cross-Project Benchmark — Task: {report.task} | Protocol: LOPO")
        print(f"{report.n_projects} held-out projects")
        print(f"{'='*72}")
        print(f"  {'Project':<40} {'AUC':>8} {'AP':>8} {'F1':>8} {'RMSE':>8}")
        print(f"  {'-'*68}")
        def _fmt(v, w=8, d=4):
            return f"{v:{w}.{d}f}" if v is not None else f"{'N/A':>{w}}"

        for r in report.project_results:
            print(f"  {r.held_out_repo[:40]:<40} "
                  f"{_fmt(r.auc)} "
                  f"{_fmt(r.ap)} "
                  f"{_fmt(r.f1)} "
                  f"{_fmt(r.rmse, d=3)}")
        print(f"  {'-'*68}")
        m = report
        print(f"  {'MEAN ± STD':<40} "
              f"{f'{m.mean_auc:.4f}±{m.std_auc:.4f}' if m.mean_auc else 'N/A':>8}")
        if report.degradation_auc is not None:
            pct = abs(report.degradation_auc) / (report.random_split_auc or 1) * 100
            print(f"\n  Random-split AUC : {report.random_split_auc:.4f}")
            print(f"  Cross-proj AUC   : {report.mean_auc:.4f}")
            print(f"  Degradation      : {report.degradation_auc:+.4f} ({pct:.1f}% drop)")
        print(f"{'='*72}\n")

    def save_report(self, report: BenchmarkReport, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info("Benchmark report saved -> %s", path)

    def save_latex_table(self, report: BenchmarkReport, path: str) -> None:
        """LaTeX per-project table for paper inclusion."""
        lines = [
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "Project & AUC & AP & F1 & RMSE \\\\",
            "\\midrule",
        ]
        def _f(v, d=4):
            return f"{v:.{d}f}" if v is not None else "--"

        for r in report.project_results:
            name = r.held_out_repo.split("/")[-1].replace("_", "\\_")
            lines.append(
                f"{name} & "
                f"{_f(r.auc)} & "
                f"{_f(r.ap)} & "
                f"{_f(r.f1)} & "
                f"{_f(r.rmse, 3)} \\\\"
            )
        m = report
        auc_str = (f"{m.mean_auc:.3f}$\\pm${m.std_auc:.3f}"
                   if m.mean_auc is not None else "--")
        lines += [
            "\\midrule",
            (f"Mean $\\pm$ std & "
             f"{auc_str} & "
             f"{_f(m.mean_ap, 3)} & "
             f"{_f(m.mean_f1, 3)} & "
             f"{_f(m.mean_rmse, 3)} \\\\"),
            "\\bottomrule",
            "\\end{tabular}",
        ]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("\n".join(lines))
        logger.info("LaTeX table saved -> %s", path)
