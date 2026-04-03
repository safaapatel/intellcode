"""
Ablation Study Framework
=========================
Systematic feature/component ablation for IntelliCode ML models.

Required for publication at ICSE/FSE/ASE. Every claim about a component's
contribution must be backed by an ablation that removes that component and
measures the performance delta.

Ablation types:
  1. Feature group ablation  — remove one feature group, retrain, measure delta
  2. Component ablation      — remove one model from the ensemble
  3. Data ablation           — reduce training set size (learning curves)
  4. Threshold ablation      — sweep decision threshold for binary classifiers

Usage:
    from evaluation.ablation_study import AblationStudy
    study = AblationStudy(task="bug")
    results = study.run_all(X, y, repos, n_trials=5)
    study.print_table(results)
    study.save_latex_table(results, "tables/ablation_bug.tex")
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature group definitions (matches metrics_to_feature_vector order)
# ---------------------------------------------------------------------------

STATIC_FEATURE_GROUPS = {
    "size":       [0, 1, 2, 3],          # sloc, blank_lines, comment_lines, docstring_lines
    "complexity": [4, 5, 6],             # cyclomatic, cognitive, halstead_bugs
    "halstead":   [7, 8, 9, 10],         # volume, difficulty, effort, time
    "functions":  [11, 12],              # n_long_functions, n_complex_functions
    "line_style": [13, 14, 15],          # max_line_len, avg_line_len, n_lines_over_80
}

JIT_FEATURE_GROUPS = {
    "diffusion":   ["NS", "ND", "NF", "Entropy"],
    "size_change": ["LA", "LD", "LT"],
    "purpose":     ["FIX"],
    "history":     ["NDEV", "AGE", "NUC"],
    "experience":  ["EXP", "REXP", "SEXP"],
}

JIT_FEATURE_NAMES = [
    "NS", "ND", "NF", "Entropy", "LA", "LD", "LT", "FIX",
    "NDEV", "AGE", "NUC", "EXP", "REXP", "SEXP",
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    """Result for a single ablation configuration."""
    config_name:  str
    ablated:      str           # what was removed ("none", "feature:complexity", etc.)
    auc:          Optional[float]
    ap:           Optional[float]
    f1:           Optional[float]
    rmse:         Optional[float]
    spearman:     Optional[float]
    n_train:      int
    n_test:       int
    delta_auc:    Optional[float] = None   # vs. full model
    pvalue:       Optional[float] = None   # Wilcoxon vs. full model
    cohens_d:     Optional[float] = None   # standardised effect size
    cliffs_delta: Optional[float] = None   # non-parametric effect size

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AblationReport:
    """Full ablation study report."""
    task:       str
    metric:     str
    baseline:   AblationResult          # full model
    ablations:  list[AblationResult]    # each ablation config

    def to_dict(self) -> dict:
        return {
            "task":      self.task,
            "metric":    self.metric,
            "baseline":  self.baseline.to_dict(),
            "ablations": [a.to_dict() for a in self.ablations],
        }


# ---------------------------------------------------------------------------
# Core ablation runner
# ---------------------------------------------------------------------------

class AblationStudy:
    """
    Runs systematic ablation experiments for a single IntelliCode task.

    Supported tasks: "bug", "complexity", "security", "pattern"
    """

    def __init__(
        self,
        task:       str,
        n_jobs:     int = 1,    # always 1 on Windows (avoids OOM/pickle errors)
        random_state: int = 42,
    ):
        assert task in ("bug", "complexity", "security", "pattern")
        self._task   = task
        self._n_jobs = n_jobs
        self._rs     = random_state

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_feature_ablation(
        self,
        X:     np.ndarray,
        y:     np.ndarray,
        repos: list[str],
        feature_groups: dict[str, list],
        n_trials: int = 5,
    ) -> AblationReport:
        """
        For each feature group:
          1. Zero out that group in X (simulate removal)
          2. Train and evaluate on cross-project split
          3. Compute Wilcoxon p-value vs. full model
        """
        from sklearn.model_selection import StratifiedKFold, KFold
        from training.dataset_builder import leave_one_project_out_splits

        # Baseline: full model
        logger.info("Running baseline (all features) ...")
        baseline_scores = self._cv_evaluate(X, y, repos, n_trials)
        baseline_result = self._aggregate_result("full_model", "none", baseline_scores, y)

        ablation_results: list[AblationResult] = []

        for group_name, group_indices in feature_groups.items():
            logger.info("Ablating feature group: %s (indices %s)", group_name, group_indices)
            X_ablated = X.copy()

            if isinstance(group_indices[0], int):
                # Index-based ablation (static features)
                X_ablated[:, group_indices] = 0.0
            else:
                # Name-based ablation (JIT features — need to know column order)
                for name in group_indices:
                    if name in JIT_FEATURE_NAMES:
                        idx = 16 + JIT_FEATURE_NAMES.index(name)
                        if idx < X_ablated.shape[1]:
                            X_ablated[:, idx] = 0.0

            scores = self._cv_evaluate(X_ablated, y, repos, n_trials)
            result = self._aggregate_result(f"ablate_{group_name}", group_name, scores, y)
            result.delta_auc = (
                (result.auc or 0.0) - (baseline_result.auc or 0.0)
                if result.auc is not None and baseline_result.auc is not None
                else None
            )

            # Wilcoxon signed-rank test + effect sizes
            base_aucs = baseline_scores.get("auc", [])
            abl_aucs  = scores.get("auc", [])
            if len(base_aucs) >= 5 and len(abl_aucs) >= 5:
                try:
                    from scipy.stats import wilcoxon
                    stat, pval = wilcoxon(
                        base_aucs, abl_aucs, alternative="greater",
                    )
                    result.pvalue = float(pval)
                except Exception:
                    pass

                # Cohen's d (standardised mean difference)
                try:
                    b = np.array(base_aucs)
                    a = np.array(abl_aucs)
                    pooled_std = np.sqrt((b.std(ddof=1)**2 + a.std(ddof=1)**2) / 2)
                    if pooled_std > 0:
                        result.cohens_d = round(float((b.mean() - a.mean()) / pooled_std), 4)
                except Exception:
                    pass

                # Cliff's delta (non-parametric, no normality assumption)
                try:
                    b = np.array(base_aucs)
                    a = np.array(abl_aucs)
                    dominance = sum(
                        1 if bi > ai else (-1 if bi < ai else 0)
                        for bi in b for ai in a
                    )
                    result.cliffs_delta = round(float(dominance / (len(b) * len(a))), 4)
                except Exception:
                    pass

            ablation_results.append(result)

        if self._task in ("bug", "security"):
            metric = "auc"
        elif self._task == "pattern":
            metric = "f1"
        else:
            metric = "rmse"
        return AblationReport(
            task=self._task,
            metric=metric,
            baseline=baseline_result,
            ablations=ablation_results,
        )

    def run_component_ablation(
        self,
        X:       np.ndarray,
        y:       np.ndarray,
        repos:   list[str],
        components: dict[str, Callable],
        n_trials: int = 5,
    ) -> AblationReport:
        """
        For each model component (e.g., "xgboost", "lr", "mlp"):
          1. Train the ensemble without that component
          2. Evaluate and compare to full ensemble
        """
        logger.info("Running component ablation for task: %s", self._task)
        baseline_scores = self._cv_evaluate(X, y, repos, n_trials)
        baseline_result = self._aggregate_result("full_ensemble", "none", baseline_scores, y)

        ablation_results: list[AblationResult] = []
        for comp_name, build_fn in components.items():
            logger.info("Ablating component: %s", comp_name)
            scores = self._cv_evaluate(X, y, repos, n_trials, model_factory=build_fn)
            result = self._aggregate_result(f"ablate_{comp_name}", comp_name, scores, y)
            result.delta_auc = (
                (result.auc or 0.0) - (baseline_result.auc or 0.0)
                if result.auc is not None and baseline_result.auc is not None else None
            )
            ablation_results.append(result)

        return AblationReport(
            task=self._task,
            metric="f1" if self._task == "pattern" else "auc",
            baseline=baseline_result,
            ablations=ablation_results,
        )

    def run_data_ablation(
        self,
        X:       np.ndarray,
        y:       np.ndarray,
        repos:   list[str],
        fractions: list[float] = (0.1, 0.25, 0.5, 0.75, 1.0),
    ) -> AblationReport:
        """
        Learning curve: train on increasing fractions of the data.
        Shows whether the model is data-starved or converged.
        """
        logger.info("Running data ablation (learning curve) ...")
        baseline_scores = self._cv_evaluate(X, y, repos, n_trials=3)
        baseline_result = self._aggregate_result("full_data_100%", "none", baseline_scores, y)

        ablation_results: list[AblationResult] = []
        for frac in fractions:
            if frac >= 1.0:
                continue
            n = max(20, int(len(X) * frac))
            idx = np.random.choice(len(X), n, replace=False)
            X_sub = X[idx]
            y_sub = y[idx]
            repos_sub = [repos[i] for i in idx] if repos else []
            scores = self._cv_evaluate(X_sub, y_sub, repos_sub, n_trials=3)
            result = self._aggregate_result(f"data_{int(frac*100)}%", f"data_frac={frac}", scores, y_sub)
            ablation_results.append(result)
            logger.info("  %d%% data: AUC=%.4f", int(frac*100), result.auc or 0)

        return AblationReport(
            task=self._task,
            metric="f1" if self._task == "pattern" else ("rmse" if self._task == "complexity" else "auc"),
            baseline=baseline_result,
            ablations=ablation_results,
        )

    # ------------------------------------------------------------------
    # Cross-validated evaluation
    # ------------------------------------------------------------------

    def _cv_evaluate(
        self,
        X:       np.ndarray,
        y:       np.ndarray,
        repos:   list[str],
        n_trials: int,
        model_factory: Optional[Callable] = None,
    ) -> dict[str, list[float]]:
        """
        Run n_trials cross-project or stratified splits, return per-fold scores.
        """
        from training.dataset_builder import leave_one_project_out_splits

        scores: dict[str, list[float]] = {"auc": [], "ap": [], "f1": [], "rmse": [], "spearman": []}

        # Build record list for LOPO splits
        records = [{"repo": r, "_idx": i} for i, r in enumerate(repos)]

        if repos and len(set(repos)) >= 3:
            splits = leave_one_project_out_splits(records)[:n_trials]
        else:
            from sklearn.model_selection import StratifiedKFold, KFold
            # Fallback to k-fold when repo info unavailable
            n_folds = min(n_trials, 5)
            if self._task in ("bug", "security"):
                kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self._rs)
                folds = list(kf.split(X, y))
            else:
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=self._rs)
                folds = list(kf.split(X))
            splits = []
            for tr_idx, te_idx in folds:
                tr_recs = [{"repo": "fold_train", "_idx": i} for i in tr_idx]
                te_recs = [{"repo": "fold_test",  "_idx": i} for i in te_idx]
                splits.append(("fold", tr_recs, te_recs))

        for held_out, train_recs, test_recs in splits:
            tr_idx = [r["_idx"] for r in train_recs]
            te_idx = [r["_idx"] for r in test_recs]
            if len(tr_idx) < 10 or len(te_idx) < 5:
                continue

            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]

            model = model_factory() if model_factory else self._default_model()
            try:
                model.fit(X_tr, y_tr)
                fold_scores = self._score(model, X_te, y_te)
                for k, v in fold_scores.items():
                    if v is not None:
                        scores[k].append(v)
            except Exception as e:
                logger.debug("Fold error: %s", e)

        return scores

    def _default_model(self):
        """
        Return the task-canonical production model for ablation evaluation.

        Using the same model architecture as the deployed system ensures that
        ablation results directly reflect production behaviour, not a generic
        GBM proxy (which would give misleading feature importance rankings).

        Task → production architecture:
          security  → RandomForestClassifier (same as EnsembleSecurityModel RF head)
          bug       → XGBClassifier          (best single model in LR+XGB ensemble)
          pattern   → RandomForestClassifier (same as PatternRFModel)
          complexity→ XGBRegressor           (same as ComplexityPredictionModel)
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        if self._task == "security":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("model",  RandomForestClassifier(
                    n_estimators=200, class_weight="balanced",
                    n_jobs=1, random_state=self._rs,
                )),
            ])
        elif self._task == "bug":
            try:
                from xgboost import XGBClassifier
                return Pipeline([
                    ("scaler", StandardScaler()),
                    ("model",  XGBClassifier(
                        n_estimators=200, max_depth=5, learning_rate=0.05,
                        subsample=0.8, n_jobs=1, random_state=self._rs,
                        eval_metric="logloss", verbosity=0,
                    )),
                ])
            except ImportError:
                pass
            # Fallback if xgboost not available
            from sklearn.ensemble import GradientBoostingClassifier
            return Pipeline([
                ("scaler", StandardScaler()),
                ("model",  GradientBoostingClassifier(
                    n_estimators=200, max_depth=5, random_state=self._rs,
                )),
            ])
        elif self._task == "pattern":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("model",  RandomForestClassifier(
                    n_estimators=300, class_weight="balanced",
                    n_jobs=1, random_state=self._rs,
                )),
            ])
        else:  # complexity
            try:
                from xgboost import XGBRegressor
                return Pipeline([
                    ("scaler", StandardScaler()),
                    ("model",  XGBRegressor(
                        n_estimators=500, max_depth=6, learning_rate=0.05,
                        subsample=0.8, n_jobs=1, random_state=self._rs,
                        verbosity=0,
                    )),
                ])
            except ImportError:
                pass
            from sklearn.ensemble import GradientBoostingRegressor
            return Pipeline([
                ("scaler", StandardScaler()),
                ("model",  GradientBoostingRegressor(
                    n_estimators=200, max_depth=6, random_state=self._rs,
                )),
            ])

    def _score(self, model, X_te: np.ndarray, y_te: np.ndarray) -> dict[str, Optional[float]]:
        """Compute all metrics for one fold."""
        from sklearn.metrics import (
            roc_auc_score, average_precision_score, f1_score,
            mean_squared_error,
        )

        out: dict[str, Optional[float]] = {
            "auc": None, "ap": None, "f1": None, "rmse": None, "spearman": None
        }
        try:
            if self._task in ("bug", "security"):
                y_prob = model.predict_proba(X_te)[:, 1]
                y_pred = (y_prob > 0.5).astype(int)
                if len(set(y_te)) > 1:
                    out["auc"] = float(roc_auc_score(y_te, y_prob))
                    out["ap"]  = float(average_precision_score(y_te, y_prob))
                out["f1"] = float(f1_score(y_te, y_pred, zero_division=0))
            elif self._task == "pattern":
                y_pred = model.predict(X_te)
                y_prob = model.predict_proba(X_te)
                out["f1"] = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
                try:
                    from sklearn.metrics import roc_auc_score as ras
                    out["auc"] = float(ras(y_te, y_prob, multi_class="ovr", average="macro"))
                except Exception:
                    pass
            else:
                y_pred = model.predict(X_te)
                out["rmse"] = float(np.sqrt(mean_squared_error(y_te, y_pred)))
                try:
                    from scipy.stats import spearmanr
                    rho, _ = spearmanr(y_te, y_pred)
                    out["spearman"] = float(rho)
                except Exception:
                    pass
        except Exception as e:
            logger.debug("Scoring error: %s", e)
        return out

    def _aggregate_result(
        self,
        config_name: str,
        ablated: str,
        scores: dict[str, list[float]],
        y: np.ndarray,
    ) -> AblationResult:
        """Aggregate per-fold scores to mean values."""
        def mean_or_none(lst):
            return round(float(np.mean(lst)), 4) if lst else None

        return AblationResult(
            config_name=config_name,
            ablated=ablated,
            auc=mean_or_none(scores.get("auc", [])),
            ap=mean_or_none(scores.get("ap", [])),
            f1=mean_or_none(scores.get("f1", [])),
            rmse=mean_or_none(scores.get("rmse", [])),
            spearman=mean_or_none(scores.get("spearman", [])),
            n_train=int(len(y) * 0.8),
            n_test=int(len(y) * 0.2),
        )

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_table(self, report: AblationReport) -> None:
        """Print ASCII ablation table to stdout."""
        metric = report.metric
        print(f"\n{'='*90}")
        print(f"Ablation Study -- Task: {report.task}  |  Primary metric: {metric}")
        print(f"{'='*90}")
        print(f"  {'Configuration':<30} {metric.upper():>8} {'Delta':>8} {'p-value':>10} {'d':>8} {'Cliff':>8}")
        print(f"  {'-'*78}")

        base = report.baseline
        base_val = getattr(base, metric) or 0.0
        print(f"  {'[BASELINE] ' + base.config_name:<30} {base_val:>8.4f} {'--':>8} {'--':>10} {'--':>8} {'--':>8}")

        for r in sorted(report.ablations, key=lambda x: getattr(x, metric) or 0, reverse=True):
            val   = getattr(r, metric)
            delta = r.delta_auc
            pval  = r.pvalue
            cd    = r.cohens_d
            clf   = r.cliffs_delta
            val_s   = f"{val:.4f}"   if val   is not None else "N/A"
            delta_s = f"{delta:+.4f}" if delta is not None else "--"
            pval_s  = f"{pval:.4f}"  if pval  is not None else "--"
            cd_s    = f"{cd:+.3f}"   if cd    is not None else "--"
            clf_s   = f"{clf:+.3f}"  if clf   is not None else "--"
            sig = "*" if pval is not None and pval < 0.05 else ""
            print(f"  {r.config_name:<30} {val_s:>8} {delta_s:>8} {pval_s+sig:>10} {cd_s:>8} {clf_s:>8}")

        print(f"  {'-'*78}")
        print(f"  d = Cohen's d (standardised effect size)  |  Cliff = Cliff's delta (non-parametric)")
        print(f"  * p < 0.05 (Wilcoxon signed-rank test, one-sided)")
        print(f"{'='*90}\n")

    def save_latex_table(self, report: AblationReport, output_path: str) -> None:
        """Write a LaTeX tabular fragment for paper inclusion."""
        metric = report.metric
        lines = [
            "\\begin{tabular}{lrrrrl}",
            "\\toprule",
            f"Configuration & {metric.upper()} & $\\Delta$ & $p$-value & $d$ & $\\delta$ \\\\",
            "\\midrule",
        ]
        base_val = getattr(report.baseline, metric) or 0.0
        lines.append(f"Full model & {base_val:.4f} & --- & --- & --- & --- \\\\")

        for r in report.ablations:
            val   = getattr(r, metric)
            delta = r.delta_auc
            pval  = r.pvalue
            cd    = r.cohens_d
            clf   = r.cliffs_delta
            val_s   = f"{val:.4f}"    if val   is not None else "--"
            delta_s = f"{delta:+.4f}" if delta is not None else "--"
            pval_s  = f"{pval:.4f}"   if pval  is not None else "--"
            cd_s    = f"{cd:+.3f}"    if cd    is not None else "--"
            clf_s   = f"{clf:+.3f}"   if clf   is not None else "--"
            sig = "$^{*}$" if pval is not None and pval < 0.05 else ""
            name = r.config_name.replace("_", "\\_")
            lines.append(
                f"\\quad w/o {name} & {val_s} & {delta_s} & {pval_s}{sig} & {cd_s} & {clf_s} \\\\"
            )

        lines += [
            "\\bottomrule",
            "\\multicolumn{6}{l}{\\footnotesize $d$ = Cohen's $d$;\\;"
            " $\\delta$ = Cliff's $\\delta$ (non-parametric);"
            " $^{*}$ $p < 0.05$ (Wilcoxon signed-rank, one-sided)}",
            "\\end{tabular}",
        ]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
        logger.info("LaTeX table saved -> %s", output_path)

    def save_json(self, report: AblationReport, output_path: str) -> None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        logger.info("Ablation JSON saved -> %s", output_path)
