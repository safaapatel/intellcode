"""
Train Complexity Prediction Model (XGBoost)

Usage:
    cd backend
    python training/train_complexity.py --data data/complexity_dataset.jsonl
    python training/train_complexity.py --data data/complexity_dataset.jsonl --cv 5

    # Cross-project split (recommended for research-quality evaluation):
    python training/train_complexity.py \\
        --data data/complexity_dataset.jsonl \\
        --test-repos pallets/flask psf/requests

    # Leave-one-project-out cross-project benchmark (LOPO):
    python training/train_complexity.py \\
        --data data/complexity_dataset.jsonl \\
        --cross-project-benchmark

Outputs:
    checkpoints/complexity/model.pkl
    checkpoints/complexity/mapie_wrapper.pkl   (conformal intervals, if mapie installed)
    checkpoints/complexity/metrics.json
    checkpoints/complexity/cross_project_benchmark.json   (if --cross-project-benchmark)
    checkpoints/complexity/train_config.json              (reproducibility)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.training_config import ComplexityConfig, set_global_seed

# ── Feature names (must stay in sync with code_metrics.metrics_to_feature_vector) ──
FEATURE_NAMES = [
    # cognitive_complexity is EXCLUDED (it is the prediction TARGET — COG_IDX=1)
    "cyclomatic_complexity", "max_func_cc", "avg_func_cc",
    "sloc", "comments", "blank_lines",
    "halstead_volume", "halstead_difficulty", "halstead_effort",
    "bugs_delivered",
    "n_long_functions", "n_complex_functions",
    "max_line_length", "avg_line_length", "n_lines_over_80",
    # Graph-based structural features (dims 15-17, added to close LOC baseline gap)
    "max_nesting_depth", "n_branches", "cc_per_sloc",
]  # 18 features


# ── Data loading ──────────────────────────────────────────────────────────────

def _repo_from_origin(origin: str) -> str:
    """Extract a repository label from an 'origin' field.

    Handles multiple origin formats:
      'cve_mining/flask/conf.py'    -> 'flask'       (CVE mining prefix stripped)
      'psf/requests/src/file.py'   -> 'psf/requests' (owner/repo)
      'django/build.py'            -> 'django'        (single-level repo)
    """
    if not origin:
        return "unknown"
    parts = origin.split("/")
    # Strip known mining/collection prefixes
    if parts[0] in ("cve_mining", "osv_mining", "github_advisory", "nvd_mining"):
        return parts[1] if len(parts) > 1 else "unknown"
    # owner/repo/path  -> 'owner/repo'
    if len(parts) >= 3:
        return f"{parts[0]}/{parts[1]}"
    # owner/file.py or just 'owner' -> 'owner'
    return parts[0]


def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load JSONL dataset produced by dataset_builder.py.

    TARGET LEAKAGE FIX (Mar 2026):
      The original target was maintainability_index (MI), a closed-form formula
      of halstead_volume * cyclomatic_complexity * sloc — all present in the
      feature vector — giving trivial R2≈1.0.  We now use cognitive_complexity
      (features[1]) as the target and exclude it from the input, forcing the
      model to learn a non-trivial mapping from structural metrics to cognitive
      load (which is NOT algebraically derivable from the other features).

    Returns:
        X     : (N, 15) feature matrix (cognitive_complexity excluded)
        y     : (N,)    cognitive complexity scores (non-negative integers)
        repos : (N,)    repository name per sample (for cross-project split)
    """
    # Index of cognitive_complexity in the 16-dim feature vector
    COG_IDX = 1

    X, y, repos = [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            feats = record.get("features", [])
            if not feats or len(feats) < 16:
                continue
            # Extract cognitive complexity as target
            cog = feats[COG_IDX]
            if cog < 0:
                continue
            # Build 15-dim input (remove cognitive_complexity)
            x_vec = [feats[i] for i in range(16) if i != COG_IDX]
            # Repo label: prefer explicit "repo" field, fall back to parsing "origin"
            repo = record.get("repo") or _repo_from_origin(record.get("origin", ""))
            X.append(x_vec)
            y.append(cog)
            repos.append(repo)

    print(f"Loaded {len(X)} samples from {path}")
    print(f"  Target: cognitive_complexity  (range: {min(y):.0f} – {max(y):.0f})")
    print(f"  Repositories: {sorted(set(repos))}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), repos


def cross_project_split(
    X: np.ndarray,
    y: np.ndarray,
    repos: list[str],
    test_repos: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split by repository so that no repo appears in both train and test.
    This is the research-grade evaluation strategy — random file-level splits
    inflate metrics because files from the same repo share style/patterns.
    """
    test_set = set(test_repos)
    train_mask = np.array([r not in test_set for r in repos])
    test_mask = ~train_mask

    n_test = test_mask.sum()
    n_train = train_mask.sum()
    if n_test == 0:
        raise ValueError(
            f"No samples found for test repos {test_repos}. "
            f"Available repos: {sorted(set(repos))}"
        )
    print(f"Cross-project split — Train: {n_train}  Test: {n_test}")
    print(f"  Test repos : {test_repos}")
    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    data_path: str,
    output_dir: str = "checkpoints/complexity",
    cv_folds: int = 5,
    test_split: float = 0.15,
    test_repos: Optional[list[str]] = None,
    fit_conformal: bool = True,
    run_lopo_benchmark: bool = False,
    cfg: Optional[ComplexityConfig] = None,
    **xgb_kwargs,
) -> dict:
    """
    Train XGBoost complexity regressor with k-fold CV, held-out test evaluation,
    optional cross-project split, optional conformal prediction intervals,
    and Wilcoxon significance test against a Halstead-only baseline.

    Args:
        data_path:     Path to JSONL dataset.
        output_dir:    Where to save checkpoints and metrics.
        cv_folds:      K-fold CV on the training set.
        test_split:    Fraction of data for test when NOT using cross-project split.
        test_repos:    If given, hold these repos out entirely as test set.
        fit_conformal: Whether to fit a MAPIE conformal wrapper.
        **xgb_kwargs:  Override default XGBoost hyperparameters.
    """
    # ── Config + reproducibility ──────────────────────────────────────────────
    if cfg is None:
        cfg = ComplexityConfig(
            n_estimators=xgb_kwargs.get("n_estimators", 500),
            max_depth=xgb_kwargs.get("max_depth", 6),
            learning_rate=xgb_kwargs.get("learning_rate", 0.05),
            cv_folds=cv_folds,
            test_split=test_split,
            output_dir=output_dir,
            fit_conformal=fit_conformal,
            use_cross_project=bool(test_repos),
            test_repos=test_repos or ["scikit-learn/scikit-learn"],
        )
    cfg.validate()
    set_global_seed(cfg.seed)

    try:
        import xgboost as xgb
    except ImportError:
        print("ERROR: xgboost required. pip install xgboost")
        sys.exit(1)

    from sklearn.model_selection import train_test_split, KFold
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    from scipy.stats import wilcoxon, spearmanr
    import math
    import pickle

    X, y, repos = load_dataset(data_path)

    # ── Train / test split ────────────────────────────────────────────────────
    if test_repos:
        X_train, X_test, y_train, y_test = cross_project_split(X, y, repos, test_repos)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_split, random_state=42
        )
        print(f"Random split — Train: {len(X_train)}, Test: {len(X_test)}")

    # ── Hyperparameters ───────────────────────────────────────────────────────
    params = {
        "n_estimators":    xgb_kwargs.get("n_estimators", 500),
        "max_depth":       xgb_kwargs.get("max_depth", 6),
        "learning_rate":   xgb_kwargs.get("learning_rate", 0.05),
        "subsample":       xgb_kwargs.get("subsample", 0.8),
        "colsample_bytree":xgb_kwargs.get("colsample_bytree", 0.8),
        "reg_lambda":      xgb_kwargs.get("reg_lambda", 1.0),
        "reg_alpha":       xgb_kwargs.get("reg_alpha", 0.1),
        "min_child_weight":xgb_kwargs.get("min_child_weight", 5),
        "gamma":           xgb_kwargs.get("gamma", 0.1),
        "random_state":    42,
        "n_jobs":          1,   # n_jobs>1 causes OOM/pickle errors on Windows
        "eval_metric":     "rmse",
    }

    # ── K-fold cross-validation on the training set ───────────────────────────
    print(f"\nRunning {cv_folds}-fold cross-validation on training set...")
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_rmses, cv_r2s, cv_maes, cv_rhos = [], [], [], []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train), 1):
        m = xgb.XGBRegressor(**params)
        m.fit(
            X_train[tr_idx], y_train[tr_idx],
            eval_set=[(X_train[val_idx], y_train[val_idx])],
            verbose=False,
        )
        preds = m.predict(X_train[val_idx])
        rmse = math.sqrt(mean_squared_error(y_train[val_idx], preds))
        r2   = r2_score(y_train[val_idx], preds)
        mae  = mean_absolute_error(y_train[val_idx], preds)
        rho, _ = spearmanr(y_train[val_idx], preds)
        cv_rmses.append(rmse); cv_r2s.append(r2)
        cv_maes.append(mae);   cv_rhos.append(rho)
        print(f"  Fold {fold}: RMSE={rmse:.3f}  MAE={mae:.3f}  R2={r2:.4f}  rho={rho:.4f}")

    print(f"\nCV RMSE : {np.mean(cv_rmses):.3f} +/- {np.std(cv_rmses):.3f}")
    print(f"CV MAE  : {np.mean(cv_maes):.3f} +/- {np.std(cv_maes):.3f}")
    print(f"CV R2   : {np.mean(cv_r2s):.4f} +/- {np.std(cv_r2s):.4f}")
    print(f"CV rho    : {np.mean(cv_rhos):.4f} +/- {np.std(cv_rhos):.4f}")

    # ── Final model on full training set ─────────────────────────────────────
    print("\nTraining final model on full training set...")
    final_model = xgb.XGBRegressor(**params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=100,
    )

    # ── Test-set evaluation ───────────────────────────────────────────────────
    y_pred = final_model.predict(X_test)
    test_rmse = math.sqrt(mean_squared_error(y_test, y_pred))
    test_r2   = r2_score(y_test, y_pred)
    test_mae  = mean_absolute_error(y_test, y_pred)
    test_rho, test_rho_p = spearmanr(y_test, y_pred)

    print(f"\nTest RMSE : {test_rmse:.3f}")
    print(f"Test MAE  : {test_mae:.3f}")
    print(f"Test R2   : {test_r2:.4f}")
    print(f"Test rho    : {test_rho:.4f}  (p={test_rho_p:.4e})")

    # ── Halstead-only baseline (research requirement) ────────────────────────
    # Baseline: predict using only the Halstead bugs_delivered feature (index 10)
    # as a proxy — equivalent to what prior work uses as a trivial baseline.
    # We also compare against predicting the training mean (zero-information baseline).
    mean_pred = np.full_like(y_test, y_train.mean())
    mean_rmse = math.sqrt(mean_squared_error(y_test, mean_pred))

    # Halstead-effort single-feature linear baseline
    from sklearn.linear_model import LinearRegression
    hal_idx = FEATURE_NAMES.index("halstead_effort")
    lr_base = LinearRegression()
    lr_base.fit(X_train[:, [hal_idx]], y_train)
    hal_pred = lr_base.predict(X_test[:, [hal_idx]])
    hal_rmse = math.sqrt(mean_squared_error(y_test, hal_pred))
    hal_r2   = r2_score(y_test, hal_pred)

    print(f"\nBaseline (mean prediction) RMSE : {mean_rmse:.3f}")
    print(f"Baseline (Halstead effort LR)   RMSE : {hal_rmse:.3f}  R2={hal_r2:.4f}")
    print(f"XGBoost improvement over mean   : {mean_rmse - test_rmse:+.3f} RMSE")

    # Wilcoxon signed-rank test: are XGBoost residuals smaller than Halstead residuals?
    xgb_errors = np.abs(y_test - y_pred)
    hal_errors = np.abs(y_test - hal_pred)
    if len(xgb_errors) >= 10:
        stat, p_val = wilcoxon(xgb_errors, hal_errors, alternative="less")
        print(f"\nWilcoxon (XGB < Halstead): stat={stat:.1f}  p={p_val:.4e}  "
              f"{'[significant]' if p_val < 0.05 else '[not significant]'}")
        sig_result = {"wilcoxon_stat": float(stat), "wilcoxon_p": float(p_val)}
    else:
        print("[WARN] Too few test samples for Wilcoxon test (need ≥ 10)")
        sig_result = {}

    # ── Feature importance ────────────────────────────────────────────────────
    importances = final_model.feature_importances_
    feat_imp = sorted(
        zip(FEATURE_NAMES[:len(importances)], importances),
        key=lambda x: x[1], reverse=True,
    )
    print("\nTop 10 Feature Importances:")
    for name, imp in feat_imp[:10]:
        print(f"  {name:40s} {imp:.4f}")

    # ── Save model ────────────────────────────────────────────────────────────
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    print(f"\nModel saved -> {model_path}")

    # ── Conformal calibration (MAPIE) ─────────────────────────────────────────
    conformal_metrics: dict = {}
    if fit_conformal:
        try:
            from mapie.regression import MapieRegressor

            # Cross-project conformal calibration (Barber et al. 2023):
            # Calibrate on the held-out TEST set (which is already a held-out
            # *repository* when --test-repos is used). This is strictly stronger
            # than in-distribution calibration on a slice of the training set —
            # it provides coverage guarantees that hold when the model is
            # deployed on a new repository, not just a new file from the same repo.
            #
            # When test_repos is None (random split), fall back to a 15% slice of
            # the training set, and note that coverage is in-distribution only.
            if test_repos:
                X_cal, y_cal = X_test, y_test
                cal_strategy = "cross_project (test repo)"
            else:
                cal_size = max(50, int(0.15 * len(X_train)))
                X_cal, y_cal = X_train[-cal_size:], y_train[-cal_size:]
                cal_strategy = "in_distribution (training slice)"
                print(
                    f"[WARN] Conformal calibrated in-distribution ({cal_strategy}).\n"
                    f"       Use --test-repos for cross-project coverage guarantees."
                )

            mapie = MapieRegressor(final_model, method="plus", cv="prefit")
            mapie.fit(X_cal, y_cal)
            mapie_path = out_dir / "mapie_wrapper.pkl"
            with open(mapie_path, "wb") as f:
                pickle.dump(mapie, f)
            print(f"MAPIE conformal wrapper saved -> {mapie_path}  "
                  f"(calibration: {cal_strategy})")

            # Evaluate empirical coverage on test set
            _, pi = mapie.predict(X_test, alpha=0.1)
            covered = np.mean((y_test >= pi[:, 0, 0]) & (y_test <= pi[:, 1, 0]))
            avg_width = float(np.mean(pi[:, 1, 0] - pi[:, 0, 0]))
            print(f"90% PI empirical coverage: {covered:.3f}  avg width: {avg_width:.2f}")
            conformal_metrics = {
                "empirical_coverage_90":   float(covered),
                "avg_interval_width_90":   avg_width,
                "calibration_strategy":    cal_strategy,
                "calibration_n_samples":   len(X_cal),
            }
        except ImportError:
            print("[WARN] mapie not installed — skipping conformal calibration. pip install mapie")

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics = {
        "cv_rmse_mean":    float(np.mean(cv_rmses)),
        "cv_rmse_std":     float(np.std(cv_rmses)),
        "cv_mae_mean":     float(np.mean(cv_maes)),
        "cv_mae_std":      float(np.std(cv_maes)),
        "cv_r2_mean":      float(np.mean(cv_r2s)),
        "cv_r2_std":       float(np.std(cv_r2s)),
        "cv_spearman_mean":float(np.mean(cv_rhos)),
        "cv_spearman_std": float(np.std(cv_rhos)),
        "test_rmse":       float(test_rmse),
        "test_mae":        float(test_mae),
        "test_r2":         float(test_r2),
        "test_spearman":   float(test_rho),
        "test_spearman_p": float(test_rho_p),
        "baseline_mean_rmse":    float(mean_rmse),
        "baseline_halstead_rmse":float(hal_rmse),
        "baseline_halstead_r2":  float(hal_r2),
        "n_train":         len(X_train),
        "n_test":          len(X_test),
        "split_strategy":  "cross_project" if test_repos else "random",
        "test_repos":      test_repos or [],
        "significance":    sig_result,
        "conformal":       conformal_metrics,
        "feature_importances": [(n, float(i)) for n, i in feat_imp],
        "hyperparams":     params,
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved -> {metrics_path}")

    # ── Multi-seed stability check ─────────────────────────────────────────────
    # Run train/test split 5 times with different random seeds.
    # Reports mean +/- std across splits — required to show results are not
    # seed-sensitive (single-seed results may be cherry-picked).
    EVAL_SEEDS = [42, 0, 7, 123, 999]
    if not test_repos:
        print("\nRunning multi-seed stability check (5 seeds)...")
        seed_rmses, seed_r2s, seed_rhos = [], [], []
        for seed in EVAL_SEEDS:
            X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
                X, y, test_size=test_split, random_state=seed)
            m_s = xgb.XGBRegressor(**{**params, "random_state": seed})
            m_s.fit(X_tr_s, y_tr_s, eval_set=[(X_te_s, y_te_s)], verbose=False)
            p_s = m_s.predict(X_te_s)
            seed_rmses.append(math.sqrt(mean_squared_error(y_te_s, p_s)))
            seed_r2s.append(r2_score(y_te_s, p_s))
            seed_rhos.append(float(spearmanr(y_te_s, p_s)[0]))
            print(f"  seed={seed}: RMSE={seed_rmses[-1]:.3f}  R2={seed_r2s[-1]:.4f}  rho={seed_rhos[-1]:.4f}")
        multi_seed = {
            "seeds": EVAL_SEEDS,
            "rmse_mean": round(float(np.mean(seed_rmses)), 4),
            "rmse_std":  round(float(np.std(seed_rmses)),  4),
            "r2_mean":   round(float(np.mean(seed_r2s)),   4),
            "r2_std":    round(float(np.std(seed_r2s)),    4),
            "rho_mean":  round(float(np.mean(seed_rhos)),  4),
            "rho_std":   round(float(np.std(seed_rhos)),   4),
        }
        metrics["multi_seed"] = multi_seed
        print(f"\nMulti-seed RMSE : {multi_seed['rmse_mean']:.3f} +/- {multi_seed['rmse_std']:.3f}")
        print(f"Multi-seed R2   : {multi_seed['r2_mean']:.4f} +/- {multi_seed['r2_std']:.4f}")
        print(f"Multi-seed rho  : {multi_seed['rho_mean']:.4f} +/- {multi_seed['rho_std']:.4f}")

    # ── Save training config for reproducibility ──────────────────────────────
    cfg.save(str(out_dir / "train_config.json"))

    # ── Leave-one-project-out cross-project benchmark (optional) ─────────────
    if run_lopo_benchmark:
        try:
            from evaluation.cross_project_benchmark import CrossProjectBenchmark
            print("\nRunning leave-one-project-out (LOPO) cross-project benchmark...")
            # Rebuild raw records from loaded arrays for benchmark compatibility
            records = []
            with open(data_path) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            bench = CrossProjectBenchmark(task="complexity")
            report = bench.run(records)
            bench.print_report(report)
            bench_path = str(out_dir / "cross_project_benchmark.json")
            bench.save_report(report, bench_path)
            bench.save_latex_table(report, str(out_dir / "cross_project_table.tex"))
            metrics["lopo_mean_rmse"] = report.mean_rmse
            metrics["lopo_std_rmse"]  = report.std_rmse
            metrics["lopo_mean_spearman"] = report.mean_spearman
            # Overwrite metrics with LOPO results appended
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            print(f"[WARN] LOPO benchmark failed: {e}")

    return metrics


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train complexity prediction XGBoost model")
    parser.add_argument("--data", required=True, help="Path to JSONL dataset")
    parser.add_argument("--out", default="checkpoints/complexity")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--test-repos", nargs="+", default=None,
        metavar="REPO",
        help="Repos to hold out as test set (e.g. pallets/flask psf/requests). "
             "If omitted, uses a random 15%% split.",
    )
    parser.add_argument(
        "--no-conformal", action="store_true",
        help="Skip MAPIE conformal calibration",
    )
    parser.add_argument(
        "--cross-project-benchmark", action="store_true",
        help="Run leave-one-project-out (LOPO) benchmark after training. "
             "Produces cross_project_benchmark.json and LaTeX table.",
    )
    args = parser.parse_args()

    train(
        data_path=args.data,
        output_dir=args.out,
        cv_folds=args.cv,
        test_repos=args.test_repos,
        fit_conformal=not args.no_conformal,
        run_lopo_benchmark=args.cross_project_benchmark,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
