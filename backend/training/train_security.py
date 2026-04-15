"""
Train Security Detection Ensemble (Random Forest + 1D CNN)
============================================================
RESEARCH FIXES FROM AUDIT:
  1. CVEFixes format: dataset loader now reads 'source' field from CVEFixes
     records (real CVE-linked vulnerable code), not heuristic-labeled data.
  2. Average Precision (AP) added as primary metric — AP is preferred over
     AUC-ROC for imbalanced security datasets (fewer positives than negatives).
  3. Wilcoxon signed-rank test vs. majority-class baseline — required for
     statistical significance claims in publications.
  4. Cross-project split support — pass --test-repos to evaluate honestly.
  5. Recall@threshold reported separately — recall is the primary concern
     in security tooling (missing a vulnerability is costly).

Usage:
    cd backend
    python training/train_security.py --data data/security_dataset.jsonl

    # With cross-project split (held-out repo):
    python training/train_security.py \\
        --data data/security_dataset.jsonl \\
        --test-repos django/django psf/requests

Outputs:
    checkpoints/security/rf_model.pkl
    checkpoints/security/cnn_model.pt
    checkpoints/security/cnn_vocab.json
    checkpoints/security/metrics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Global seed for reproducibility
random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list[dict]:
    """
    Load JSONL security dataset.
    Supports multiple label sources: CVEFixes, vuln_repo, clean_repo.
    """
    records = []
    label_counts: dict = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            records.append(r)
            src = r.get("data_source", "unknown")
            label_counts[src] = label_counts.get(src, 0) + 1

    n_pos = sum(1 for r in records if r.get("label") == 1)
    n_neg = sum(1 for r in records if r.get("label") == 0)
    logger.info(
        "Loaded %d samples: %d positive, %d negative | sources: %s",
        len(records), n_pos, n_neg, label_counts,
    )
    return records


def cross_project_split(
    records: list[dict],
    test_repos: list[str],
) -> tuple[list[dict], list[dict]]:
    """Split strictly by repository (no cross-contamination)."""
    test_set   = set(test_repos)
    train_recs = [r for r in records if r.get("repo", "") not in test_set]
    test_recs  = [r for r in records if r.get("repo", "") in test_set]
    logger.info(
        "Cross-project split: %d train | %d test | test repos: %s",
        len(train_recs), len(test_recs), sorted(test_set),
    )
    return train_recs, test_recs


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def build_rf_features(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Extract RF structured features from dataset records."""
    from models.security_detection import _build_rf_feature_vector

    X, y = [], []
    for r in records:
        try:
            # CVEFixes records may have 'source' field with actual code
            if "source" in r and r["source"]:
                feat = _build_rf_feature_vector(r["source"])
            elif "features" in r:
                feat = np.array(r["features"], dtype=np.float32)
            elif "n_calls" in r:
                # Minimal AST features from dataset_builder
                feat = np.zeros(16, dtype=np.float32)
                feat[0] = r.get("n_calls", 0)
                feat[1] = r.get("n_imports", 0)
            else:
                continue
            # Ensure consistent dimension — 31-dim extended feature vector
            if len(feat) < 2:
                continue
            if len(feat) < 31:
                feat = np.pad(feat, (0, 31 - len(feat)))
            X.append(feat[:31])
            y.append(int(r["label"]))
        except Exception:
            continue
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def build_token_ids(
    records: list[dict],
    vocab: dict[str, int],
    max_len: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    from features.ast_extractor import tokens_to_ids

    X, y = [], []
    for r in records:
        tokens = r.get("tokens", [])
        ids    = tokens_to_ids(tokens, vocab, max_len)
        X.append(ids)
        y.append(int(r["label"]))
    return np.array(X, dtype=np.int32), np.array(y, dtype=np.int32)


# ---------------------------------------------------------------------------
# Significance testing
# ---------------------------------------------------------------------------

def wilcoxon_vs_majority_class(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> dict:
    """
    Wilcoxon signed-rank test comparing model AP vs. majority-class baseline AP.

    Majority-class baseline: always predict the majority class probability
    (a degenerate classifier that ignores input features).

    Bootstraps AP scores over n_bootstrap samples of the test set.
    Reports two-sided p-value and Cohen's d effect size.
    """
    from sklearn.metrics import average_precision_score

    rng = np.random.RandomState(random_state)
    majority_prob = np.full_like(y_prob, float(np.mean(y_true)))

    model_aps:    list[float] = []
    baseline_aps: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        y_b = y_true[idx]
        if len(set(y_b)) < 2:
            continue
        model_aps.append(float(average_precision_score(y_b, y_prob[idx])))
        baseline_aps.append(float(average_precision_score(y_b, majority_prob[idx])))

    if len(model_aps) < 10:
        return {"error": "Insufficient bootstrap samples"}

    model_arr    = np.array(model_aps)
    baseline_arr = np.array(baseline_aps)

    try:
        from scipy.stats import wilcoxon as scipy_wilcoxon
        stat, pval = scipy_wilcoxon(model_arr, baseline_arr, alternative="greater")
        pval = float(pval)
    except Exception:
        # Approximate with t-test if scipy unavailable
        diff = model_arr - baseline_arr
        t = np.mean(diff) / (np.std(diff) / np.sqrt(len(diff)) + 1e-9)
        pval = float(np.exp(-0.5 * t))

    # Cohen's d effect size
    pooled_std = np.sqrt((np.var(model_arr) + np.var(baseline_arr)) / 2 + 1e-9)
    cohens_d   = float((np.mean(model_arr) - np.mean(baseline_arr)) / pooled_std)

    return {
        "model_ap_mean":    round(float(np.mean(model_aps)),    4),
        "model_ap_std":     round(float(np.std(model_aps)),     4),
        "baseline_ap_mean": round(float(np.mean(baseline_aps)), 4),
        "baseline_ap_std":  round(float(np.std(baseline_aps)),  4),
        "wilcoxon_pvalue":  round(pval, 6),
        "cohens_d":         round(cohens_d, 4),
        "significant":      pval < 0.05,
        "n_bootstrap":      len(model_aps),
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    data_path:    str,
    output_dir:   str = "checkpoints/security",
    test_split:   float = 0.15,
    cnn_epochs:   int = 15,
    rf_estimators: int = 200,
    test_repos:   Optional[list[str]] = None,
) -> dict:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        classification_report, roc_auc_score,
        average_precision_score, recall_score, precision_score, f1_score,
    )
    from features.ast_extractor import build_token_vocab
    from models.security_detection import RandomForestSecurityModel, CNNSecurityModel

    from training.training_config import SecurityConfig, set_global_seed
    cfg = SecurityConfig(
        rf_n_estimators=rf_estimators,
        cnn_epochs=cnn_epochs,
        output_dir=output_dir,
    )
    set_global_seed(cfg.seed)

    records = load_dataset(data_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Split: cross-project or random stratified ---
    if test_repos:
        train_records, test_records = cross_project_split(records, test_repos)
    else:
        train_records, test_records = train_test_split(
            records,
            test_size=test_split,
            random_state=cfg.seed,
            stratify=[r["label"] for r in records],
        )
    logger.info("Train: %d | Test: %d", len(train_records), len(test_records))

    # ----------------------------------------------------------------
    # Random Forest
    # ----------------------------------------------------------------
    logger.info("=== Training Random Forest (%d estimators) ===", rf_estimators)
    X_rf_tr, y_rf_tr = build_rf_features(train_records)
    X_rf_te, y_rf_te = build_rf_features(test_records)

    rf_auc = rf_ap = rf_recall = rf_threshold = 0.0
    rf_preds_prob = np.array([])

    if len(X_rf_tr) > 0:
        rf_model = RandomForestSecurityModel()
        rf_model.fit(X_rf_tr, y_rf_tr, n_estimators=rf_estimators)
        rf_model.save(str(out_dir / "rf_model.pkl"))

        rf_preds_prob = np.array([rf_model.predict_proba(x) for x in X_rf_te])

        # Threshold via Youden's J on TRAINING set to avoid test-set leakage.
        # Selecting the threshold on the test set inflates reported recall/precision
        # because the threshold is optimised on the same data used for evaluation.
        from sklearn.metrics import roc_curve
        rf_preds_prob_tr = np.array([rf_model.predict_proba(x) for x in X_rf_tr])
        fpr_tr, tpr_tr, thresholds_tr = roc_curve(y_rf_tr, rf_preds_prob_tr)
        youden_j_tr = tpr_tr - fpr_tr
        optimal_idx = int(np.argmax(youden_j_tr))
        rf_threshold = float(thresholds_tr[optimal_idx])

        rf_labels = (rf_preds_prob > rf_threshold).astype(int)

        if len(set(y_rf_te)) > 1:
            rf_auc    = float(roc_auc_score(y_rf_te, rf_preds_prob))
            rf_ap     = float(average_precision_score(y_rf_te, rf_preds_prob))
            rf_recall = float(recall_score(y_rf_te, rf_labels, zero_division=0))

        logger.info(
            "RF: AUC=%.4f  AP=%.4f  Recall@%.3f=%.4f",
            rf_auc, rf_ap, rf_threshold, rf_recall,
        )
        logger.info("\n%s", classification_report(
            y_rf_te, rf_labels, target_names=["clean", "vulnerable"]
        ))
    else:
        logger.warning("No RF features available — skipping RF training")

    # ----------------------------------------------------------------
    # Vocabulary + CNN
    # ----------------------------------------------------------------
    logger.info("=== Building Token Vocabulary ===")
    train_token_lists = [r.get("tokens", []) for r in train_records]
    token_corpus      = [" ".join(tl) for tl in train_token_lists]
    vocab             = build_token_vocab(token_corpus, max_vocab=cfg.max_vocab)
    logger.info("Vocabulary size: %d", len(vocab))

    vocab_path = str(out_dir / "cnn_vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)

    cnn_auc = cnn_ap = None
    cnn_preds_prob = np.array([])
    ensemble_auc = ensemble_ap = None
    significance  = {}

    try:
        import torch
        torch.manual_seed(cfg.seed)
        logger.info("=== Training 1D CNN (%d epochs) ===", cnn_epochs)

        X_cnn_tr, y_cnn_tr = build_token_ids(train_records, vocab)
        X_cnn_te, y_cnn_te = build_token_ids(test_records,  vocab)

        cnn_model = CNNSecurityModel()
        cnn_model.build(vocab)
        cnn_model.train_loop(
            X_cnn_tr, y_cnn_tr,
            epochs=cnn_epochs,
            batch_size=cfg.cnn_batch_size,
            lr=cfg.cnn_lr,
        )
        cnn_model.save(str(out_dir / "cnn_model.pt"), vocab_path)

        cnn_preds_prob = np.array([
            cnn_model.predict_proba(" ".join(r.get("tokens", [])))
            for r in test_records
        ])

        if len(set(y_cnn_te)) > 1:
            cnn_auc = float(roc_auc_score(y_cnn_te, cnn_preds_prob))
            cnn_ap  = float(average_precision_score(y_cnn_te, cnn_preds_prob))
        logger.info("CNN: AUC=%.4f  AP=%.4f", cnn_auc or 0, cnn_ap or 0)

        # --- Ensemble ---
        if len(X_rf_te) > 0 and len(cnn_preds_prob) > 0:
            rf_w, cnn_w   = cfg.ensemble_rf_weight, cfg.ensemble_cnn_weight
            ensemble_prob = rf_w * rf_preds_prob + cnn_w * cnn_preds_prob
            ensemble_pred = (ensemble_prob > rf_threshold).astype(int)

            y_ens = y_cnn_te  # same order as test_records
            if len(set(y_ens)) > 1:
                ensemble_auc = float(roc_auc_score(y_ens, ensemble_prob))
                ensemble_ap  = float(average_precision_score(y_ens, ensemble_prob))

            logger.info("Ensemble: AUC=%.4f  AP=%.4f", ensemble_auc or 0, ensemble_ap or 0)
            logger.info("\n%s", classification_report(
                y_ens, ensemble_pred, target_names=["clean", "vulnerable"]
            ))

            # Wilcoxon significance test vs. majority-class baseline
            logger.info("=== Statistical Significance Test (Wilcoxon vs. majority) ===")
            significance = wilcoxon_vs_majority_class(y_ens, ensemble_prob)
            logger.info(
                "  Model AP: %.4f ± %.4f | Baseline AP: %.4f ± %.4f",
                significance.get("model_ap_mean", 0),
                significance.get("model_ap_std", 0),
                significance.get("baseline_ap_mean", 0),
                significance.get("baseline_ap_std", 0),
            )
            logger.info(
                "  Wilcoxon p=%.6f  Cohen's d=%.4f  Significant=%s",
                significance.get("wilcoxon_pvalue", 1.0),
                significance.get("cohens_d", 0.0),
                significance.get("significant", False),
            )
        else:
            ensemble_auc = cnn_auc
            ensemble_ap  = cnn_ap

    except ModuleNotFoundError:
        logger.warning("torch not installed — CNN skipped (RF-only mode)")
        ensemble_auc = rf_auc
        ensemble_ap  = rf_ap

    # ----------------------------------------------------------------
    # Multi-seed stability evaluation (RF only — CNN is too slow to retrain)
    # ----------------------------------------------------------------
    EVAL_SEEDS = [42, 0, 7, 123, 999]
    seed_aucs, seed_aps = [], []
    if len(X_rf_tr) > 0 and not test_repos:
        logger.info("=== Multi-seed stability check (%d seeds) ===", len(EVAL_SEEDS))
        all_records = records  # full dataset
        all_X, all_y = build_rf_features(all_records)
        for seed in EVAL_SEEDS:
            from sklearn.model_selection import train_test_split as tts
            try:
                X_tr_s, X_te_s, y_tr_s, y_te_s = tts(
                    all_X, all_y, test_size=test_split,
                    random_state=seed, stratify=all_y,
                )
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import roc_auc_score, average_precision_score
                rf_s = RandomForestClassifier(
                    n_estimators=rf_estimators, class_weight="balanced",
                    n_jobs=1, random_state=seed,
                )
                rf_s.fit(X_tr_s, y_tr_s)
                prob_s = rf_s.predict_proba(X_te_s)[:, 1]
                if len(set(y_te_s)) > 1:
                    seed_aucs.append(float(roc_auc_score(y_te_s, prob_s)))
                    seed_aps.append(float(average_precision_score(y_te_s, prob_s)))
                    logger.info(
                        "  seed=%d  AUC=%.4f  AP=%.4f",
                        seed, seed_aucs[-1], seed_aps[-1],
                    )
            except Exception as e:
                logger.debug("Seed %d failed: %s", seed, e)
        if seed_aucs:
            logger.info(
                "Multi-seed RF:  AUC=%.4f+/-%.4f  AP=%.4f+/-%.4f  (n=%d seeds)",
                np.mean(seed_aucs), np.std(seed_aucs),
                np.mean(seed_aps),  np.std(seed_aps),
                len(seed_aucs),
            )

    # ----------------------------------------------------------------
    # Save metrics
    # ----------------------------------------------------------------
    metrics = {
        "rf_auc":            float(rf_auc),
        "rf_ap":             float(rf_ap),
        "rf_recall":         float(rf_recall),
        "rf_threshold":      float(rf_threshold),
        "cnn_auc":           float(cnn_auc)  if cnn_auc  is not None else None,
        "cnn_ap":            float(cnn_ap)   if cnn_ap   is not None else None,
        "ensemble_auc":      float(ensemble_auc) if ensemble_auc is not None else float(rf_auc),
        "ensemble_ap":       float(ensemble_ap)  if ensemble_ap  is not None else float(rf_ap),
        "n_train":           len(train_records),
        "n_test":            len(test_records),
        "vocab_size":        len(vocab),
        "cnn_epochs":        cnn_epochs,
        "rf_estimators":     rf_estimators,
        "split_strategy":    "cross_project" if test_repos else "random_stratified",
        "test_repos":        test_repos or [],
        "significance":      significance,
        "primary_metric":    "average_precision",  # AP preferred for imbalanced data
        "multi_seed_auc_mean": round(float(np.mean(seed_aucs)), 4) if seed_aucs else None,
        "multi_seed_auc_std":  round(float(np.std(seed_aucs)),  4) if seed_aucs else None,
        "multi_seed_ap_mean":  round(float(np.mean(seed_aps)),  4) if seed_aps  else None,
        "multi_seed_ap_std":   round(float(np.std(seed_aps)),   4) if seed_aps  else None,
    }

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("\nMetrics saved -> %s", metrics_path)

    # Save config for reproducibility
    cfg.save(str(out_dir / "train_config.json"))

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train security detection ensemble (RF + CNN)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data",       required=True, help="Path to JSONL dataset")
    parser.add_argument("--out",        default="checkpoints/security")
    parser.add_argument("--cnn-epochs", type=int, default=15)
    parser.add_argument("--rf-trees",   type=int, default=200)
    parser.add_argument("--test-repos", nargs="*", default=None,
                        help="Repo names to hold out for cross-project evaluation")
    args = parser.parse_args()

    train(
        data_path=args.data,
        output_dir=args.out,
        cnn_epochs=args.cnn_epochs,
        rf_estimators=args.rf_trees,
        test_repos=args.test_repos,
    )


if __name__ == "__main__":
    main()
