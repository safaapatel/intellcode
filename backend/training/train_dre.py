"""
Train Differential Risk Encoder (DRE)
Usage:
    cd backend
    python training/train_dre.py --data data/bug_dataset.jsonl
Outputs: checkpoints/dre/dre.pkl, checkpoints/dre/metrics.json
"""
from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

STATIC_FEAT_KEYS = [
    "cyclomatic_complexity","cognitive_complexity","max_function_complexity",
    "avg_function_complexity","sloc","comments","blank_lines",
    "halstead_volume","halstead_difficulty","halstead_effort","halstead_bugs",
    "n_long_functions","n_complex_functions","max_line_length","avg_line_length",
]  # 15-dim


def _rec_to_static(rec: dict) -> np.ndarray | None:
    sf = rec.get("static_features", [])
    if len(sf) >= 15:
        return np.array(sf[:15], dtype=np.float32)
    gf = rec.get("git_features", {})
    vals = [float(rec.get(k, gf.get(k, 0))) for k in STATIC_FEAT_KEYS]
    if any(v != 0 for v in vals):
        return np.array(vals, dtype=np.float32)
    return None


def build_pairs(data_path: str) -> tuple[list, np.ndarray]:
    """Group records by (repo, file), sort by timestamp, build consecutive pairs."""
    from collections import defaultdict
    groups: dict[str, list] = defaultdict(list)

    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = f"{rec.get('repo','')}/{rec.get('file_path','')}"
            ts = rec.get("author_date_unix_timestamp", 0) or 0
            label = int(rec.get("label", rec.get("is_buggy", 0)))
            feat = _rec_to_static(rec)
            if feat is not None:
                groups[key].append((ts, feat, label))

    pairs, labels = [], []
    for recs in groups.values():
        recs.sort(key=lambda x: x[0])
        for i in range(1, len(recs)):
            _, x_prev, _ = recs[i-1]
            _, x_curr, lbl = recs[i]
            pairs.append((x_curr, x_prev))
            labels.append(lbl)

    if not pairs:
        # Fallback: treat each record as first-commit (prev = zeros)
        logger.info("No consecutive pairs found — using first-commit mode")
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                rec = json.loads(line)
                feat = _rec_to_static(rec)
                if feat is None: continue
                lbl = int(rec.get("label", rec.get("is_buggy", 0)))
                pairs.append((feat, np.zeros_like(feat)))
                labels.append(lbl)

    logger.info("DRE dataset: %d pairs  (%.1f%% buggy)", len(pairs), 100*np.mean(labels))
    return pairs, np.array(labels, dtype=np.float32)


def train(data_path: str, output_dir: str = "checkpoints/dre") -> dict:
    from models.differential_risk_encoder import DifferentialRiskEncoder

    pairs, y = build_pairs(data_path)
    n = len(pairs)
    n_test = max(1, int(n * 0.2))
    pairs_tr, pairs_te = pairs[:-n_test], pairs[-n_test:]
    y_tr, y_te = y[:-n_test], y[-n_test:]

    model = DifferentialRiskEncoder()
    metrics = model.fit(pairs_tr, y_tr, output_dir=output_dir)

    # Evaluate
    from models.differential_risk_encoder import build_delta_features
    X_te_delta = np.array([build_delta_features(xc, xp) for xc, xp in pairs_te], dtype=np.float32)
    if model._scaler_mean is not None:
        mean = model._scaler_mean
        std  = model._scaler_std
        if X_te_delta.shape[1] > len(mean):
            mean = np.pad(mean, (0, X_te_delta.shape[1] - len(mean)))
            std  = np.pad(std,  (0, X_te_delta.shape[1] - len(std)), constant_values=1.0)
        elif X_te_delta.shape[1] < len(mean):
            X_te_delta = X_te_delta[:, :len(mean)]
        X_te_scaled = (X_te_delta - mean) / std
    else:
        X_te_scaled = X_te_delta

    preds = model._predict_scaled(X_te_scaled)
    auc = DifferentialRiskEncoder._roc_auc(y_te, preds)

    eval_metrics = {**metrics, "test_auc": round(auc, 4), "n_test": n_test}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)

    logger.info("DRE test AUC=%.4f  model=%s", auc, metrics.get("model", "?"))
    return eval_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/bug_dataset.jsonl")
    p.add_argument("--out",  default="checkpoints/dre")
    args = p.parse_args()

    if not Path(args.data).exists():
        logger.error("Dataset not found: %s", args.data)
        sys.exit(1)

    metrics = train(args.data, args.out)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
