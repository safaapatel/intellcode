import json
from pathlib import Path

print("=== Real Datasets ===")
for name in ["security", "complexity", "bug", "pattern"]:
    p = Path(f"data/{name}_dataset.jsonl")
    if p.exists():
        lines = [l for l in p.read_text(encoding="utf-8").strip().splitlines() if l]
        has_origin = sum(1 for l in lines if "origin" in l)
        print(f"  {name}: {len(lines)} samples, {has_origin} with real origin")
    else:
        print(f"  {name}: MISSING")

print()
print("=== Model Checkpoints (Metrics) ===")
for name, path in [
    ("Security", "checkpoints/security/metrics.json"),
    ("Complexity", "checkpoints/complexity/metrics.json"),
    ("Bug", "checkpoints/bug_predictor/metrics.json"),
]:
    p = Path(path)
    if p.exists():
        m = json.loads(p.read_text())
        if name == "Security":
            print(f"  {name}: RF_AUC={m.get('rf_auc', '?'):.3f}, "
                  f"CNN_AUC={m.get('cnn_auc', 0):.3f}, "
                  f"Ensemble_AUC={m.get('ensemble_auc', '?'):.3f}")
        elif name == "Complexity":
            print(f"  {name}: R2={m.get('test_r2', '?'):.4f}, "
                  f"RMSE={m.get('test_rmse', '?'):.3f}")
        elif name == "Bug":
            lr  = m.get("logistic_regression", {})
            xgb = m.get("xgboost", {})
            mlp = m.get("mlp", {})
            ens = m.get("ensemble_auc", m.get("test_auc", "?"))
            print(f"  {name}: LR_AUC={lr.get('test_auc','?')}, "
                  f"XGB_AUC={xgb.get('test_auc','?')}, "
                  f"MLP_AUC={mlp.get('test_auc','?')}, "
                  f"Ensemble_AUC={ens}")
    else:
        print(f"  {name}: checkpoint missing at {path}")
