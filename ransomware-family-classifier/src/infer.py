import os
import json
import argparse
import numpy as np
import pandas as pd
from utils import load_config, load_artifacts

def main(config_path: str, input_csv: str, topk: int):
    cfg = load_config(config_path)
    output_dir = cfg.get("output_dir", "./artifacts")
    model, le, feats = load_artifacts(output_dir)

    df = pd.read_csv(input_csv)
    # If input contains target columns or id columns, ignore them safely
    cols_available = [c for c in feats if c in df.columns]
    missing = [c for c in feats if c not in df.columns]
    if missing:
        print(f"[WARN] Missing {len(missing)} features not found in input. Filling with -1.")
    for m in missing:
        df[m] = -1

    X = df[feats].fillna(-1)

    proba = model.predict(X, num_iteration=getattr(model, "best_iteration", None))
    preds = proba.argmax(axis=1)
    labels = le.inverse_transform(preds)

    # Top-k per row
    topk = max(1, int(topk))
    topk_indices = np.argsort(-proba, axis=1)[:, :topk]
    topk_labels = [[le.classes_[j] for j in row] for row in topk_indices]
    topk_scores = [[float(proba[i, j]) for j in row] for i, row in enumerate(topk_indices)]

    out = df.copy()
    out["pred_family"] = labels
    out["pred_topk_labels"] = topk_labels
    out["pred_topk_scores"] = topk_scores

    out_path = os.path.join(output_dir, "inference_results.csv")
    out.to_csv(out_path, index=False)
    print(f"Inference done. Saved to {out_path}")
    print("Preview:")
    print(out.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--input", type=str, required=True, help="CSV file with same schema as training features")
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()
    main(args.config, args.input, args.topk)
