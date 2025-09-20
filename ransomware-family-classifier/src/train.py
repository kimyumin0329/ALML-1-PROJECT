import os
import json
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from utils import (
    load_config, load_dataset, prepare_labels, split_data, encode_labels,
    get_class_weights, save_artifacts, evaluation_report, macro_f1
)

EXCLUDE_TYPES = ["object"]  # non-numeric columns will be auto-handled

def build_feature_list(df: pd.DataFrame, drop_cols):
    cols = [c for c in df.columns if c not in drop_cols]
    # Keep numeric columns only (LightGBM can handle categorical via category dtype,
    # but here we assume most features are numeric; extend if needed).
    num_cols = df[cols].select_dtypes(include=["number", "bool"]).columns.tolist()
    return num_cols

def main(config_path: str):
    print("[DEBUG] main() 시작")   # 추가

    cfg = load_config(config_path)
    print("[DEBUG] config 로드 완료")

    csv_path = cfg["dataset_path"]
    print(f"[DEBUG] dataset_path = {csv_path}")

    df = load_dataset(csv_path)
    print(f"[DEBUG] 데이터셋 shape = {df.shape}")
    print(df.head())

    cfg = load_config(config_path)
    csv_path = cfg["dataset_path"]
    target_col = cfg["target_col"]
    drop_cols = set(cfg.get("drop_cols", [])) | {target_col}
    label_min_count = int(cfg.get("label_min_count", 0))
    use_other = bool(cfg.get("use_other_class", True))
    other_label_name = cfg.get("other_label_name", "Other")
    test_size = float(cfg.get("test_size", 0.2))
    random_state = int(cfg.get("random_state", 42))
    n_splits = int(cfg.get("n_splits", 5))
    lgbm_params = cfg.get("lgbm_params", {})
    output_dir = cfg.get("output_dir", "./artifacts")

    df = load_dataset(csv_path)

    # Drop obvious identifier columns if present in dataset
    # (The config already includes common ones; extend as needed)
    # Prepare labels (handles rare families -> 'Other' or drop)
    y, df = prepare_labels(df, target_col, label_min_count, use_other, other_label_name)

    # Build features list
    feature_cols = build_feature_list(df, drop_cols)
    if len(feature_cols) == 0:
        raise RuntimeError("No numeric feature columns found. Check drop_cols or dataset schema.")

    # Handle NaN
    df[feature_cols] = df[feature_cols].fillna(-1)

    # Split
    X_train, X_test, y_train, y_test = split_data(df, feature_cols, y, test_size, random_state)

    # Label encoding
    y_train_enc, y_test_enc, le = encode_labels(y_train, y_test)

    # Class weighting (inverse frequency) if enabled
    weights = None
    if cfg.get("class_weighting", True):
        class_weights = get_class_weights(y_train_enc)
        weights = np.array([class_weights[y] for y in y_train_enc])


    # LightGBM dataset
    num_classes = len(le.classes_)
    lgbm_params["num_class"] = num_classes
    lgbm_params.setdefault("objective", "multiclass")

    dtrain = lgb.Dataset(X_train, label=y_train_enc,weight=weights, free_raw_data=False)
    dvalid = lgb.Dataset(X_test, label=y_test_enc, reference=dtrain, free_raw_data=False)

    # Train
    callbacks = [
    lgb.early_stopping(stopping_rounds=100),
    lgb.log_evaluation(period=50)  # <- 예전 verbose_eval=50 역할
    ]

    model = lgb.train(
    lgbm_params,
    dtrain,
    valid_sets=[dtrain, dvalid],
    valid_names=["train", "valid"],
    callbacks=callbacks
    )

    # Predict & evaluate
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = y_pred_proba.argmax(axis=1)

    print("\n=== Evaluation (Holdout) ===")
    print(evaluation_report(y_test_enc, y_pred, le))
    print(f"Macro-F1: {macro_f1(y_test_enc, y_pred):.4f}")

    # Save artifacts
    save_artifacts(output_dir, model, le, feature_cols)
    with open(os.path.join(output_dir, "label_classes.json"), "w") as f:
        json.dump(le.classes_.tolist(), f, indent=2)

    # Save a small feature-importance report
    importances = model.feature_importance(importance_type="gain")
    fi = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)[:50]
    with open(os.path.join(output_dir, "feature_importance_top50.json"), "w") as f:
        json.dump([{ "feature": k, "gain": float(v)} for k, v in fi], f, indent=2)

    print(f"\nArtifacts saved to: {output_dir}")

if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument("--config", type=str, default="configs/config.yaml")
     args = parser.parse_args()
     main(args.config)
