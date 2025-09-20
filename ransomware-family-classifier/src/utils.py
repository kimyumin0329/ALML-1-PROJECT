import os
import json
import yaml
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, f1_score, confusion_matrix

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df

def prepare_labels(df: pd.DataFrame, target_col: str, label_min_count: int,
                   use_other: bool, other_label_name: str) -> pd.Series:
    # Handle rare classes
    counts = df[target_col].value_counts()
    if label_min_count > 0:
        rare = set(counts[counts < label_min_count].index)
        if use_other:
            df[target_col] = df[target_col].apply(lambda x: other_label_name if x in rare else x)
        else:
            df = df[~df[target_col].isin(rare)].copy()
    return df[target_col], df

def split_data(df: pd.DataFrame, X_cols: List[str], y: pd.Series,
               test_size: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df[X_cols]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def encode_labels(y_train: pd.Series, y_test: pd.Series) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc, le

def get_class_weights(y_enc: np.ndarray) -> Dict[int, float]:
    counts = Counter(y_enc)
    total = sum(counts.values())
    weights = {cls: total/(len(counts)*cnt) for cls, cnt in counts.items()}
    return weights

def save_artifacts(output_dir: str, model, label_encoder: LabelEncoder, feature_cols: List[str]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, "model_lgbm.pkl"))
    joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.pkl"))
    with open(os.path.join(output_dir, "features.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)

def load_artifacts(output_dir: str):
    model = joblib.load(os.path.join(output_dir, "model_lgbm.pkl"))
    le = joblib.load(os.path.join(output_dir, "label_encoder.pkl"))
    with open(os.path.join(output_dir, "features.json"), "r") as f:
        feats = json.load(f)
    return model, le, feats

def evaluation_report(y_true_enc, y_pred_enc, label_encoder) -> str:
    report = classification_report(y_true_enc, y_pred_enc, target_names=label_encoder.classes_)
    return report

def macro_f1(y_true_enc, y_pred_enc) -> float:
    return f1_score(y_true_enc, y_pred_enc, average="macro")

def compute_confusion(y_true_enc, y_pred_enc) -> np.ndarray:
    return confusion_matrix(y_true_enc, y_pred_enc)
