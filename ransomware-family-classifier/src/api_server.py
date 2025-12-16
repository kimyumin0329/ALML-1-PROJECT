from fastapi import FastAPI
from fastapi import Body
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.utils import load_config, load_artifacts


app = FastAPI(title="Ransomware Family Classifier")

CONFIG_PATH = "configs/config.yaml"
cfg = load_config(CONFIG_PATH)
OUTPUT_DIR = cfg.get("output_dir", "./artifacts")

model, le, feats = load_artifacts(OUTPUT_DIR)

class PredictRequest(BaseModel):
    # 1) 권장: {"features": {...}, "topk": 5}
    features: Optional[Dict[str, Any]] = None
    topk: int = 5

    # 2) 호환: features 없이도, 바디에 바로 key/value 넣으면 처리(추후 편하려고)
    class Config:
        extra = "allow"

class PredictResponseItem(BaseModel):
    family: str
    prob: float

class PredictResponse(BaseModel):
    topk: List[PredictResponseItem]
    message: Optional[str] = None

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # features가 없으면 바디의 나머지(extra)를 features로 취급
    features = req.features
    if features is None:
        raw = req.model_dump()
        raw.pop("features", None)
        raw.pop("topk", None)
        features = raw

    # 모델이 기대하는 feature list 기준으로 정렬 + 누락은 -1로 채움
    missing = [c for c in feats if c not in features]
    row = {c: features.get(c, -1) for c in feats}

    X = pd.DataFrame([row]).fillna(-1)

    proba = model.predict(X, num_iteration=getattr(model, "best_iteration", None))
    proba = np.array(proba)
    if proba.ndim == 1:
        proba = proba.reshape(1, -1)

    topk = max(1, int(req.topk))
    idxs = np.argsort(-proba[0])[:topk]

    items = []
    for i in idxs:
        fam = le.classes_[i]
        items.append(PredictResponseItem(family=str(fam), prob=float(proba[0, i])))

    msg = None
    if missing:
        msg = f"Missing {len(missing)} features filled with -1."

    return PredictResponse(topk=items, message=msg)

@app.post("/api/analyze")
def api_analyze(payload: Dict[str, Any] = Body(...)):
    # Spring이 보내는 AiPayload(snake_case) 그대로를 features로 사용
    pred = predict(PredictRequest(features=payload, topk=5))

    top1 = pred.topk[0] if pred.topk else None
    score = float(top1.prob) if top1 else 0.0
    fam = top1.family if top1 else "UNKNOWN"

    # 임계값(일단 동작용, 나중에 튜닝 가능)
    if score >= 0.85:
        label = "DANGER"
    elif score >= 0.60:
        label = "WARNING"
    else:
        label = "SAFE"

    return {
        "status": "ok",
        "label": label,
        "score": score,
        "detail": f"top_family={fam}",
        "message": pred.message,
    }
