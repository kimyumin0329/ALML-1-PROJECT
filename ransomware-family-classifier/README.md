# Ransomware Family Classifier (Scenario C)

VS Code-ready project to train a ransomware **Family** classifier from a CSV like `ransom.csv`.
It uses LightGBM with stratified split, class-weighting, and Macro-F1 evaluation.
Artifacts (model + encoders) are saved under `artifacts/`.

## Quickstart

```bash
# 1) Create venv (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Put your dataset (e.g., ransom.csv) under the project root
#    or point to it in configs/config.yaml

# 4) Train
python src/train.py --config configs/config.yaml

# 5) Inference on a CSV with the same schema (or a subset with target Family missing)
python src/infer.py --config configs/config.yaml --input sample_infer.csv --topk 5
```

## Notes
- We drop identifier columns like `md5`, `sha1` from features.
- We keep both static(PE) and dynamic(behavior) features from the CSV.
- You may set `label_min_count` to group rare families into "Other" or drop them.
- For production, replace the CSV-based inference with real **feature extraction** from uploaded binaries.
