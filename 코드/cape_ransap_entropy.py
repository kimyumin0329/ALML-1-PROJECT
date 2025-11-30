#carpe_json 폴더의 JSON 모두 로딩

#엔트로피 기반 RanSAP 파생 특징 생성

#ransomware / benign / packed category 분리

#family 라벨 삽입

#최종 CSV: cape_ransap_entropy_features.csv 생성


import os
import json
import pandas as pd

# CAPE JSON이 들어있는 절대 경로
INPUT_DIR = r"C:\Users\rladb\Desktop\carpe_json"

def extract_entropy_features(report):
    static = report.get("static", {}) or {}
    sections = static.get("pe_sections", []) or []

    # 엔트로피 값만 추출
    ent = [float(s.get("entropy")) for s in sections if s.get("entropy") is not None]

    if not ent:
        return {
            "pe_avg_entropy": 0.0,
            "pe_max_entropy": 0.0,
            "pe_high_entropy_section_ratio": 0.0,
            "pe_has_high_entropy_section": 0
        }

    avg = sum(ent) / len(ent)
    mx = max(ent)
    high = sum(1 for e in ent if e >= 7.2)
    ratio = high / len(ent)

    return {
        "pe_avg_entropy": avg,
        "pe_max_entropy": mx,
        "pe_high_entropy_section_ratio": ratio,
        "pe_has_high_entropy_section": 1 if high > 0 else 0
    }

rows = []

for fname in os.listdir(INPUT_DIR):
    if not fname.endswith(".json"):
        continue

    fpath = os.path.join(INPUT_DIR, fname)

    with open(fpath, "r", encoding="utf-8") as f:
        report = json.load(f)

    feats = extract_entropy_features(report)

    # label 정보 추가
    info = report.get("info", {})
    meta = report.get("metadata", {})

    feats["sample_id"] = fname
    feats["category"] = info.get("category")
    feats["family"] = meta.get("family")
    rows.append(feats)

df = pd.DataFrame(rows)
df.to_csv("cape_ransap_entropy_features.csv", index=False)

print("[+] cape_ransap_entropy_features.csv 생성 완료!")
