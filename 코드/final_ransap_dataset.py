import pandas as pd

CAPE_CSV = r"C:\Users\rladb\Desktop\cape_ransap_entropy_features.csv"
MORDOR_CSV = r"C:\Users\rladb\Desktop\mordor_ransap_features.csv"
OUTPUT = r"C:\Users\rladb\Desktop\final_ransap_dataset.csv"

print("[+] Loading CSV files...")

cape_df = pd.read_csv(CAPE_CSV)
mordor_df = pd.read_csv(MORDOR_CSV)

print("[+] CAPE samples:", len(cape_df))
print("[+] Mordor samples:", len(mordor_df))

# 두 데이터셋 모두 동일한 RanSAP feature 컬럼을 가짐
final_df = pd.concat([cape_df, mordor_df], axis=0, ignore_index=True)

print("[+] Total samples:", len(final_df))

# 결측치 처리 (없어도 안전)
final_df = final_df.fillna(0)

final_df.to_csv(OUTPUT, index=False)

print("[✓] Final dataset saved:", OUTPUT)
