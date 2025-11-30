import os, json

INPUT_DIR = r"c:\Users\rladb\Desktop\carpe_json"


for fname in os.listdir(INPUT_DIR):
    print("[+] Checking", fname)
    try:
        with open(os.path.join(INPUT_DIR, fname), "r", encoding="utf-8") as f:
            json.load(f)
        print("   → OK")
    except Exception as e:
        print("   → ERROR:", e)
