import os
import requests
import zipfile
import json
import pandas as pd
from collections import Counter, defaultdict

# ==========================================
# 1. codeload 기반 Mordor ZIP 다운로드 URL
# ==========================================
MORDOR_URLS = [
    "https://codeload.github.com/OTRF/Security-Datasets/legacy.zip/master?path=datasets/small/windows/collection/msf_record_mic_2020-06-09225055",
    "https://codeload.github.com/OTRF/Security-Datasets/legacy.zip/master?path=datasets/small/windows/collection/atomic_T1119_auto_data_access_2020-07-15220533",
    "https://codeload.github.com/OTRF/Security-Datasets/legacy.zip/master?path=datasets/small/windows/collection/atomic_T1005_local_data_access_2020-07-17164921",
    "https://codeload.github.com/OTRF/Security-Datasets/legacy.zip/master?path=datasets/small/windows/collection/empire_data_access_2020-06-09142250",
    "https://codeload.github.com/OTRF/Security-Datasets/legacy.zip/master?path=datasets/small/windows/execution/msf_shell_psh_2020-06-01103042",
    "https://codeload.github.com/OTRF/Security-Datasets/legacy.zip/master?path=datasets/small/windows/execution/atomic_T1059_powershell_exec_2020-07-15215920",
    "https://codeload.github.com/OTRF/Security-Datasets/legacy.zip/master?path=datasets/small/windows/execution/empire_native_api_exec_2020-06-09152250",
    "https://codeload.github.com/OTRF/Security-Datasets/legacy.zip/master?path=datasets/small/windows/persistence/empire_registry_runkey_2020-06-09184520",
    "https://codeload.github.com/OTRF/Security-Datasets/legacy.zip/master?path=datasets/small/windows/persistence/atomic_T1112_reg_mod_2020-07-15223521",
    "https://codeload.github.com/OTRF/Security-Datasets/legacy.zip/master?path=datasets/small/windows/impact/atomic_T1490_inhibit_system_recovery_2020-07-10153102"
]

DOWNLOAD_DIR = "mordor_zips"
EXTRACT_DIR = "mordor_extracted"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(EXTRACT_DIR, exist_ok=True)


# ==========================================
# 2. ZIP download
# ==========================================
def download_zip(url):
    fname = os.path.join(DOWNLOAD_DIR, url.split("?path=")[-1].replace("/", "_") + ".zip")
    if not os.path.exists(fname):
        print(f"[+] Downloading: {fname}")
        r = requests.get(url, stream=True)
        with open(fname, "wb") as f:
            f.write(r.content)
    return fname


# ==========================================
# 3. unzip
# ==========================================
def unzip_file(zip_path):
    print(f"[+] Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(EXTRACT_DIR)


# ==========================================
# 4. JSON 파서
# ==========================================
def parse_json_logs(path):
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except:
                pass
    return events


# ==========================================
# 5. RanSAP 특징 14개 계산
# ==========================================
def compute_ransap_features(events):
    c = Counter()
    touched = set()
    parent_map = defaultdict(str)

    for e in events:
        eid = int(e.get("EventID", -1))

        # File Write/Create
        if eid == 11:
            c["file_write_count"] += 1
            touched.add(e.get("TargetFilename", ""))

        # File Rename
        elif eid == 13:
            c["file_rename_count"] += 1
            touched.add(e.get("TargetFilename", ""))

        # Registry modification
        elif eid == 12:
            c["registry_mod_count"] += 1

        # Process Create
        elif eid == 1:
            c["process_spawn_count"] += 1
            parent_map[e.get("ProcessId")] = e.get("ParentImage", "")

    feats = {
        "file_write_count": c["file_write_count"],
        "file_rename_count": c["file_rename_count"],
        "registry_mod_count": c["registry_mod_count"],
        "process_spawn_count": c["process_spawn_count"],
        "touched_file_count": len(touched),
    }

    # Shadow copy delete
    shadow_kw = ["vssadmin", "shadowcopy", "wmic shadowcopy"]
    feats["shadow_copy_delete"] = 1 if any(kw in str(events).lower() for kw in shadow_kw) else 0

    # Rapid multi-file access
    feats["rapid_multi_file_access"] = 1 if (feats["file_write_count"] + feats["file_rename_count"]) > 50 else 0

    # Abnormal parent process
    normal = ["explorer.exe", "svchost.exe", "services.exe"]
    feats["abnormal_parent_process"] = 1 if any(p not in normal for p in parent_map.values()) else 0

    return feats


# ==========================================
# 6. 전체 실행
# ==========================================
print("[+] Starting Mordor pipeline...")

# Download & unzip
for url in MORDOR_URLS:
    z = download_zip(url)
    unzip_file(z)

# JSON 탐색 (중첩 폴더까지 검색)
feature_rows = []
for root, dirs, files in os.walk(EXTRACT_DIR):
    for fname in files:
        if fname.endswith(".json"):
            full_path = os.path.join(root, fname)
            try:
                events = parse_json_logs(full_path)
                feats = compute_ransap_features(events)
                feats["sample_id"] = fname
                feature_rows.append(feats)
                print(f"[OK] Extracted: {fname}")
            except:
                print(f"[ERROR] Failed: {fname}")

# Save CSV
df = pd.DataFrame(feature_rows)
df.to_csv("mordor_ransap_features.csv", index=False)
print("\n[+] Done! → mordor_ransap_features.csv 생성 완료")
