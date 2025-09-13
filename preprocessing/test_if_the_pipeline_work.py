import pandas as pd
import numpy as np
import nibabel as nib
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import napari
from create_volumes import stacking2D, save_as_nifti, run_hd_bet
import helper

# ---------- CONFIG ----------
LABEL_FILE = "../labelled_patients.csv"
TMP_DIR = "tmp"
OUT_DIR = "volumes/test"
INDEX = 0           # 改这里选择测试样本
VISUALISE = True    # 是否使用 napari 查看每个阶段
# ----------------------------

os.makedirs(TMP_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(LABEL_FILE)
row = df.iloc[INDEX]

subject_id = row['SubjectID']
label = row['Class']
modality = row['Type']
file_path = row['FilePath']

print(f"▶ Processing Subject {subject_id} ({modality}): {file_path}")

# Step 1: stacking 2D to volume
vol_dict = stacking2D(file_path)
if not vol_dict:
    print("[❌] Stacking failed")
    exit()

for seq, volume in vol_dict.items():
    print(f"✔ Stacked volume shape: {volume.shape}")
    helper.debug_pipeline("stacking2D", volume, visualise=VISUALISE)

    # Step 2: Save raw NIfTI
    safe_seq = seq.replace('/', '_').replace('\\', '_')
    raw_nii_path = os.path.join(TMP_DIR, f"{subject_id}_{safe_seq}_raw.nii.gz")
    save_as_nifti(volume, raw_nii_path)

    # Step 3: Skull Stripping
    stripped_nii_path = os.path.join(TMP_DIR, f"{subject_id}_{safe_seq}_stripped.nii.gz")
    run_hd_bet(raw_nii_path, stripped_nii_path)

    # Step 4: Load stripped NIfTI
    if not os.path.exists(stripped_nii_path):
        print("[❌] HD-BET failed, skipping this sample.")
        continue

    stripped_volume = nib.load(stripped_nii_path).get_fdata()
    helper.debug_pipeline("HD-BET output", stripped_volume, visualise=VISUALISE)

    # Step 5: Intensity Normalisation
    if modality == "MRI":
        norm = helper.z_score_norm(stripped_volume, np.mean(stripped_volume), np.std(stripped_volume))
    elif modality == "DAT":
        norm = helper.min_max_norm(stripped_volume, np.max(stripped_volume), np.min(stripped_volume))
    else:
        print("[❌] Unknown modality")
        continue

    helper.debug_pipeline("Normalised", norm, visualise=VISUALISE)

    # Step 6: Save
    norm = np.expand_dims(norm, axis=0)
    out_name = f"volume_{subject_id}_{modality}_{safe_seq}_label{label}.npy"
    out_path = os.path.join(OUT_DIR, out_name)
    np.save(out_path, norm)
    print(f"[✓] Saved: {out_name}, shape={norm.shape}")
