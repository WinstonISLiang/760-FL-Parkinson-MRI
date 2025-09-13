# test_preprocess.py — Main script to run full MRI preprocessing pipeline
# This script executes volume stacking, filtering, normalization, and saves preprocessed .npy files.

import os
from pathlib import Path
from create_volumes_v2 import preprocess_all_volumes

# === Configurable Parameters ===
LABEL_CSV    = "../labelled_patients.csv"   # Path to CSV file containing labels and file paths
OUT_DIR      = "preprocessed_vols"          # Directory where output .npy files will be saved
ORIG_SPACING = (3.0, 2.0, 2.0)              # Original voxel spacing (z, y, x) in mm
TARGET_SHAPE = (64, 128, 128)               # Desired (D, H, W) shape for output volumes
USE_16BIT    = True                         # Whether to output volumes as uint16
STD_THRESH   = 0.1                          # Minimum std threshold to exclude blank slices
# ===============================

def main():
    print("➤ Starting batch preprocessing …")
    preprocess_all_volumes(out_dir          = OUT_DIR,
                           label_file       = LABEL_CSV,
                           original_spacing = ORIG_SPACING,
                           target_shape     = TARGET_SHAPE,
                           use_16bit        = USE_16BIT,
                           std_threshold    = STD_THRESH)

    # List results in the output directory
    out_path = Path(OUT_DIR)
    if out_path.exists():
        files = sorted(out_path.glob("*.npy"))
        print(f"\n✓ Preprocessing complete. Generated {len(files)} .npy files:")
        for f in files[:10]:  # Preview first 10 files
            print("  •", f.name)
        if len(files) > 10:
            print("  …")
    else:
        print("⚠️  Output directory not found — all samples may have been skipped.")

if __name__ == "__main__":
    main()
