from create_volumes import stacking2D
import pandas as pd
import numpy as np

input_file = "../labelled_patients.csv"

df = pd.read_csv(input_file)

for _, row in df.iterrows():
    file_path = row['FilePath']
    volumes = stacking2D(file_path)

    if not volumes:
        print(f"[❌] Stacking failed for: {file_path}")
        break

    for seq_name, vol in volumes.items():
        print(f"[{seq_name}] shape={vol.shape}, dtype={vol.dtype}")
        print(f"  min={np.min(vol):.4f}, max={np.max(vol):.4f}, mean={np.mean(vol):.4f}, std={np.std(vol):.4f}")
    break
