# view_npy.py — Visualize .npy volume data with Napari
# This script is used to inspect the effect of preprocessing on 3D medical images.

import numpy as np
import napari

# === Modify this path to point to the file you want to visualize ===
NPU_PATH = "preprocessed_vols/1_Reg_-_DWI_SENSE_label0.npy"

# 1) Load .npy volume file
vol = np.load(NPU_PATH)
print("Loaded", NPU_PATH, "shape:", vol.shape, "dtype:", vol.dtype)

# 2) Launch Napari viewer
viewer = napari.Viewer(title=NPU_PATH)

# If the data has shape (C, D, H, W), show only the first channel; otherwise show directly
if vol.ndim == 4:
    viewer.add_image(vol[0], name="volume")
else:
    viewer.add_image(vol, name="volume")

napari.run()
