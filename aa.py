import cv2
import numpy as np

# Load an image
img_path = "ntua-parkinson-dataset/pd-patients/Subject4/1.MRI/dReg_-_sDW_SSh_SENSE_001.png"
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

# Check bit depth
if img.dtype == np.uint8:
    print("8-bit image")
    # Convert to 16-bit
    img_16bit = img.astype(np.uint16) * 256
elif img.dtype == np.uint16:
    print("16-bit image")
