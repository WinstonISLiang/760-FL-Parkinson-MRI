import numpy as np
from PIL import Image
import glob
import os

def min_max_norm(img_arr: np.array, max_int: int, min_int: int, epsilon=1e-8):
    return (img_arr - min_int) / (max_int - min_int + epsilon)

def z_score_norm(img_arr, mean_val, std):
    return (img_arr - mean_val) / (std + 1e-8)

def normalise(img: Image):
    '''
    Normalising to [0,1], assumes 8-bit PNGs.
    If PNGs are 16-bit, can clip intensities; constrain to [0,1]

    Fix: Using min-max normalisation to detect intense pixels
    '''
    img_array = np.array(img, dtype=np.float32)
    if img_array.max() > 255:
        img_array = min_max_norm(img_array, img_array.max(), img_array.min())
    else:
        img_array = img_array / 255.0

    return img_array

def sort_img_files(img_dir: str):
    '''
    .sorted() assumes filesnames are lexicographically ordered (eg. slice_001.png, slice_002.png)
    If filesnames are inconsistent (eg. 1.png, 10.png, 2.png), sorting will fail.

    Fix: Using glob (Python library for searching files through directories, efficiently)
    '''
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not img_files:
        raise ValueError(f"No PNG files found in {img_dir}")

    return img_files