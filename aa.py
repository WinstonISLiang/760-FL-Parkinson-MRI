import re
import glob
import os
import pandas as pd

# Example filenames
filenames = [
    "T1W_FFE_005.png",
    "T2W_FLAIR_001.png",
    "DUAL_TSE_010.png",
    "dReg_..._SENSE_012.png"
]

# Function to get the prefix
def get_prefix_and_suffix(filename):
    match = re.match(r'(.*?)(\d{3})\.png', filename)
    if match:
        prefix = match.group(1)
        suffix = match.group(2)
        return prefix, suffix

    return None, None

def group_by_prefix(filenames):
    prefix_map = {}
    for filename in filenames:
        prefix, suffix = get_prefix_and_suffix(filename)

        if prefix is not None:
            if prefix not in prefix_map:
                prefix_map[prefix] = []
            prefix_map[prefix].append(suffix)

    return prefix_map

# Extract prefixes
for filename in filenames:
    prefix, suffix = get_prefix_and_suffix(filename)
    print(f"Filename: {filename}, Prefix: {prefix}, Suffix: {suffix}")


# def sort_img_files(img_dir: str):
#     img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))

#     if not img_files:
#         raise ValueError(f"No PNG files found in {img_dir}")

#     return img_files

# x = sort_img_files("ntua-parkinson-dataset/pd-patients/Subject4/1.MRI")
# print(x)