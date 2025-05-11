import numpy as np
from PIL import Image
import glob
import nibabel as nib
import os
import napari
import re
import SimpleITK as sitk

'''
    Helper functions for create_volumes.py:
    - get_bit_depth(img_path): checks the bit of image to determine which preprocessing method to apply.
    - visualise(img): 3D visualiser of image
    - get_prefix_and_suffix(filename): gets the prefix, suffix of a filename
    - group_by_prefix(filenames): groups suffixes by a common prefix using hashing

    stacking2D()
      : get_bit_depth(img_path)
      : visualise(img)

'''

def get_bit_depth(image_path: str):
    mode_to_bits = {
        '1': 1,
        'L': 8,
        'P': 8,
        'RGB': 24,
        'RGBA': 32,
        'CMYK': 32,
        'I': 32,
        'F': 32
    }

    try:
        img = Image.open(image_path)
        bit = mode_to_bits.get(img.mode, None)
        return bit
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

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




def visualise(img, title:str ="Untitled"):
    viewer = napari.Viewer(title=title)
    viewer.add_image(sitk.GetArrayFromImage(img), name='3D volume')
    napari.run()



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
    img_files = sorted(glob.glob(os.path.join('../' + img_dir, "*.png"))) # don't remove ../
    print(img_files)
    # print(os.path.join(img_dir, "*.png"))

    if not img_files:
        raise ValueError(f"No PNG files found in {img_dir}")

    return img_files


def nii_to_npy(nii_path, npy_path):
    """
    Convert a .nii.gz file to .npy.

    Args:
        nii_path (str): Path to input .nii.gz file.
        npy_path (str): Path to output .npy file.

    Returns:
        np.ndarray: 3D volume as NumPy array, or None if failed.
    """
    try:
        # Load .nii.gz
        nii_img = nib.load(nii_path)
        data = nii_img.get_fdata().astype(np.float32)

        # Validate
        if not np.isfinite(data).all():
            print(f"Invalid values (NaN/inf) in {nii_path}")
            return None
        if data.ndim != 3:
            print(f"Expected 3D volume, got shape {data.shape}")
            return None

        # Save as .npy
        np.save(npy_path, data)
        print(f"Converted {nii_path} to {npy_path}, shape={data.shape}")
        return data
    except Exception as e:
        print(f"Failed to convert {nii_path}: {e}")
        return None


def apply_n4_bias_correction(img_path: str, mask_fissure: bool = True) -> sitk.Image:
    """
    Apply N4 Bias Field Correction to a single NIfTI image.

    Args:
        img_path: Path to input .nii or .nii.gz file.
        mask_fissure: If True, use Otsu thresholding to create a mask for correction.

    Returns:
        Corrected SimpleITK Image.
    """
    img = sitk.ReadImage(img_path)
    img_float = sitk.Cast(img, sitk.sitkFloat32)
    if mask_fissure:
        mask = sitk.OtsuThreshold(img_float, 0, 1, 200)
    else:
        mask = None
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_img = corrector.Execute(img_float, mask)
    return corrected_img


def batch_bias_correction(input_dir: str, output_dir: str, pattern: str = "*.nii.gz") -> None:
    """
    Apply N4 bias correction to all NIfTI files in input_dir and save results to output_dir.

    Args:
        input_dir: Directory containing .nii or .nii.gz files.
        output_dir: Directory where corrected images will be saved.
        pattern: Glob pattern to match input files.
    """
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join(input_dir, pattern))
    if not files:
        raise ValueError(f"No files matching {pattern} in {input_dir}")
    for fpath in files:
        print(f'--> Applying Bias Correction on: {fpath}')
        corrected = apply_n4_bias_correction(fpath)
        base = os.path.splitext(os.path.splitext(os.path.basename(fpath))[0])[0]
        save_path = os.path.join(output_dir, f"{base}_bias_corrected.nii.gz")
        sitk.WriteImage(corrected, save_path)
        print(f'[✓] Saved corrected image: {save_path}')
