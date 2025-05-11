from skimage.transform import resize
from PIL import Image
from typing import Tuple, Dict, List

import os
import numpy as np
import pandas as pd
import nibabel as nib
import subprocess
import glob
import SimpleITK as sitk
import napari
import helper
# import non_iid_split
import re

def stacking2D(img_dir: str, target_shape: Tuple[int,int,int] = (64,128,128),
               std_threshold: float = 0.1, use_16bit: bool = True):
    '''
    Stacking 2D slices into 3D volumes.
    '''
    img_files = helper.sort_img_files(img_dir)

    # group slices by sequence prefix (e.g., T1W_FFE, T2W_FLAIR)
    grouped_slices: Dict[str, List[str]] = {}
    for img_path in img_files:
        filename = os.path.basename(img_path)
        match = re.match(r'^(T\dW_[A-Z]+)', filename)  # extract sequence prefix (e.g., 'T1W_FFE' from 'T1W_FFE_001.png')
        print(match)

        if match:
            seq_prefix = match.group(1)
        else:
            seq_prefix = 'unknown'
            print(f"Warning: Could not extract sequence prefix from {filename}, using 'unknown'")

        if seq_prefix not in grouped_slices:
            grouped_slices[seq_prefix] = []
        grouped_slices[seq_prefix].append(img_path)

    # process each sequence group separately
    volumes: Dict[str, np.ndarray] = {}
    for seq_prefix, group_paths in grouped_slices.items():
        print(f"Processing group '{seq_prefix}' with {len(group_paths)} slices")

        slices = []
        for img_path in group_paths:
            img = Image.open(img_path).convert('L') # grayscale
            img_array = np.array(img)

            # build volumes using the same bit dtype
            bit_depth = helper.get_bit_depth(img_path)
            if use_16bit:
                if bit_depth == 8:
                    img_array = (img_array * 256).astype(np.uint16)
                elif bit_depth == 16:
                    img_array = img_array.astype(np.uint16)
            else:
                if bit_depth == 16:
                    img_array = (img_array / 256).astype(np.uint8)
                elif bit_depth == 8:
                    img_array = img_array.astype(np.uint8)

            img_array = helper.normalise(img)
            img_array = resize(img_array, (target_shape[1], target_shape[2]), anti_aliasing=True)

            # check for blank slice
            if np.std(img_array) < std_threshold:
                print(f"No valid slices in group '{seq_prefix}' for {img_dir}")
                continue

            slices.append(img_array)

        if not slices:
            print(f"No valid slices in {img_dir}")
            return

        data = np.stack(slices, axis=0) # ensures (D, H, W); not (H, D, W); D = num_slices
        print(f"Stacked {len(slices)} in {img_dir}, shape={data.shape}")

        # resize depth
        if data.shape[0] != target_shape[0]:
            data = resize(data, target_shape, anti_aliasing=True)
            print(f"Resized depth to {target_shape[0]}, new shape={data.shape}")

        # ensure correct dtype
        data = data.astype(np.uint16 if use_16bit else np.uint8)

        volumes[seq_prefix] = data

    if not volumes:
        print(f"No valid volumes created for {img_dir}")
        return {}

    return volumes

## todo: implement bias correction somwhere

def skull_strip_volume(data, nifti_dir="../nifti-data", vol_in="../volumes", vol_out="../skull-stripped"):
    os.makedirs(nifti_dir, exist_ok=True)
    os.makedirs(vol_out, exist_ok=True)

    for file in os.listdir(vol_in):
        if file.endswith('.npy'):
            file_path = os.path.join(vol_in, file)

            base_name = os.path.splitext(file)[0]
            nii_name = base_name + '.nii.gz'
            nii_path = os.path.join(nifti_dir, nii_name)

            npy_data = np.load(file_path)
            nii = nib.Nifti1Image(npy_data, affine=np.eye(4))
            nib.save(nii, nii_path)
            print(f'[✓] Saved NIfTI: {nii_path}')

            output_path = os.path.join(vol_out, base_name + '_stripped.nii.gz')
            cmd = [
                'hd-bet',
                '-i', nii_path,
                '-o', output_path,
                '-device', 'cpu'
            ]
            print(f'--> Running skull stripping on: {nii_path}')
            subprocess.run(cmd)

def preprocess_all_volumes(out_dir, label_file):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(label_file)

    for _, row in df.iterrows():
        modality = row['Type']
        file_path = row['FilePath']
        print(f"Preprocessing {file_path} ({modality})")

        # preprocess and resample volume
        result = stacking2D(file_path)
        helper.visualise(sitk.GetImageFromArray(result))
        if result is None:
            print(f"Skipping {file_path} due to preprocessing failure")
            continue

        # TODO: need to resample to 1mm {here}

        # stripped_nii_path = skull_strip_volume(result, file_path)
        # if stripped_nii_path is None:
        #     print(f"Skipping {file_path} due to skull stripping failure")
        #     continue

        # # convert to .npy
        # base_name = os.path.basename(file_path)
        # npy_path = os.path.join(out_dir, f"{base_name}_skull_stripped.npy")
        # data = helper.nii_to_npy(stripped_nii_path, npy_path)
        # if data is None:
        #     print(f"Skipping {file_path} due to .npy conversion error")
        #     continue

        # if modality == 'MRI':
        #     bias_corrected = bias_correction(stripped)


        # img = sitk.GetImageFromArray(stripped)
        # visualise(img)


        # TODO: will need to look at other normalisation methods for different use-cases
        if modality == 'MRI':
            stripped_normal = helper.z_score_norm(stripped, np.mean(stripped), np.std(stripped))
        elif modality == 'DAT':
            stripped_normal = helper.min_max_norm(stripped, np.max(stripped), np.min(stripped))

        # to ensure compatibility for frameworks - channel-first convention (N, C, D, H, W)
        stripped_normal = np.expand_dims(stripped_normal, axis=0)

        out_filename = f"volume_{x[1]['SubjectID']}_labelled_{x[1]['Class']}.npy"
        save_path = os.path.join(volume_output_dir, out_filename)
        np.save(save_path, volume)
        print(f"Saved: {out_filename} at {save_path}")


# non_iid_split()

