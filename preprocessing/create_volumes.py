import os
import numpy as np
import pandas as pd
import nibabel as nib
import subprocess
from skimage.transform import resize
from PIL import Image
import glob
import SimpleITK as sitk
import napari

# import non_iid_split
import helper

def visualise(img, title="Untitled"):
    viewer = napari.Viewer(title=title)
    viewer.add_image(sitk.GetArrayFromImage(img), name='3D volume')
    napari.run()

def preprocess_volume(img_dir, target_shape=(64,128,128)):
    '''
    TODO: add comment
    '''
    img_files = helper.sort_img_files(img_dir)

    # stacking 2D images to build 3D volumes
    slices = []
    for img_path in img_files:
        img = Image.open(img_path).convert('L') # grayscale
        img_array = helper.normalise(img)
        img_array = resize(img_array, (target_shape[1], target_shape[2]), anti_aliasing=True)

        # check for blank slice
        if np.std(img_array) < 1e-8:
            print(f"Empty slice detected in {img_path}")
            return None # if empty, what should i do with the volume?

        slices.append(img_array)

    data = np.stack(slices, axis=0) # ensures (D, H, W); not (H, D, W); D = num_slices
    print(f"Stacked {len(slices)} in {img_dir}, shape={data.shape}")

    # resize depth
    if data.shape[0] != target_shape[0]:
        data = resize(data, target_shape, anti_aliasing=True)
        print(f"Resized depth to {target_shape[0]}, new shape={data.shape}")

    return data

def bias_correction(data, img_dir, spacing=(1.0, 1.0, 1.0)):
    """Apply N4 bias field correction to a 3D volume."""
    if not np.isfinite(data).all():
        print(f"Invalid values (NaN/inf) before bias correction for {img_dir}")
        return None
    img = sitk.GetImageFromArray(data)
    img.SetSpacing(spacing)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    try:
        corrected_img = corrector.Execute(img)
        data = sitk.GetArrayFromImage(corrected_img)
        print(f"Bias correction completed for {img_dir}, shape={data.shape}")
        return data
    except Exception as e:
        print(f"Bias correction failed for {img_dir}: {e}")
        return None

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
        result = preprocess_volume(file_path)
        visualise(sitk.GetImageFromArray(result))
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

if __name__ == '__main__':
    input_file = "../labelled_patients.csv"
    volume_output_dir = "volumes/temp"

    preprocess_all_volumes(volume_output_dir, input_file) # skull stripping, z-score norm
    # non_iid_split()

    df = pd.read_csv(input_file)
