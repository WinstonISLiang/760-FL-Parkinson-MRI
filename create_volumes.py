import os
import numpy as np
from PIL import Image
import nibabel as nib
import pandas as pd
import subprocess
from skimage.transform import resize

def preprocess_volume(img_dir, target_shape=(64,128,128)):
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    if not img_files:
        print(f"No PNG files found in {img_dir}")
        return

    # stacking 2D images to make 3D volumes w format as (num_slices, 128, 128)
    slices = []
    for img_path in img_files:
        img_path = os.path.join(img_dir, img_path)
        img = Image.open(img_path).convert('L') # grayscale
        img_array = np.array(img) / 255.0 # normalise to [0,1]
        img_array = resize(img_array, (target_shape[1], target_shape[2]), anti_aliasing=True)
        slices.append(img_array)

    data = np.stack(slices, axis=0) # ensures (D, H, W); not (H, D, W); D = num_slices
    print(f"Stacked {len(slices)} in {img_dir}, shape={data.shape}")

    # resize depth
    if data.shape[0] != target_shape[0]:
        data = resize(data, target_shape, anti_aliasing=True)
        print(f"Resized depth to {target_shape[0]}, new shape={data.shape}")

    # check for empty volume
    if np.std(data) < 1e-8:
        print(f"Empty volume detected in {img_dir}, std={np.std(data)}")
        return None

    return data

def skull_strip_volume(data, temp_dir="fsl_bet_volumes"):
    os.makedirs(temp_dir, exist_ok=True)

    # save 3D vol as temp .nii.gz
    temp_in = os.path.join(temp_dir, "temp_input.nii.gz")
    temp_out = os.path.join(temp_dir, "temp_output.nii.gz")

    # create NIfTI image
    nii_img = nib.Nifti1Image(data, affine=np.eye(4))
    nib.save(nii_img, temp_in)

    # FSL BET for skull stripping
    try:
        subprocess.run(["bet", temp_in, temp_out, "-F"], check=True)
    except subprocess.CalledProcessError:
        print("FSL BET Failed. ")
        return

    stripped_img = nib.load(temp_out)
    stripped_data = stripped_img.get_fdata()

    # Clean up temporary files
    os.remove(temp_in)
    os.remove(temp_out)

    return stripped_data

def preprocess_all_volumes(out_dir, normalize_method="z-score", label_file="labelled_patients.csv"):
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(label_file)
    for x in df.iterrows():
        subject_id = x[1]['SubjectID']
        label = x[1]['Class']
        volume = preprocess_volume(x[1]['FilePath'])
        stripped = skull_strip_volume(volume)

        # TODO: will need to look at other normalisation methods for different use-cases
        if normalize_method == 'z-score':
            stripped_normal = (stripped - np.mean(stripped)) / (np.std(stripped) + 1e-8)
        elif normalize_method == 'minmax':
            min_val, max_val = np.min(stripped), np.max(stripped)
            stripped_normal = (stripped - min_val) / (max_val - min_val + 1e-8)

        # to ensure compatibility for frameworks - channel-first convention (N, C, D, H, W)
        stripped_normal = np.expand_dims(stripped_normal, axis=0)

        out_filename = f"volume_{x[1]['SubjectID']}_labelled_{x[1]['Class']}.npy"
        save_path = os.path.join(volume_output_dir, out_filename)
        np.save(save_path, volume)
        print(f"Saved: {out_filename} at {save_path}")

# process each patient
main_path = "ntua-parkinson-dataset/pd-patients"
input_file = "labelled_patients.csv"
volume_output_dir = "volumes/temp"

preprocess_all_volumes(volume_output_dir)

df = pd.read_csv(input_file)
os.makedirs(volume_output_dir, exist_ok=True) # create output directory

for x in df.iterrows():
    volume = preprocess_volume(x[1]['FilePath'])
    filename = f"volume_{x[1]['SubjectID']}_labelled_{x[1]['Class']}.npy"
    save_path = os.path.join(volume_output_dir, filename)
    np.save(save_path, volume)
    print(f"Saved at {save_path}")




# for i, (folder, group) in enumerate(grouped):
#     label = group['label'].iloc[0]
#     volume = load_volume(group['image_path'].tolist())
#     filename = f"volume_{i:02d}_label_{label}.npy"
#     save_path = os.path.join(volume_output_dir, filename)
#     np.save(save_path, volume)
#     print(f"saved: {save_path}")