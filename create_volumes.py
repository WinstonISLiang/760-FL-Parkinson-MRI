import os
import numpy as np
from PIL import Image
import nibabel as nib
import pandas as pd
import subprocess
from skimage.transform import resize

def preprocess_volume(img_path, target_shape=(64,128,128)):
    img_files = sorted([os.path.join(img_path, f) for f in os.listdir(img_path) if f.endswith(".png")])
    if not img_files:
        return

    # stacking 2D images to make 3D volumes w format as (num_slices, 128, 128)
    slices = []
    for img_path in img_files:
        img = Image.open(img_path).convert('L') # grayscale
        img_array = np.array(img) / 255.0 # normalise to [0,1]
        img_array = resize(img_array, (target_shape[0], target_shape[1]), anti_aliasing=True)
        slices.append(img_array)

    data = np.stack(slices, axis=0) # ensures (D, H, W); not (H, D, W); D = num_slices

    # resize depth
    if data.shape[0] != target_shape[0]:
        data = resize(data, target_shape, anti_aliasing=True)

    # check for empty volume
    if np.std(data) < 1e-8:
        return None

    return data

def skull_strip_volume(data, temp_dir="temp"):
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

def preprocess_all_volumes(label_file="labelled_patients.csv"):
    labels_df = pd.read_csv(label_file)


# process each patient
main_path = "ntua-parkinson-dataset/pd-patients"
input_file = "labelled_patients.csv"
volume_output_dir = "volumes/temp"

df = pd.read_csv(input_file)
os.makedirs(volume_output_dir, exist_ok=True) # create output directory

for x in df.iterrows():
    volume = preprocess_volume(x[1]['FilePath'])
    filename = f"volume_{x[1]['SubjectID']}.npy"
    save_path = os.path.join(volume_output_dir, filename)
    np.save(save_path, volume)




# for i, (folder, group) in enumerate(grouped):
#     label = group['label'].iloc[0]
#     volume = load_volume(group['image_path'].tolist())
#     filename = f"volume_{i:02d}_label_{label}.npy"
#     save_path = os.path.join(volume_output_dir, filename)
#     np.save(save_path, volume)
#     print(f"saved: {save_path}")