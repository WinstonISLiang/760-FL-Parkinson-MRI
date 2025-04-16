import os
import numpy as np
import nibabel as nib
import subprocess

volume_dir = 'volumes'
nifti_dir = 'nifti-data'
output_dir = 'stripped-output'

os.makedirs(nifti_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

for file in os.listdir(volume_dir):
    if file.endswith('.npy'):
        file_path = os.path.join(volume_dir, file)

        base_name = os.path.splitext(file)[0]
        nii_name = base_name + '.nii.gz'
        nii_path = os.path.join(nifti_dir, nii_name)

        npy_data = np.load(file_path)
        nii = nib.Nifti1Image(npy_data, affine=np.eye(4))
        nib.save(nii, nii_path)
        print(f'[✓] Saved NIfTI: {nii_path}')

        output_path = os.path.join(output_dir, base_name + '_stripped.nii.gz')
        cmd = [
            'hd-bet',
            '-i', nii_path,
            '-o', output_path,
            '-device', 'cpu'
        ]
        print(f'--> Running skull stripping on: {nii_path}')
        subprocess.run(cmd)
