import os
import SimpleITK as sitk

# input and output directories
stripped_dir = 'stripped-output'
bias_corrected_dir = 'bias_correction_results'

# make sure output directory exists
os.makedirs(bias_corrected_dir, exist_ok=True)

def apply_n4_bias_correction(img_path):
    '''
    N4 Bias Field Correction for single image
    '''
    img = sitk.ReadImage(img_path)
    img_float = sitk.Cast(img, sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(img_float, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_img = corrector.Execute(img_float, mask)
    return corrected_img

# main function
for file in os.listdir(stripped_dir):
    if file.endswith('.nii.gz'):
        stripped_path = os.path.join(stripped_dir, file)

        print(f'--> Applying Bias Correction on: {stripped_path}')
        corrected_img = apply_n4_bias_correction(stripped_path)

        # save the corrected image
        base_name = os.path.splitext(os.path.splitext(file)[0])[0]  # remove .nii.gz
        save_path = os.path.join(bias_corrected_dir, base_name + '_bias_corrected.nii.gz')

        sitk.WriteImage(corrected_img, save_path)
        print(f'[✓] Saved corrected image: {save_path}')
