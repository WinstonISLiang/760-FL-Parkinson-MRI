import os
import shutil
import subprocess
import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage.transform import resize
from PIL import Image
from preprocessing import helper
import napari

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def preprocess_volume(img_dir: str, target_shape=(64,128,128)) -> np.ndarray:
    """
    从切片目录 img_dir 读取 PNG 切片，归一化并重采样，重建固定尺寸的 3D 体。
    """
    files = helper.sort_img_files(img_dir)
    slices = []
    for path in files:
        img = Image.open(path).convert('L')
        arr = helper.normalise(img)
        arr = resize(arr, (target_shape[1], target_shape[2]), anti_aliasing=True)
        if np.std(arr) < 1e-8:
            print(f"[Warning] 空切片：{path}")
            return None
        slices.append(arr)
    vol = np.stack(slices, axis=0)
    if vol.shape[0] != target_shape[0]:
        vol = resize(vol, target_shape, anti_aliasing=True)
    return vol


def bias_correction(data: np.ndarray, spacing=(1.0,1.0,1.0)) -> np.ndarray:
    """
    对 NumPy 3D 体执行 N4 偏场校正。
    """
    if not np.isfinite(data).all():
        print("[Error] 数据中包含 NaN 或 inf，跳过偏场校正")
        return None
    img = sitk.GetImageFromArray(data.astype(np.float32))
    img.SetSpacing(spacing)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    try:
        out = corrector.Execute(img)
        return sitk.GetArrayFromImage(out)
    except Exception as e:
        print(f"[Error] 偏场校正失败：{e}")
        return None


def skull_strip_volume_array(data: np.ndarray,
                              subject_id: str,
                              nifti_dir: str,
                              strip_dir: str,
                              spacing=(1.0,1.0,1.0)) -> np.ndarray:
    """
    将 NumPy 体保存为临时 NIfTI，调用 hd-bet 做 Skull-Stripping，
    并加载剥离后的 NIfTI 返回数组。如果 hd-bet 失败就返回原 data。
    """
    # 如果跑不了 hd-bet，就直接跳过
    if shutil.which('hd-bet') is None:
        print("[Warning] hd-bet not found, skipping skull-strip.")
        return data

    # 从环境里读取想用的 device（cpu / cuda / mps）
    device = os.environ.get('HD_BET_DEVICE', 'cpu')
    if device == 'mps':
        print("[Warning] MPS cannot run 3D ConvTranspose; skipping hd-bet.")
        return data

    os.makedirs(nifti_dir, exist_ok=True)
    os.makedirs(strip_dir, exist_ok=True)
    tmp_nii = os.path.join(nifti_dir, f"{subject_id}.nii.gz")
    sitk.WriteImage(sitk.GetImageFromArray(data.astype(np.float32)), tmp_nii)
    out_nii = os.path.join(strip_dir, f"{subject_id}_stripped.nii.gz")

    cmd = [
        'hd-bet',
        '-i', tmp_nii,
        '-o', out_nii,
        '-device', device,
        '--disable_tta'
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[Warning] hd-bet failed ({e}); skipping skull-strip.")
        return data

    stripped = sitk.ReadImage(out_nii)
    return sitk.GetArrayFromImage(stripped)



def visualize_itk(img: sitk.Image, title: str):
    """Napari 交互式可视化 SITK Image"""
    arr = sitk.GetArrayFromImage(img)
    v = napari.Viewer(title=title)
    v.add_image(arr, name=title)
    napari.run()


def process_subject(row: pd.Series,
                    out_dir: str,
                    tmp_nifti: str,
                    tmp_stripped: str,
                    spacing=(1.0,1.0,1.0)):
    subject = str(row.get('SubjectID', 'sub'))
    modality = row['Type']
    path = row['FilePath']  # PNG 目录路径
    print(f"\n[Start] process {subject} ({modality})")

    vol = preprocess_volume(path)
    if vol is None:
        return

    vol = bias_correction(vol, spacing)
    if vol is None:
        return

    vol = skull_strip_volume_array(vol, subject, tmp_nifti, tmp_stripped, spacing)
    if vol is None:
        return

    # 根据模态选择归一化方式
    if modality.upper() == 'MRI':
        normed = helper.z_score_norm(vol, np.mean(vol), np.std(vol))
    else:
        normed = helper.min_max_norm(vol, np.max(vol), np.min(vol))

    # 扩展通道维度 (N,C,D,H,W)
    final = np.expand_dims(normed, axis=0)

    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{subject}_{modality}.npy")
    np.save(save_path, final)
    print(f"[Saved] {save_path}")


def main():
    label_csv = "labelled_patients.csv"
    output_dir = "volumes/test_processed"
    tmp_nii_dir = "tmp_nifti"
    tmp_strip_dir = "tmp_stripped"

    df = pd.read_csv(label_csv)
    for _, row in df.iterrows():
        process_subject(row, output_dir, tmp_nii_dir, tmp_strip_dir)
    print("\n[Done] 所有样本处理完毕")


if __name__ == '__main__':
    main()
