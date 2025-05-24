from skimage.transform import resize
from PIL import Image
from typing import Tuple, Dict, List
from collections import defaultdict
from typing import Dict
import os
import numpy as np
import pandas as pd
import nibabel as nib
import subprocess
import glob
import SimpleITK as sitk
import napari
import helper
import tempfile
# import non_iid_split

skip_keywords = [
    "copy", "localizer", "survey", "calibration", "scout", "mpr_thick_range",
    "mip", "screensave", "asset", "topogram", "head_"
]
keep_keywords = ["t1", "mprage", "flair", "dat", "spekt", "dopamine"]



def stacking2D(
    img_dir: str,
    target_shape: Tuple[int, int, int] = (64, 128, 128),
    std_threshold: float = 0.1,
    use_16bit: bool = True,
):
    """
    Stack 2-D slices into 3-D volumes.
    """
    img_files = helper.sort_img_files(img_dir)
    num_slices = len(img_files)
    print(f"Number of slices in {img_dir}: {num_slices}")

    # group slices by sequence prefix (e.g. T1W_FFE, T2W_FLAIR)
    prefix_map = defaultdict(list)
    for file in img_files:
        prefix, suffix = helper.get_prefix_and_suffix(file)

        if prefix is not None:
            prefix_map[prefix].append(suffix)

    # process each sequence group separately
    volumes: Dict[str, np.ndarray] = {}
    for seq_prefix, suffix_values in prefix_map.items():
        prefix_lower = seq_prefix.lower()
        if any(k in prefix_lower for k in skip_keywords) or not any(k in prefix_lower for k in keep_keywords):
            print(f"[🗑️] Skipping '{seq_prefix}' — rejected by filters")
            continue

        print(f"Processing group '{seq_prefix}' with {len(suffix_values)} slices")

        slices = []
        for suffix in suffix_values:
            img_path = os.path.join(seq_prefix + suffix + ".png")
            print(img_path)
            img = Image.open(img_path).convert("L")  # grayscale
            img_array = np.array(img)

            # convert to a common bit depth
            bit_depth = helper.get_bit_depth(img_path)
            if use_16bit:
                if bit_depth == 8:
                    img_array = (img_array * 255).astype(np.uint16)
                elif bit_depth == 16:
                    img_array = img_array.astype(np.uint16)
            else:
                if bit_depth == 16:
                    img_array = (img_array / 256).astype(np.uint8)
                elif bit_depth == 8:
                    img_array = img_array.astype(np.uint8)

            img_array = helper.normalise(img)
            img_array = resize(
                img_array, (target_shape[1], target_shape[2]), anti_aliasing=True
            )

            # skip blank slices
            if np.std(img_array) < std_threshold:
                print(f"No valid slices in group '{seq_prefix}' for {img_dir}")
                continue

            slices.append(img_array)

        if not slices:
            print(f"No valid slices in {img_dir}")
            return

        data = np.stack(
            slices, axis=0
        )  # (D, H, W) where D = num_slices, not (H, D, W)
        print(f"Stacked {len(slices)} in {img_dir}, shape={data.shape}")

        # resize the depth dimension
        if data.shape[0] != target_shape[0]:
            data = resize(data, target_shape, anti_aliasing=True)
            print(f"Resized depth to {target_shape[0]}, new shape={data.shape}")

        # ensure integer dtype
        if use_16bit:
            data = (data * 65535).round().astype(np.uint16)  # 0–65535
        else:
            data = (data * 255).round().astype(np.uint8)  # 0–255


        volumes[seq_prefix] = data

    if not volumes:
        print(f"No valid volumes created for {img_dir}")
        return {}

    return volumes


def skull_strip_array(volume: np.ndarray) -> np.ndarray:
    """
    Use HD-BET to skull-strip a 3D numpy volume in-memory.
    Returns the skull-stripped volume as a NumPy array.
    Automatically sets env variable to avoid OpenMP conflict on macOS.
    """
    # set OpenMP override
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    if volume.ndim == 4 and volume.shape[0] == 1:
        volume = volume[0]  # (1, D, H, W) → (D, H, W)

    with tempfile.TemporaryDirectory() as tmpdir:
        nifti_path = os.path.join(tmpdir, "input.nii.gz")
        output_path = os.path.join(tmpdir, "input_stripped.nii.gz")

        # Write input volume to NIfTI
        sitk.WriteImage(sitk.GetImageFromArray(volume.astype(np.float32)), nifti_path)

        try:
            result = subprocess.run(
                ["hd-bet", "-i", nifti_path, "-o", output_path, "-device", "cpu"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                timeout=300
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"❌ HD-BET failed for {nifti_path}:\n"
                f"stderr:\n{e.stderr.decode()}"
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("❌ HD-BET timed out.")

        # Load stripped image
        stripped_img = sitk.ReadImage(output_path)
        stripped_vol = sitk.GetArrayFromImage(stripped_img)

        if np.std(stripped_vol) < 1e-3:
            raise ValueError("⚠️ Stripped volume appears blank — check input image quality.")

        return stripped_vol.astype(np.float32)



# ------------------------------------------------------------------
def resample_isotropic(
    volume: np.ndarray,
    original_spacing=(3.0, 2.0, 2.0),
    target_spacing=(1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Resample (D, H, W) volumes to isotropic spacing using SimpleITK (B-spline).
    """
    img = sitk.GetImageFromArray(volume)
    img.SetSpacing(original_spacing)  # (z, y, x) order in our arrays

    new_size = (
        np.array(img.GetSize())
        * np.array(original_spacing)
        / np.array(target_spacing)
    ).round().astype(int)

    res = sitk.Resample(
        img,
        size=[int(x) for x in new_size],
        outputSpacing=target_spacing,
        interpolator=sitk.sitkBSpline,
    )
    return sitk.GetArrayFromImage(res)


# ------------------------------------------------------------------
def preprocess_all_volumes(
    out_dir: str,
    label_file: str,
    original_spacing=(3.0, 2.0, 2.0),
    target_shape=(64, 128, 128),
    use_16bit: bool = True,
    std_threshold: float = 0.1,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(label_file)


    for idx, row in df.iterrows():
        
        subj = row["SubjectID"]
        label = row["Class"]
        modality = row["Type"]
        path = row["FilePath"]




        # skip dat files temporarily
        if modality.strip().lower() == "dat":
            print(f"  ↳ Skipping {subj} ({modality}) [DAT skipped]")
            continue









        print(f"\n[{idx+1}/{len(df)}] {path}  ({modality})")

        # ① build 3-D volumes
        vol_dict = stacking2D(
            img_dir=path,
            target_shape=target_shape,
            std_threshold=std_threshold,
            use_16bit=use_16bit,
        )

        if not vol_dict:
            print("  ↳ stacking failed, skip")
            continue
        
        # 优先级序列
        priority = ["t1", "mprage", "flair"]

# 对 MRI 模态进行结构序列优选
        if modality.upper() == "MRI":
            def rank(seq_name):
                lower = seq_name.lower()
                for i, p in enumerate(priority):
                    if p in lower:
                        return i
                return len(priority)

    # 排序所有序列（rank 优先 + 切片数量倒序）
            sorted_vols = sorted(
            vol_dict.items(),
                key=lambda item: (rank(item[0]), -item[1].shape[0])
            )

            # 保留排序后的第一个（最优序列）
            sorted_vols = sorted_vols[:1]
        else:
            # DAT 数据保留所有序列
            sorted_vols = vol_dict.items()

        # ② iterate over sequences (T1, T2, FLAIR…)
        for seq, vol_int in sorted_vols:
            # cast to float32 before resampling
            vol_float = vol_int.astype(np.float32)
            vol_iso = resample_isotropic(vol_float, original_spacing)

            # ③ volume-level intensity normalisation
            if modality.upper() == "MRI":
                vol_iso = skull_strip_array(vol_iso)
                vol_iso = n4_bias_correct(vol_iso)    
                
                mu, sigma = vol_iso.mean(), vol_iso.std()
                vol_norm = helper.z_score_norm(vol_iso, mu, sigma)
            else:  # DAT
                vmin, vmax = vol_iso.min(), vol_iso.max()
                vol_norm = helper.min_max_norm(vol_iso, vmax, vmin)

            # ④ add channel dimension (C,D,H,W) and save
            vol_norm = np.expand_dims(vol_norm.astype(np.float32), axis=0)

            seq_safe = os.path.basename(seq).rstrip("_")  # drop path and trailing '_'
            fname = f"{subj}_{seq_safe}_label{label}.npy"
            np.save(os.path.join(out_dir, fname), vol_norm)
            print(f"  ↳ saved {fname}  shape={vol_norm.shape}")

    print("\n✓ All volumes processed.")




# ------------------------------------------------------------------
def n4_bias_correct(vol_float: np.ndarray,
                    mask: np.ndarray | None = None,
                    iter_list=(50, 50, 30, 20)) -> np.ndarray:
    """
    直接在原分辨率做 N4BiasFieldCorrection（CPU 会慢一些）
    vol_float 必须全为正数；建议输入 0–1 或 0–65535 区间
    """
    # SimpleITK Image
    img = sitk.GetImageFromArray(vol_float)

    # 自动生成粗掩模（> 5% 最大值）
    if mask is None:
        mask = (vol_float > vol_float.max() * 0.05).astype(np.uint8)
    mask_img = sitk.GetImageFromArray(mask)

    # N4 filter
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(iter_list)   # 每金字塔层迭代次数
    img_corr  = corrector.Execute(img, mask_img)

    return sitk.GetArrayFromImage(img_corr).astype(np.float32)
