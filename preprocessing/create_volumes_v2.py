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
# import non_iid_split


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
        # changed from original code: data = data.astype(np.uint16 if use_16bit else np.uint8)
        if use_16bit:
            data = (data * 65535).round().astype(np.uint16)  # 0–65535
        else:
            data = (data * 255).round().astype(np.uint8)  # 0–255

        volumes[seq_prefix] = data

    if not volumes:
        print(f"No valid volumes created for {img_dir}")
        return {}

    return volumes


def skull_strip_array(
        volume: np.ndarray,
        device: str = "cpu",
        mode: str = "fast"
) -> np.ndarray:
    """
    Run HD-BET skull stripping on a 3-D (D, H, W) or 4-D (C, D, H, W) NumPy
    volume and return the brain-extracted volume as a NumPy array.

    Parameters
    ----------
    volume : np.ndarray
        Input MRI volume.  Accepts either:
        * shape = (D, H, W)  – single-channel 3-D volume, or
        * shape = (C, D, H, W) with C == 1 – channel-first convention.
        The data are **not** copied; the array is only cast to float32
        when written to disk for HD-BET.
    device : str, optional
        Which compute device HD-BET should use.
        * "cpu"  – force CPU inference (portable, default).
        * "0", "1", … – CUDA GPU ID.
    mode : str, optional
        HD-BET preset:
        * "fast"  – lower accuracy, higher speed.
        * "accurate" – default HD-BET setting (slower).

    Returns
    -------
    np.ndarray
        Brain-extracted volume with the **same dtype** and spatial shape
        as the input.  If the input had an extra channel dimension the
        returned array keeps that dimension.

    Notes
    -----
    1. The function writes a temporary NIfTI file under a randomly
       generated directory (e.g. /tmp/hd_bet_abcdef/).
       All intermediate files are removed in the `finally` block.
    2. HD-BET is called via its command-line interface, so the
       `hd-bet` executable must be available in the current environment.
    3. Only single-channel volumes are supported because HD-BET expects
       normal structural MRI input.  If multi-contrast stripping is
       required, run the function on each channel separately.
    """
    # ------------------------------------------------------------------
    # 0. Handle a possible leading channel dimension (C, D, H, W).
    #    HD-BET only accepts a 3-D volume, so squeeze it out and remember
    #    to restore it afterwards.
    # ------------------------------------------------------------------
    need_unsqueeze = False
    if volume.ndim == 4:
        if volume.shape[0] != 1:
            raise ValueError(
                "Only single-channel skull stripping is supported. "
                f"Got volume.shape={volume.shape}."
            )
        volume = volume[0]           # shape -> (D, H, W)
        need_unsqueeze = True

    # ------------------------------------------------------------------
    # 1. Create a unique temporary directory for all intermediate files.
    #    Using tempfile ensures automatic name collision avoidance.
    # ------------------------------------------------------------------
    tmp_dir = tempfile.mkdtemp(prefix="hd_bet_")

    try:
        # Full paths used by HD-BET
        in_path  = os.path.join(tmp_dir, "in.nii.gz")
        out_pref = os.path.join(tmp_dir, "out")        # HD-BET adds suffixes

        # ------------------------------------------------------------------
        # 2. Dump the NumPy array to NIfTI.  HD-BET expects float32 input.
        # ------------------------------------------------------------------
        nib.save(
            nib.Nifti1Image(volume.astype(np.float32), affine=np.eye(4)),
            in_path
        )

        # ------------------------------------------------------------------
        # 3. Call HD-BET via subprocess.  Stderr/stdout are captured so that
        #    any failure raises CalledProcessError for easier debugging.
        # ------------------------------------------------------------------
        cmd = [
            "hd-bet",
            "-i", in_path,
            "-o", out_pref,
            "-device", device,
            "-mode", mode
        ]
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # ------------------------------------------------------------------
        # 4. Read the stripped result back into memory.
        #    HD-BET appends '_stripped.nii.gz' to the output prefix.
        # ------------------------------------------------------------------
        stripped_path = f"{out_pref}_stripped.nii.gz"
        stripped = nib.load(stripped_path).get_fdata(dtype=volume.dtype)

        # ------------------------------------------------------------------
        # 5. Restore the original channel dimension if it was present.
        # ------------------------------------------------------------------
        if need_unsqueeze:
            stripped = stripped[np.newaxis, ...]  # shape -> (1, D, H, W)

        return stripped

    finally:
        # ------------------------------------------------------------------
        # 6. Always remove the temporary directory, even if an exception
        #    occurred, to keep the file system clean.
        # ------------------------------------------------------------------
        shutil.rmtree(tmp_dir)




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


def preprocess_all_volumes(
    out_dir: str,
    label_file: str,
    original_spacing=(3.0, 2.0, 2.0),
    target_shape=(64, 128, 128),
    use_16bit: bool = True,
    std_threshold: float = 0.1,
) -> None:
    """
    Parameters
    ----------
    out_dir : str
        Output directory where the .npy files are saved.
    label_file : str
        CSV containing columns SubjectID, Class, Type, FilePath.
    original_spacing : tuple[float, float, float]
        Original voxel spacing (z, y, x) in millimetres.
    target_shape : tuple[int, int, int]
        Desired (D, H, W) shape passed to `stacking2D`.
    use_16bit : bool
        If True, save uint16 volumes; otherwise save uint8.
    std_threshold : float
        Standard-deviation threshold below which a slice is considered blank.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(label_file)

    for idx, row in df.iterrows():
        subj = row["SubjectID"]
        label = row["Class"]
        modality = row["Type"]
        path = row["FilePath"]

        print(f"\n[{idx+1}/{len(df)}] {path}  ({modality})")

        # 1. build 3-D volumes
        vol_dict = stacking2D(
            img_dir=path,
            target_shape=target_shape,
            std_threshold=std_threshold,
            use_16bit=use_16bit,
        )

        if not vol_dict:
            print("  ↳ stacking failed, skip")
            continue

        # 2. iterate over sequences (T1, T2, FLAIR…)
        for seq, vol_int in vol_dict.items():
            # cast to float32 before resampling
            vol_float = vol_int.astype(np.float32)
            vol_iso = resample_isotropic(vol_float, original_spacing)

            # 3. volume-level intensity normalisation
            if modality.upper() == "MRI":
                # vol_iso = skull_strip_array(vol_iso) # skull stripping, haven't tested with full data set yet, but works pretty good with small dataset. 
                vol_iso = n4_bias_correct(vol_iso)    
                mu, sigma = vol_iso.mean(), vol_iso.std()
                vol_norm = helper.z_score_norm(vol_iso, mu, sigma)
            else:  # DAT,  reason why I skip bias correction for DAT:
                   # SPECT signals are sparse and high-contrast; applying N4 would likely smooth out genuine hotspots by mistaking them for artefacts.   
                vmin, vmax = vol_iso.min(), vol_iso.max()
                vol_norm = helper.min_max_norm(vol_iso, vmax, vmin)

            # 4. add channel dimension (C,D,H,W) and save
            vol_norm = np.expand_dims(vol_norm.astype(np.float32), axis=0)

            seq_safe = os.path.basename(seq).rstrip("_")  # drop path and trailing '_'
            fname = f"{subj}_{seq_safe}_label{label}.npy"
            np.save(os.path.join(out_dir, fname), vol_norm)
            print(f"  ↳ saved {fname}  shape={vol_norm.shape}")

    print("\n✓ All volumes processed.")




def n4_bias_correct(vol_float: np.ndarray,
                    mask: np.ndarray | None = None,
                    iter_list=(50, 50, 30, 20)) -> np.ndarray:
   """
    Perform N4 bias-field correction at full resolution (CPU-only, therefore slower).
    The input volume must contain only positive values; recommend scaling to 0–1 or 0–65535.
    """

    # SimpleITK Image
    img = sitk.GetImageFromArray(vol_float)

    # Automatically generate a coarse mask (> 5 % of the maximum value)
    if mask is None:
        mask = (vol_float > vol_float.max() * 0.05).astype(np.uint8)
    mask_img = sitk.GetImageFromArray(mask)

    # N4 filter
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations(iter_list)   # Number of iterations for each pyramid level

    img_corr  = corrector.Execute(img, mask_img)

    return sitk.GetArrayFromImage(img_corr).astype(np.float32)
