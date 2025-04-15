# Privacy-Preserving Federated Learning for Early Parkinson's Detection through Decentralized MRI Analysis

## Setup

Download the dataset via

```
git pull https://github.com/ails-lab/ntua-parkinson-dataset.git
```
## Preprocessing

- The `001.png` of a higher `sY` is a higher resolution image, given `001.png` appears in all `sY` of `pd-patients/SubjectX/0.DAT/sY` subdirectories. We select the largest to ensure we work with the most detailed version, despite possible added computation and overfit risk.

- Currently, using z-score normalisation; results in empty hollows, will need to review.
- HD-BET for skull stripping (more performant version of FSL BET)

**preprocessed_data files are ready for direct use in 3D CNN training pipeline.**


The folder contains the MRI volumes after preprocessing. The preprocessing steps included:

###### Data Cleaning:
Volumes with near-zero intensity variation (empty volumes) have been filtered out.

###### Intensity Normalization:
Z-score normalization was applied to each volume (subtract mean, divide by standard deviation).

###### Channel Dimension Added:
Each volume has been reshaped from (64, 128, 128) to (1, 64, 128, 128) (channel-first format).

###### File Format:
The preprocessed volumes are stored as .npy files. The file names correspond to the original .nii.gz files.

## Methodology
###### Base 3D CNNs


## Results
