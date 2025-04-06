import os
import pandas as pd

def label_patients(base_path: str, res: list):
    '''
    Labelling subjects in pd as label 1; subjects in non-pd as label 0
    '''
    for subject in os.listdir(base_path):
        subject_path = os.path.join(base_path, subject)

        # collecting 0.DAT images
        if os.path.isdir(subject_path):
            dat_path = os.path.join(subject_path, "0.DAT")
            subject_id = subject.split('t')[1]
            label = 0 if 'non' in base_path else 1

            # if multiple dirs in 0.DAT under subject, take the latest subdir
            if os.path.isdir(dat_path):
                subdirs = [d for d in os.listdir(dat_path) if os.path.isdir(os.path.join(dat_path, d))]

                if subdirs:
                    latest_subdir = sorted(subdirs)[-1]
                    latest_subdir_path = os.path.join(dat_path, latest_subdir)
                    res.append((subject_id, label, "DAT", latest_subdir_path))

            # collecting 1.MRI images
            mri_path = os.path.join(subject_path, "1.MRI")
            if os.path.isdir(mri_path):
                res.append((subject_id, label, "MRI", mri_path))

    return res

data = []

pd_dir = "ntua-parkinson-dataset/pd-patients"
non_pd_dir = "ntua-parkinson-dataset/non-pd-patients"

data = label_patients(pd_dir, data)
data = label_patients(non_pd_dir, data)

df = pd.DataFrame(data, columns=["SubjectID", "Class", "Type", "FilePath"])
df.to_csv("labelled_patients.csv", index=False)
print(f"labelled_patients.csv {'updated' if os.path.exists('labelled_patients.csv') else 'generated'}.")