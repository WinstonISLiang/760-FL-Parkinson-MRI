from create_volumes import stacking2D
import pandas as pd

input_file = "../labelled_patients.csv"
volume_output_dir = "volumes/temp"

df = pd.read_csv(input_file)

for _, row in df.iterrows():
    file_path = row['FilePath']
    x = stacking2D(file_path)
    # print(x)
    break
