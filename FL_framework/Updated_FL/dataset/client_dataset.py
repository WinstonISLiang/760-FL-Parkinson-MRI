import os
import numpy as np
import torch
from torch.utils.data import Dataset
from fedml.data.data_loader import DefaultDataLoader

def extract_label_from_filename(filename):
    match = re.search(r'label([01])', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Label not found in filename: {filename}")

class ClientDataset(Dataset):
    def __init__(self, data_folder):
        self.files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.npy')]
        self.labels = [extract_label_from_filename(os.path.basename(f)) for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        volume = np.load(self.files[idx])
        volume = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return volume, label

def load_partition_data(args):
    client_data_paths = [os.path.join(args.data_dir, f"client_{i+1}") for i in range(args.client_num)]
    partition_dict = {}
    for i, path in enumerate(client_data_paths):
        dataset = ClientDataset(path)
        partition_dict[i] = dataset
    return partition_dict, partition_dict
