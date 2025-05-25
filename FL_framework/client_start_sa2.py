# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import flwr as fl

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
data_path = os.path.join(base_dir, "client_data1", "client_2")

X, y = [], []
for file in os.listdir(data_path):
    if file.endswith(".npy"):
        label = int(file.split("_")[3])
        vol = np.load(os.path.join(data_path, file))
        X.append(vol)
        y.append(label)

# horizontal flip
X_aug = []
y_aug = []
for i in range(len(X)):
    flipped = np.flip(X[i], axis=3)
    X_aug.append(flipped)
    y_aug.append(y[i])

# vertical flip
X = np.concatenate([X, np.array(X_aug)])
y = np.concatenate([y, np.array(y_aug)])
X_flip_v = [np.flip(x, axis=2) for x in X]
y_flip_v = y.copy()

# Gaussian noise
X_noise = [x + np.random.normal(0, 0.01, x.shape) for x in X]
y_noise = y.copy()
X = np.concatenate([X, np.array(X_flip_v), np.array(X_noise)])
y = np.concatenate([y, np.array(y_flip_v), np.array(y_noise)])

# Normalization and split into training and test sets
X = np.array(X, dtype=np.float32)
y = np.array(y)
X = (X - np.mean(X)) / np.std(X)

# train:80%, test:20%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Convert NumPy data to PyTorch tensors
# Convert the feature matrices in the training and test sets from NumPy to PyTorch tensors of 32-bit floating point numbers (float32), which is a common data type for CNN input.The labels are also converted to tensors, specifying the data type as long (int64), which is the type required by CrossEntropyLoss().
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Design a sampler for class imbalance, count the number of samples for each class, calculate and assign weights, and create a weighted sampler
class_sample_count = np.array([np.sum(y_train == t) for t in np.unique(y_train)])
weights = 1. / class_sample_count
sample_weights = weights[y_train]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


# dataloader
train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_tensor),
    batch_size=8,
    sampler=sampler  # no shuffle needed
)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=8)

from model import Deeper3DCNN  # assuming you put model class in model.py
from client_sa import ParkinsonSAClient as ParkinsonClient
import torch

model = Deeper3DCNN(dropout=0.5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Assume train_loader and test_loader are loaded already as in your original code
client = ParkinsonClient(model, train_loader, test_loader, device, client_id=1, seed=456)
client.peer_seeds = [123]
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)