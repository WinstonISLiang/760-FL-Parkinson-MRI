# -*- coding: utf-8 -*-

import flwr as fl
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score
from secure_masking import get_shared_key, create_mask, apply_mask
import shutil
import os

tmp = "tmp_param"
if not os.path.exists(tmp):
    os.makedirs(tmp)
else:
    # clean when first start
    if not hasattr(globals(), "_tmp_cleared"):
        shutil.rmtree(tmp)
        os.makedirs(tmp)
        globals()["_tmp_cleared"] = True



class ParkinsonSAClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device, client_id=0, seed=None):
        self.client_id = client_id
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.mask = None
        self.round_number = 0
        self.peer_seeds = []
        self.seed = seed

    def get_parameters(self, config):
        params = [val.cpu().numpy() for val in self.model.state_dict().values()]

        # origin parameter
        if not hasattr(self, "param_saved"):
            np.save(os.path.join(tmp, f"param_client{self.client_id}.npy"), params[0])
            self.param_saved = True

        # parameter after mask
        if self.mask is not None:
            masked_params = apply_mask(params, self.mask)
            if not hasattr(self, "masked_param_saved"):
                np.save(os.path.join(tmp, f"masked_param_client{self.client_id}.npy"), masked_params[0])
                self.masked_param_saved = True
            return masked_params

        return params

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0

        for epoch in range(config.get("local_epochs", 1)):
            for inputs, labels in self.train_loader:
                if inputs.size(0) == 0:
                    continue
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        # create logic of mask
        # The round number is increased by 1 in each round
        self.round_number += 1
        # Extract the shape of the current model parameters
        param_shapes = [p.shape for p in self.get_parameters({})]
        mask_list = []
        # Use two seeds to generate a shared key, then use this key to generate a mask,
        # add all the masks together, and save the synthesized "total mask" to self.mask
        for peer_seed in self.peer_seeds:
            key = get_shared_key(self.seed, peer_seed)
            mask = create_mask(key, param_shapes, self.round_number)
            mask_list.append(mask)
        self.mask = [sum(m[i] for m in mask_list) for i in range(len(mask_list[0]))]

        # Calculate training metrics
        self.model.eval()
        y_true, y_pred = [], []

        with torch.no_grad():
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

        print(
            f"[Client] Training Loss: {total_loss / len(self.train_loader):.4f} | Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")

        return self.get_parameters(config={}), len(self.train_loader.dataset), {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        y_true = []
        y_pred = []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

        print(f"[Client] Evaluation - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        return float(1 - accuracy), len(self.test_loader.dataset), {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        }