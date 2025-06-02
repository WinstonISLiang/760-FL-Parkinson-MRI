import flwr as fl
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from sklearn.metrics import accuracy_score, precision_score, recall_score

class ParkinsonClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        noise_multiplier = config.get("noise_multiplier", 1.0)
        local_epochs = config.get("local_epochs", 1)

        # Make the model DP-compatible
        self.model = ModuleValidator.fix(self.model)
        privacy_engine = PrivacyEngine()
        self.model, optimizer, self.train_loader = privacy_engine.make_private(
            module=self.model,
            optimizer=optim.Adam(self.model.parameters(), lr=3e-4),
            data_loader=self.train_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=3.0,
        )

        criterion = nn.CrossEntropyLoss()

        for epoch in range(local_epochs):
            for inputs, labels in self.train_loader:
                if inputs.size(0) == 0:
                    continue
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Only log locally for debugging
        epsilon = privacy_engine.get_epsilon(delta=1e-5)
        print(f"[Client] DP ε = {epsilon:.2f}; model update completed with SA simulation.")

        # Do NOT return metrics to server
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        y_true, y_pred = [], []
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
