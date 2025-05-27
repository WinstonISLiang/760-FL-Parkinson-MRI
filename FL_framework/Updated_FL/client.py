from fedml import mlops
import torch
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from model import Deeper3DCNN
from dataset.client_dataset import ClientDataset
from torch.utils.data import DataLoader

def train(args, train_data, model, device, **kwargs):
    model.to(device)
    model.train()

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # Attach DP engine
    privacy_engine = PrivacyEngine(
        model,
        batch_size=args.batch_size,
        sample_size=len(train_loader.dataset),
        alphas=[10, 100],
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
    )
    privacy_engine.attach(optimizer)

    for epoch in range(args.epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.squeeze(2)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    epsilon, _ = privacy_engine.get_privacy_spent(delta=1e-5)
    mlops.log_metric(args.run_id, "epsilon", epsilon)
    return model

def test(args, test_data, model, device, **kwargs):
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    model.to(device)
    model.eval()

    loader = DataLoader(test_data, batch_size=args.batch_size)
    preds, targets = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.squeeze(2)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, average='weighted', zero_division=0)
    recall = recall_score(targets, preds, average='weighted', zero_division=0)
    return acc, prec, recall
