# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn

class Deeper3DCNN(nn.Module):
    def __init__(self, dropout=0.5):
        super(Deeper3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 4, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(4)
        self.conv2 = nn.Conv3d(4, 8, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(8)
        self.conv3 = nn.Conv3d(8, 16, 3, padding=1)
        self.bn3 = nn.BatchNorm3d(16)
        self.conv4 = nn.Conv3d(16, 32, 3, padding=1)
        self.bn4 = nn.BatchNorm3d(32)
        self.conv5 = nn.Conv3d(32, 64, 3, padding=1)
        self.bn5 = nn.BatchNorm3d(64)

        self.pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = None
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = self.pool(torch.relu(self.bn4(self.conv4(x))))
        x = self.pool(torch.relu(self.bn5(self.conv5(x))))

        x = x.view(x.size(0), -1)

        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 32).to(x.device)

        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)