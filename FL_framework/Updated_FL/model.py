# -*- coding: utf-8 -*-

import torch.nn as nn
import torch


class Deeper3DCNN(nn.Module):
    def __init__(self, dropout=0.5):
        super(Deeper3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 4, 3, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=2, num_channels=4)    # replace BatchNorm3d(4)
        
        self.conv2 = nn.Conv3d(4, 8, 3, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=4, num_channels=8)    # replace BatchNorm3d(8)
        
        self.conv3 = nn.Conv3d(8, 16, 3, padding=1)
        self.gn3 = nn.GroupNorm(num_groups=8, num_channels=16)   # replace BatchNorm3d(16)
        
        self.conv4 = nn.Conv3d(16, 32, 3, padding=1)
        self.gn4 = nn.GroupNorm(num_groups=16, num_channels=32)  # replace BatchNorm3d(32)
        
        self.conv5 = nn.Conv3d(32, 64, 3, padding=1)
        self.gn5 = nn.GroupNorm(num_groups=32, num_channels=64)  # replace BatchNorm3d(64)

        self.pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = None
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.gn1(self.conv1(x))))
        x = self.pool(torch.relu(self.gn2(self.conv2(x))))
        x = self.pool(torch.relu(self.gn3(self.conv3(x))))
        x = self.pool(torch.relu(self.gn4(self.conv4(x))))
        x = self.pool(torch.relu(self.gn5(self.conv5(x))))

        x = x.view(x.size(0), -1)

        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 32).to(x.device)

        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)
