import torch
import torch.nn as nn

class Improved3DCNN(nn.Module):
    def __init__(self):
        super(Improved3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 4, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(4)
        self.conv2 = nn.Conv3d(4, 8, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(8)
        self.conv3 = nn.Conv3d(8, 16, 3, padding=1)
        self.bn3 = nn.BatchNorm3d(16)
        self.pool = nn.MaxPool3d(2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16 * 8 * 16 * 16, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)