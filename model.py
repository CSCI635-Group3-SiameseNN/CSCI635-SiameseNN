import torch
import torch.nn as nn
import torch.nn.functional as F
from kafnets import KAF

class Siamese(nn.Module):
    # Original model unchanged
    def __init__(self):
        super(Siamese, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 10),  # 64@96*96
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),  # 64@48*48
            nn.Conv2d(64, 128, 7),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),   # 128@21*21
            nn.Conv2d(128, 128, 4),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2), # 128@9*9
            nn.Conv2d(128, 256, 4),
            nn.SiLU(inplace=True),
        )
        self.liner = nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid())
        self.out = nn.Linear(4096, 1)

    def forward_one(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1)
        x = self.liner(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        return out

class SiameseATTReLU(nn.Module):
    # Architecture for AT&T dataset with ReLU as previously described
    def __init__(self):
        super(SiameseATTReLU, self).__init__()
        # Layer 1: Conv1, Conv2, Conv3 each followed by ReLU and MaxPool
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # After 3 pooling layers on a 92x112 image:
        # Approx size after conv3: 8 x 11 x 14 = 1232 features (flattened)
        self.fc1 = nn.Sequential(
            nn.Linear(1232, 80000),
            nn.SiLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(80000, 500),
            nn.SiLU(inplace=True)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(500, 250),
            nn.SiLU(inplace=True)
        )
        self.fc4 = nn.Sequential(
            nn.Linear(250, 5),
            nn.SiLU(inplace=True)
        )
        self.out = nn.Linear(5, 1)

    def forward_one(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        return out
