import torch.nn as nn
import torch.nn.functional as F
from models.base import BASE
from data.dataset import SpectroDataset

class CNN(BASE):
    def __init__(self, input_channels=1, H=128, W=32, num_classes=30):
        super(CNN, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # after 4 pooling layers of (2,2): H/16, W/16
        final_h = H // 2 
        final_w = W // 2 

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * final_h * final_w, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.fc1(x)

        return x

    @classmethod
    def supported_dataset(cls):
        return SpectroDataset
