from __future__ import annotations

from torch import nn


class SimpleModel(nn.Module):
    def __init__(self, feature_size, num_classes):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(feature_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x


class SimpleAudioModel(nn.Module):
    def __init__(self, input_size, feature_size, num_classes):
        super().__init__()
        self.feature_size = feature_size
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64 * ((feature_size - 2) // 2), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(-1, 64 * ((self.feature_size - 2) // 2))
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
