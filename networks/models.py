import torch.nn as nn


class MNISTModelLight(nn.Module):
    def __init__(self):
        super(MNISTModelLight, self).__init__()
        self.fc1 = nn.Linear(in_features=28*28, out_features=128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten изображение
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = nn.functional.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)
