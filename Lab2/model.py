import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTConvNet(nn.Module):
    """
    Architecture Design:
    - Phase 3 (Reduce Bias): 2 Conv layers with 32 and 64 filters to capture complexity.
    - Phase 4 (Reduce Variance): Dropout(0.5) to prevent overfitting on training data.
    """
    def __init__(self, dropout_rate=0.5):
        super(MNISTConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x