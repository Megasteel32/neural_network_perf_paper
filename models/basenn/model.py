import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseNN(nn.Module):
    """Pure Neural Network model following 784-800-10 architecture."""

    BATCH_SIZE = 256
    EPOCHS = 10
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.5

    def __init__(self):
        super().__init__()

        # Simple 2-layer architecture
        self.flatten = nn.Flatten()

        # First fully connected layer (784 -> 800)
        self.fc1 = nn.Sequential(
            nn.Linear(784, 800),
            nn.ReLU(),
            nn.Dropout(p=self.DROPOUT_RATE),
            nn.BatchNorm1d(800)
        )

        # Output layer (800 -> 10)
        self.fc2 = nn.Linear(800, 10)

    def forward(self, x):
        # Flatten the input
        x = self.flatten(x)

        # Forward pass through layers
        x = self.fc1(x)
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)