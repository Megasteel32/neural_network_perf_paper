import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialCNN(nn.Module):
    """CNN model based on reference architectures with larger FC layers."""

    BATCH_SIZE = 256
    EPOCHS = 10
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.5

    def __init__(self):
        super().__init__()

        # First convolutional block (784 -> 50)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 50, kernel_size=5, padding=2),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.Dropout2d(p=self.DROPOUT_RATE)
        )

        # Second convolutional block (50 -> 100)
        self.conv2 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=5, padding=2),
            nn.BatchNorm2d(100),
            nn.ReLU(),
            nn.Dropout2d(p=self.DROPOUT_RATE)
        )

        # Transition to fully connected layers
        self.flatten = nn.Flatten()

        # Fully connected layers matching reference architectures
        self.fc1 = nn.Sequential(
            nn.Linear(100 * 7 * 7, 500),  # After 2 max pooling operations: 28x28 -> 7x7
            nn.ReLU(),
            nn.Dropout(p=self.DROPOUT_RATE)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(500, 1000),
            nn.ReLU(),
            nn.Dropout(p=self.DROPOUT_RATE)
        )

        # Output layer
        self.fc3 = nn.Linear(1000, 10)

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Transition to fully connected layers
        x = self.flatten(x)

        # Fully connected layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return F.log_softmax(x, dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)