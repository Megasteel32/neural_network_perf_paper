import torch
import torch.nn as nn
import torch.nn.functional as F


class FullCNN(nn.Module):
    """Full CNN model following 784-50-100-500-1000-10-10 architecture using only convolutional layers."""

    BATCH_SIZE = 256
    EPOCHS = 10
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.5

    def __init__(self):
        super().__init__()

        # First convolutional block (1 -> 50)
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

        # Third convolutional block (100 -> 500)
        self.conv3 = nn.Sequential(
            nn.Conv2d(100, 500, kernel_size=3, padding=1),
            nn.BatchNorm2d(500),
            nn.ReLU(),
            nn.Dropout2d(p=self.DROPOUT_RATE)
        )

        # Fourth convolutional block (500 -> 1000)
        self.conv4 = nn.Sequential(
            nn.Conv2d(500, 1000, kernel_size=3, padding=1),
            nn.BatchNorm2d(1000),
            nn.ReLU(),
            nn.Dropout2d(p=self.DROPOUT_RATE)
        )

        # Fifth convolutional block (1000 -> 10)
        self.conv5 = nn.Sequential(
            nn.Conv2d(1000, 10, kernel_size=3, padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Dropout2d(p=self.DROPOUT_RATE)
        )

        # Final classification layer (10 -> 10)
        self.classifier = nn.Sequential(
            nn.Conv2d(10, 10, kernel_size=1),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        # Convolutional layers with pooling
        x = self.conv1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        # Global average pooling and classification
        x = self.classifier(x)
        x = torch.flatten(x, 1)

        return F.log_softmax(x, dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)