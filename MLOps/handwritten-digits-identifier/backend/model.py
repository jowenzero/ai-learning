"""
CNN Model Architecture for Handwritten Digit Recognition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitCNN(nn.Module):
    """
    Convolutional Neural Network for handwritten digit recognition.

    Architecture:
    - Conv Layer 1: 1 input channel -> 32 output channels, 3x3 kernel
    - Conv Layer 2: 32 -> 64 channels, 3x3 kernel
    - Conv Layer 3: 64 -> 128 channels, 3x3 kernel
    - Fully Connected Layer 1: 128 * 3 * 3 -> 256 neurons
    - Fully Connected Layer 2: 256 -> 10 neurons (output classes)
    """

    def __init__(self):
        super(DigitCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout for regularization
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)  # 28x28 -> 14x14

        # Second convolutional block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)  # 14x14 -> 7x7

        # Third convolutional block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)  # 7x7 -> 3x3

        x = self.dropout1(x)

        # Flatten the tensor
        x = x.view(-1, 128 * 3 * 3)

        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        return x
