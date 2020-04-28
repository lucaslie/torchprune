"""Module containing a custom LeNet5 implementation."""
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """A classic LeNet5 architecture.

    from
    https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    """

    def __init__(self, num_classes, num_in_channels=3):
        """Initialize the LeNet5 with the desired number of output classes."""
        super().__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(num_in_channels, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        # we model first fc1 layer as conv layer to avoid reshaping
        self.fc1 = nn.Conv2d(16, 120, 6)
        # self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        """Pytorch forward function."""
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.fc1(x).view(x.shape[0], -1))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def lenet5(num_classes, num_in_channels=3, **kwargs):
    """Initialize and return a LeNet-5."""
    return LeNet5(num_classes, num_in_channels)
