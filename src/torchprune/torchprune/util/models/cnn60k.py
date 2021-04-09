"""A module for a special CNN60K implementation, see below."""
import torch.nn as nn


class CNN60K(nn.Module):
    """CNN60K implementation for particular paper comparison.

    This CNN was used in the Net-Trim paper (https://arxiv.org/abs/1611.05162)
    We implemented it here for comparison.
    """

    def __init__(self, num_classes=1, num_in_channels=3):
        """Initialize the network."""
        super().__init__()
        # conv parameters
        self.channels = [num_in_channels, 32, 32]
        self.kernels = [[5, 5], [5, 5]]
        self.strides = [[1, 1], [1, 1]]

        # max pool
        self.pool_kernel = [2, 2]

        # model first fc layer as conv layer...
        self.channels.append(512)
        # self.kernels.append([10, 10])  # MNIST
        self.kernels.append([12, 12])  # CIFAR10
        self.strides.append([1, 1])

        # fc parameters
        self.units = [512, num_classes]

        # construct network
        self.conv = None
        self.linear = None
        self._construct()

    def _construct(self):
        """Construct the network after we initialize parameters."""
        # construct conv layers
        self.conv = nn.Sequential()
        for ell in range(len(self.kernels)):
            self.conv.add_module(
                "conv" + str(ell),
                nn.Conv2d(
                    in_channels=self.channels[ell],
                    out_channels=self.channels[ell + 1],
                    kernel_size=self.kernels[ell],
                    stride=self.strides[ell],
                    padding=0,
                ),
            )
            self.conv.add_module("relu" + str(ell), nn.ReLU(inplace=True))

            # add a max pool one before the last conv layer
            if ell == len(self.kernels) - 2:
                self.conv.add_module(
                    "pool", nn.MaxPool2d(self.pool_kernel, padding=0)
                )

        # construct linear layers
        self.linear = nn.Sequential()
        num_lin_layers = len(self.units) - 1
        for ell in range(num_lin_layers):
            self.linear.add_module(
                "linear" + str(ell),
                nn.Linear(
                    in_features=self.units[ell],
                    out_features=self.units[ell + 1],
                ),
            )
            # don't add ReLU at the end
            if ell < num_lin_layers - 1:
                self.linear.add_module(
                    "relu" + str(ell), nn.ReLU(inplace=True)
                )

    def forward(self, x):
        """Pytorch forward function."""
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x


def cnn60k(num_classes, num_in_channels=3, **kwargs):
    """Return a fully-initialized CNN60K."""
    return CNN60K(num_classes, num_in_channels)
