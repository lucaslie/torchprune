"""Module with Deepknight network."""
import torch.nn as nn


class Deepknight(nn.Module):
    """The deepknight network for Alex Amini's paper."""

    def __init__(self, num_classes=1):
        """Initialize the deepknight with the right number of classes."""
        super().__init__()
        # conv parameters
        self.channels = [3, 24, 36, 48, 64, 64]
        self.kernels = [[5, 5], [5, 5], [3, 3], [3, 3], [3, 3]]
        self.strides = [[2, 2], [2, 2], [2, 2], [1, 1], [1, 1]]

        # first fc layer: we will model the first linear layer as conv
        #                 operations to avoid reshaping (reshaping messes up
        #                 our compression step)
        self.channels.append(1000)
        self.kernels.append([1, 24])
        self.strides.append([1, 1])

        # fc parameters
        self.units = [1000, 100, num_classes]

        # construct network
        self.conv = None
        self.linear = None
        self._construct()

    def _construct(self):
        """Construct the network after initialization."""
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


def deepknight(num_classes, **kwargs):
    """Return an intialized deepknight net."""
    return Deepknight(num_classes)
