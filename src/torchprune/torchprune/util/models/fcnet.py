"""Module that contains a generic fully-connected network implementation."""
import torch.nn as nn
import torch


class FCNet(nn.Module):
    """A vanilla fully-connected network."""

    def __init__(self, etas, batch_norm):
        """Initialize FCNet with number of neurons and possible batch norm.

        Args:
            etas(list): number of neurons per layer
            batch_norm(bool): whether to include batch norm between layers

        """
        super(FCNet, self).__init__()

        # a few quantities
        self.num_layers = len(etas)
        self.etas = etas

        # Sum of neurons from l = 2 to L
        self.eta = sum(etas[1:])

        # Input mask
        self.input_mask = torch.ones(self.etas[0], dtype=torch.bool)

        # max neurons from layers
        self.eta_star = max(etas)

        # check if we want batch normalization
        self.batch_norm = batch_norm

        # add layers
        self.layers = nn.Sequential()
        for ell in range(1, self.num_layers):
            self.layers.add_module(
                "linear" + str(ell - 1),
                nn.Linear(etas[ell - 1], etas[ell], bias=True),
            )
            # Don't add ReLU to output layer
            if ell is self.num_layers - 1:
                continue
            # add batch normalization if desired
            if self.batch_norm:
                self.layers.add_module(
                    "bn" + str(ell - 1), nn.BatchNorm1d(etas[ell])
                )
            self.layers.add_module(
                "relu" + str(ell - 1), nn.ReLU(inplace=True)
            )

    def forward(self, x):
        """Pytorch forward function."""
        # Reshape torch tensor
        x = x.view(x.shape[0], -1)

        # Apply input mask
        x = x[:, self.input_mask]

        # Return output
        return self.layers(x)


def lenet300_100(num_classes, **kwargs):
    """Initialize a LeNet300-100 with the FCNet class."""
    return FCNet([784, 300, 100, num_classes], False)


def lenet500_300_100(num_classes, **kwargs):
    """Initialize a LeNet500-300-100 with the FCNet class."""
    return FCNet([784, 500, 300, 100, num_classes], False)


def fcnet_nettrim(num_classes, **kwargs):
    """Return a FC architectures according to Net-Trim.

    Net-Trim paper: https://epubs.siam.org/doi/pdf/10.1137/19M1246468
    """
    return FCNet([784, 300, 1000, 100, num_classes], False)
