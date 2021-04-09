"""A module for the CNN5 model from double-descent, see below."""
import torch.nn as nn


class CNN5(nn.Module):
    """The CNN5 sequential convoluational architecture from double descent.

    This architecture is described and was used for experiments in:

    Deep Double Descent: Where Bigger Models and More Data Hurt
    https://openai.com/blog/deep-double-descent/
    https://arxiv.org/abs/1912.02292

    """

    def __init__(self, num_classes, k_scale=64):
        """Initialize the network.

        Args:
            num_classes (int): number of classes in the output.
            k_scale (int, optional): scaling factor for width. Defaults to 64.
        """
        super().__init__()

        def _get_conv_module(in_channels, out_channels, pool_kernel):
            """Generate conv, bn, relu, maxpool sequence for one layer."""
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(pool_kernel),
            )

        # some fixed parameters for the four conv layers
        self._channels = [3]
        self._channels.extend([k_scale * (2 ** i) for i in range(4)])
        self._pool_kernels = [1, 2, 2, 8]

        # now generate the conv layers.
        self._conv_layers = nn.Sequential(
            *[
                _get_conv_module(in_c, out_c, p_k)
                for in_c, out_c, p_k in zip(
                    self._channels[:-1], self._channels[1:], self._pool_kernels
                )
            ]
        )

        # model the linear layer as conv layer as well to make it easier on us
        self._linear = nn.Conv2d(self._channels[-1], num_classes, 1)

    def forward(self, x):
        """Pytorch forward function."""
        x = self._conv_layers(x)
        x = self._linear(x)
        return x.view(x.shape[0], -1)


def cnn5(num_classes, k_scale=64):
    """Return a fully-initialized CNN60K."""
    return CNN5(num_classes, k_scale)
