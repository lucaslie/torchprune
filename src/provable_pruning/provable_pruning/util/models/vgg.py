"""Module with VGG wrapper that is compatible with filter pruning."""
import torch.nn as nn


class VGGWrapper(nn.Module):
    """A custom wrapper to make sure it's compatible with Filter Pruning."""

    def __init__(self, vgg_net):
        """Initialize the class with a vanilla VGG network."""
        super().__init__()
        # create a conv layer that corresponds to the first linear layer
        linear1 = vgg_net.classifier[0]
        conv = nn.Conv2d(512, 4096, 7, 7)

        # copy data into it
        conv.bias.data.copy_(linear1.bias.data)
        conv.weight.data.view(4096, -1).copy_(linear1.weight.data)

        # replace the layer in the sequential classifier part
        vgg_net.classifier = nn.Sequential(
            conv, nn.Flatten(1), *vgg_net.classifier[1:]
        )

        self.vgg_net = vgg_net

    def forward(self, x):
        """Pytorch forward function."""
        x = self.vgg_net.features(x)
        x = self.vgg_net.avgpool(x)
        x = self.vgg_net.classifier(x)
        return x
