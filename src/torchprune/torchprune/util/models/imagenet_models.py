"""Module with all ImageNet models."""
import torch.nn as nn

# simply import all torchvision models here
from torchvision.models import *  # noqa: F401, F403

# also import torchvision models with name
import torchvision.models as models

# below we overwrite the torchvision.models.vgg models since we have to wrap
# them.


class _VGGWrapper(nn.Module):
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


def vgg11(*args):
    """Return a wrapped torchvision.models.vgg11."""
    return _VGGWrapper(models.vgg11(*args))


def vgg11_bn(*args):
    """Return a wrapped torchvision.models.vgg11_bn."""
    return _VGGWrapper(models.vgg11_bn(*args))


def vgg13(*args):
    """Return a wrapped torchvision.models.vgg13."""
    return _VGGWrapper(models.vgg13(*args))


def vgg13_bn(*args):
    """Return a wrapped torchvision.models.vgg13_bn."""
    return _VGGWrapper(models.vgg13_bn(*args))


def vgg16(*args):
    """Return a wrapped torchvision.models.vgg16."""
    return _VGGWrapper(models.vgg16(*args))


def vgg16_bn(*args):
    """Return a wrapped torchvision.models.vgg16_bn."""
    return _VGGWrapper(models.vgg16_bn(*args))


def vgg19(*args):
    """Return a wrapped torchvision.models.vgg19."""
    return _VGGWrapper(models.vgg19(*args))


def vgg19_bn(*args):
    """Return a wrapped torchvision.models.vgg19_bn."""
    return _VGGWrapper(models.vgg19_bn(*args))
