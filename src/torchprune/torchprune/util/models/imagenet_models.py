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


class AlexNetBn(nn.Module):
    """An AlexNet with batch normalization added after each linear layer."""

    def __init__(self, num_classes=1000):
        """Initialize with batch normalization now."""
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(256, 4096, 6, bias=False),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, 1, bias=False),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True),
            nn.Flatten(1),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        """Regular forward function."""
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


def alexnet_bn(*args, **kwargs):
    """Return a batch-normalized AlexNet with desired number of classes."""
    return AlexNetBn(*args, **kwargs)
