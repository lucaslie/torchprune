"""Module with Deeplab v3 segmentation networks."""

import torch
import torch.nn as nn
import torchvision.models.segmentation as seg_models


class SingleOutNet(nn.Module):
    """A wrapper module to only return "out" from output dictionary in eval."""

    def __init__(self, network):
        """Initialize with the network that needs to be wrapped."""
        super().__init__()
        self.network = network

    def forward(self, x):
        """Only return the "out" of all the outputs."""
        if self.training or torch.is_grad_enabled():
            return self.network.forward(x)
        else:
            return self.network.forward(x)["out"]


def fcn_resnet50(num_classes):
    """Return torchvision fcn_resnet50 and wrap it in SingleOutNet."""
    return SingleOutNet(
        seg_models.fcn_resnet50(num_classes=num_classes, aux_loss=True)
    )


def fcn_resnet101(num_classes):
    """Return torchvision fcn_resnet101 and wrap it in SingleOutNet."""
    return SingleOutNet(
        seg_models.fcn_resnet101(num_classes=num_classes, aux_loss=True)
    )


def deeplabv3_resnet50(num_classes):
    """Return torchvision deeplabv3_resnet50 and wrap it in SingleOutNet."""
    return SingleOutNet(
        seg_models.deeplabv3_resnet50(num_classes=num_classes, aux_loss=True)
    )


def deeplabv3_resnet101(num_classes):
    """Return torchvision deeplabv3_resnet101 and wrap it in SingleOutNet."""
    return SingleOutNet(
        seg_models.deeplabv3_resnet101(num_classes=num_classes, aux_loss=True)
    )
