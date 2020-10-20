# flake8: noqa: F403,F401
"""Package that contains all datasets we use for compression."""

from torchvision.datasets import (
    CIFAR10,
    CIFAR100,
    FashionMNIST,
    MNIST,
    ImageFolder,
)

from .driving import Driving
from .imagenet import ImageNet
from .dds import DownloadDataset
