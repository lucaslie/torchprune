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
from .imagenet_c import *  # noqa: F403,F401
from .objectnet import ObjectNet
from .cifar10 import *  # noqa: F403,F401
from .dds import DownloadDataset
