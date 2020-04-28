# flake8: noqa: F401, F403
"""Package with all custom net implementation and CIFAR nets."""
# import custom nets
from .fcnet import FCNet, lenet300_100, lenet500_300_100
from .lenet5 import lenet5
from .deepknight import deepknight
from .cnn60k import cnn60k
from .cnn5 import cnn5
from .vgg import VGGWrapper

# import cifar nets
from .cnn.models.cifar import *
