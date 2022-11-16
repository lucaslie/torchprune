# flake8: noqa: F401, F403
"""Package with all custom net implementation and CIFAR nets."""
# import custom nets
from .fcnet import FCNet, lenet300_100, lenet500_300_100, fcnet_nettrim
from .lenet5 import lenet5
from .deepknight import deepknight
from .deeplab import *
from .cnn60k import cnn60k
from .cnn5 import cnn5
from .bert import bert
from .node import *
from .cnf import *
from .ffjord import *
from .ffjord_tabular import *
from .ffjord_cnf import *

# import cifar nets
from ..external.cnn.models.cifar import *

# import imagenet models
# (give them a sub-module because of name-clashes otherwise)
from . import imagenet_models as imagenet
