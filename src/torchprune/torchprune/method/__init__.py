# flake8: noqa: F403,F401
"""The compression module with all the compression methods."""

# we properly import the nets in each sub-package so * is fine here
from .alds import *
from .base import BaseCompressedNet, CompressedNet, WeightNet, FilterNet
from .base_decompose import BaseDecomposeNet
from .base_sens import BaseSensNet
from .messi import *
from .norm import *
from .pca import *
from .pfp import *
from .rank_learned import *
from .ref import *
from .sipp import *
from .snip import *
from .svd import *
from .thi import *
from .thres_filter import *
from .thres_weight import *
from .uni_filter import *
from .uni_weight import *
