# flake8: noqa: F403,F401
"""The compression module with all the compression methods."""

# we properly import the nets in each sub-package so * is fine here
from .base import BaseCompressedNet, CompressedNet, WeightNet, FilterNet
from .base_sens import BaseSensNet
from .norm import *
from .pfp import *
from .ref import *
from .sipp import *
from .snip import *
from .thi import *
from .thres_filter import *
from .thres_weight import *
from .uni_filter import *
from .uni_weight import *
