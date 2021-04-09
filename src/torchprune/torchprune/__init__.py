# flake8: noqa: F403,F401
"""The torchprune package.

This package contains the implementation of various pruning methods and
utilities to train, prune, retrain compressed networks.
"""

# import all pruning methods straight to the top-level
from .method import *

# import util as well
from . import util
