"""Module containing implementations for provable filter pruning methods."""

from ..base import FilterNet
from ..base_sens import BaseSensNet

from .pfp_allocator import PFPAllocator, PFPRandAllocator, PFPTopAllocator
from .pfp_pruner import PFPPruner, PFPRandPruner, PFPTopPruner
from .pfp_sparsifier import PFPSparsifier, PFPRandSparsifier, PFPTopSparsifier
from .pfp_tracker import PFPTracker


class PFPNet(BaseSensNet, FilterNet):
    """This class implements the deterministic version of PFP."""

    @property
    def out_mode(self):
        """Return the indicator for out mode or in mode."""
        return False

    @property
    def deterministic(self):
        """Return the indicator for deterministic compression."""
        return True

    @property
    def _tracker(self):
        return PFPTracker

    @property
    def _allocator(self):
        return PFPAllocator

    @property
    def _pruner(self):
        return PFPPruner

    @property
    def _sparsifier(self):
        return PFPSparsifier


class PFPNetRand(BaseSensNet, FilterNet):
    """This class implements the randomized version of PFP."""

    @property
    def out_mode(self):
        """Return the indicator for out mode or in mode."""
        return False

    @property
    def deterministic(self):
        """Return the indicator for deterministic compression."""
        return False

    @property
    def _tracker(self):
        return PFPTracker

    @property
    def _allocator(self):
        return PFPRandAllocator

    @property
    def _pruner(self):
        return PFPRandPruner

    @property
    def _sparsifier(self):
        return PFPRandSparsifier


class PFPNetTop(BaseSensNet, FilterNet):
    """This class implements the partially-derandomized version of PFP."""

    @property
    def out_mode(self):
        """Return the indicator for out mode or in mode."""
        return False

    @property
    def deterministic(self):
        """Return the indicator for deterministic compression."""
        return False

    @property
    def _tracker(self):
        return PFPTracker

    @property
    def _allocator(self):
        return PFPTopAllocator

    @property
    def _pruner(self):
        return PFPTopPruner

    @property
    def _sparsifier(self):
        return PFPTopSparsifier
