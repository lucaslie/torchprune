"""Module containing SiPP pruning methods."""

from ..base import WeightNet
from ..base_sens import BaseSensNet

from .sipp_allocator import (
    SiPPAllocator,
    SiPPHybridAllocator,
    SiPPRandAllocator,
)
from .sipp_pruner import SiPPPruner, SiPPHybridPruner, SiPPRandPruner
from .sipp_sparsifier import (
    SiPPSparsifier,
    SiPPRandSparsifier,
    SiPPHybridSparsifier,
)
from .sipp_tracker import SiPPRandTracker, SiPPTracker


class SiPPNet(BaseSensNet, WeightNet):
    """Deterministic, classic SiPP."""

    @property
    def deterministic(self):
        """Return the indicator for deterministic compression."""
        return True

    @property
    def _tracker(self):
        return SiPPTracker

    @property
    def _allocator(self):
        return SiPPAllocator

    @property
    def _pruner(self):
        return SiPPPruner

    @property
    def _sparsifier(self):
        return SiPPSparsifier


class SiPPNetHybrid(BaseSensNet, WeightNet):
    """Hybrid, classic SiPP."""

    @property
    def deterministic(self):
        """Return the indicator for deterministic compression."""
        return False

    @property
    def _tracker(self):
        return SiPPRandTracker

    @property
    def _allocator(self):
        return SiPPHybridAllocator

    @property
    def _pruner(self):
        return SiPPHybridPruner

    @property
    def _sparsifier(self):
        return SiPPHybridSparsifier


class SiPPNetRand(BaseSensNet, WeightNet):
    """Purely random, sampling-based, classic SiPP."""

    @property
    def deterministic(self):
        """Return the indicator for deterministic compression."""
        return False

    @property
    def _tracker(self):
        return SiPPRandTracker

    @property
    def _allocator(self):
        return SiPPRandAllocator

    @property
    def _pruner(self):
        return SiPPRandPruner

    @property
    def _sparsifier(self):
        return SiPPRandSparsifier
