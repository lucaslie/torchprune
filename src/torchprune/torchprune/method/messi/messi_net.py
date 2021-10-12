"""Module containing the MessiNet implementations."""

from ..base_decompose import BaseDecomposeNet
from .messi_allocator import MessiAllocator
from .messi_sparsifier import MessiSparsifier


class MessiNet(BaseDecomposeNet):
    """Projective clustering for any weight layer."""

    @property
    def _allocator_type(self):
        """Get allocator type."""
        return MessiAllocator

    @property
    def _sparsifier_type(self):
        """Get sparsifier type."""
        return MessiSparsifier

    @property
    def _k_split(self):
        """Get number of k splits in each layer."""
        return 3


class MessiNet5(MessiNet):
    """Projective clustering for any weight layer with k=5."""

    @property
    def _k_split(self):
        """Get number of k splits in each layer."""
        return 5
