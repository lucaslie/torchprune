"""Module containing allocators for Messi."""

import torch

from ..base_decompose import BaseDecomposeAllocator


class MessiAllocator(BaseDecomposeAllocator):
    """The allocator for Messi pruning."""

    def _get_k_splits(self, desired_k_split):
        """Get desired k for each layer."""
        return torch.ones_like(self._in_features) * desired_k_split

    def _compute_ranks_j_for_arg(self, arg, ranks, num_weights_per_j):
        """Get the desired rank for each layer (const relative rank)."""
        return (arg * ranks).round()
