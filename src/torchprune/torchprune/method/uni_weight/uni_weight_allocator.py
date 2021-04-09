"""Module for uniform layer allocation for weight pruning."""
import torch
from ..base import BaseAllocator


class UniAllocator(BaseAllocator):
    """The allocator for uniform weight-based layer allocation."""

    def __init__(self, modules):
        """Initialize with the available modules for each layer."""
        super().__init__()

        self._num_layers = len(modules)
        self.register_buffer("_num_filters", torch.Tensor())
        self.register_buffer("_num_per_filter", torch.Tensor())
        self._num_filters = torch.zeros(
            self._num_layers, dtype=torch.int, device=modules[0].weight.device
        )
        self._num_per_filter = torch.zeros_like(self._num_filters)

        # for easier book keeping
        self._offsets[-1] = [0, 0]

        for ell, module in enumerate(modules):
            self._num_filters[ell] = module.weight.shape[0]
            self._num_per_filter[ell] = module.weight[0].numel()
            end_prev = self._offsets[ell - 1][1]
            self._offsets[ell] = [end_prev, end_prev + self._num_filters[ell]]

    def _allocate_method(self, budget):
        self._allocation = torch.zeros(
            torch.sum(self._num_filters),
            dtype=torch.int,
            device=self._num_filters.device,
        )

        size_original = int((self._num_filters * self._num_per_filter).sum())
        keep_ratio = float(budget) / float(size_original)

        for ell in range(self._num_layers):
            alloc_l = keep_ratio * self._num_per_filter[ell].float()
            self._allocation[
                self._offsets[ell][0] : self._offsets[ell][1]
            ] = torch.max(alloc_l, torch.ones_like(alloc_l)).int()

    def _extract(self, arr, offsets, ell):
        arr_layer = arr[offsets[ell][0] : offsets[ell][1]]
        return arr_layer
