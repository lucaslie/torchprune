"""A module containing uniform weight pruning."""

import torch
from ..base import RandFeaturePruner, RandFeatureSparsifier, WeightNet

from .uni_weight_allocator import UniAllocator


class UniNet(WeightNet):
    """A simple weight pruning algorithm based on uniform sampling."""

    @property
    def deterministic(self):
        """Return the indicator for deterministic compression."""
        return False

    @property
    def retrainable(self):
        """Return the indicator whether we can retrain afterwards."""
        return True

    def _start_preprocessing(self):
        pass

    def _finish_preprocessing(self):
        pass

    def _get_allocator(self):
        return UniAllocator(self.compressed_net.compressible_layers)

    def _get_pruner(self, ell):
        weight = self.compressed_net.compressible_layers[ell].weight
        sensitivity = torch.ones(weight.shape).to(weight.device)
        pruner = RandFeaturePruner(weight, sensitivity)
        pruner.uniform = True
        return pruner

    def _get_sparsifier(self, pruner):
        return RandFeatureSparsifier(pruner)
