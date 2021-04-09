"""Module containing the classic weight thresholding pruning heuristic."""

import torch.nn as nn
from ..base import DetFeatureSparsifier, DetFeaturePruner, WeightNet

from .thres_weight_tracker import ThresTracker
from .thres_weight_allocator import ThresAllocator


class ThresNet(WeightNet):
    """Classic prune heuristic of removing smallest-norm weights globally."""

    @property
    def deterministic(self):
        """Return the indicator for deterministic compression."""
        return True

    @property
    def retrainable(self):
        """Return the indicator whether we can retrain afterwards."""
        return True

    def __init__(self, original_net, loader_s, loss_handle):
        """Initialize with orignal_net and be done with it."""
        super().__init__(original_net, loader_s, loss_handle)

        # have some pseudo trackers
        self._trackers_weight = nn.ModuleList()

    def _start_preprocessing(self):
        # create for each layer
        for ell in self.layers:
            module = self.compressed_net.compressible_layers[ell]
            self._trackers_weight.append(ThresTracker(module))

    def _finish_preprocessing(self):
        del self._trackers_weight
        self._trackers_weight = nn.ModuleList()

    def _get_allocator(self):
        return ThresAllocator(self._trackers_weight)

    def _get_pruner(self, ell):
        module = self.compressed_net.compressible_layers[ell]
        return DetFeaturePruner(
            module.weight, self._trackers_weight[ell].sensitivity
        )

    def _get_sparsifier(self, pruner):
        return DetFeatureSparsifier(pruner)
