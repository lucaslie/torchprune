"""Module containing SNIP pruning heuristic.

Read more about it here:
https://arxiv.org/abs/1810.02340
"""

import torch
import torch.nn as nn
from ...util import tensor
from ..base import DetFeatureSparsifier, DetFeaturePruner, WeightNet
from ..thres_weight.thres_weight_allocator import ThresAllocator

from .snip_tracker import SnipTracker


class SnipNet(WeightNet):
    """Prune heuristic removing weights with smallest norm gradient product."""

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

        # have some SNIP trackers
        self._trackers = nn.ModuleList()

    def _start_preprocessing(self):
        self._trackers = nn.ModuleList()
        # create and enable tracker for each layer
        with torch.enable_grad():
            for ell in self.layers:
                module = self.compressed_net.compressible_layers[ell]
                self._trackers.append(SnipTracker(module))
                self._trackers[ell].enable_tracker()

                # do a forward pass and backward pass to obtain sensitivities
                device = module.weight.device
                for images, targets in self._loader_s:
                    images = tensor.to(images, device, non_blocking=True)
                    targets = tensor.to(targets, device, non_blocking=True)
                    outs = self.compressed_net(images)
                    loss = self._loss_handle(outs, targets)
                    loss.backward()

                self._trackers[ell].disable_tracker()

    def _finish_preprocessing(self):
        del self._trackers
        self._trackers = nn.ModuleList()

    def _get_allocator(self):
        return ThresAllocator(self._trackers)

    def _get_pruner(self, ell):
        module = self.compressed_net.compressible_layers[ell]
        return DetFeaturePruner(module.weight, self._trackers[ell].sensitivity)

    def _get_sparsifier(self, pruner):
        return DetFeatureSparsifier(pruner)
