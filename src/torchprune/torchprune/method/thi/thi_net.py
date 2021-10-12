"""Module containing the ThiNet implementation which is also data-informed."""

import torch.nn as nn
from ...util import tensor
from ..uni_filter.uni_filter_allocator import FilterUniAllocator
from ..base import DetFilterPruner, FilterSparsifier, FilterNet

from .thi_tracker import ThiTracker


class ThiNet(FilterNet):
    """ThiNet, a data-informed heuristic for filter pruning."""

    @property
    def out_mode(self):
        """Return the indicator for out mode or in mode."""
        return False

    @property
    def deterministic(self):
        """Return the indicator for deterministic compression."""
        return True

    @property
    def retrainable(self):
        """Return the indicator whether we can retrain afterwards."""
        return True

    def __init__(self, original_net, loader_s, loss_handle):
        """Initialize with uncompressed net and data loader."""
        super().__init__(original_net, loader_s, loss_handle)

        # a few required objects
        self._trackers = nn.ModuleList()

    def _get_pruner(self, ell):
        weight = self.compressed_net.compressible_layers[ell].weight
        pruner = DetFilterPruner(weight, self._trackers[ell].sensitivity_in)
        return pruner

    def _get_sparsifier(self, pruner):
        return FilterSparsifier(pruner, self.out_mode)

    def _get_allocator(self):
        return FilterUniAllocator(self.compressed_net, self.out_mode)

    def _start_preprocessing(self):
        self._trackers = nn.ModuleList()
        # create and enable tracker for each layer
        for ell, module in enumerate(self.compressed_net.compressible_layers):
            self._trackers.append(ThiTracker(module, len(self._loader_s)))
            self._trackers[ell].enable_tracker()

        device = self.compressed_net.compressible_layers[0].weight.device

        num_batches = len(self._loader_s)

        # do a forward pass to obtain sensitivities
        for i_batch, (images, _) in enumerate(self._loader_s):
            if not any(trkr.more_data_needed() for trkr in self._trackers):
                break
            # compute thi-stats now
            self.compressed_net(tensor.to(images, device))
            print(f"Processed thi batch: [{i_batch+1}/{num_batches}]")

        num_layers = len(self.layers)

        # post-process via greedy approach
        for ell in self.layers:
            self._trackers[ell].finish_sensitivity()
            self._trackers[ell].disable_tracker()
            print(f"Post-processed thi stats, layer: [{ell+1}/{num_layers}]")

    def _finish_preprocessing(self):
        del self._trackers
        self._trackers = nn.ModuleList()
