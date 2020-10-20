"""Module containing the ThiNet implementation which is also data-informed."""

import torch.nn as nn
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

    def __init__(self, original_net, loader_s):
        """Initialize with uncompressed net and data loader."""
        super().__init__(original_net)

        # a few required objects
        self._loader_s = loader_s
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
        for ell in self.layers:
            module = self.compressed_net.compressible_layers[ell]
            # self._trackers[l] = ThiTracker(module)
            self._trackers.append(ThiTracker(module))
            self._trackers[ell].enable_tracker()

            # do a forward pass to obtain sensitivities
            device = module.weight.device
            for images, _ in self._loader_s:
                self.compressed_net(images.to(device))

            self._trackers[ell].finish_sensitivity()
            self._trackers[ell].disable_tracker()

    def _finish_preprocessing(self):
        del self._trackers
        self._trackers = nn.ModuleList()

    def compress(self, keep_ratio, from_original=True, initialize=True):
        """Execute the compression step."""
        if (
            self.__class__.__name__ == "ThiNet"
            and "ResNet" in self.compressed_net.torchnet._get_name()
        ):
            keep_ratio *= 0.75
        if (
            self.__class__.__name__ == "ThiNet"
            and "DenseNet" in self.compressed_net.torchnet._get_name()
        ):
            keep_ratio *= 2.0

        return super().compress(keep_ratio, from_original, initialize)
