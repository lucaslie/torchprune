"""Module implements filter thresholding but with uniform sampling."""

import torch
from ..base import FilterNet, FilterSparsifier, DetFilterPruner

from .uni_filter_allocator import FilterUniAllocator


class FilterUniNet(FilterNet):
    """Filter thresholding with uniform filter pruning.

    This class simulates simply randomly choosing a subset of neurons from
    the big network. This corresponds to directly initializing a small
    network and thus helps establishing a "ground truth" for compression
    experiments without prior training, i.e., pruning on randomly initialized
    networks.

    """

    @property
    def deterministic(self):
        """Return the indicator for deterministic compression."""
        return False

    @property
    def retrainable(self):
        """Return the indicator whether we can retrain afterwards."""
        return True

    @property
    def out_mode(self):
        """Return the indicator for out mode or in mode."""
        return False

    def _start_preprocessing(self):
        pass

    def _finish_preprocessing(self):
        pass

    def _get_allocator(self):
        return FilterUniAllocator(self.compressed_net, self.out_mode)

    def _get_pruner(self, ell):
        weight = self.compressed_net.compressible_layers[ell].weight
        sensitivity = torch.rand(weight.shape[~self.out_mode]).to(
            weight.device
        )
        pruner = DetFilterPruner(weight, sensitivity)
        pruner.uniform = True
        return pruner

    def _get_sparsifier(self, pruner):
        return FilterSparsifier(pruner, self.out_mode)
