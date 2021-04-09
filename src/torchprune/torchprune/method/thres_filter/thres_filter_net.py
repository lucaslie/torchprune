"""Module containing norm-based filter thresholding heuristics."""
import torch

from ..uni_filter.uni_filter_allocator import FilterUniAllocator
from ..base import DetFilterPruner, FilterSparsifier, FilterNet


class SoftNet(FilterNet):
    """Filter thresholding for pruning based on l_2-norm of filter.

    Pruning Step based on
    https://arxiv.org/abs/1808.06866
    Soft Filter Pruning for Accelerating Deep Convolutional Neural Networks.

    """

    @property
    def out_mode(self):
        """Return the indicator for out mode or in mode."""
        return True

    @property
    def deterministic(self):
        """Return the indicator for deterministic compression."""
        return True

    @property
    def retrainable(self):
        """Return the indicator whether we can retrain afterwards."""
        return True

    @property
    def _p_norm(self):
        return 2

    def _start_preprocessing(self):
        pass

    def _finish_preprocessing(self):
        pass

    def _get_allocator(self):
        return FilterUniAllocator(self.compressed_net, self.out_mode)

    def _get_pruner(self, ell):
        # define lambda to get filter sensitivity
        def _get_filter_sens(weight):
            return torch.norm(
                weight.view(weight.shape[0], -1), dim=-1, p=self._p_norm
            )

        # compute sensitivity of out features
        weight = self.compressed_net.compressible_layers[ell].weight
        sens_out = _get_filter_sens(weight)

        # create sparsifier
        sparsifier = DetFilterPruner(weight, sens_out)

        return sparsifier

    def _get_sparsifier(self, pruner):
        return FilterSparsifier(pruner, self.out_mode)


class FilterThresNet(SoftNet):
    """Filter thresholding for pruning based on l_1-norm of filter.

    Pruning Step based on
    https://arxiv.org/abs/1608.08710
    Pruning Filters for Efficient ConvNets.
    """

    @property
    def _p_norm(self):
        return 1
