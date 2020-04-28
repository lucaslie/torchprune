"""Module containing implementations for uniform filter layer allocation."""
import copy
import torch
from ..base import BaseFilterAllocator


class FilterUniAllocator(BaseFilterAllocator):
    """The allocator for uniform filter-based layer allocation.

    This allocator allocates number of filters per layer such that the same
    fraction of filters per layer is retained.

    """

    def _get_boundaries(self, keep_ratio):
        # for bisection method
        size_rel_min = 1e-10 * keep_ratio
        size_rel_max = 1.0

        return size_rel_min, size_rel_max

    def _get_proposed_num_features(self, arg):
        size_rel = arg
        out_features = copy.deepcopy(self._out_features)
        in_features = copy.deepcopy(self._in_features)

        # depending on out_mode cannot reduce everything...
        mask = torch.ones_like(out_features).bool()
        if self._out_mode:
            reducing = out_features
            mask[-1] = False  # never reduce output out_features
        else:
            reducing = in_features
            mask[0] = False  # never reduce input in_features

        # check proposed number of features according to relative size and
        # have at least one everywhere
        proposed = (reducing[mask].float() * size_rel).round().int()
        proposed[proposed < 1] = 1

        reducing[mask] = proposed

        if self._out_mode:
            in_features[1:] = out_features[:-1]
        else:
            out_features[:-1] = in_features[1:]

        return out_features, in_features
