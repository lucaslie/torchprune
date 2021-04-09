"""Module containing implementations for uniform filter layer allocation."""
import copy
from ..base import BaseFilterAllocator


class FilterUniAllocator(BaseFilterAllocator):
    """The allocator for uniform filter-based layer allocation.

    This allocator allocates number of filters per layer such that the same
    fraction of filters per layer is retained.

    """

    def _get_boundaries(self):
        # for bisection method
        size_rel_min = 1e-12
        size_rel_max = 1.0

        return size_rel_min, size_rel_max

    def _get_proposed_num_features(self, arg):
        # set up proposed features (depending on in/out mode)
        size_rel = arg
        out_overlap = slice(None, -1)  # just like [:-1]
        in_overlap = slice(1, None)  # just like [1:]

        if self._out_mode:
            features = self._out_features
            other_features = self._in_features
            idx_same = -1  # don't reduce output feature of last layer
            overlap = out_overlap
            other_overlap = in_overlap
        else:
            features = self._in_features
            other_features = self._out_features
            idx_same = 0  # never reduce in features of input first layer
            overlap = in_overlap
            other_overlap = out_overlap

        # prune each layer by the same ratio
        pruned_features = (features * size_rel).round().int()

        # re-set pruned features of the "unprunable layer"
        pruned_features[idx_same] = features[idx_same]

        # sanity check
        pruned_features[pruned_features < 1] = 1

        # check feature reduction.
        feature_reduction = features - pruned_features

        # "propagate" compression by reducing other features by same amount
        pruned_other_features = copy.deepcopy(other_features)
        pruned_other_features[other_overlap] -= feature_reduction[overlap]
        pruned_other_features[pruned_other_features < 1] = 1

        # now return according to in / out mode
        if self._out_mode:
            pruned_out_features = pruned_features
            pruned_in_features = pruned_other_features
        else:
            pruned_out_features = pruned_other_features
            pruned_in_features = pruned_features

        return pruned_out_features, pruned_in_features
