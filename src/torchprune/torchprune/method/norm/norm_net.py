"""Module implementing norm-based sensitivity for sample-based pruning."""
from abc import ABC, abstractmethod

import torch

from ..uni_weight.uni_weight_allocator import UniAllocator
from ..base import WeightNet, RandFeaturePruner, RandFeatureSparsifier


class NormNet(WeightNet, ABC):
    """Base class constructing the norm-based sensitivities for sampling."""

    @property
    def deterministic(self):
        """Return the indicator for deterministic compression."""
        return False

    @property
    def retrainable(self):
        """Return the indicator whether we can retrain afterwards."""
        return True

    @property
    @abstractmethod
    def _p_norm(self):
        raise NotImplementedError

    def _start_preprocessing(self):
        pass

    def _finish_preprocessing(self):
        pass

    def _get_allocator(self):
        return UniAllocator(self.compressed_net.compressible_layers)

    def _get_pruner(self, ell):
        weight = self.compressed_net.compressible_layers[ell].weight.data
        if self._p_norm == -1:
            # If p == -1, then we use a combination of ell_1 and ell_2
            sensitivity = 0.5 * (
                torch.abs(weight) ** 1 + torch.abs(weight) ** 2
            )
        else:
            sensitivity = torch.abs(weight) ** self._p_norm
        pruner = RandFeaturePruner(weight, sensitivity)
        return pruner

    def _get_sparsifier(self, pruner):
        return RandFeatureSparsifier(pruner)


class EllOneNet(NormNet):
    """The l_1-based sensitivity version of NormNet."""

    @property
    def _p_norm(self):
        return 1


class EllTwoNet(NormNet):
    """The l_2-based sensitivity version of NormNet."""

    @property
    def _p_norm(self):
        return 2


class EllOneAndTwoNet(NormNet):
    """The mixture l_1 and l_2 sensitivity version of NormNet."""

    @property
    def _p_norm(self):
        return -1
