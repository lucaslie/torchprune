"""Module containing the pruner implementations for SiPP."""
from abc import abstractmethod

import torch.nn as nn
import torch

from ..base import BasePruner


class BaseSensPruner(BasePruner):
    """A class for commonalities for pruners for sensitivity-based methods."""

    def __init__(self, tensor, tracker, **kwargs):
        """Initialize with tensor, tracker and flexible arguments."""
        super().__init__(tensor, tracker.sensitivity, **kwargs)

        # keep the indices
        self.idx_plus = tracker.idx_plus
        self.idx_minus = tracker.idx_minus

        # have a pruner for each
        self.pruner_plus = nn.Module()
        self.pruner_minus = nn.Module()

        self.bias = tracker.module.bias

    def compute_probabilities(self):
        """Compute the probabilities associated with neurons in the layer."""
        # get positive side
        self.pruner_plus = self._get_pruner_pm(self.idx_plus)

        # get negative side
        self.pruner_minus = self._get_pruner_pm(self.idx_minus)

    def prune(self, size_pruned):
        """Prune across neurons within the layer to the desired size."""
        num_samples_plus = self.pruner_plus.prune(size_pruned[0])
        num_samples_minus = self.pruner_minus.prune(size_pruned[1])
        return torch.stack((num_samples_plus, num_samples_minus))

    def _get_pruner_pm(self, idx_pm):
        # short hand notations
        weight = self.tensor.data
        sens = self.sensitivity

        # computations
        weight_pm = torch.zeros(weight.shape).to(weight.device)
        weight_pm[idx_pm] = weight[idx_pm]
        sens_pm = torch.zeros(sens.shape).to(sens.device)
        sens_pm[idx_pm] = sens[idx_pm]

        pruner_pm = self._pruner_class(weight_pm, sens_pm)
        pruner_pm.compute_probabilities()

        return pruner_pm

    @property
    @abstractmethod
    def _pruner_class(self):
        raise NotImplementedError("Not implemented by the abstract class!")
