"""Module containing the pruner implementations for SiPP."""

import torch.nn as nn

from ..base import BasePruner, RandFeaturePruner, DetFeaturePruner
from ..base_sens import BaseSensPruner


class SiPPRandPruner(BaseSensPruner):
    """The pruner for random SiPPNetRand."""

    @property
    def _pruner_class(self):
        return RandFeaturePruner


class SiPPPruner(BaseSensPruner):
    """The pruner for deterministic SiPPNet."""

    @property
    def _pruner_class(self):
        return DetFeaturePruner


class SiPPHybridPruner(BasePruner):
    """The pruner for hybrid SiPPNetHybrid."""

    def __init__(self, tensor, tracker, **kwargs):
        """Initialize with tensor, tracker and flexible arguments."""
        super().__init__(tensor, tracker.sensitivity, **kwargs)

        self.pruner_sipp_det = SiPPPruner(
            tensor=tensor, tracker=tracker, **kwargs
        )
        self.pruner_sipp_rand = SiPPRandPruner(
            tensor=tensor, tracker=tracker, **kwargs
        )

    def compute_probabilities(self):
        """Compute the probabilities associated with neurons in the layer."""
        self.pruner_sipp_rand.compute_probabilities()
        self.pruner_sipp_det.compute_probabilities()

    def prune(self, size_pruned):
        """Prune across neurons within the layer to the desired size.

        Note that
        positive sample number --> do deterministic
        negative sample number --> do randomized

        """
        size_pruned_det = nn.functional.relu(size_pruned)
        size_pruned_rand = nn.functional.relu(-size_pruned)

        num_samples_det = self.pruner_sipp_det.prune(size_pruned_det)
        num_samples_rand = self.pruner_sipp_rand.prune(size_pruned_rand)

        return num_samples_det - num_samples_rand
