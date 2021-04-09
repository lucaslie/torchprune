"""Module containing sparsifiers for SiPP."""
from abc import abstractmethod

import torch.nn as nn

from ..base import (
    BaseSparsifier,
    DetFeatureSparsifier,
    RandFeatureSparsifier,
)


class BaseSiPPSparsifier(BaseSparsifier):
    """The base sparsifier for SiPP."""

    def __init__(self, pruner):
        """Initialize the sparsifier from the pruner of the same layer."""
        super().__init__(pruner)

        # get positive side
        self._sparsifier_plus = self._sparsifier_class(pruner.pruner_plus)

        # get negative side
        self._sparsifier_minus = self._sparsifier_class(pruner.pruner_minus)
        self._updated_k = False

    @property
    @abstractmethod
    def _sparsifier_class(self):
        raise NotImplementedError("Not implemented by the abstract class!")

    def sparsify(self, num_samples):
        """Sparsify the edges of the associated neurons with num_samples.

        Sparsify weights for an entire weight matrix by sampling separately
        both the pos. and neg. weights.

        """
        weight_hat = self._sparsifier_plus.sparsify(num_samples[0])
        weight_hat += self._sparsifier_minus.sparsify(num_samples[1])
        return weight_hat


class SiPPSparsifier(BaseSiPPSparsifier):
    """The sparsifier for deterministic SiPP."""

    @property
    def _sparsifier_class(self):
        return DetFeatureSparsifier


class SiPPRandSparsifier(BaseSiPPSparsifier):
    """The sparsifier for random SiPPRand."""

    @property
    def _sparsifier_class(self):
        return RandFeatureSparsifier


class SiPPHybridSparsifier(BaseSparsifier):
    """The sparsifier for hybrid SiPPHybrid."""

    def __init__(self, pruner):
        """Initialize the sparsifier from the pruner of the same layer."""
        super().__init__(pruner)
        self._sparsifier_sipp_det = SiPPSparsifier(pruner.pruner_sipp_det)
        self._sparsifier_sipp_rand = SiPPRandSparsifier(
            pruner.pruner_sipp_rand
        )

    def sparsify(self, num_samples):
        """Sparsify the edges of the associated neurons with num_samples.

        positive sample number --> do deterministic
        negative sample number --> do randomized

        """
        num_samples_det = nn.functional.relu(num_samples)
        num_samples_rand = nn.functional.relu(-num_samples)

        weight_hat_det = self._sparsifier_sipp_det.sparsify(num_samples_det)
        weight_hat_rand = self._sparsifier_sipp_rand.sparsify(num_samples_rand)
        return weight_hat_det + weight_hat_rand
