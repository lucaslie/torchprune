"""Module with abstract interface for sparsifiers."""
from abc import ABC, abstractmethod
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.multinomial import Multinomial


class BaseSparsifier(ABC, nn.Module):
    """The basic interface for a sparsifier.

    A sparsifier sparsifies the weights incoming to neurons in a given layer.

    """

    def __init__(self, pruner):
        """Initialize the sparsifier from the pruner of the same layer."""
        super().__init__()
        # Cache tensor and sensitivity of each parameter of tensor
        self._tensor = pruner.tensor
        self._probability = pruner.probability
        self._probability_div = pruner.probability_div

    @abstractmethod
    def sparsify(self, num_samples):
        """Sparsify the edges of the associated neurons with num_samples."""
        raise NotImplementedError

    def forward(self, x):
        """It's a nn.Module, so strictly speaking it needs a forward func."""


class SimpleSparsifier(BaseSparsifier):
    """The interface for a simple sparsifier without sensitivity."""

    @property
    @abstractmethod
    def _do_reweighing(self):
        raise NotImplementedError

    @abstractmethod
    def _reweigh(self, counts, num_samples, probs_div):
        raise NotImplementedError(
            "The base class does not implement this " "method."
        )


class RandSparsifier(SimpleSparsifier):
    """The partial implementation for the random sparsification."""

    @property
    def _do_reweighing(self):
        return True

    def _reweigh(self, counts, num_samples, probs_div):
        gammas = counts.float() / num_samples.float() / probs_div
        return gammas

    def _generate_counts(self, num_samples, probs):
        distribution = Multinomial(num_samples.item(), probs.view(-1))
        counts = distribution.sample()
        return counts.view(probs.shape)


class DetSparsifier(SimpleSparsifier):
    """The partial implementation for the deterministic sparsification."""

    @property
    def _do_reweighing(self):
        return False

    def _reweigh(self, counts, num_samples, probs_div):
        sens_sum = max(0, 1 - torch.sum(probs_div[counts]).item())
        kappa = 0
        if sens_sum < 1:
            kappa = sens_sum / (1 - sens_sum)
        # Under the i.i.d. assumption this works, otherwise no.
        gammas = (1 + kappa) * counts.float()

        return gammas

    def _generate_counts(self, num_samples, probs):
        mask = torch.zeros_like(probs, dtype=torch.bool)
        numel = probs.numel()
        num_samples = int(np.clip(0, int(num_samples), numel))
        if num_samples > 0:
            idx_top = np.argpartition(
                probs.view(-1).cpu().numpy(), -num_samples
            )[-num_samples:]
        else:
            idx_top = []
        mask.view(-1)[idx_top] = True

        return mask


class FeatureSparsifier(SimpleSparsifier):
    """The partial implementation for the feature-wise sparsifier."""

    def __init__(self, pruner):
        """Initialize the sparsifier from the pruner of the same layer."""
        super().__init__(pruner)

    @abstractmethod
    def _generate_counts(self, num_samples_f, probs_f):
        raise NotImplementedError(
            "The base class does not implement this " "method."
        )

    def sparsify(self, num_samples):
        """Sparsify the edges of the associated feature with num_samples."""
        # short notation
        weight_original = self._tensor
        probs = self._probability

        # pre-allocate gammas
        gammas = torch.ones_like(probs)

        idx = (num_samples).nonzero()
        gammas[(num_samples < 1).nonzero(), :] = 0.0

        # loop through all filters from which we should sample from
        for idx_f in idx:
            # generate counts for this filter
            counts = self._generate_counts(num_samples[idx_f], probs[idx_f])
            # only use approximation when it effectively reduces size
            less = (
                counts.nonzero().shape[0]
                < self._tensor[idx_f].nonzero().shape[0]
            )

            if less:
                # if it does, reweigh appropriately
                if self._do_reweighing:
                    gammas[idx_f] = self._reweigh(
                        counts,
                        num_samples[idx_f],
                        self._probability_div[idx_f],
                    )
                else:
                    gammas[idx_f] = (counts > 0).float()

        # return approximation
        return gammas * weight_original


class RandFeatureSparsifier(RandSparsifier, FeatureSparsifier):
    """A sparsifier for random weight sparsification per feature."""


class DetFeatureSparsifier(DetSparsifier, FeatureSparsifier):
    """The sparsifier for deterministic weight sparsification per feature."""


class FilterSparsifier(SimpleSparsifier):
    """The implementation for the fake sparsifier for filter pruning."""

    @property
    def _do_reweighing(self):
        return False

    def _reweigh(self, counts, num_samples, probs_div):
        gammas = counts.float() / num_samples.float() / probs_div
        return gammas

    def __init__(self, pruner, out_mode):
        """Initialize the sparsifier from the pruner of the same layer."""
        super().__init__(pruner)
        self._out_mode = out_mode

    def sparsify(self, num_samples):
        """Fake-sparsify the edges (we don't do sparsification for filters)."""
        # short notation
        weight_original = self._tensor

        # pre-allocate gammas
        gammas = copy.deepcopy(num_samples).float()

        # check for reweighing
        if self._do_reweighing and num_samples.sum() > 0:
            gammas = self._reweigh(
                gammas, num_samples.sum(), self._probability_div
            )
        else:
            gammas = (gammas > 0).float()

        # make gammas compatible with Woriginal
        gammas = gammas.unsqueeze(int(self._out_mode)).unsqueeze(-1)

        # make Woriginal compatible with gammas and return
        weight_hat = (
            gammas
            * weight_original.view(
                weight_original.shape[0], weight_original.shape[1], -1
            )
        ).view_as(weight_original)
        return weight_hat
