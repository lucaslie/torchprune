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
    def _reweigh(self, counts, num_samples_normalized_f, probs_div_f):
        raise NotImplementedError(
            "The base class does not implement this " "method."
        )


class RandSparsifier(SimpleSparsifier):
    """The partial implementation for the random SimpleSparsifier."""

    def _reweigh(self, counts, num_samples_normalized_f, probs_div_f):
        gammas_f = (
            counts.float() / num_samples_normalized_f.float() / probs_div_f
        )
        return gammas_f


class DetSparsifier(SimpleSparsifier):
    """The partial implementation for the deterministic SimpleSparsifier."""

    def _reweigh(self, counts, num_samples_normalized_f, probs_div_f):
        sens_sum = max(0, 1 - torch.sum(probs_div_f[counts]).item())
        kappa = 0
        if sens_sum < 1:
            kappa = sens_sum / (1 - sens_sum)
        # Under the i.i.d. assumption this works, otherwise no.
        gammas_f = (1 + kappa) * counts.float()

        return gammas_f


class WeightSparsifier(SimpleSparsifier):
    """The partial implementation for the weight-based SimpleSparsifier."""

    def __init__(self, pruner):
        """Initialize the sparsifier from the pruner of the same layer."""
        super().__init__(pruner)

    @abstractmethod
    def _generate_counts(self, num_samples_f, probs_f):
        raise NotImplementedError(
            "The base class does not implement this " "method."
        )

    def sparsify(self, num_samples):
        """Sparsify the edges of the associated neurons with num_samples."""
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


class FilterSparsifier(SimpleSparsifier):
    """The partial implementation for the filter-based SimpleSparsifier."""

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


class RandWeightSparsifier(WeightSparsifier, RandSparsifier):
    """A sparsifier for random weight sparsification."""

    @property
    def _do_reweighing(self):
        return True

    def _generate_counts(self, num_samples_f, probs_f):
        distribution = Multinomial(num_samples_f.item(), probs_f.view(-1))

        counts = distribution.sample()

        return counts.view(probs_f.shape)


class RandFilterSparsifier(FilterSparsifier, RandSparsifier):
    """A sparsifier for random filter sparsification (fake)."""

    @property
    def _do_reweighing(self):
        return False


class DetFilterSparsifier(FilterSparsifier, DetSparsifier):
    """The fake sparsifier for filter thresholding."""

    @property
    def _do_reweighing(self):
        return False


class DetWeightSparsifier(WeightSparsifier, DetSparsifier):
    """The sparsifier for weight thresholding."""

    @property
    def _do_reweighing(self):
        return False

    def _generate_counts(self, num_samples_f, probs_f):
        mask = torch.zeros_like(probs_f, dtype=torch.bool)
        numel = probs_f.numel()
        num_samples_f = int(np.clip(1, int(num_samples_f), numel))
        idx_top = np.argpartition(
            probs_f.view(-1).cpu().numpy(), -num_samples_f
        )[-num_samples_f:]
        mask.view(-1)[idx_top] = True

        return mask
