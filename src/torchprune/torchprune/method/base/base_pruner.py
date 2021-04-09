"""Module  for base pruner.

Pruners are designed to allocate layer samples to each neuron.

"""
from abc import ABC, abstractmethod
import copy

import torch
import torch.nn as nn
from torch.distributions.multinomial import Multinomial
import numpy as np

from .base_util import get_prob_stats, adapt_sample_size


class BasePruner(ABC, nn.Module):
    """Abstract interface to allocate samples for a layer to each neuron."""

    def __init__(self, tensor, sensitivity, **kwargs):
        """Initialize with tensor and sensitivities and flexible arguments."""
        super().__init__()

        # Cache tensors and sensitivity of each *parameter* of tensor
        self.tensor = tensor
        self.sensitivity = sensitivity

        # initialize the standard stuff
        self.register_buffer("probability", torch.Tensor())
        self.register_buffer("probability_div", torch.Tensor())

        # also fake-initialize so linter doesn't complain
        self.probability = torch.Tensor()
        self.probability_div = torch.Tensor()

    @abstractmethod
    def prune(self, size_pruned):
        """Prune across neurons within the layer to the desired size."""
        raise NotImplementedError

    @abstractmethod
    def compute_probabilities(self):
        """Compute the probabilities associated with neurons in the layer."""
        raise NotImplementedError

    def forward(self, x):
        """It's a nn.Module, so strictly speaking it needs a forward func."""


class DetPruner(BasePruner, ABC):
    """A class containing all implementations for deterministic pruners."""

    def compute_probabilities(self):
        """Compute the probabilities associated with neurons in the layer."""
        self.probability = self.sensitivity
        self.probability_div = self.sensitivity


class RandPruner(BasePruner, ABC):
    """A class containing all implementations for randomized pruners."""

    def __init__(self, tensor, sensitivity, **kwargs):
        """Initialize with tensor and sensitivities and flexible arguments."""
        super().__init__(tensor, sensitivity, **kwargs)
        self.uniform = False

    @property
    def _start_dim(self):
        return 1

    def compute_probabilities(self):
        """Compute the probabilities associated with neurons in the layer."""
        dims_to_sum = list(range(self._start_dim, self.sensitivity.dim()))

        # query probabilities
        _, self.probability = get_prob_stats(self.sensitivity, dims_to_sum)

        # Also save a version with 0 mapped to Inf so division works
        eps = torch.Tensor([np.finfo(np.float32).eps]).to(
            self.sensitivity.device
        )
        self.probability_div = copy.deepcopy(self.probability)
        self.probability_div.masked_fill_(self.probability_div <= eps, np.Inf)


class RandFeaturePruner(RandPruner):
    """Pruner for any randomized, weight-based pruning method."""

    def __init__(self, tensor, sensitivity, **kwargs):
        """Initialize with tensor and sensitivities and flexible arguments."""
        super().__init__(tensor, sensitivity, **kwargs)
        self._num_samples = [{} for x in range(self.tensor.shape[0])]

    def prune(self, size_pruned):
        """Prune across neurons within the layer to the desired size."""
        probs = self.probability
        num_filters = probs.shape[0]
        num_samples = torch.zeros_like(size_pruned)

        for idx_filt in range(num_filters):
            key = int(size_pruned[idx_filt])
            if key not in self._num_samples[idx_filt]:
                self._num_samples[int(idx_filt)][key] = adapt_sample_size(
                    probs[idx_filt].view(-1),
                    size_pruned[idx_filt],
                    self.uniform,
                )
            num_samples[idx_filt] = self._num_samples[int(idx_filt)][key]

        return num_samples


class RandFilterPruner(RandPruner):
    """Pruner for any randomized, filter-based pruning method."""

    @property
    def _start_dim(self):
        return 0

    def prune(self, size_pruned):
        """Prune across neurons within the layer to the desired size."""
        probs = self.probability
        num_filters = probs.shape[0]

        if size_pruned > 0:
            num_to_sample = adapt_sample_size(
                probs.view(-1), size_pruned, self.uniform
            )
            num_to_sample = int(num_to_sample)
            num_samples = Multinomial(num_to_sample, probs).sample().int()
        else:
            num_samples = torch.zeros(num_filters).int().to(size_pruned.device)

        return num_samples


class DetFeaturePruner(DetPruner):
    """Pruner for classic weight thresholding heuristic."""

    def prune(self, size_pruned):
        """Prune across neurons within the layer to the desired size."""
        num_samples = copy.deepcopy(size_pruned)
        return num_samples


class DetFilterPruner(DetPruner):
    """Pruner for classic norm-based filter pruning heuristic."""

    def prune(self, size_pruned):
        """Prune across neurons within the layer to the desired size."""
        probs = self.probability
        mask = torch.zeros_like(probs, dtype=torch.bool)
        if size_pruned > 0:
            size_pruned = size_pruned.cpu().numpy()
            idx_top = np.argpartition(
                probs.view(-1).cpu().numpy(), -size_pruned
            )[-size_pruned:]
            mask.view(-1)[idx_top] = True
        num_samples = mask.view(mask.shape[0], -1).sum(dim=-1)

        return num_samples


class TensorPruner(RandPruner):
    """A fake pruner for tensor-based sparsification."""

    @property
    def _start_dim(self):
        return 0

    def prune(self, size_pruned):
        """Just return the size given (by the allocator)."""
        return torch.sum(size_pruned)


class TensorPruner2(TensorPruner):
    """Fake pruner that does not alter proposed size."""

    def prune(self, size_pruned):
        """Just return the size given (by the allocator)."""
        # original TensorPruner returns sum instead...
        return size_pruned
