"""Module containing the pruner implementations for PFP."""
import copy

import torch
from ..base import BasePruner, RandFilterPruner, DetFilterPruner


class PFPRandPruner(RandFilterPruner):
    """The pruner for random PFPNetRand."""

    def __init__(self, tensor, tracker, **kwargs):
        """Initialize with tensor, tracker and flexible arguments."""
        super().__init__(tensor, tracker.sensitivity_in, **kwargs)


class PFPPruner(DetFilterPruner):
    """The pruner for deterministic PFPNet."""

    def __init__(self, tensor, tracker, **kwargs):
        """Initialize with tensor, tracker and flexible arguments."""
        super().__init__(tensor, tracker.sensitivity_in, **kwargs)


class PFPTopPruner(BasePruner):
    """The pruner for hybrid PFPNetTop."""

    def __init__(self, tensor, tracker, **kwargs):
        """Initialize with tensor, tracker and flexible arguments."""
        super().__init__(tensor, tracker.sensitivity_in, **kwargs)
        self.pruner_pfp_det = PFPPruner(tensor, tracker)
        self.pruner_pfp_rand = PFPRandPruner(tensor, tracker)

    def compute_probabilities(self):
        """Compute the probabilities associated with neurons in the layer."""
        self.pruner_pfp_rand.compute_probabilities()
        self.pruner_pfp_det.compute_probabilities()

        # store original probability in this class
        self.sensitivity = copy.deepcopy(self.pruner_pfp_rand.sensitivity)

    def prune(self, size_pruned):
        """Prune across neurons within the layer to the desired size."""
        top_k = size_pruned[0]
        remaining = size_pruned[1]

        # zero out top probs in RFS pruner and re-compute probs
        self.pruner_pfp_rand.sensitivity = copy.deepcopy(self.sensitivity)
        if top_k > 0:
            _, idx_top = torch.topk(self.sensitivity, top_k)
            self.pruner_pfp_rand.sensitivity[idx_top] = 0.0
            self.pruner_pfp_rand.compute_probabilities()

        # get num samples for each scenario
        num_samples_det = self.pruner_pfp_det.prune(top_k).int()
        num_samples_rand = self.pruner_pfp_rand.prune(remaining)

        # pass it on
        return torch.stack((num_samples_det, num_samples_rand))
