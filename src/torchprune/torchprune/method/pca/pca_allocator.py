"""Module containing the SVD allocators."""

import torch

from ..base_decompose import BaseDecomposeAllocator, FoldScheme


class PCAAllocator(BaseDecomposeAllocator):
    """PCA Allocator based on greedy allocation."""

    @property
    def _folding_scheme_value(self):
        return FoldScheme.KERNEL_ENCODE.value

    def __init__(self, net, trackers, **kwargs):
        """Initialize allocator with trackers."""
        super().__init__(net, k_split=1, **kwargs)

        self.register_buffer("_sensitivities", None)
        self._sensitivities = self._process_sensitivities(trackers)

    def _process_sensitivities(self, trackers):
        """Process sensitivities greedily and return greedy selection order."""
        device = trackers[0].sensitivity.device

        # collect sensitivities
        rank_max = max(len(tracker.sensitivity) for tracker in trackers)
        sensitivities = torch.zeros(self._num_layers, rank_max, device=device)
        for sens_ell, tracker in zip(sensitivities, trackers):
            sens_ell[: len(tracker.sensitivity)].copy_(tracker.sensitivity)

        # normalize sensitivities with flops per rank and layer
        num_weights_per_j = self._get_weight_stats()[1]
        num_flops_per_j = num_weights_per_j * self._num_patches
        sensitivities = sensitivities / num_flops_per_j[:, None]

        # return new and normalized sensitivities
        return sensitivities

    def _get_weight_stats(self):
        _, num_weights_per_j = super()._get_weight_stats()

        # ranks in this case are actually always out_dims since it's PCA
        out_dims = self._out_features * self._kernel_size_out

        return out_dims, num_weights_per_j

    def _get_boundaries(self):
        # use min and max sensitivities ...
        return self._sensitivities.min(), self._sensitivities.max() * 1.01

    def _compute_ranks_j_for_arg(self, arg, ranks, num_weights_per_j):
        """Get the desired rank for each layer based on arg."""
        # check the threshold for each layer now and record the number of ranks
        bigger = self._sensitivities > arg

        # get ranks now per layer
        ranks_j = bigger.sum(dim=-1)

        return ranks_j
