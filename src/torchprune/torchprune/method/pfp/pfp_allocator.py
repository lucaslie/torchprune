"""Module containing the allocators for PFP pruning."""
import copy

import torch

from ..base import BaseFilterAllocator
from ..base.base_util import expected_unique, adapt_sample_size
from ..sipp.sipp_allocator import SiPPRandAllocator


class PFPRandAllocator(BaseFilterAllocator, SiPPRandAllocator):
    """Allocator for randomized PFP."""

    def __init__(self, net, trackers, delta_failure, c_constant, **kwargs):
        """Initialize the filter based allocator so it's compliant."""
        super().__init__(
            net=net,
            out_mode=False,
            trackers=trackers,
            delta_failure=delta_failure,
            c_constant=c_constant,
            **kwargs
        )

    def _get_boundaries(self):
        # for bisection method
        eps_min = 1e-300
        eps_max = 1e150
        return eps_min, eps_max

    def _get_unique_samples(self, m_budget):
        for i, _ in enumerate(m_budget):
            # Reverse calibration
            m_budget[i] = expected_unique(self._probability[i], m_budget[i])
        return m_budget

    def _get_proposed_num_features(self, arg):
        # get budget according to sample complexity
        eps = arg
        m_budget = self._get_sample_complexity(eps)
        m_budget = self._get_unique_samples(m_budget).to(self._in_features)

        # assign budget to in features if smaller
        in_features = copy.deepcopy(self._in_features)
        in_features[m_budget < in_features] = m_budget[m_budget < in_features]
        in_features[in_features < 1] = 1

        # propagate compression to out features by reducing by the same amount
        in_feature_reduction = self._in_features - in_features
        out_features = copy.deepcopy(self._out_features)
        out_features[:-1] -= in_feature_reduction[1:]
        out_features[out_features < 1] = 1

        return out_features, in_features

    def _adapt_l_coeffs(self, l_coeffs, num_filters, num_patches):
        return l_coeffs

    def _get_sens_stats(self, tracker):
        # short-hand
        sens_in = tracker.sensitivity_in

        nnz = (sens_in != 0.0).sum().view(-1)
        sum_sens = sens_in.sum().view(-1)
        probs = sens_in / sum_sens

        return nnz, sum_sens, probs

    def _get_sample_complexity(self, eps, sens_tilde=None):
        if sens_tilde is None:
            sens_tilde = self._coeffs
        k_constant = 3.0
        m_budget = k_constant * (6 + 2 * eps) * sens_tilde / (eps ** 2)
        m_budget = m_budget.ceil()
        m_budget[m_budget == 0] = 1
        return m_budget


class PFPAllocator(PFPRandAllocator):
    """The alloctor for deterministic PFP."""


class PFPTopAllocator(PFPRandAllocator):
    """The allocator for hybrid PFP."""

    def _allocate_method(self, budget):
        # do the standard allocation for filters
        super()._allocate_method(budget)

        # now check which one we should keep deterministically
        alloc_new = torch.zeros_like(self._allocation)
        alloc_new = torch.stack((alloc_new, alloc_new), dim=1)
        for ell in range(self._num_layers):
            k_best = self._get_split(ell, self._allocation[ell])
            alloc_new[ell][0] = k_best
            alloc_new[ell][1] = self._allocation[ell] - k_best

        # store allocation
        self._allocation = alloc_new

    def _get_split(self, ell, size_pruned):
        if size_pruned <= 1:
            return size_pruned
        probs = self._probability[ell]

        # maybe prunedSize / 2?
        adapted = float(adapt_sample_size(probs, size_pruned))

        idx = probs >= 1.0 / adapted
        ret = min(torch.sum(idx).to(size_pruned), size_pruned)
        return ret
