"""Module containing the allocator for the rank learning algorithm."""

import torch

from ..base_decompose import FoldScheme
from ..svd.svd_allocator import SVDFrobeniusEnergyAllocator


class LearnedRankAllocator(SVDFrobeniusEnergyAllocator):
    """Allocation based on C-Step in Eq (5).

    min_r lambda C_k(r) + sum_{i=r+1}^R_k s_{ki}^2
    with
    r       ... chosen rank
    C_k(r)  ... FLOPs in layer k when having rank r
    R_k     ... full rank of layer k
    s_{ki}  ... i^th largest singular value in layer k

    We search over possible values of lambda ("arg") to find the solution that
    corresponds to our desired prune ratio.


    Eq (5) was originally introduced in:
    http://openaccess.thecvf.com/content_CVPR_2020/html/Idelbayev_Low-Rank_Compression_of_Neural_Nets_Learning_the_Rank_of_Each_CVPR_2020_paper.html
    """

    @property
    def _folding_scheme_value(self):
        return FoldScheme.KERNEL_SPLIT1.value

    def _compute_rel_error_for_weight(self, weight, k_split, scheme):
        # this is the "relative Frobenious norm" (relative energy)
        rel_e = super()._compute_rel_error_for_weight(weight, k_split, scheme)

        # here we want the absolute norm though and we will abuse the relative
        # error notation to leverage the parent class
        fro_norm = self._compute_norm_for_weight(weight, scheme, ord="fro")
        abs_e = rel_e * fro_norm * fro_norm
        return abs_e

    def _get_boundaries(self):
        return torch.tensor(0.0), torch.tensor(1.0e12)

    def _compute_ranks_j_for_arg(self, arg, ranks, num_weights_per_j):
        """Compute the ranks based on the desired cost function."""
        flops_per_j = num_weights_per_j * self._num_patches

        # compute cost per rank and layer, shape == (num_layers x rank_max)
        compute_cost = (
            torch.arange(self._rel_error.shape[1])[None, :].to(flops_per_j)
            * flops_per_j[:, None]
        )
        assert compute_cost.min() >= 0

        # now get total cost
        cost_total = arg * compute_cost + self._rel_error

        # argmin per layer are the desired ranks
        ranks_j = cost_total.argmin(dim=1)

        return ranks_j


class LearnedRankAllocatorScheme0(LearnedRankAllocator):
    """Learned rank allocation with scheme 0 instead."""

    @property
    def _folding_scheme_value(self):
        return FoldScheme.KERNEL_ENCODE.value
