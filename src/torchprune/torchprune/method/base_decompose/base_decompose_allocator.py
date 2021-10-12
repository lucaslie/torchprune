"""Module containing the low-rank compression allocators."""

from abc import abstractmethod, ABC
import numpy as np
import torch

from .base_decompose_util import FoldScheme
from ..base import BaseFilterAllocator


class BaseDecomposeAllocator(BaseFilterAllocator, ABC):
    """The base allocator for decomposition-based compression."""

    @property
    def _folding_scheme_value(self):
        return FoldScheme.get_default().value

    def __init__(self, net, k_split, **kwargs):
        """Initialize the allocator with the desired split in k."""
        super().__init__(net, out_mode=True, **kwargs)
        self._desired_k_split = k_split

        # keep a ref of the network around
        self._net = net

        # check number of groups in each layer
        for module in net.compressible_layers:
            if hasattr(module, "groups"):
                assert module.groups == 1, "Only ungrouped modules supported."

        # now get per-layer k-split
        self.register_buffer("_k_splits", None)
        self._k_splits = self._get_k_splits(self._desired_k_split)

        # in/out kernel sizes which will depend on scheme
        self.register_buffer("_kernel_size_in", None)
        self.register_buffer("_kernel_size_out", None)
        self._kernel_size_in = torch.ones_like(self._kernel_size)
        self._kernel_size_out = torch.ones_like(self._kernel_size)

        # desired scheme for each layer (use default scheme)
        self.register_buffer("_scheme_values_raw", None)
        self._scheme_values_raw = torch.zeros_like(self._k_splits)

        # update scheme values and kernels accordingly
        self._scheme_values = self._folding_scheme_value

    @abstractmethod
    def _compute_ranks_j_for_arg(self, arg, ranks, num_weights_per_j):
        """Get the desired rank for each layer based on arg."""

    def _get_possible_k(self, w_shape_in):
        """Get possible k-splits for some in shape of weight."""
        # divisors are a numpy array in desceding order
        divisors = np.arange(w_shape_in, 0, -1)
        return divisors[np.remainder(w_shape_in, divisors) == 0]

    def _get_k_splits(self, desired_k_split):
        """Return k that is closest to desired k and divisor of in features."""
        k_splits = torch.ones_like(self._in_features)

        # get an array for desired_k_split
        if isinstance(desired_k_split, torch.Tensor):
            desired_k_split = desired_k_split.cpu().numpy()
        else:
            desired_k_split = (
                np.zeros(len(k_splits), dtype=np.int) + desired_k_split
            )

        for ell, mod in enumerate(self._net.compressible_layers):
            # divisors are in desceding order
            # ties will thus be broken by picking the larger value of k since
            # torch min convention is to return first minimal index for ties
            divisors = self._get_possible_k(mod.weight.shape[1])
            i_closest = np.argmin(np.abs(divisors - desired_k_split[ell]))
            k_splits[ell] = int(divisors[i_closest])

        return k_splits

    def _scheme(self, ell):
        """Return scheme for layer ell."""
        return FoldScheme(self._scheme_values[ell].item())

    @property
    def _scheme_values(self):
        """Return values for decomposition scheme applied in each layer."""
        return self._scheme_values_raw

    @_scheme_values.setter
    def _scheme_values(self, scheme_values):
        """Set new scheme values for each layer and update in/out kernels."""
        if isinstance(scheme_values, torch.Tensor):
            self._scheme_values_raw.copy_(scheme_values)
        else:
            self._scheme_values_raw[:] = scheme_values
        self._compute_kernels_in_out()

    @property
    def _schemes(self):
        """Yield schemes layer-by-layer."""
        for scheme_value in self._scheme_values:
            yield FoldScheme(scheme_value.item())

    def _compute_kernels_in_out(self):
        """Compute the resulting in/out kernel sizes based on schemes."""
        for ell, scheme in enumerate(self._schemes):
            kernel_size = self._kernel_shapes[ell]
            k_out, k_in = scheme.get_decomposed_kernel_sizes(kernel_size)
            self._kernel_size_out[ell] = k_out
            self._kernel_size_in[ell] = k_in

    def _get_boundaries(self):
        """Get boundaries for bisection method."""
        rank_rel_min = 1e-12
        rank_rel_max = 1.1

        return rank_rel_min, rank_rel_max

    def _get_size(self, rank_stats, num_weights_per_j):
        """Get the resulting size as function of ranks_j and weights per j."""
        ranks_j = rank_stats[:, 0]
        num_w_orig = self._out_features * self._in_features * self._kernel_size
        num_w_proj = ranks_j * num_weights_per_j
        return sum(torch.min(num_w_orig, num_w_proj))

    def _get_weight_stats(self):
        """Compute and return per-layer weight statistics."""
        out_dims = self._out_features * self._kernel_size_out
        in_dims = self._in_features * self._kernel_size_in
        ranks = torch.min(out_dims, (in_dims // self._k_splits))

        # some weight statistics
        num_weights_per_j = self._k_splits * out_dims  # U, decoding
        num_weights_per_j += in_dims  # V, encoding

        return ranks, num_weights_per_j

    def _get_proposed_num_features(self, arg):
        """Compute resulting features (j, k) per layer."""
        # some weight statistics
        ranks, num_weights_per_j = self._get_weight_stats()

        # compute resulting ranks
        ranks_j = self._compute_ranks_j_for_arg(arg, ranks, num_weights_per_j)
        ranks_j = ranks_j.to(ranks)

        # some sanity checks for the ranks
        ranks_j[ranks_j < 1] = 1
        ranks_j[ranks_j > ranks] = ranks[ranks_j > ranks]

        # construct resulting statistics and return
        rank_stats = torch.stack(
            (ranks_j, self._k_splits, self._scheme_values), dim=1
        )
        return rank_stats, num_weights_per_j


class DecomposeRankAllocator(BaseDecomposeAllocator):
    """Decomposition allocator with const relative rank reduction per layer."""

    def _compute_ranks_j_for_arg(self, arg, ranks, num_weights_per_j):
        """Get the desired rank for each layer (const relative rank)."""
        return (arg * ranks).round()


class DecomposeRankAllocatorScheme0(DecomposeRankAllocator):
    """Simple rank-based allocator with folding scheme 0."""

    @property
    def _folding_scheme_value(self):
        return FoldScheme.KERNEL_ENCODE.value


class DecomposeRankAllocatorScheme1(DecomposeRankAllocator):
    """Simple rank-based allocator with folding scheme 1."""

    @property
    def _folding_scheme_value(self):
        return FoldScheme.KERNEL_SPLIT1.value


class DecomposeRankAllocatorScheme2(DecomposeRankAllocator):
    """Simple rank-based allocator with folding scheme 2."""

    @property
    def _folding_scheme_value(self):
        return FoldScheme.KERNEL_SPLIT2.value


class DecomposeRankAllocatorScheme3(DecomposeRankAllocator):
    """Simple rank-based allocator with folding scheme 3."""

    @property
    def _folding_scheme_value(self):
        return FoldScheme.KERNEL_DECODE.value
