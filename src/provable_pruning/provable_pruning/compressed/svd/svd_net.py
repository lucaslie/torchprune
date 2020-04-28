"""Module implementing network compression with classic SVD."""
import copy
import numpy as np
import torch
import torch.nn as nn

from ..base import BaseCompressedNet


class SVDNet(BaseCompressedNet):
    """The network implementing SVD-based compression."""

    @property
    def deterministic(self):
        """Return the indicator for deterministic compression."""
        return True

    @property
    def retrainable(self):
        """Return the indicator whether we can retrain afterwards."""
        return False

    def __init__(self, original_net):
        """Simply initialize with the original network."""
        super().__init__(original_net)
        self._svd = {}
        self._ranks = {}
        self._size_total = 0
        self._boost = 1.0

    def _low_rank_approx(self, svd=None, weight=None, rank=1):
        """Compute an r-rank approximation of the weight matrix W."""
        if not svd:
            svd = np.linalg.svd(weight, full_matrices=False)

        u_svd, s_svd, v_svd = svd
        s_diag = np.diag(s_svd)
        s_diag[s_diag < s_svd[rank]] = 0
        weight_low_rank = u_svd @ s_diag @ v_svd

        weight_low_rank = torch.tensor(weight_low_rank)
        return weight_low_rank

    def _initialize_compression(self):
        """Initialize the SVD decompositions for each weight matrix."""
        # printing initialization message
        print(f"Initializing {type(self).__name__}...")

        # Initialize the SVD decomposition of the weight matrix of each layer
        # for later low-rank approx computation.
        for ell in self.layers:
            module = self.compressed_net.compressible_layers[ell]
            weight = copy.deepcopy(module.weight.data)

            # The matrix W for which we should perform SVD on will depend on
            # whether we are dealing with a convolutional layer.
            if isinstance(module, nn.Linear):
                pass
            elif isinstance(module, nn.Conv2d):
                # W is of dimensions
                # c^ell x c^{ell-1} x kappa_1^ell x kappa_2^ell
                shape = weight.shape

                m_shape = shape[1] * shape[2] * shape[3]

                # Reshape W so that it is of size c^ell x m
                weight = weight.view(-1, m_shape)
            else:
                raise ValueError(
                    "Only nn.Linear and nn.Conv2d modules can be compressed right now!"
                )

            u_svd, s_svd, v_svd = np.linalg.svd(
                weight.cpu().numpy(), full_matrices=False
            )
            self._svd[ell] = (u_svd, s_svd, v_svd)

        print("Done")
        print("")

    def _get_closest_rank(self, svd, sample_size):
        size_this = 0
        (u_svd, s_svd, v_svd) = svd
        n_size = s_svd.shape[0]

        for i in range(n_size):
            size_this += (
                np.count_nonzero(u_svd.T[i, :])
                + np.count_nonzero(v_svd[i, :])
                + 1
            )
            if size_this >= sample_size:
                return i + 1, size_this

        return n_size, size_this

    def compress(self, keep_ratio, from_original=True, initialize=True):
        """Execute the compression step."""
        # start fresh
        device = next(self.compressed_net.parameters()).device
        del self.compressed_net
        self.compressed_net = copy.deepcopy(self.original_net[0])
        self.compressed_net.to(device)

        if initialize:
            self._initialize_compression()

        self._size_total = 0

        # track budget per layer
        budget_per_layer = [0 for _ in self.layers]

        for ell in self.layers:
            module = self.compressed_net.compressible_layers[ell]
            weight_original = module.weight.data
            weight_copy = copy.deepcopy(module.weight.data)

            # The matrix W for which we should perform SVD on will depend on
            # whether we are dealing with a convolutional layer.
            if isinstance(module, nn.Linear):
                num_entries = weight_copy.shape[0] * weight_copy.shape[1]
            elif isinstance(module, nn.Conv2d):
                # W is of dimensions
                # c^ell x c^{ell-1} x kappa_1^ell x kappa_2^ell
                shape = weight_copy.shape

                m_shape = shape[1] * shape[2] * shape[3]
                # Reshape W so that it is of size c^ell x m
                weight_copy = weight_copy.view(-1, m_shape)
                num_entries = shape[0] * m_shape
            else:
                raise ValueError(
                    "Only nn.Linear and nn.Conv2d modules can be compressed right now!"
                )

            svd = self._svd[ell]
            sample_size_this = (
                keep_ratio * num_entries * self._boost
            )  # need a little boost
            rank_approx, uv_nonzero = self._get_closest_rank(
                svd, sample_size_this
            )

            num_weights_nonzero = np.count_nonzero(weight_copy.cpu().numpy())
            if num_weights_nonzero > uv_nonzero:
                # Compute the low-rank approximation using the cached SVD
                # decomposition.
                weight_hat = self._low_rank_approx(
                    svd, weight_copy, rank_approx
                )

                # If we are dealing with a convolutional layer, we need to
                # reshape the low-rank approximation.
                if isinstance(module, nn.Conv2d):
                    weight_hat = weight_hat.view(module.weight.data.shape)

                size_layer = uv_nonzero
            else:
                weight_hat = weight_original
                size_layer = num_weights_nonzero

            module.state_dict()["weight"].copy_(weight_hat)

            # Also update the size with the number of non-zero entries in the
            # bias vector.
            size_layer += (
                np.count_nonzero(module.bias.data.cpu().numpy())
                if module.bias is not None
                else 0
            )

            # Update the rank of matrix W^l for later size computation.
            self._ranks[ell] = rank_approx

            # update total size
            self._size_total += size_layer

            # tracking "samples" per layer
            budget_per_layer[ell] = size_layer

        return budget_per_layer

    def size(self):
        """Return the total number of non-zero parameters of the network.

        An n x d matrix A is described by O(n*d) numbers. But, with
        k-truncated SVD where we take the top k singular values, we can
        describe A by k*(n+d + 1) numbers. To reflect this we need to do a
        different size comparison.

        """
        return self._size_total
