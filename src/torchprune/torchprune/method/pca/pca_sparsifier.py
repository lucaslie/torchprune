"""Module containing the sparsifier for PCA-based pruning."""

from ..base_decompose import BaseDecomposeSparsifier, FoldScheme


class PCASparsifier(BaseDecomposeSparsifier):
    """The PCA sparsifier that projects the weights into the PCA space."""

    def __init__(self, pruner):
        """Initialize with pruner and retrieve stats."""
        super().__init__(pruner)
        self._principle_components = pruner.principle_components
        self._data_mean = pruner.data_mean
        self._bias = pruner.bias

        self.register_buffer("_full_bias", None)
        self._full_bias = None

    def sparsify(self, rank_stats):
        """Sparsify to the desired number of features (rank_j)."""
        # get compressed weights
        weights_hat = super().sparsify(rank_stats)

        # retrieve stored bias, "unstore" it, and wrap return
        if len(weights_hat) > 0:
            bias = self._full_bias.detach().clone()
        else:
            bias = None
        self._full_bias = None

        return (weights_hat, bias)

    def _sparsify(self, tensor, rank_j, k_split, scheme):
        """Sparsify to the desired number of features (rank_j)."""
        assert k_split == 1, "No splits allowed"
        assert scheme == FoldScheme.KERNEL_ENCODE, "Wrong folding scheme"

        # fold weight and compute approximations from principle components
        w_folded = scheme.fold(tensor)

        # get "u" and "v"
        u_hat = self._principle_components[:, :rank_j].detach().clone()
        v_hat = u_hat.T @ w_folded

        # store full bias
        bias = self._data_mean - u_hat @ u_hat.T @ self._data_mean
        if self._bias is not None:
            bias += self._bias
        self._full_bias = bias

        # return now weights
        return [(u_hat, v_hat)]
