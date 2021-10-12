"""Module containing the sparsifier (core algorithm) for Messi."""

import torch

from ..base_decompose import GroupedDecomposeSparsifier
from .messi_util import factor


class MessiSparsifier(GroupedDecomposeSparsifier):
    """A Messi-based sparsifier for tensors (generalized to conv)."""

    @property
    def _em_steps(self):
        return 5

    @property
    def _num_init_em(self):
        return 5

    def _sparsify_with_messi(self, tensor, rank_j, k_split, scheme):
        """Sparsify to the desired number of features (rank_j) with Messi."""
        # get projective clustering
        # Convention:
        #  1. y = A^t * x
        #  2. A = UV
        #  3. y = V^T * (U^T * x)
        partition, list_u, list_v = factor.raw(
            scheme.fold(tensor.detach()).t().cpu().numpy(),
            j=rank_j,
            k=k_split,
            steps=self._em_steps,
            NUM_INIT_FOR_EM=self._num_init_em,
        )
        u_stitched, v_stitched = factor.stitch(partition, list_u, list_v)

        # Convert it to pytorch/numpy convention ...
        # Pytorch convention:
        #  1. y = A * x
        #  2. y = U * (V * x)
        # weights_hat = [U, V]
        return [
            (
                torch.tensor(v_stitched.T).float().to(tensor.device),
                torch.tensor(u_stitched.T).float().to(tensor.device),
            )
        ]

    def _sparsify(self, tensor, rank_j, k_split, scheme):
        """Sparsify to the desired number of features (rank_j)."""
        # in some rare occassion Messi fails
        # --> then let's just use grouped projective sparsifier and stitch it
        try:
            return self._sparsify_with_messi(tensor, rank_j, k_split, scheme)
        except ValueError:
            weights_hat = super()._sparsify(tensor, rank_j, k_split, scheme)
            u_chunks, v_chunks = zip(*weights_hat)
            print(
                "Messi failed."
                " Falling back to grouped decomposition sparsification"
            )
            return [(torch.cat(u_chunks, dim=1), torch.block_diag(*v_chunks))]
