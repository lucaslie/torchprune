"""Module containing the SVD allocators."""

from ..alds.alds_allocator import ALDSErrorAllocator


class SVDNuclearEnergyAllocator(ALDSErrorAllocator):
    """Relative nuclear energy allocator.

    We minimize the maximum relative reduction in energy based on the nuclear
    norm, where the relative energy reduction is defined as:

    rel_energy_reduction = ||W - What||_nuclear // ||W||_nuclear

    Note that || . ||_nuclear is the sum of singular values, i.e., l1-norm of
    singular values.
    """

    def __init__(self, *args, **kwargs):
        """Initialize like parent class and assert that we don't have k>1."""
        super().__init__(*args, **kwargs)
        assert self._desired_k_split == 1, "No k-split allowed for SVD"

    def _compute_rel_error_for_weight(self, weight, k_split, scheme):
        # grab all singular values for decomposition
        # has shape [k_split x rank_per_k]
        singular_values = self._compute_sv_for_weight(weight, k_split, scheme)

        # compute nuclear norm for current operator
        # has shape []
        nuc_norm = self._compute_norm_for_weight(weight, scheme, ord="nuc")

        # compute nuclear norm of "residual operator" W - What
        # --> corresponds to sum of singular values not included.
        # --> we thus take a cumulative sum
        # We take "max" over k-splits since k-splits is always 1 (vanilla SVD)
        nuc_norm_residual = singular_values.flip(1).cumsum(dim=1).flip(1)
        nuc_norm_residual = nuc_norm_residual.max(dim=0)[0]

        # resulting "relative energy" for each layer and each possible rank_j!
        # shape is [max(num_sv)]
        rel_energy_reduction = nuc_norm_residual / nuc_norm

        return rel_energy_reduction


class SVDFrobeniusEnergyAllocator(SVDNuclearEnergyAllocator):
    """Relative Frobenius energy allocator.

    We minimize the maximum relative reduction in energy based on the Frobenius
    norm, where the relative energy reduction is defined as:

    rel_energy_reduction = ||W - What||_fro // ||W||_fro

    Note that || . ||_fro is the square root of the sum of squared singular
    values, i.e., l2-norm of singular values.
    """

    def _compute_rel_error_for_weight(self, weight, k_split, scheme):
        # grab all singular values for decomposition
        # has shape [k_split x rank_per_k]
        singular_values = self._compute_sv_for_weight(weight, k_split, scheme)

        # compute Frobenius norm for current operator
        # has shape []
        fro_norm = self._compute_norm_for_weight(weight, scheme, ord="fro")
        fro_norm_squared = fro_norm * fro_norm

        # compute Frobenius norm of "residual operator" W - What
        # --> corresponds to sum of squared singular values not included.
        # --> we thus take a cumulative sum of the squares
        # We take "max" over k-splits since k-splits is always 1 (vanilla SVD)
        sv_squared = singular_values * singular_values
        fro_norm_residual = sv_squared.flip(1).cumsum(dim=1).flip(1)
        fro_norm_residual = fro_norm_residual.max(dim=0)[0]

        # resulting "relative energy" for each layer and each possible rank_j!
        # shape is [max(num_sv)]
        rel_energy_reduction = fro_norm_residual / fro_norm_squared

        return rel_energy_reduction
