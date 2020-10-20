"""Module containing (fake) sparsifiers for PFP."""

from ..base import BaseSparsifier, FilterSparsifier


class PFPRandSparsifier(FilterSparsifier):
    """The fake sparsifier for random PFPRand."""

    @property
    def _do_reweighing(self):
        return True

    def __init__(self, pruner):
        """Initialize the sparsifier from the pruner of the same layer."""
        super().__init__(pruner, out_mode=False)


class PFPSparsifier(FilterSparsifier):
    """The fake sparsifier for deterministic PFP."""

    def __init__(self, pruner):
        """Initialize the sparsifier from the pruner of the same layer."""
        super().__init__(pruner, out_mode=False)


class PFPTopSparsifier(BaseSparsifier):
    """The fake sparsifier for hybrid PFPTop."""

    def __init__(self, pruner):
        """Initialize the sparsifier from the pruner of the same layer."""
        super().__init__(pruner)
        self._sparsifier_pfp_det = PFPSparsifier(pruner.pruner_pfp_det)
        self._sparsifier_pfp_rand = PFPRandSparsifier(pruner.pruner_pfp_rand)
        self.pruner = pruner

    def sparsify(self, num_samples):
        """Fake-sparsify the edges (we don't do sparsification for filters)."""
        # This fixed the bug for PopNet
        self._sparsifier_pfp_rand._probability_div = (
            self.pruner.pruner_pfp_rand.probability_div
        )

        weight_hat_det = self._sparsifier_pfp_det.sparsify(num_samples[0])
        weight_hat_rand = self._sparsifier_pfp_rand.sparsify(num_samples[1])

        return weight_hat_det + weight_hat_rand
