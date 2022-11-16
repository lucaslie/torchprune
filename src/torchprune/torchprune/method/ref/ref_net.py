"""Module implementing fake network compression for reference stats."""
import torch
from ..base import BaseCompressedNet


class ReferenceNet(BaseCompressedNet):
    """The network implementing the reference network."""

    def __init__(self, original_net):
        """Initialize the compression with a uncompressed network."""
        super().__init__(original_net)

        self.register_buffer("_keep_ratio_latest", torch.Tensor())
        self._keep_ratio_latest = torch.tensor(1.0)

    @property
    def deterministic(self):
        """Indicate whether compression method is deterministic."""
        return True

    @property
    def retrainable(self):
        """Indicate whether we can retrain after applying this method."""
        return False

    def compress(self, keep_ratio, from_original=True, initialize=True):
        """Execute a fake compression step."""
        budget_per_layer = [
            keep_ratio * float(module.weight.data.numel())
            for module in self.compressed_net.compressible_layers
        ]
        self._keep_ratio_latest = torch.tensor(keep_ratio)
        return budget_per_layer

    def size(self):
        """Fake size with current keep_ratio."""
        return super().size() * float(self._keep_ratio_latest)

    def flops(self):
        """Fake flops with current keep_ratio."""
        return super().flops() * float(self._keep_ratio_latest)


class FakeNet(ReferenceNet):
    """A reference net that is retrainable.

    By enabling retraining we can simulate an unpruned network that gets the
    exact same training and retraining as the pruned networks.
    """

    @property
    def retrainable(self):
        """Indicate whether we can retrain after applying this method."""
        return True
