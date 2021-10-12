"""Module implementing network compression with data-dependent PCA."""

import torch.nn as nn

from ...util import tensor
from ..base_decompose import BaseDecomposeNet

from .pca_allocator import PCAAllocator
from .pca_pruner import PCAPruner
from .pca_sparsifier import PCASparsifier
from .pca_tracker import PCATracker


class PCANet(BaseDecomposeNet):
    """Data-dependent PCA for weight decomposition.

    This is a procedure following the following setup:
    * data-dependent low-rank decomposition based on scheme 0
    * greedy rank selection for budget allocation

    This was used in the following work:
    * https://arxiv.org/abs/1505.06798
    """

    @property
    def _allocator_type(self):
        """Get allocator type."""
        return PCAAllocator

    @property
    def _sparsifier_type(self):
        """Get sparsifier type."""
        return PCASparsifier

    @property
    def _k_split(self):
        """Get number of k splits in each layer."""
        return 1

    def __init__(self, *args, **kwargs):
        """Add trackers to initialization."""
        super().__init__(*args, **kwargs)

        # get a list of trackers
        self._trackers = nn.ModuleList()

    def _start_preprocessing(self):
        device = self.compressed_net.compressible_layers[0].weight.device
        num_batches = len(self._loader_s)
        num_layers = len(self.layers)

        self._trackers = nn.ModuleList()
        # create and enable tracker for each layer
        for ell, module in enumerate(self.compressed_net.compressible_layers):
            tracker = PCATracker(module, len(self._loader_s))
            tracker.enable_tracker()
            self._trackers.append(tracker)

        # do a forward pass to collect data
        for i_batch, (images, _) in enumerate(self._loader_s):
            if not any(trkr.more_data_needed() for trkr in self._trackers):
                break
            # keep track of PCA stats now
            self.compressed_net(tensor.to(images, device))
            print(f"Processed PCA batch: [{i_batch+1}/{num_batches}]")

        # post-process and compute PCA
        for ell, tracker in enumerate(self._trackers):
            tracker.finish_pca()
            tracker.disable_tracker()
            print(f"Post-processed PCA, layer: [{ell+1}/{num_layers}]")

    def _finish_preprocessing(self):
        del self._trackers
        self._trackers = nn.ModuleList()

    def _get_allocator(self):
        return self._allocator_type(self.compressed_net, self._trackers)

    def _get_pruner(self, ell):
        """Return a fake pruner since we don't need a pruner."""
        module = self.compressed_net.compressible_layers[ell]
        tracker = self._trackers[ell]
        return PCAPruner(module.weight, tracker)
