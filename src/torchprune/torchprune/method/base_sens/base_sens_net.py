"""Implementation for abstract base class for sensitivity-based nets.

DESIGN PHILOSOPHY OF THE COMPRESSED NET CLASSES:

1.) All functions are implemented in a modular, hierarchical Base classes
2.) Actual classes just define properties and inject various functionality by
    deriving from all required base classes
3.) Overwrite small functions if deemed absolutely necessary
"""

from abc import ABC, abstractmethod

import torch.nn as nn
from ..base import CompressedNet
from ...util import tensor


class BaseSensNet(CompressedNet, ABC):
    """This class contains basic interfaces required for sensitivity."""

    @property
    @abstractmethod
    def _allocator(self):
        """Return class used for allocator."""
        raise NotImplementedError

    @property
    @abstractmethod
    def _sparsifier(self):
        """Return class used for sparsifier."""
        raise NotImplementedError

    @property
    @abstractmethod
    def _pruner(self):
        """Return class used for pruner."""
        raise NotImplementedError

    @property
    @abstractmethod
    def _tracker(self):
        raise NotImplementedError

    @property
    def retrainable(self):
        """Return whether compressed net is retrainable."""
        return True

    def _get_allocator(self):
        # initialize sample size allocator
        return self._allocator(
            net=self.compressed_net,
            trackers=self._sens_trackers,
            delta_failure=self._delta_failure,
            c_constant=self._c_constant,
        )

    def __init__(
        self,
        original_net,
        loader_s,
        loss_handle,
        delta_failure=1.0e-16,
        c_constant=3,
    ):
        """Initialize this class with additional data and hyperparameters."""
        super().__init__(original_net, loader_s, loss_handle)

        # a few parameters
        self._delta_failure = delta_failure
        self._c_constant = c_constant

        # a few required objects
        self._sens_trackers = nn.ModuleList()

    def _get_pruner(self, ell):
        module = self.compressed_net.compressible_layers[ell]
        return self._pruner(
            tensor=module.weight, tracker=self._sens_trackers[ell]
        )

    def _get_sparsifier(self, pruner):
        return self._sparsifier(pruner)

    def _start_preprocessing(self):
        self._sens_trackers = nn.ModuleList()
        # create and enable tracker for each layer
        for ell, module in enumerate(self.compressed_net.compressible_layers):
            self._sens_trackers.append(self._tracker(module))
            self._sens_trackers[ell].enable_tracker()

        device = self.compressed_net.compressible_layers[0].weight.device

        # get a loader with mini-batch size 1
        loader_mini = tensor.MiniDataLoader(self._loader_s, 1)
        num_batches = len(loader_mini)

        # loop through batches and track sensitivity
        for i_batch, (images, _) in enumerate(loader_mini):
            # compute sensitivity for this batch
            self._compute_sensitivity(tensor.to(images, device))
            print(f"Processed sensitivity batch: [{i_batch+1}/{num_batches}]")

        # disable trackers
        for ell in self.layers:
            self._sens_trackers[ell].disable_tracker()

    def _compute_sensitivity(self, images):
        """Simply do a forward pass to obtain edge/neuron sensitivities."""
        self.compressed_net(images)

    def _finish_preprocessing(self):
        del self._sens_trackers
        self._sens_trackers = nn.ModuleList()
