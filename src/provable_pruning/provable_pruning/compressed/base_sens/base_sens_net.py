"""Implementation for abstract base class for sensitivity-based nets.

DESIGN PHILOSOPHY OF THE COMPRESSED NET CLASSES:

1.) All functions are implemented in a modular, hierarchical Base classes
2.) Actual classes just define properties and inject various functionality by
    deriving from all required base classes
3.) Overwrite small functions if deemed absolutely necessary
"""

from abc import ABC, abstractmethod
import math

import torch.nn as nn
from ..base import CompressedNet


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
        self, original_net, loader_s, delta_failure=1.0e-16, c_constant=3
    ):
        """Initialize this class with additional data and hyperparameters."""
        super().__init__(original_net)

        # a few parameters
        self._delta_failure = delta_failure
        self._c_constant = c_constant

        # a few required objects
        self._loader_s = loader_s
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
        for i, ell in enumerate(self.layers):
            module = self.compressed_net.compressible_layers[ell]

            self._sens_trackers.append(self._tracker(module))
            self._sens_trackers[ell].enable_tracker()

        device = module.weight.device

        # loop through batches and track sensitivity
        for images, _ in self._loader_s:
            # put on correct device
            images = images.to(device)

            # further split it into mini-batches ...
            sens_batch_size = 4
            batch_size = images.data.shape[0]
            sens_batch_size = min(sens_batch_size, batch_size)

            # process sensitivity in smaller batches
            for i in range(math.ceil(batch_size / sens_batch_size)):
                # compute start and end index for indexing into ins, outs
                idx_start = i * sens_batch_size
                idx_end = min((i + 1) * sens_batch_size, batch_size)
                # compute the sensivity with the mini-batch
                self._compute_sensitivity(images[idx_start:idx_end])

        # disable trackers
        for ell in self.layers:
            self._sens_trackers[ell].disable_tracker()

    def _compute_sensitivity(self, images):
        """Simply do a forward pass to obtain edge/neuron sensitivities."""
        self.compressed_net(images)

    def _finish_preprocessing(self):
        del self._sens_trackers
        self._sens_trackers = nn.ModuleList()
