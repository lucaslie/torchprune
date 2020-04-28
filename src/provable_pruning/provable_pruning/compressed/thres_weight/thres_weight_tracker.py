"""Module with tracker for ThresNet."""

import torch

from ..base import BaseTracker


class ThresTracker(BaseTracker):
    """A simple tracker for weights only."""

    def __init__(self, module):
        """Initialize with the module to track."""
        super().__init__(module)

        # some stuff
        self.register_buffer("sensitivity", torch.Tensor())
        self.register_buffer("idx_plus", torch.Tensor())
        self.register_buffer("idx_minus", torch.Tensor())

        # compute sensitivity from abs weights
        self.sensitivity = torch.abs(self.module.weight.data)

        # also have some "fake indices"
        self.idx_plus = torch.ones_like(self.sensitivity).bool()
        self.idx_minus = torch.zeros_like(self.sensitivity).bool()

    def reset(self):
        """Reset the internal statistics of the tracker."""
        pass

    def _hook(self, module, ins, outs):
        pass

    def _backward_hook(self, grad):
        pass
