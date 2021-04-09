"""Module with tracker for SnipNet."""

import torch

from ..base import BaseTracker


class SnipTracker(BaseTracker):
    """A simple tracker for SNIP derivative.

    NOTE: This implementation assumes a loss function that is additively
    decomposable over the data points (i.e. we can do batched computation over
    the total loss by summing up the individual batched loss terms).
    """

    def __init__(self, module):
        """Initialize with the module to track."""
        super().__init__(module)

        # some stuff
        self.register_buffer("sensitivity", torch.Tensor())
        self.register_buffer("sensitivity_sign", torch.Tensor())
        self.register_buffer("idx_plus", torch.Tensor())
        self.register_buffer("idx_minus", torch.Tensor())

        # make sure we can remember current batch size
        self._batch_size_current = 0
        self._batch_size_total = 0

        # now reset values
        self.reset()

    def reset(self):
        """Reset the internal statistics of the tracker."""
        # store sensitivity
        weight = self.module.weight.data
        self.sensitivity = torch.zeros(weight.shape).to(weight.device)

        # store sensitivity sign separately
        self.sensitivity_sign = torch.sign(self.sensitivity)

        # also have some "fake indices"
        self.idx_plus = torch.ones_like(self.sensitivity).bool()
        self.idx_minus = torch.zeros_like(self.sensitivity).bool()

        # keep track of batch size
        self._batch_size_current = 0
        self._batch_size_total = 0

    def _hook(self, module, ins, outs):
        # store current batch size during forward pass
        self._batch_size_current = len(ins[0])

    def _backward_hook(self, grad):
        alpha = self._batch_size_total / (
            self._batch_size_current + self._batch_size_total
        )
        self._batch_size_total += self._batch_size_current

        # recover sign
        self.sensitivity *= self.sensitivity_sign

        # keep track of average sensitivity over batches
        self.sensitivity *= alpha
        self.sensitivity += (1.0 - alpha) * grad * self.module.weight

        # remove and store sign
        self.sensitivity_sign = torch.sign(self.sensitivity)
        self.sensitivity = torch.abs(self.sensitivity)

        # reset current batch size since we already incorporated the batch
        self._batch_size_current = 0
