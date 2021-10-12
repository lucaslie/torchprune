"""Module for PCAPruner (simple wrapper to keep track of stats)."""

from ..base import TensorPruner2


class PCAPruner(TensorPruner2):
    """A fake pruner that stores reference to tracker stats."""

    def __init__(self, tensor, tracker, **kwargs):
        """Initialize with tracker to reference relevant stats."""
        super().__init__(tensor, tensor.detach().clone().abs(), **kwargs)
        self.principle_components = tracker.principle_components
        self.sensitivity = tracker.sensitivity
        self.data_mean = tracker.data_mean
        self.bias = tracker.module.bias
