"""Module containing the base interface for a tracker."""
from abc import ABC, abstractmethod
import torch.nn as nn


class BaseTracker(ABC, nn.Module):
    """A basic interface for trackers."""

    def __init__(self, module):
        """Initialize with the module to track."""
        super().__init__()
        self.module = module

        # store hooks
        self._hook_handle = None
        self._backward_hook_handle = None

    def enable_tracker(self):
        """Enable the tracker, do nothing if enabled."""
        self._set_tracking(True)

    def disable_tracker(self):
        """Disable the tracker, do nothing if disabled."""
        self._set_tracking(False)

    def _set_tracking(self, tracking_mode=True):
        if tracking_mode and self._hook_handle is None:
            self._hook_handle = self.module.register_forward_hook(self._hook)
            self._backward_hook_handle = self.module.weight.register_hook(
                self._backward_hook
            )
        elif tracking_mode is False and self._hook_handle is not None:
            self._hook_handle.remove()
            self._backward_hook_handle.remove()
            self._hook_handle = None
            self._backward_hook_handle = None

    def forward(self, x):
        """Fake forward function since it is inheriting from nn.Module."""

    @abstractmethod
    def reset(self):
        """Reset the internal statistics of the tracker."""
        raise NotImplementedError

    @abstractmethod
    def _hook(self, module, ins, outs):
        """Execute this function after outs=module.forward(ins) is computed."""
        raise NotImplementedError

    @abstractmethod
    def _backward_hook(self, grad):
        """Execute this function after grad of module.weight is computed."""
        raise NotImplementedError
