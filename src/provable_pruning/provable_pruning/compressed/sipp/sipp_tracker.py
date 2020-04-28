"""All sipp-related trackers."""

import torch

from ..base_sens import BaseSensTracker

__all__ = ["SiPPRandTracker", "SiPPTracker"]


class SiPPRandTracker(BaseSensTracker):
    """The tracker for randomized SiPPRand."""

    def _reduction(self, g_sens_f, dim):
        return torch.max(g_sens_f, dim=dim)[0]


# Deterministic SiPP tracker should use the mean
class SiPPTracker(BaseSensTracker):
    """The tracker for deterministic SiPP."""

    def _reduction(self, g_sens_f, dim):
        return torch.mean(g_sens_f, dim=dim)
