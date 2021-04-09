"""All sensitivity-related trackers."""

import torch

from ..base_sens import BaseSensTracker


class PFPTracker(BaseSensTracker):
    """The tracker for all PFP variations."""

    def _reduction(self, g_sens_f, dim):
        return torch.max(g_sens_f, dim=dim)[0]
