"""A module summarizing all the custom losses and the torch.nn losses."""

import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss  # noqa: F403,F401


class CrossEntropyLossWithAuxiliary(nn.CrossEntropyLoss):
    """Cross-entropy loss that can add auxiliary loss if present."""

    def forward(self, input, target):
        """Return cross-entropy loss and add auxiliary loss if possible."""
        if isinstance(input, dict):
            loss = super().forward(input["out"], target)
            if "aux" in input:
                loss += 0.5 * super().forward(input["aux"], target)
        else:
            loss = super().forward(input, target)
        return loss


class LossFromInput(nn.Module):
    """Loss that is directly extracted from the input dictionary."""

    def forward(self, input, target):
        """Return loss from the inputs and ignore targets."""
        return input["loss"] if isinstance(input, dict) else input[0]
