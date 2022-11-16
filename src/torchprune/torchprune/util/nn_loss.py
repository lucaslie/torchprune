"""A module summarizing all the custom losses and the torch.nn losses."""

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss  # noqa: F403,F401

from .external.ffjord.train_misc import standard_normal_logprob


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


class NLLPriorLoss(nn.Module):
    """Loss corresponding to NLL between output and prior."""

    def forward(self, input, target):
        """Return average NLL."""
        prior = input["prior"]
        out = input["out"]
        logprob = prior.log_prob(out[:, 1:]).to(out) - out[:, 0]
        return -logprob.mean()


class NLLNatsLoss(nn.Module):
    """Loss corresponding to standard normal logprob loss.

    Check out util.external.ffjord.train_tabular.compute_loss for more info.
    """

    def _compute_logprob(self, input):
        """Compute and return standard normal log prob."""
        z_out = input["out"]
        delta_logp = input["delta_logp"]
        logpz = (
            standard_normal_logprob(z_out)
            .view(z_out.shape[0], -1)
            .sum(1, keepdim=True)
        )
        logpx = logpz - delta_logp
        loss = -torch.mean(logpx)
        return loss

    def forward(self, input, target):
        """Return average standard normal logprob loss."""
        loss = self._compute_logprob(input)

        # add regularizer loss if needed
        if "reg_loss" in input:
            loss += input["reg_loss"]

        # return overall loss
        return loss


class NLLBitsLoss(NLLNatsLoss):
    """Loss corresponding to logprob loss expressed in "bits/dim"."""

    def _compute_logprob(self, input):
        """Compute and return log prob normalized as bits/dim."""
        z_out = input["out"]
        delta_logp = input["delta_logp"]
        logpz = (
            standard_normal_logprob(z_out)
            .view(z_out.shape[0], -1)
            .sum(1, keepdim=True)
        )
        # averaged over batches
        logpx = logpz - delta_logp
        logpx_per_dim = torch.sum(logpx) / input["nelement"]
        bits_per_dim = -(logpx_per_dim - np.log(256)) / np.log(2)
        return bits_per_dim
