"""The module with the tracker for ThiNet."""

import copy

import torch
import torch.nn as nn
import numpy as np

from ..base import BaseTracker


class ThiTracker(BaseTracker):
    """Tracker for ThiNet.

    With this class we keep track of the sensitivity definition as used in
    the paper
    https://arxiv.org/abs/1707.06342
    ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression

    """

    def __init__(self, module):
        """Initialize with module."""
        if (not isinstance(module, nn.Conv2d)) and (
            not isinstance(module, nn.Linear)
        ):
            raise NotImplementedError(
                "ThiTracker for {} is not implemented!".format(
                    type(module).__name__
                )
            )
        super().__init__(module)

        # some stuff for later
        self.register_buffer("sensitivity_in", torch.Tensor())
        self.sensitivity_in = torch.Tensor()

        self.register_buffer("_features", torch.Tensor())
        self._features = torch.Tensor()

        self._current_batch = None

        # initialize the standard stuff:
        self.reset()

    def _unfold(self, activations):
        # shape= (batchSize, inChannels, kernel1 * kernel2, num_patches)
        if isinstance(self.module, nn.Conv2d):
            f_unfold = nn.Unfold(
                kernel_size=self.module.kernel_size,
                stride=self.module.stride,
                padding=self.module.padding,
                dilation=self.module.dilation,
            )
            a_unfolded = f_unfold(activations)
            a_unfolded = a_unfolded.view(
                activations.shape[0],
                activations.shape[1],
                -1,
                a_unfolded.shape[-1],
            )
        elif isinstance(self.module, nn.Linear):
            a_unfolded = activations.unsqueeze(-1).unsqueeze(-1)

        return a_unfolded

    def reset(self):
        """Reset the internal statistics of the tracker."""
        weight = self.module.weight.data
        self.sensitivity_in = torch.zeros(weight.shape[1]).to(weight.device)
        self._features = torch.Tensor()
        self._current_batch = 1

    def _hook(self, module, ins, outs):
        activations = copy.deepcopy(ins[0].data)
        self._update_sensitivity(activations)

    def _backward_hook(self, grad):
        pass

    def finish_sensitivity(self):
        """Finish the sensitivity computation after gathering all the data."""
        # do at most 1000 features
        idx = torch.randperm(self._features.shape[1])[:100]
        self._features = self._features[:, idx]

        weight = self.module.weight.data
        num_features_in = weight.shape[1]
        selected_in = torch.zeros(num_features_in).bool()

        # greedy approach to rank in features
        for rank in reversed(range(num_features_in)):
            error_best = torch.Tensor([np.Inf])
            best = None

            # loop through remaining features to see which to add next
            for idx_in in range(num_features_in):
                # it's already in the set, no need trying to add it...
                if selected_in[idx_in]:
                    continue

                # try adding in feature j and compute error
                selected_in[idx_in] = 1
                error_with_j = (
                    self._features[selected_in].sum(dim=0) ** 2
                ).sum()

                # see if it's better than previous best
                if error_with_j < error_best:
                    error_best = error_with_j
                    best = idx_in

                # remove j from selectedIn for now
                selected_in[idx_in] = 0

            # add best one from this round to selectedIn
            selected_in[best] = 1

            # also note the rank of best in the sensitivities
            self.sensitivity_in[best] = rank

    def _update_sensitivity(self, activations):
        # Wunfold.shape = (outFeature, inFeatures, kernel1*kernel2)
        weight = self.module.weight.data
        w_unfolded = weight.view(weight.shape[0], weight.shape[1], -1)

        self._current_batch += 1

        # shape = (batchSize, inFeatures, kernel1*kernel2, num_patches)
        a_unfolded = self._unfold(activations)

        # keep track of features for each output
        # (inFeatures, batchSize*outFeatures*num_patches)
        features = torch.zeros(
            a_unfolded.shape[1],
            a_unfolded.shape[0] * w_unfolded.shape[0] * a_unfolded.shape[-1],
        )

        # populate sensitivity for each input channel
        for idx_f in range(w_unfolded.shape[1]):
            # compute features for one f
            # (batchSize, outFeatures, kernel1*kernel2, num_patches)
            features_f = w_unfolded[:, idx_f].unsqueeze(0).unsqueeze(
                -1
            ) * a_unfolded[:, idx_f].unsqueeze(1)
            features[idx_f] = features_f.sum(dim=2).view(-1)

        self._features = torch.cat((features, self._features), dim=-1)
