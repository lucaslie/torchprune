"""The module with the tracker for ThiNet."""

import copy

import torch
import torch.nn as nn
import numpy as np

from ..base import BaseTracker
from ...util import tensor


class ThiTracker(BaseTracker):
    """Tracker for ThiNet.

    With this class we keep track of the sensitivity definition as used in
    the paper
    https://arxiv.org/abs/1707.06342
    ThiNet: A Filter Level Pruning Method for Deep Neural Network Compression

    """

    def __init__(self, module, num_batches):
        """Initialize with module."""
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            raise NotImplementedError(
                "ThiTracker for {} is not implemented!".format(
                    type(module).__name__
                )
            )
        super().__init__(module)

        self._current_batch = None

        # by passing in an estimate of the number of batches the tracker will
        # see we can start to sub-sample during tracking to avoid overflow...
        self._num_batches = num_batches
        self._num_features_max = 100  # tracking at most 100 features...
        self._num_features_min_per_batch = 10

        # some stuff for later
        weight = self.module.weight.data
        self.register_buffer("sensitivity_in", torch.Tensor())
        self.sensitivity_in = torch.zeros(weight.shape[1]).to(weight.device)

        self.register_buffer("_features", torch.Tensor())
        self._features = torch.zeros(
            weight.shape[1], self._num_features_max
        ).to(weight.device)

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
            # flatten all batch dimensions ...
            a_unfolded = tensor.flatten_all_but_last(activations)
            # then unsqueeze for kernel and patch dimension
            a_unfolded = a_unfolded.unsqueeze(-1).unsqueeze(-1)

        return a_unfolded

    def reset(self):
        """Reset the internal statistics of the tracker."""
        self.sensitivity_in.fill_(0.0)
        self._features.fill_(0.0)
        self._current_batch = 1
        self._num_f_processed = 0

    def more_data_needed(self):
        """Return whether we already gathered enough features."""
        if self._num_f_processed < self._num_features_max:
            return True
        return False

    def _hook(self, module, ins, outs):
        if not self.more_data_needed():
            return
        activations = copy.deepcopy(ins[0].data)
        self._update_sensitivity(activations)

    def _backward_hook(self, grad):
        pass

    def finish_sensitivity(self):
        """Finish the sensitivity computation after gathering all the data."""
        # some stats
        weight = self.module.weight.data
        num_features_in = weight.shape[1]

        # things to keep track off
        selected_in = torch.zeros(num_features_in).bool().to(weight.device)
        errors_with_j = torch.zeros(num_features_in, device=weight.device)

        # reset sensitivity to 0
        self.sensitivity_in.fill_(0.0)

        # number of top values to add during each round
        k_per_round = max(1, int(np.sqrt(num_features_in)))

        # current rank we are trying to add here
        rank_current = num_features_in

        # (lazily evaluated) greedy approach to rank in features
        while not torch.all(selected_in):
            # reset errors with j
            errors_with_j.fill_(np.Inf)

            # loop through remaining features to see which to add next
            for idx_in in range(num_features_in):
                # it's already in the set, no need trying to add it...
                if selected_in[idx_in]:
                    continue

                # try adding in feature j
                selected_in[idx_in] = True

                # compute error
                errors_with_j[idx_in] = (
                    self._features[selected_in].sum(dim=0) ** 2
                ).sum()

                # remove j from selectedIn for now
                selected_in[idx_in] = False

            # try to pick at least sqrt(num_features_in) or what is left
            k_this_round = min(
                k_per_round, torch.isfinite(errors_with_j).sum().item()
            )

            # sanity check to break
            if k_this_round < 1:
                print("Sensitivity computation stopped with incomplete ranks.")
                break

            # pick topk to reduce runtime and them in
            _, idxs_top = errors_with_j.topk(
                k_this_round, largest=False, sorted=True
            )
            for idx_top in idxs_top:
                self.sensitivity_in[idx_top] = rank_current
                rank_current -= 1

            # set selected indices to be in now
            selected_in[idxs_top] = True

        # already pruned features should have sensitivity 0!
        per_feature_sum = weight.view(*weight.shape[:2], -1).abs().sum((0, 2))
        self.sensitivity_in[per_feature_sum == 0.0] = 0.0

    def _update_sensitivity(self, activations):
        # Wunfold.shape = (outFeature, inFeatures, kernel1*kernel2)
        weight = self.module.weight.data
        w_unfolded = weight.view(weight.shape[0], weight.shape[1], -1)

        # shape = (batchSize, inFeatures, kernel1*kernel2, num_patches)
        a_unfolded = self._unfold(activations)

        # total features we receive per in_feature would be
        # batch_size * out_features * num_patches
        total_features = (
            a_unfolded.shape[0] * w_unfolded.shape[0] * a_unfolded.shape[-1]
        )

        # features will be sub-sampled to num_f_keep
        # num_f_keep represents the anticipated average number of features to
        # keep per batch that is incoming. That way we avoid storing too many
        # unnecessary features.
        num_f_keep = int(self._num_features_max / self._num_batches) + 1
        num_f_keep = max(self._num_features_min_per_batch, num_f_keep)
        num_f_keep = min(
            num_f_keep, self._num_features_max - self._num_f_processed
        )
        num_f_keep = min(num_f_keep, total_features)

        # create subset from total features
        idx_keep = torch.randperm(total_features)[:num_f_keep]

        # populate sensitivity for each input channel
        for idx_f in range(w_unfolded.shape[1]):
            # compute features for one f
            # (batchSize, outFeatures, kernel1*kernel2, num_patches)
            features_f = (
                w_unfolded[None, :, idx_f, :, None]
                * a_unfolded[:, idx_f : idx_f + 1]
            )

            # now store them
            self._features[
                idx_f,
                self._num_f_processed : self._num_f_processed + num_f_keep,
            ] = (
                features_f.sum(dim=2).view(-1)[idx_keep].clone().detach()
            )

        # update running counts
        self._current_batch += 1
        self._num_f_processed += num_f_keep
