"""The module with the tracker for ThiNet."""


import torch
import torch.nn as nn

from ..base import BaseTracker


class PCATracker(BaseTracker):
    """Tracker for PCANet.

    With this class we keep track of data matrix so we can compute PCA
    afterwards.
    """

    def __init__(self, module, num_batches):
        """Initialize with module."""
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            raise NotImplementedError(
                f"PCATracker for {type(module).__name__} is not implemented!"
            )
        super().__init__(module)

        self._current_batch = None

        # by passing in an estimate of the number of batches the tracker will
        # see we can start to sub-sample during tracking to avoid overflow...
        self._num_batches = num_batches
        self._num_features_max = 30000  # tracking at most 30,000 patches
        self._num_features_min_per_batch = 10

        # some stuff for later
        self.register_buffer("principle_components", None)
        self.principle_components = None

        self.register_buffer("sensitivity", None)
        self.sensitivity = None

        self.register_buffer("data_mean", torch.Tensor())
        self.data_mean = None

        self.register_buffer("_data", torch.Tensor())
        self._data = None

        # initialize the standard stuff:
        self.reset()

    def reset(self):
        """Reset the internal statistics of the tracker."""
        self.principle_components = None
        self.sensitivity = None
        self.data_mean = None
        self._data = None
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
        self._update_data(outs)

    def _backward_hook(self, grad):
        pass

    def _update_data(self, outs):
        """Store a subsampled set from the outputs of the layer."""
        # outs has dimension batch_size * out_features * height * width
        # where height * width is optional in case of linear layers
        total_features = outs[:, 0].numel()

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

        # initialize data matrix if necessary
        if self._data is None:
            # out_channels x num_features_max
            self._data = torch.zeros(
                outs.shape[1],
                self._num_features_max,
                dtype=outs.dtype,
                device=outs.device,
            )

        # reshape outs
        # out_features x (batch_size * height * width)
        outs = outs.detach().transpose(0, 1).reshape(outs.shape[1], -1)

        # sub-sample and store outs
        idx_keep = torch.randperm(total_features)[:num_f_keep]
        self._data[
            :, self._num_f_processed : self._num_f_processed + num_f_keep
        ].copy_(outs[:, idx_keep])

        # update running counts
        self._current_batch += 1
        self._num_f_processed += num_f_keep

    def finish_pca(self):
        """Compute PCA from the data matrix."""
        # check that all data is valid or sub-sample
        idx_valid = self._data.sum(dim=0) != 0.0
        data = self._data[:, idx_valid]

        # center data and store mean
        data_mean = data.mean(dim=1, keepdim=True)
        data = data - data_mean
        self.data_mean = data_mean[:, 0]

        # covariance matrix
        cov_mat = data @ data.T

        # solve eigen-decomposition for "principle components"
        self.principle_components, s_values = torch.svd(cov_mat)[:2]

        # compute and store sensitivity
        # note that we have convention that "bigger is better"
        self.sensitivity = s_values / s_values.cumsum(0)
