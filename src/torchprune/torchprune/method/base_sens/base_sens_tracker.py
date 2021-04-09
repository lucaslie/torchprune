"""The sensitivity-related base tracker."""
import copy
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn

from ..base import BaseTracker
from ...util import tensor


class BaseSensTracker(BaseTracker, ABC):
    """The classic sensitivity tracker for SiPP and PFP."""

    @abstractmethod
    def _reduction(self, g_sens_f, dim):
        raise NotImplementedError

    def __init__(self, module, absolute=False):
        """Initialize with module to track and whether we track abs values."""
        if (not isinstance(module, nn.Conv2d)) and (
            not isinstance(module, nn.Linear)
        ):
            raise NotImplementedError(
                "BaseSensTracker for {} is not implemented!".format(
                    type(module).__name__
                )
            )
        super().__init__(module)

        self._absolute = absolute

        # have a direct reference to the weights as well
        self.weight = module.weight

        # some stuff for later
        self.register_buffer("sensitivity", torch.Tensor())
        self.sensitivity = torch.Tensor()

        self.register_buffer("sensitivity_in", torch.Tensor())
        self.sensitivity_in = torch.Tensor()

        self.register_buffer("idx_plus", torch.Tensor())
        self.idx_plus = torch.Tensor()

        self.register_buffer("idx_minus", torch.Tensor())
        self.idx_minus = torch.Tensor()

        self._num_points_processed = None
        self.num_patches = None
        self._current_batch = None

        # initialize the standard stuff
        if self._absolute:
            weight = torch.abs(self.module.weight.data)
        else:
            weight = self.module.weight.data
        self.reset()

        # remove weights we are not interested in and have two modules ...
        idx_plus = weight > 0.0
        idx_minus = weight < 0.0
        weight_plus = torch.zeros_like(weight)
        weight_minus = torch.zeros_like(weight)
        weight_plus[idx_plus] = weight[idx_plus]
        weight_minus[idx_minus] = weight[idx_minus]

        # save a deepcopy of module with WPlus ...
        self._module_plus = copy.deepcopy(self.module)
        self._module_plus.weight.data = weight_plus
        self._remove_bias(self._module_plus)

        # save a deepcopy of module with WPlus ...
        self._module_minus = copy.deepcopy(self.module)
        self._module_minus.weight.data = weight_minus
        self._remove_bias(self._module_minus)

        # save indices
        self.idx_plus = idx_plus
        self.idx_minus = idx_minus

    def unfold(self, x):
        """Unfold depending on type of layer.

        After unfolding we want shape [batch_size, feature_size, patch_size]
        """
        if isinstance(self.module, nn.Conv2d):
            return nn.functional.unfold(
                x,
                kernel_size=self.module.kernel_size,
                stride=self.module.stride,
                padding=self.module.padding,
                dilation=self.module.dilation,
            )
        else:
            # flatten all batch dimensions, then unsqueeze last dim
            return tensor.flatten_all_but_last(x).unsqueeze(-1)

    def reset(self):
        """Reset the internal statistics of the tracker."""
        weight = self.module.weight.data
        self.sensitivity = torch.zeros(weight.shape).to(weight.device)
        self.sensitivity_in = torch.zeros(weight.shape[1]).to(weight.device)

        self._num_points_processed = 0
        self.num_patches = None
        self._current_batch = 1

    def _hook(self, module, ins, outs):
        a_adapted = self._adapt_activations(ins[0].data)

        if self.module.bias is not None:
            shape = np.array(outs.data.shape)
            shape_div = copy.deepcopy(shape)
            if isinstance(self.module, nn.Linear):
                shape_div[-1] = 1
            else:
                shape_div[-3] = 1
            shape_bias = (shape / shape_div).astype(int).tolist()
            outs_no_bias = outs.data - self.module.bias.view(shape_bias)
        else:
            outs_no_bias = outs.data.clone().detach()

        # get the new g
        g_sens, g_sens_in = self._update_g_sens(a_adapted, outs_no_bias, outs)

        # now update the sensitivity with the new g values.
        self._update_sensitivity(g_sens, g_sens_in)

    def _backward_hook(self, grad):
        pass

    def _adapt_activations(self, a_original):

        if self._absolute:
            return torch.abs(a_original)

        bigger = a_original >= 0.0
        if torch.all(bigger):
            return a_original

        # change activations matrix to be only positive if necessary
        a_pos = torch.zeros_like(a_original)
        a_neg = torch.zeros_like(a_original)
        a_pos[bigger] = a_original[bigger]
        a_neg[~bigger] = a_original[~bigger]

        return torch.cat((a_pos, -a_neg))

    def _remove_bias(self, module):
        # remove bias since we don't need it for our computations
        if module.bias is not None:
            bias_original = module.bias
            bias_zeros = torch.zeros(bias_original.shape).to(
                bias_original.device
            )
            module.state_dict()["bias"].copy_(bias_zeros)

    def _process_denominator(self, z_values):
        # processing
        eps = torch.Tensor([np.finfo(np.float32).eps]).to(z_values.device)
        mask = torch.le(torch.abs(z_values), eps)
        z_values.masked_fill_(mask, np.Inf)
        return z_values

    def _get_g_sens_f(self, weight_f, activations, z_values_f):
        # compute g
        g_sens_f = weight_f.unsqueeze(0).unsqueeze(-1) * activations
        g_sens_f /= z_values_f.unsqueeze(1)

        return g_sens_f.clone().detach()

    def _reshape_z(self, z_values):
        # flatten all batch dimensions first if it's linear...
        if isinstance(self.module, nn.Linear):
            z_values = tensor.flatten_all_but_last(z_values)
        z_values = z_values.view(z_values.shape[0], z_values.shape[1], -1)
        return z_values

    def _compute_g_sens_f(
        self, idx_f, w_unfold_plus, w_unfold_minus, a_unfold, z_plus, z_minus
    ):

        g_sens_f = torch.max(
            self._get_g_sens_f(
                w_unfold_plus[idx_f], a_unfold, z_plus[:, idx_f]
            ),
            self._get_g_sens_f(
                w_unfold_minus[idx_f], a_unfold, z_minus[:, idx_f]
            ),
        )
        return g_sens_f

    def _update_g_sens(self, activations, z_no_bias, outs):

        # Wunfold.shape = (outFeature, filterSize)
        # where filterSize = inFeatures*kappa1*kappa2 for conv2d
        # or filterSize = inNeurons for linear
        weight_plus = self._module_plus.weight.data
        weight_minus = self._module_minus.weight.data
        w_unfold_plus = weight_plus.view((weight_minus.shape[0], -1))
        w_unfold_minus = weight_minus.view((weight_minus.shape[0], -1))

        # now compute Z, shape = [batchSize, outFeatures, outExamples], where
        # outExamples = 1 for linear or
        # outExamples = number of patches for conv2d
        z_plus = self._module_plus(activations)
        z_minus = self._module_minus(activations)

        outs = self._reshape_z(outs)
        z_plus = self._reshape_z(z_plus)
        z_minus = self._reshape_z(z_minus)
        z_no_bias = self._reshape_z(z_no_bias)

        self._num_points_processed += activations.shape[0]

        self._current_batch += 1
        z_plus = self._process_denominator(z_plus)
        z_minus = self._process_denominator(z_minus)

        # shape = (batchSize, filterSize, outExamples) as above
        a_unfolded = self.unfold(activations)

        if self.num_patches is None:
            self.num_patches = a_unfolded.shape[-1]

        # preallocate g
        batch_size = a_unfolded.shape[0]
        device = self.sensitivity.device
        g_sens = torch.zeros((batch_size,) + self.sensitivity.shape).to(device)
        g_sens_in = torch.zeros((batch_size,) + self.sensitivity_in.shape).to(
            device
        )

        # populate g for this batch
        for idx_f in range(w_unfold_plus.shape[0]):
            # compute g
            g_sens_f = self._compute_g_sens_f(
                idx_f,
                w_unfold_plus,
                w_unfold_minus,
                a_unfolded,
                z_plus,
                z_minus,
            )

            # Reduction over outExamples
            g_sens_f = self._reduction(g_sens_f, dim=-1)

            # Reduction over outputChannels
            g_sens_in_f = self._reduction(
                g_sens_f.view((g_sens_f.shape[0], weight_plus.shape[1], -1)),
                dim=-1,
            )

            # store results
            g_sens[:, idx_f] = g_sens_f.view_as(g_sens[:, idx_f])
            g_sens_in = torch.max(g_sens_in, g_sens_in_f)

        # return g
        return g_sens, g_sens_in

    def _update_sensitivity(self, g_sens, g_sens_in):
        # get a quick reference
        sens = self.sensitivity
        sens_in = self.sensitivity_in

        # Max over this batch
        sens_batch = torch.max(g_sens, dim=0)[0]
        sens_in_batch = torch.max(g_sens_in, dim=0)[0]

        # store sensitivity
        self.sensitivity.copy_(torch.max(sens, sens_batch.view(sens.shape)))

        # store sensitivity over input channels
        self.sensitivity_in.copy_(torch.max(sens_in, sens_in_batch))
