# %%

import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn

# import torch.optim as optim
from torch.autograd import Variable
from torchvision.transforms import transforms


import time
import json
import argparse, os


import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import nn_ as nn_
from nn_ import log
from torch.autograd import Variable
import numpy as np


sum_from_one = nn_.sum_from_one


#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:58:12 2017

@author: CW
"""


import numpy as np

import torch
import torch.nn as nn
from torch.nn import Module


from functools import reduce

# aliasing
N_ = None


delta = 1e-6
softplus_ = nn.Softplus()
softplus = lambda x: softplus_(x) + delta


tile = lambda x, r: np.tile(x, r).reshape(x.shape[0], x.shape[1] * r)


# %------------ MADE ------------%


def get_rank(max_rank, num_out):
    rank_out = np.array([])
    while len(rank_out) < num_out:
        rank_out = np.concatenate([rank_out, np.arange(max_rank)])
    excess = len(rank_out) - num_out
    remove_ind = np.random.choice(max_rank, excess, False)
    rank_out = np.delete(rank_out, remove_ind)
    np.random.shuffle(rank_out)
    return rank_out.astype("float32")


def get_mask_from_ranks(r1, r2):
    return (r2[:, None] >= r1[None, :]).astype("float32")


def get_masks_all(ds, fixed_order=False, derank=1):
    # ds: list of dimensions dx, d1, d2, ... dh, dx,
    #                       (2 in/output + h hidden layers)
    # derank only used for self connection, dim > 1
    dx = ds[0]
    ms = list()
    rx = get_rank(dx, dx)
    if fixed_order:
        rx = np.sort(rx)
    r1 = rx
    if dx != 1:
        for d in ds[1:-1]:
            r2 = get_rank(dx - derank, d)
            ms.append(get_mask_from_ranks(r1, r2))
            r1 = r2
        r2 = rx - derank
        ms.append(get_mask_from_ranks(r1, r2))
    else:
        ms = [
            np.zeros([ds[i + 1], ds[i]]).astype("float32")
            for i in range(len(ds) - 1)
        ]
    if derank == 1:
        assert np.all(np.diag(reduce(np.dot, ms[::-1])) == 0), "wrong masks"

    return ms, rx


def get_masks(dim, dh, num_layers, num_outlayers, fixed_order=False, derank=1):
    ms, rx = get_masks_all(
        [
            dim,
        ]
        + [dh for i in range(num_layers - 1)]
        + [
            dim,
        ],
        fixed_order,
        derank,
    )
    ml = ms[-1]
    ml_ = (
        (
            ml.transpose(1, 0)[:, :, None]
            * (
                [
                    np.cast["float32"](1),
                ]
                * num_outlayers
            )
        )
        .reshape(dh, dim * num_outlayers)
        .transpose(1, 0)
    )
    ms[-1] = ml_
    return ms, rx


class MADE(Module):
    def __init__(
        self,
        dim,
        hid_dim,
        num_layers,
        num_outlayers=1,
        activation=nn.ELU(),
        fixed_order=False,
        derank=1,
    ):
        super(MADE, self).__init__()

        oper = nn_.WNlinear

        self.dim = dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.num_outlayers = num_outlayers
        self.activation = activation

        ms, rx = get_masks(
            dim, hid_dim, num_layers, num_outlayers, fixed_order, derank
        )
        ms = [m for m in map(torch.from_numpy, ms)]
        self.rx = rx

        sequels = list()
        for i in range(num_layers - 1):
            if i == 0:
                sequels.append(oper(dim, hid_dim, True, ms[i], False))
                sequels.append(activation)
            else:
                sequels.append(oper(hid_dim, hid_dim, True, ms[i], False))
                sequels.append(activation)

        self.input_to_hidden = nn.Sequential(*sequels)
        self.hidden_to_output = oper(
            hid_dim, dim * num_outlayers, True, ms[-1]
        )

    def forward(self, input):
        hid = self.input_to_hidden(input)
        return self.hidden_to_output(hid).view(
            -1, self.dim, self.num_outlayers
        )

    def randomize(self):
        ms, rx = get_masks(
            self.dim, self.hid_dim, self.num_layers, self.num_outlayers
        )
        for i in range(self.num_layers - 1):
            mask = torch.from_numpy(ms[i])
            if self.input_to_hidden[i * 2].mask.is_cuda:
                mask = mask.cuda()
            self.input_to_hidden[i * 2].mask.data.zero_().add_(mask)
        self.rx = rx


class cMADE(Module):
    def __init__(
        self,
        dim,
        hid_dim,
        context_dim,
        num_layers,
        num_outlayers=1,
        activation=nn.ELU(),
        fixed_order=False,
        derank=1,
    ):
        super(cMADE, self).__init__()

        oper = nn_.CWNlinear

        self.dim = dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.context_dim = context_dim
        self.num_outlayers = num_outlayers
        self.activation = nn_.Lambda(lambda x: (activation(x[0]), x[1]))

        ms, rx = get_masks(
            dim, hid_dim, num_layers, num_outlayers, fixed_order, derank
        )
        ms = [m for m in map(torch.from_numpy, ms)]
        self.rx = rx

        sequels = list()
        for i in range(num_layers - 1):
            if i == 0:
                sequels.append(oper(dim, hid_dim, context_dim, ms[i], False))
                sequels.append(self.activation)
            else:
                sequels.append(
                    oper(hid_dim, hid_dim, context_dim, ms[i], False)
                )
                sequels.append(self.activation)

        self.input_to_hidden = nn.Sequential(*sequels)
        self.hidden_to_output = oper(
            hid_dim, dim * num_outlayers, context_dim, ms[-1]
        )

    def forward(self, inputs):
        input, context = inputs
        hid, _ = self.input_to_hidden((input, context))
        out, _ = self.hidden_to_output((hid, context))
        return out.view(-1, self.dim, self.num_outlayers), context

    def randomize(self):
        ms, rx = get_masks(
            self.dim, self.hid_dim, self.num_layers, self.num_outlayers
        )
        for i in range(self.num_layers - 1):
            mask = torch.from_numpy(ms[i])
            if self.input_to_hidden[i * 2].mask.is_cuda:
                mask = mask.cuda()
            self.input_to_hidden[i * 2].mask.zero_().add_(mask)
        self.rx = rx


class BaseFlow(Module):
    def sample(self, n=1, context=None, **kwargs):
        dim = self.dim
        if isinstance(self.dim, int):
            dim = [
                dim,
            ]

        spl = Variable(torch.FloatTensor(n, *dim).normal_())
        lgd = Variable(torch.from_numpy(np.zeros(n).astype("float32")))
        if context is None:
            context = Variable(
                torch.from_numpy(
                    np.ones((n, self.context_dim)).astype("float32")
                )
            )

        if hasattr(self, "gpu"):
            if self.gpu:
                spl = spl.cuda()
                lgd = lgd.cuda()
                context = context.gpu()

        return self.forward((spl, lgd, context))

    def cuda(self):
        self.gpu = True
        return super(BaseFlow, self).cuda()


class LinearFlow(BaseFlow):
    def __init__(
        self, dim, context_dim, oper=nn_.ResLinear, realify=nn_.softplus
    ):
        super(LinearFlow, self).__init__()
        self.realify = realify

        self.dim = dim
        self.context_dim = context_dim

        if type(dim) is int:
            dim_ = dim
        else:
            dim_ = np.prod(dim)

        self.mean = oper(context_dim, dim_)
        self.lstd = oper(context_dim, dim_)

        self.reset_parameters()

    def reset_parameters(self):
        if isinstance(self.mean, nn_.ResLinear):
            self.mean.dot_01.scale.data.uniform_(-0.001, 0.001)
            self.mean.dot_h1.scale.data.uniform_(-0.001, 0.001)
            self.mean.dot_01.bias.data.uniform_(-0.001, 0.001)
            self.mean.dot_h1.bias.data.uniform_(-0.001, 0.001)
            self.lstd.dot_01.scale.data.uniform_(-0.001, 0.001)
            self.lstd.dot_h1.scale.data.uniform_(-0.001, 0.001)
            if self.realify == nn_.softplus:
                inv = np.log(np.exp(1 - nn_.delta) - 1) * 0.5
                self.lstd.dot_01.bias.data.uniform_(inv - 0.001, inv + 0.001)
                self.lstd.dot_h1.bias.data.uniform_(inv - 0.001, inv + 0.001)
            else:
                self.lstd.dot_01.bias.data.uniform_(-0.001, 0.001)
                self.lstd.dot_h1.bias.data.uniform_(-0.001, 0.001)
        elif isinstance(self.mean, nn.Linear):
            self.mean.weight.data.uniform_(-0.001, 0.001)
            self.mean.bias.data.uniform_(-0.001, 0.001)
            self.lstd.weight.data.uniform_(-0.001, 0.001)
            if self.realify == nn_.softplus:
                inv = np.log(np.exp(1 - nn_.delta) - 1) * 0.5
                self.lstd.bias.data.uniform_(inv - 0.001, inv + 0.001)
            else:
                self.lstd.bias.data.uniform_(-0.001, 0.001)

    def forward(self, inputs):
        x, logdet, context = inputs
        mean = self.mean(context)
        lstd = self.lstd(context)
        std = self.realify(lstd)

        if type(self.dim) is int:
            x_ = mean + std * x
        else:
            size = x.size()
            x_ = mean.view(size) + std.view(size) * x
        logdet_ = sum_from_one(torch.log(std)) + logdet
        return x_, logdet_, context


class BlockAffineFlow(Module):
    # NICE, volume preserving
    # x2' = x2 + nonLinfunc(x1)

    def __init__(self, dim1, dim2, context_dim, hid_dim, activation=nn.ELU()):
        super(BlockAffineFlow, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.actv = activation

        self.hid = nn_.WNBilinear(dim1, context_dim, hid_dim)
        self.shift = nn_.WNBilinear(hid_dim, context_dim, dim2)

    def forward(self, inputs):
        x, logdet, context = inputs
        x1, x2 = x

        hid = self.actv(self.hid(x1, context))
        shift = self.shift(hid, context)

        x2_ = x2 + shift

        return (x1, x2_), 0, context


class IAF(BaseFlow):
    def __init__(
        self,
        dim,
        hid_dim,
        context_dim,
        num_layers,
        activation=nn.ELU(),
        realify=nn_.sigmoid,
        fixed_order=False,
    ):
        super(IAF, self).__init__()
        self.realify = realify

        self.dim = dim
        self.context_dim = context_dim

        if type(dim) is int:
            self.mdl = cMADE(
                dim,
                hid_dim,
                context_dim,
                num_layers,
                2,
                activation,
                fixed_order,
            )
            self.reset_parameters()

    def reset_parameters(self):
        self.mdl.hidden_to_output.cscale.weight.data.uniform_(-0.001, 0.001)
        self.mdl.hidden_to_output.cscale.bias.data.uniform_(0.0, 0.0)
        self.mdl.hidden_to_output.cbias.weight.data.uniform_(-0.001, 0.001)
        self.mdl.hidden_to_output.cbias.bias.data.uniform_(0.0, 0.0)
        if self.realify == nn_.softplus:
            inv = np.log(np.exp(1 - nn_.delta) - 1)
            self.mdl.hidden_to_output.cbias.bias.data[1::2].uniform_(inv, inv)
        elif self.realify == nn_.sigmoid:
            self.mdl.hidden_to_output.cbias.bias.data[1::2].uniform_(2.0, 2.0)

    def forward(self, inputs):
        x, logdet, context = inputs
        out, _ = self.mdl((x, context))
        if isinstance(self.mdl, cMADE):
            mean = out[:, :, 0]
            lstd = out[:, :, 1]

        std = self.realify(lstd)

        if self.realify == nn_.softplus:
            x_ = mean + std * x
        elif self.realify == nn_.sigmoid:
            x_ = (-std + 1.0) * mean + std * x
        elif self.realify == nn_.sigmoid2:
            x_ = (-std + 2.0) * mean + std * x
        logdet_ = sum_from_one(torch.log(std)) + logdet
        return x_, logdet_, context


class IAF_VP(BaseFlow):
    def __init__(
        self,
        dim,
        hid_dim,
        context_dim,
        num_layers,
        activation=nn.ELU(),
        fixed_order=True,
    ):
        super(IAF_VP, self).__init__()

        self.dim = dim
        self.context_dim = context_dim

        if type(dim) is int:
            self.mdl = cMADE(
                dim,
                hid_dim,
                context_dim,
                num_layers,
                1,
                activation,
                fixed_order,
            )
            self.reset_parameters()

    def reset_parameters(self):
        self.mdl.hidden_to_output.cscale.weight.data.uniform_(-0.001, 0.001)
        self.mdl.hidden_to_output.cscale.bias.data.uniform_(0.0, 0.0)
        self.mdl.hidden_to_output.cbias.weight.data.uniform_(-0.001, 0.001)
        self.mdl.hidden_to_output.cbias.bias.data.uniform_(0.0, 0.0)

    def forward(self, inputs):
        x, logdet, context = inputs
        out, _ = self.mdl((x, context))
        mean = out[:, :, 0]
        x_ = mean + x
        return x_, logdet, context


class IAF_DSF(BaseFlow):

    mollify = 0.0

    def __init__(
        self,
        dim,
        hid_dim,
        context_dim,
        num_layers,
        activation=nn.ELU(),
        fixed_order=False,
        num_ds_dim=4,
        num_ds_layers=1,
        num_ds_multiplier=3,
    ):
        super(IAF_DSF, self).__init__()

        self.dim = dim
        self.context_dim = context_dim
        self.num_ds_dim = num_ds_dim
        self.num_ds_layers = num_ds_layers

        if type(dim) is int:
            self.mdl = cMADE(
                dim,
                hid_dim,
                context_dim,
                num_layers,
                num_ds_multiplier * (hid_dim // dim) * num_ds_layers,
                activation,
                fixed_order,
            )
            self.out_to_dsparams = nn.Conv1d(
                num_ds_multiplier * (hid_dim // dim) * num_ds_layers,
                3 * num_ds_layers * num_ds_dim,
                1,
            )
            self.reset_parameters()

        self.sf = SigmoidFlow(num_ds_dim)

    def reset_parameters(self):
        self.out_to_dsparams.weight.data.uniform_(-0.001, 0.001)
        self.out_to_dsparams.bias.data.uniform_(0.0, 0.0)

        inv = np.log(np.exp(1 - nn_.delta) - 1)
        for l in range(self.num_ds_layers):
            nc = self.num_ds_dim
            nparams = nc * 3
            s = l * nparams
            self.out_to_dsparams.bias.data[s : s + nc].uniform_(inv, inv)

    def forward(self, inputs):
        x, logdet, context = inputs
        out, _ = self.mdl((x, context))
        if isinstance(self.mdl, cMADE):
            out = out.permute(0, 2, 1)
            dsparams = self.out_to_dsparams(out).permute(0, 2, 1)
            nparams = self.num_ds_dim * 3

        mollify = self.mollify
        h = x.view(x.size(0), -1)
        for i in range(self.num_ds_layers):
            params = dsparams[:, :, i * nparams : (i + 1) * nparams]
            h, logdet = self.sf(h, logdet, params, mollify)

        return h, logdet, context


class SigmoidFlow(BaseFlow):
    def __init__(self, num_ds_dim=4):
        super(SigmoidFlow, self).__init__()
        self.num_ds_dim = num_ds_dim

        self.act_a = lambda x: nn_.softplus(x)
        self.act_b = lambda x: x
        self.act_w = lambda x: nn_.softmax(x, dim=2)

    def forward(self, x, logdet, dsparams, mollify=0.0, delta=nn_.delta):

        ndim = self.num_ds_dim
        a_ = self.act_a(dsparams[:, :, 0 * ndim : 1 * ndim])
        b_ = self.act_b(dsparams[:, :, 1 * ndim : 2 * ndim])
        w = self.act_w(dsparams[:, :, 2 * ndim : 3 * ndim])

        a = a_ * (1 - mollify) + 1.0 * mollify
        b = b_ * (1 - mollify) + 0.0 * mollify

        pre_sigm = a * x[:, :, None] + b
        sigm = torch.sigmoid(pre_sigm)
        x_pre = torch.sum(w * sigm, dim=2)
        x_pre_clipped = x_pre * (1 - delta) + delta * 0.5
        x_ = log(x_pre_clipped) - log(1 - x_pre_clipped)
        xnew = x_

        logj = (
            F.log_softmax(dsparams[:, :, 2 * ndim : 3 * ndim], dim=2)
            + nn_.logsigmoid(pre_sigm)
            + nn_.logsigmoid(-pre_sigm)
            + log(a)
        )

        logj = torch.exp(logj, 2).sum(2)
        logdet_ = (
            logj
            + np.log(1 - delta)
            - (log(x_pre_clipped) + log(-x_pre_clipped + 1))
        )
        logdet = logdet_.sum(1) + logdet

        return xnew, logdet


class IAF_DDSF(BaseFlow):
    def __init__(
        self,
        dim,
        hid_dim,
        context_dim,
        num_layers,
        activation=nn.ELU(),
        fixed_order=False,
        num_ds_dim=4,
        num_ds_layers=1,
        num_ds_multiplier=3,
    ):
        super(IAF_DDSF, self).__init__()

        self.dim = dim
        self.context_dim = context_dim
        self.num_ds_dim = num_ds_dim
        self.num_ds_layers = num_ds_layers

        if type(dim) is int:
            self.mdl = cMADE(
                dim,
                hid_dim,
                context_dim,
                num_layers,
                int(num_ds_multiplier * (hid_dim / dim) * num_ds_layers),
                activation,
                fixed_order,
            )

        num_dsparams = 0
        for i in range(num_ds_layers):
            if i == 0:
                in_dim = 1
            else:
                in_dim = num_ds_dim
            if i == num_ds_layers - 1:
                out_dim = 1
            else:
                out_dim = num_ds_dim

            u_dim = in_dim
            w_dim = num_ds_dim
            a_dim = b_dim = num_ds_dim
            num_dsparams += u_dim + w_dim + a_dim + b_dim

            self.add_module(
                "sf{}".format(i), DenseSigmoidFlow(in_dim, num_ds_dim, out_dim)
            )
        if type(dim) is int:
            self.out_to_dsparams = nn.Conv1d(
                int(num_ds_multiplier * (hid_dim / dim) * num_ds_layers),
                int(num_dsparams),
                1,
            )
        else:
            self.out_to_dsparams = nn.Conv1d(
                num_ds_multiplier * (hid_dim / dim[0]) * num_ds_layers,
                num_dsparams,
                1,
            )

        self.reset_parameters()

    def reset_parameters(self):
        self.out_to_dsparams.weight.data.uniform_(-0.001, 0.001)
        self.out_to_dsparams.bias.data.uniform_(0.0, 0.0)

    def forward(self, inputs):
        x, logdet, context = inputs
        out, _ = self.mdl((x, context))
        out = out.permute(0, 2, 1)
        dsparams = self.out_to_dsparams(out).permute(0, 2, 1)

        start = 0

        h = x.view(x.size(0), -1)[:, :, None]
        n = x.size(0)
        dim = self.dim if type(self.dim) is int else self.dim[0]
        lgd = Variable(
            torch.from_numpy(np.zeros((n, dim, 1, 1)).astype("float32"))
        )
        if self.out_to_dsparams.weight.is_cuda:
            lgd = lgd.cuda()
        for i in range(self.num_ds_layers):
            if i == 0:
                in_dim = 1
            else:
                in_dim = self.num_ds_dim
            if i == self.num_ds_layers - 1:
                out_dim = 1
            else:
                out_dim = self.num_ds_dim

            u_dim = in_dim
            w_dim = self.num_ds_dim
            a_dim = b_dim = self.num_ds_dim
            end = start + u_dim + w_dim + a_dim + b_dim

            params = dsparams[:, :, start:end]
            h, lgd = getattr(self, "sf{}".format(i))(h, lgd, params)
            start = end

        assert out_dim == 1, "last dsf out dim should be 1"
        return h[:, :, 0], lgd[:, :, 0, 0].sum(1) + logdet, context


class DenseSigmoidFlow(BaseFlow):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(DenseSigmoidFlow, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.act_a = lambda x: nn_.softplus(x)
        self.act_b = lambda x: x
        self.act_w = lambda x: nn_.softmax(x, dim=3)
        self.act_u = lambda x: nn_.softmax(x, dim=3)

        self.u_ = Parameter(torch.Tensor(hidden_dim, in_dim))
        self.w_ = Parameter(torch.Tensor(out_dim, hidden_dim))

        self.reset_parameters()

    def reset_parameters(self):
        self.u_.data.uniform_(-0.001, 0.001)
        self.w_.data.uniform_(-0.001, 0.001)

    def forward(self, x, logdet, dsparams):
        inv = np.log(np.exp(1 - nn_.delta) - 1)
        ndim = self.hidden_dim
        pre_u = (
            self.u_[None, None, :, :]
            + dsparams[:, :, -self.in_dim :][:, :, None, :]
        )
        pre_w = (
            self.w_[None, None, :, :]
            + dsparams[:, :, 2 * ndim : 3 * ndim][:, :, None, :]
        )
        a = self.act_a(dsparams[:, :, 0 * ndim : 1 * ndim] + inv)
        b = self.act_b(dsparams[:, :, 1 * ndim : 2 * ndim])
        w = self.act_w(pre_w)
        u = self.act_u(pre_u)

        pre_sigm = torch.sum(u * a[:, :, :, None] * x[:, :, None, :], 3) + b
        sigm = torch.sigmoid(pre_sigm)
        x_pre = torch.sum(w * sigm[:, :, None, :], dim=3)
        x_pre_clipped = x_pre * (1 - nn_.delta) + nn_.delta * 0.5
        x_ = log(x_pre_clipped) - log(1 - x_pre_clipped)
        xnew = x_

        logj = (
            F.log_softmax(pre_w, dim=3)
            + nn_.logsigmoid(pre_sigm[:, :, None, :])
            + nn_.logsigmoid(-pre_sigm[:, :, None, :])
            + log(a[:, :, None, :])
        )
        # n, d, d2, dh

        logj = (
            logj[:, :, :, :, None]
            + F.log_softmax(pre_u, dim=3)[:, :, None, :, :]
        )
        # n, d, d2, dh, d1

        logj = torch.exp(logj, 3).sum(3)
        # n, d, d2, d1

        logdet_ = (
            logj
            + np.log(1 - nn_.delta)
            - (log(x_pre_clipped) + log(-x_pre_clipped + 1))[:, :, :, None]
        )

        logdet = torch.exp(
            logdet_[:, :, :, :, None] + logdet[:, :, None, :, :], 3
        ).sum(3)
        # n, d, d2, d1, d0 -> n, d, d2, d0

        return xnew, logdet


class FlipFlow(BaseFlow):
    def __init__(self, dim):
        self.dim = dim
        super(FlipFlow, self).__init__()

    def forward(self, inputs):
        input, logdet, context = inputs

        dim = self.dim
        index = Variable(
            getattr(
                torch.arange(input.size(dim) - 1, -1, -1),
                ("cpu", "cuda")[input.is_cuda],
            )().long()
        )

        output = torch.index_select(input, dim, index)

        return output, logdet, context


class Sigmoid(BaseFlow):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, inputs):
        if len(inputs) == 2:
            input, logdet = inputs
        elif len(inputs) == 3:
            input, logdet, context = inputs
        else:
            raise (Exception("inputs length not correct"))

        output = F.sigmoid(input)
        logdet += sum_from_one(-F.softplus(input) - F.softplus(-input))

        if len(inputs) == 2:
            return output, logdet
        elif len(inputs) == 3:
            return output, logdet, context
        else:
            raise (Exception("inputs length not correct"))


class Logit(BaseFlow):
    def __init__(self):
        super(Logit, self).__init__()

    def forward(self, inputs):
        if len(inputs) == 2:
            input, logdet = inputs
        elif len(inputs) == 3:
            input, logdet, context = inputs
        else:
            raise (Exception("inputs length not correct"))

        output = log(input) - log(1 - input)
        logdet -= sum_from_one(log(input) + log(-input + 1))

        if len(inputs) == 2:
            return output, logdet
        elif len(inputs) == 3:
            return output, logdet, context
        else:
            raise (Exception("inputs length not correct"))


class Shift(BaseFlow):
    def __init__(self, b):
        self.b = b
        super(Shift, self).__init__()

    def forward(self, inputs):
        if len(inputs) == 2:
            input, logdet = inputs
        elif len(inputs) == 3:
            input, logdet, context = inputs
        else:
            raise (Exception("inputs length not correct"))

        output = input + self.b

        if len(inputs) == 2:
            return output, logdet
        elif len(inputs) == 3:
            return output, logdet, context
        else:
            raise (Exception("inputs length not correct"))


class Scale(BaseFlow):
    def __init__(self, g):
        self.g = g
        super(Scale, self).__init__()

    def forward(self, inputs):
        if len(inputs) == 2:
            input, logdet = inputs
        elif len(inputs) == 3:
            input, logdet, context = inputs
        else:
            raise (Exception("inputs length not correct"))

        output = input * self.g
        logdet += np.log(np.abs(self.g)) * np.prod(input.size()[1:])

        if len(inputs) == 2:
            return output, logdet
        elif len(inputs) == 3:
            return output, logdet, context
        else:
            raise (Exception("inputs length not correct"))


class MAF(object):
    def __init__(self, args, p):

        self.args = args
        self.__dict__.update(args.__dict__)
        self.p = p

        dim = p
        dimc = 1
        dimh = args.dimh
        flowtype = args.flowtype
        num_flow_layers = args.num_flow_layers
        num_ds_dim = args.num_ds_dim
        num_ds_layers = args.num_ds_layers
        fixed_order = args.fixed_order

        act = nn.ELU()
        if flowtype == "affine":
            flow = IAF
        elif flowtype == "dsf":
            flow = lambda **kwargs: IAF_DSF(
                num_ds_dim=num_ds_dim, num_ds_layers=num_ds_layers, **kwargs
            )
        elif flowtype == "ddsf":
            flow = lambda **kwargs: IAF_DDSF(
                num_ds_dim=num_ds_dim, num_ds_layers=num_ds_layers, **kwargs
            )

        sequels = [
            nn_.SequentialFlow(
                flow(
                    dim=dim,
                    hid_dim=dimh,
                    context_dim=dimc,
                    num_layers=args.num_hid_layers + 1,
                    activation=act,
                    fixed_order=fixed_order,
                ),
                FlipFlow(1),
            )
            for i in range(num_flow_layers)
        ] + [
            LinearFlow(dim, dimc),
        ]

        self.flow = nn.Sequential(*sequels)


# =============================================================================
# main
# =============================================================================


"""parsing and configuration"""


def parse_args():
    desc = "MAF"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument(
        "--dataset",
        type=str,
        default="miniboone",
        choices=["power", "gas", "hepmass", "miniboone", "bsds300"],
    )
    parser.add_argument(
        "--epoch", type=int, default=400, help="The number of epochs to run"
    )
    parser.add_argument(
        "--batch_size", type=int, default=100, help="The size of batch"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="models",
        help="Directory name to save the model",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="results",
        help="Directory name to save the generated images",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory name to save training logs",
    )
    parser.add_argument("--seed", type=int, default=1993, help="Random seed")
    parser.add_argument(
        "--fn", type=str, default="0", help="Filename of model to be loaded"
    )
    parser.add_argument(
        "--to_train", type=int, default=1, help="1 if to train 0 if not"
    )
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--clip", type=float, default=5.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--amsgrad", type=int, default=0)
    parser.add_argument("--polyak", type=float, default=0.0)
    parser.add_argument("--cuda", type=bool, default=False)

    parser.add_argument("--dimh", type=int, default=100)
    parser.add_argument("--flowtype", type=str, default="ddsf")
    parser.add_argument("--num_flow_layers", type=int, default=10)
    parser.add_argument("--num_hid_layers", type=int, default=1)
    parser.add_argument("--num_ds_dim", type=int, default=16)
    parser.add_argument("--num_ds_layers", type=int, default=1)
    parser.add_argument(
        "--fixed_order",
        type=bool,
        default=True,
        help="Fix the made ordering to be the given order",
    )

    return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
    # --save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # --result_dir
    if not os.path.exists(args.result_dir + "_" + args.dataset):
        os.makedirs(args.result_dir + "_" + args.dataset)

    # --result_dir
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print("number of epochs must be larger than or equal to one")

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print("batch size must be larger than or equal to one")

    return args


datasets = {
    "power": {"d": 6, "dimh": 100, "num_hid_layers": 2},
    "gas": {"d": 8, "dimh": 100, "num_hid_layers": 2},
    "hepmass": {"d": 21, "dimh": 512, "num_hid_layers": 2},
    "miniboone": {"d": 43, "dimh": 512, "num_hid_layers": 1},
    "bsds300": {"d": 63, "dimh": 1024, "num_hid_layers": 2},
}


def format_as_str(num):
    if num / 1e9 > 1:
        factor, suffix = 1e9, "B"
    elif num / 1e6 > 1:
        factor, suffix = 1e6, "M"
    elif num / 1e3 > 1:
        factor, suffix = 1e3, "K"
    else:
        factor, suffix = 1e0, ""

    num_factored = num / factor

    if num_factored / 1e2 > 1 or True:
        num_rounded = str(int(round(num_factored)))
    elif num_factored / 1e1 > 1:
        num_rounded = f"{num_factored:.1f}"
    else:
        num_rounded = f"{num_factored:.2f}"

    return f"{num_rounded}{suffix} % {num}"


def naf_size(d, **kwargs):
    from torchprune.util.net import NetHandle

    args = parse_args()
    for key, val in kwargs.items():
        setattr(args, key, val)
    model = MAF(args, d)
    model = NetHandle(model.flow)
    return model.size()


for dset, s_kwargs in datasets.items():
    print(f"{dset}: #params: {format_as_str(naf_size(**s_kwargs))}")
    print("\n")
