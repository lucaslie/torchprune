#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:58:12 2017

@author: CW
"""


import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from torch.autograd import Variable

# aliasing
N_ = None


delta = 1e-6
softplus_ = nn.Softplus()
softplus = lambda x: softplus_(x) + delta
sigmoid_ = nn.Sigmoid()
sigmoid = lambda x: sigmoid_(x) * (1 - delta) + 0.5 * delta
sigmoid2 = lambda x: sigmoid(x) * 2.0
logsigmoid = lambda x: -softplus(-x)
logit = lambda x: torch.log
log = lambda x: torch.log(x * 1e2) - np.log(1e2)
logit = lambda x: log(x) - log(1 - x)


def softmax(x, dim=-1):
    e_x = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
    out = e_x / e_x.sum(dim=dim, keepdim=True)
    return out


sum1 = lambda x: x.sum(1)
sum_from_one = (
    lambda x: sum_from_one(sum1(x)) if len(x.size()) > 2 else sum1(x)
)


class Sigmoid(Module):
    def forward(self, x):
        return sigmoid(x)


class WNlinear(Module):
    def __init__(
        self, in_features, out_features, bias=True, mask=N_, norm=True
    ):
        super(WNlinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("mask", mask)
        self.norm = norm
        self.direction = Parameter(torch.Tensor(out_features, in_features))
        self.scale = Parameter(torch.Tensor(out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", N_)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.direction.size(1))
        self.direction.data.uniform_(-stdv, stdv)
        self.scale.data.uniform_(1, 1)
        if self.bias is not N_:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if self.norm:
            dir_ = self.direction
            direction = dir_.div(dir_.pow(2).sum(1).sqrt()[:, N_])
            weight = self.scale[:, N_].mul(direction)
        else:
            weight = self.scale[:, N_].mul(self.direction)
        if self.mask is not N_:
            # weight = weight * getattr(self.mask,
            #                          ('cpu', 'cuda')[weight.is_cuda])()
            weight = weight * Variable(self.mask)
        return F.linear(input, weight, self.bias)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in_features="
            + str(self.in_features)
            + ", out_features="
            + str(self.out_features)
            + ")"
        )


class CWNlinear(Module):
    def __init__(
        self, in_features, out_features, context_features, mask=N_, norm=True
    ):
        super(CWNlinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.context_features = context_features
        self.register_buffer("mask", mask)
        self.norm = norm
        self.direction = Parameter(torch.Tensor(out_features, in_features))
        self.cscale = nn.Linear(context_features, out_features)
        self.cbias = nn.Linear(context_features, out_features)
        self.reset_parameters()
        self.cscale.weight.data.normal_(0, 0.001)
        self.cbias.weight.data.normal_(0, 0.001)

    def reset_parameters(self):
        self.direction.data.normal_(0, 0.001)

    def forward(self, inputs):
        input, context = inputs
        scale = self.cscale(context)
        bias = self.cbias(context)
        if self.norm:
            dir_ = self.direction
            direction = dir_.div(dir_.pow(2).sum(1).sqrt()[:, N_])
            weight = direction
        else:
            weight = self.direction
        if self.mask is not N_:
            # weight = weight * getattr(self.mask,
            #                          ('cpu', 'cuda')[weight.is_cuda])()
            weight = weight * Variable(self.mask)
        return scale * F.linear(input, weight, None) + bias, context

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in_features="
            + str(self.in_features)
            + ", out_features="
            + str(self.out_features)
            + ")"
        )


class WNBilinear(Module):
    def __init__(self, in1_features, in2_features, out_features, bias=True):
        super(WNBilinear, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.direction = Parameter(
            torch.Tensor(out_features, in1_features, in2_features)
        )
        self.scale = Parameter(torch.Tensor(out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", N_)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.direction.size(1))
        self.direction.data.uniform_(-stdv, stdv)
        self.scale.data.uniform_(1, 1)
        if self.bias is not N_:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input1, input2):
        dir_ = self.direction
        direction = dir_.div(dir_.pow(2).sum(1).sum(1).sqrt()[:, N_, N_])
        weight = self.scale[:, N_, N_].mul(direction)
        return F.bilinear(input1, input2, weight, self.bias)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "in1_features="
            + str(self.in1_features)
            + ", in2_features="
            + str(self.in2_features)
            + ", out_features="
            + str(self.out_features)
            + ")"
        )


class _WNconvNd(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        bias,
    ):
        super(_WNconvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups

        # weight – filters tensor (out_channels x in_channels/groups x kH x kW)
        if transposed:
            self.direction = Parameter(
                torch.Tensor(in_channels, out_channels // groups, *kernel_size)
            )
            self.scale = Parameter(torch.Tensor(in_channels))
        else:
            self.direction = Parameter(
                torch.Tensor(out_channels, in_channels // groups, *kernel_size)
            )
            self.scale = Parameter(torch.Tensor(out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", N_)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.direction.data.uniform_(-stdv, stdv)
        self.scale.data.uniform_(1, 1)
        if self.bias is not N_:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = (
            "{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is N_:
            s += ", bias=False"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)


class WNconv2d(_WNconvNd):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        mask=N_,
        norm=True,
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(WNconv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _pair(0),
            groups,
            bias,
        )

        self.register_buffer("mask", mask)
        self.norm = norm

    def forward(self, input):
        if self.norm:
            dir_ = self.direction
            direction = dir_.div(
                dir_.pow(2).sum(1).sum(1).sum(1).sqrt()[:, N_, N_, N_]
            )
            weight = self.scale[:, N_, N_, N_].mul(direction)
        else:
            weight = self.scale[:, N_, N_, N_].mul(self.direction)
        if self.mask is not None:
            weight = weight * Variable(self.mask)
        return F.conv2d(
            input,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class CWNconv2d(_WNconvNd):
    def __init__(
        self,
        context_features,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        mask=N_,
        norm=True,
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(CWNconv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _pair(0),
            groups,
            False,
        )

        self.register_buffer("mask", mask)
        self.norm = norm
        self.cscale = nn.Linear(context_features, out_channels)
        self.cbias = nn.Linear(context_features, out_channels)

    def forward(self, inputs):
        input, context = inputs
        scale = self.cscale(context)[:, :, N_, N_]
        bias = self.cbias(context)[:, :, N_, N_]
        if self.norm:
            dir_ = self.direction
            direction = dir_.div(
                dir_.pow(2).sum(1).sum(1).sum(1).sqrt()[:, N_, N_, N_]
            )
            weight = direction
        else:
            weight = self.direction
        if self.mask is not None:
            weight = weight * Variable(self.mask)
        pre = F.conv2d(
            input,
            weight,
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return pre * scale + bias, context


class ResConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        activation=nn.ReLU(),
        oper=WNconv2d,
    ):
        super(ResConv2d, self).__init__()

        self.conv_0h = oper(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.conv_h1 = oper(out_channels, out_channels, 3, 1, 1, 1, 1, True)
        self.conv_01 = oper(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

        self.activation = activation

    def forward(self, input):
        h = self.activation(self.conv_0h(input))
        out_nonlinear = self.conv_h1(h)
        out_skip = self.conv_01(input)
        return out_nonlinear + out_skip


class ResLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        same_dim=False,
        activation=nn.ReLU(),
        oper=WNlinear,
    ):
        super(ResLinear, self).__init__()

        self.same_dim = same_dim

        self.dot_0h = oper(in_features, out_features, bias)
        self.dot_h1 = oper(out_features, out_features, bias)
        if not same_dim:
            self.dot_01 = oper(in_features, out_features, bias)

        self.activation = activation

    def forward(self, input):
        h = self.activation(self.dot_0h(input))
        out_nonlinear = self.dot_h1(h)
        out_skip = input if self.same_dim else self.dot_01(input)
        return out_nonlinear + out_skip


class GatingLinear(nn.Module):
    def __init__(self, in_features, out_features, oper=WNlinear, **kwargs):
        super(GatingLinear, self).__init__()

        self.dot = oper(in_features, out_features, **kwargs)
        self.gate = oper(in_features, out_features, **kwargs)

    def forward(self, input):
        h = self.dot(input)
        s = sigmoid_(self.gate(input))
        return s * h


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)


class Slice(nn.Module):
    def __init__(self, slc):
        super(Slice, self).__init__()
        self.slc = slc

    def forward(self, input):
        return input.__getitem__(self.slc)


class SliceFactory(object):
    def __init__(self):
        pass

    def __getitem__(self, slc):
        return Slice(slc)


slicer = SliceFactory()


class Lambda(nn.Module):
    def __init__(self, function):
        super(Lambda, self).__init__()
        self.function = function

    def forward(self, input):
        return self.function(input)


class SequentialFlow(nn.Sequential):
    def sample(self, n=1, context=None, **kwargs):
        dim = self[0].dim
        if isinstance(dim, int):
            dim = [
                dim,
            ]

        spl = torch.autograd.Variable(torch.FloatTensor(n, *dim).normal_())
        lgd = torch.autograd.Variable(
            torch.from_numpy(np.random.rand(n).astype("float32"))
        )
        if context is None:
            context = torch.autograd.Variable(
                torch.from_numpy(
                    np.zeros((n, self[0].context_dim)).astype("float32")
                )
            )

        if hasattr(self, "gpu"):
            if self.gpu:
                spl = spl.cuda()
                lgd = lgd.cuda()
                context = context.cuda()

        return self.forward((spl, lgd, context))

    def cuda(self):
        self.gpu = True
        return super(SequentialFlow, self).cuda()


class ContextWrapper(nn.Module):
    def __init__(self, module):
        super(ContextWrapper, self).__init__()
        self.module = module

    def forward(self, inputs):
        input, context = inputs
        output = self.module.forward(input)
        return output, context


if __name__ == "__main__":

    mdl = CWNlinear(2, 5, 3)

    inp = torch.autograd.Variable(
        torch.from_numpy(np.random.rand(2, 2).astype("float32"))
    )
    con = torch.autograd.Variable(
        torch.from_numpy(np.random.rand(2, 3).astype("float32"))
    )

    print(mdl((inp, con))[0].size())
