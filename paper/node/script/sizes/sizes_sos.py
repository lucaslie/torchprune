# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from torchprune.util.net import NetHandle


def get_mask(in_features, out_features, in_flow_features, mask_type=None):
    """
    mask_type: input | None | output

    See Figure 1 for a better illustration:
    https://arxiv.org/pdf/1502.03509.pdf
    """
    if mask_type == "input":
        in_degrees = torch.arange(in_features) % in_flow_features
    else:
        in_degrees = torch.arange(in_features) % (in_flow_features - 1)

    if mask_type == "output":
        out_degrees = torch.arange(out_features) % in_flow_features - 1
    else:
        out_degrees = torch.arange(out_features) % (in_flow_features - 1)

    return (out_degrees.unsqueeze(-1) >= in_degrees.unsqueeze(0)).float()


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer("mask", mask)

    def forward(self, inputs):
        return F.linear(inputs, self.weight * self.mask, self.bias)


class ConditionerNet(nn.Module):
    def __init__(self, input_size, hidden_size, k, m, n_layers=1):
        super().__init__()
        self.k = k
        self.m = m
        self.input_size = input_size
        self.output_size = k * self.m * input_size + input_size
        self.network = self._make_net(
            input_size, hidden_size, self.output_size, n_layers
        )

    def _make_net(self, input_size, hidden_size, output_size, n_layers):
        if self.input_size > 1:
            input_mask = get_mask(
                input_size, hidden_size, input_size, mask_type="input"
            )
            hidden_mask = get_mask(hidden_size, hidden_size, input_size)
            output_mask = get_mask(
                hidden_size, output_size, input_size, mask_type="output"
            )

            network = nn.Sequential(
                MaskedLinear(input_size, hidden_size, input_mask),
                nn.ReLU(),
                MaskedLinear(hidden_size, hidden_size, hidden_mask),
                nn.ReLU(),
                MaskedLinear(hidden_size, output_size, output_mask),
            )
        else:
            network = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size),
            )

        """
        for module in network.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                module.bias.data.fill_(0)
        """

        return network

    def forward(self, inputs):
        batch_size = inputs.size(0)
        params = self.network(inputs)
        i = self.k * self.m * self.input_size
        c = (
            params[:, :i]
            .view(batch_size, -1, self.input_size)
            .transpose(1, 2)
            .view(batch_size, self.input_size, self.k, self.m, 1)
        )
        const = params[:, i:].view(batch_size, self.input_size)
        C = torch.matmul(c, c.transpose(3, 4))
        return C, const


#
#   SOS Block:
#


class SOSFlow(nn.Module):
    @staticmethod
    def power(z, k):
        return z ** (torch.arange(k).float().to(z.device))

    def __init__(self, input_size, hidden_size, k, r, n_layers=1):
        super().__init__()
        self.k = k
        self.m = r + 1

        self.conditioner = ConditionerNet(
            input_size, hidden_size, k, self.m, n_layers
        )
        self.register_buffer("filter", self._make_filter())

    def _make_filter(self):
        n = torch.arange(self.m).unsqueeze(1)
        e = torch.ones(self.m).unsqueeze(1).long()
        filter = (n.mm(e.transpose(0, 1))) + (e.mm(n.transpose(0, 1))) + 1
        return filter.float()

    def forward(self, inputs, mode="direct"):
        batch_size, input_size = inputs.size(0), inputs.size(1)
        C, const = self.conditioner(inputs)
        X = SOSFlow.power(inputs.unsqueeze(-1), self.m).view(
            batch_size, input_size, 1, self.m, 1
        )  # bs x d x 1 x m x 1
        Z = self._transform(X, C / self.filter) * inputs + const
        logdet = torch.log(torch.abs(self._transform(X, C))).sum(
            dim=1, keepdim=True
        )
        return Z, logdet

    def _transform(self, X, C):
        CX = torch.matmul(C, X)  # bs x d x k x m x 1
        XCX = torch.matmul(X.transpose(3, 4), CX)  # bs x d x k x 1 x 1
        summed = XCX.squeeze(-1).squeeze(-1).sum(-1)  # bs x d
        return summed

    def _jacob(self, inputs, mode="direct"):
        X = inputs.clone()
        X.requires_grad_()
        X.retain_grad()
        d = X.size(0)

        X_in = X.unsqueeze(0)
        C, const = self.conditioner(X_in)
        Xpow = SOSFlow.power(X_in.unsqueeze(-1), self.m).view(
            1, d, 1, self.m, 1
        )  # bs x d x 1 x m x 1
        Z = (self._transform(Xpow, C / self.filter) * X_in + const).view(-1)

        J = torch.zeros(d, d)
        for i in range(d):
            self.zero_grad()
            Z[i].backward(retain_graph=True)
            J[i, :] = X.grad

        del X, X_in, C, const, Xpow, Z
        return J


class MADE(nn.Module):
    """An implementation of MADE
    (https://arxiv.org/abs/1502.03509s).
    """

    def __init__(self, num_inputs, num_hidden, act="relu", pre_exp_tanh=False):
        super(MADE, self).__init__()

        activations = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
        act_func = activations[act]

        input_mask = get_mask(
            num_inputs, num_hidden, num_inputs, mask_type="input"
        )
        hidden_mask = get_mask(num_hidden, num_hidden, num_inputs)
        output_mask = get_mask(
            num_hidden, num_inputs * 2, num_inputs, mask_type="output"
        )

        self.joiner = MaskedLinear(num_inputs, num_hidden, input_mask)

        self.trunk = nn.Sequential(
            act_func(),
            MaskedLinear(num_hidden, num_hidden, hidden_mask),
            act_func(),
            MaskedLinear(num_hidden, num_inputs * 2, output_mask),
        )

    def forward(self, inputs, mode="direct"):
        if mode == "direct":
            h = self.joiner(inputs)
            m, a = self.trunk(h).chunk(2, 1)
            u = (inputs - m) * torch.exp(-a)
            return u, -a.sum(-1, keepdim=True)

        else:
            x = torch.zeros_like(inputs)
            for i_col in range(inputs.shape[1]):
                h = self.joiner(x)
                m, a = self.trunk(h).chunk(2, 1)
                x[:, i_col] = (
                    inputs[:, i_col] * torch.exp(a[:, i_col]) + m[:, i_col]
                )
            return x, -a.sum(-1, keepdim=True)

    def _jacob(self, inputs):
        X = inputs.clone()
        X.requires_grad_()
        X.retain_grad()
        d = X.size(0)

        X_in = X.unsqueeze(0)

        h = self.joiner(X_in)
        m, a = self.trunk(h).chunk(2, 1)
        u = ((X_in - m) * torch.exp(-a)).view(-1)

        J = torch.zeros(d, d)
        for i in range(d):
            self.zero_grad()
            u[i].backward(retain_graph=True)
            J[i, :] = X.grad

        del X, X_in, h, m, a, u
        return J


class BatchNormFlow(nn.Module):
    """An implementation of a batch normalization layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super(BatchNormFlow, self).__init__()

        self.log_gamma = nn.Parameter(torch.zeros(num_inputs))
        self.beta = nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer("running_mean", torch.zeros(num_inputs))
        self.register_buffer("running_var", torch.ones(num_inputs))

    def forward(self, inputs, mode="direct"):
        if mode == "direct":
            if True:  # self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (inputs - self.batch_mean).pow(2).mean(
                    0
                ) + self.eps

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)

                self.running_mean.add_(
                    self.batch_mean.data * (1 - self.momentum)
                )
                self.running_var.add_(
                    self.batch_var.data * (1 - self.momentum)
                )

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, (self.log_gamma - 0.5 * torch.log(var)).sum(
                -1, keepdim=True
            )
        else:
            if True:  # self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * var.sqrt() + mean

            return y, (-self.log_gamma + 0.5 * torch.log(var)).sum(
                -1, keepdim=True
            )

    def _jacob(self, X):
        return None


class Reverse(nn.Module):
    """An implementation of a reversing layer from
    Density estimation using Real NVP
    (https://arxiv.org/abs/1605.08803).
    """

    def __init__(self, num_inputs):
        super(Reverse, self).__init__()
        self.perm = np.array(np.arange(0, num_inputs)[::-1])
        self.inv_perm = np.argsort(self.perm)

    def forward(self, inputs, mode="direct"):
        if mode == "direct":
            return inputs[:, self.perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device
            )
        else:
            return inputs[:, self.inv_perm], torch.zeros(
                inputs.size(0), 1, device=inputs.device
            )

    def _jacob(self, X):
        return None


class FlowSequential(nn.Sequential):
    """A sequential container for flows.
    In addition to a forward pass it implements a backward pass and
    computes log jacobians.
    """

    def forward(self, inputs, mode="direct", logdets=None):
        """Performs a forward or backward pass for flow modules.
        Args:
            inputs: a tuple of inputs and logdets
            mode: to run direct computation or inverse
        """
        self.num_inputs = inputs.size(-1)

        if logdets is None:
            logdets = torch.zeros(inputs.size(0), 1, device=inputs.device)

        assert mode in ["direct", "inverse"]
        if mode == "direct":
            for module in self._modules.values():
                inputs, logdet = module(inputs, mode)
                logdets += logdet
        else:
            for module in reversed(self._modules.values()):
                inputs, logdet = module(inputs, mode)
                logdets += logdet

        return inputs, logdets

    def evaluate(self, inputs):
        N = len(self._modules)
        outputs = torch.zeros(
            N + 1, inputs.size(0), inputs.size(1), device=inputs.device
        )
        outputs[0, :, :] = inputs
        logdets = torch.zeros(N, inputs.size(0), 1, device=inputs.device)
        for i in range(N):
            outputs[i + 1, :, :], logdets[i, :, :] = self._modules[str(i)](
                outputs[i, :, :], mode="direct"
            )
        return outputs, logdets

    def log_probs(self, inputs):
        u, log_jacob = self(inputs)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True
        )
        return (log_probs + log_jacob).sum(-1, keepdim=True)

    def sample(self, num_samples=None, noise=None, cond_inputs=None):
        if noise is None:
            noise = torch.Tensor(num_samples, self.num_inputs).normal_()
        device = next(self.parameters()).device
        noise = noise.to(device)
        if cond_inputs is not None:
            cond_inputs = cond_inputs.to(device)
        samples = self.forward(noise, cond_inputs, mode="inverse")[0]
        return samples

    def jacobians(self, X):
        assert len(X.size()) == 1
        N = len(self._modules)
        num_inputs = X.size(-1)
        jacobians = torch.zeros(N, num_inputs, num_inputs)

        n_jacob = 0
        for i in range(N):
            J_i = self._modules[str(i)]._jacob(X)
            if J_i is not None:
                jacobians[n_jacob, :, :] = J_i
                n_jacob += 1
            del J_i

        return jacobians[:n_jacob, :, :]


def build_model(input_size, hidden_size, k, r, n_blocks):
    modules = []
    for i in range(n_blocks):
        modules += [
            SOSFlow(input_size, hidden_size, k, r),
            BatchNormFlow(input_size),
            Reverse(input_size),
        ]
    model = FlowSequential(*modules)

    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)

    return model


datasets = {
    "power": {"hidden_size": 100, "k": 5, "r": 4, "d": 6, "n_blocks": 8},
    "gas": {"hidden_size": 100, "k": 5, "r": 4, "d": 8, "n_blocks": 8},
    "hepmass": {"hidden_size": 512, "k": 5, "r": 4, "d": 21, "n_blocks": 8},
    "miniboone": {"hidden_size": 512, "k": 5, "r": 4, "d": 43, "n_blocks": 8},
    "bsds300": {"hidden_size": 512, "k": 5, "r": 4, "d": 63, "n_blocks": 8},
    "mnist": {"hidden_size": 100, "k": 5, "r": 4, "d": 784, "n_blocks": 8},
    "cifar": {"hidden_size": 100, "k": 5, "r": 4, "d": 3072, "n_blocks": 8},
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


def sos_size(d, hidden_size, k, r, n_blocks):
    model = build_model(d, hidden_size, k, r, n_blocks)
    model = NetHandle(model)
    return model.size()


for dset, s_kwargs in datasets.items():
    print(f"{dset}: #params: {format_as_str(sos_size(**s_kwargs))}")
