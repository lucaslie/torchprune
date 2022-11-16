"""Module containing various FFjord NODE configurations with torchdyn lib."""

import torch
import torch.nn as nn
from torch.distributions.utils import lazy_property
from torchdyn.models import CNF, hutch_trace, NeuralODE
from torchdyn.nn import Augmenter


def distribution_module(cls):
    """Return a class representing a "modulified" torch distributions."""
    # get the actual distribution class from the mro()
    cls_distribution = []
    for parent in cls.mro()[1:]:
        if (
            issubclass(parent, torch.distributions.Distribution)
            and parent != torch.distributions.Distribution
        ):
            cls_distribution.append(parent)
    assert len(cls_distribution) == 1
    cls_distribution = cls_distribution[0]

    class DistributionModule:
        """A class prototype for modulifing a desired distribution."""

        def __init__(self, *args, **kwargs):
            """Init distribution and register plain tensors as buffers."""
            super().__init__(*args, **kwargs)

            # after initializing re-register distribution parameters as buffers
            k_tensors = []
            for k in self.__dict__:

                # check if it's a lazy_property
                # in this case we should simply move on instead of evaluating it.
                if self._is_lazy_property(k):
                    continue

                # check if it's a "plain" tensor
                if isinstance(getattr(self, k), torch.Tensor):
                    k_tensors.append(k)

            for k in k_tensors:
                # for a "plain "tensor we will now register it as buffer.
                val = getattr(self, k)
                delattr(self, k)
                self.register_buffer(k, val)

        def __getattribute__(self, name):
            """Return attribute with lazy_property special check."""
            if type(self)._is_lazy_property(name):
                # deleting the attribute from the instance will simply "reset"
                # the lazy_property
                delattr(self, name)
            return super().__getattribute__(name)

        @classmethod
        def _is_lazy_property(cls, name):
            return isinstance(getattr(cls, name, object), lazy_property)

    return type(
        cls.__name__, (DistributionModule, cls_distribution, nn.Module), {}
    )


@distribution_module
class MultiVariateNormalModule(torch.distributions.MultivariateNormal):
    """An "modulified" MultiVariateNormalDistribution."""


class NeuralODEClassic(NeuralODE):
    """A wrapper for NeuralODE for the interface we are used to."""

    @property
    def defunc(self):
        """Return an old-school defunc which is now a vector field."""
        return self.vf.vf

    def __init__(self, cnf, s_span, sensitivity, solver, atol, rtol):
        """Initialize the wrapper."""
        super().__init__(
            cnf,
            sensitivity=sensitivity,
            solver=solver,
            atol=atol,
            rtol=rtol,
            atol_adjoint=atol,
            rtol_adjoint=rtol,
        )

        # fake assign s_span as before
        self.register_buffer("s_span", s_span)

        # make sure classic "defunc" has "m" field
        self.vf.vf.m = self.vf.vf.vf

    def forward(self, x, s_span=None):
        """Forward in the classic style."""
        if s_span is None:
            s_span = self.s_span
        return super().forward(x, s_span)[1][1]

    def trajectory(self, x, s_span):
        """Compute trajectory in the classic style."""
        return super().trajectory(x, s_span)


class Ffjord(nn.Module):
    """Neural ODEs for CNFs via ffjord hutchuson trace estimator."""

    @property
    def trace_estimator(self):
        """Return the desired trace estimator."""
        return hutch_trace

    def __init__(
        self,
        num_in,
        num_layers,
        hidden_size,
        module_activate,
        s_span=torch.linspace(0, 1, 20),
        sensitivity="autograd",
        solver="rk4",
        atol=1e-5,
        rtol=1e-5,
    ):
        """Initialize ffjord with the desired parameterization."""
        super().__init__()
        if num_layers < 2:
            raise ValueError("Node must be initialized with min 2 layers.")
        if not issubclass(module_activate, nn.Module):
            raise ValueError("Please provide valid module as activation.")

        # build up layers
        layers = [nn.Linear(num_in, hidden_size), module_activate()]
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(module_activate())
        layers.append(nn.Linear(hidden_size, num_in))

        # wrap in sequential module
        self.f_forward = nn.Sequential(*layers)

        # get prior
        self.prior = MultiVariateNormalModule(
            torch.zeros(num_in), torch.eye(num_in), validate_args=False
        )

        # wrap in cnf
        cnf = CNF(
            self.f_forward,
            trace_estimator=self.trace_estimator,
            noise_dist=self.prior,
        )

        # wrap in neural ode
        nde = NeuralODEClassic(
            cnf,
            s_span=s_span,
            sensitivity=sensitivity,
            solver=solver,
            atol=atol,
            rtol=rtol,
        )

        # wrap in augmenter
        self.model = nn.Sequential(
            Augmenter(augment_idx=1, augment_dims=1), nde
        )

    def forward(self, x):
        """Forward by passing it on to NeuralODE.

        We wrap output into dictionary and also return prior so that downstream
        tasks (e.g. loss and metrics) have access to the prior.
        """
        self.model[1].nfe = 0
        return {"out": self.model(x), "prior": self.prior}


def ffjord_l4_h64_sigmoid(num_classes):
    """Return a ffjord with 4 layers, 64 neurons, and sigmoid."""
    return Ffjord(
        num_in=num_classes,
        num_layers=4,
        hidden_size=64,
        module_activate=nn.Sigmoid,
    )


def ffjord_l4_h64_softplus(num_classes):
    """Return a ffjord with 4 layers, 64 neurons, and softplus."""
    return Ffjord(
        num_in=num_classes,
        num_layers=4,
        hidden_size=64,
        module_activate=nn.Softplus,
    )


def ffjord_l4_h64_tanh(num_classes):
    """Return a ffjord with 4 layers, 64 neurons, and tanh."""
    return Ffjord(
        num_in=num_classes,
        num_layers=4,
        hidden_size=64,
        module_activate=nn.Tanh,
    )


def ffjord_l4_h64_relu(num_classes):
    """Return a ffjord with 4 layers, 64 neurons, and relu."""
    return Ffjord(
        num_in=num_classes,
        num_layers=4,
        hidden_size=64,
        module_activate=nn.ReLU,
    )


def ffjord_l8_h64_sigmoid(num_classes):
    """Return a ffjord with 8 layers, 64 neurons, and sigmoid."""
    return Ffjord(
        num_in=num_classes,
        num_layers=8,
        hidden_size=64,
        module_activate=nn.Sigmoid,
    )


def ffjord_l2_h128_sigmoid(num_classes):
    """Return a ffjord with 2 layers, 128 neurons, and sigmoid."""
    return Ffjord(
        num_in=num_classes,
        num_layers=2,
        hidden_size=128,
        module_activate=nn.Sigmoid,
    )


def ffjord_l2_h64_sigmoid(num_classes):
    """Return a ffjord with 2 layers, 64 neurons, and sigmoid."""
    return Ffjord(
        num_in=num_classes,
        num_layers=2,
        hidden_size=64,
        module_activate=nn.Sigmoid,
    )


def ffjord_l4_h64_sigmoid_dopri_adjoint(num_classes):
    """Return a ffjord with 4 layers, 64 neurons, and sigmoid."""
    return Ffjord(
        num_in=num_classes,
        num_layers=4,
        hidden_size=64,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def ffjord_l4_h64_sigmoid_dopri_autograd(num_classes):
    """Return a ffjord with 4 layers, 64 neurons, and sigmoid."""
    return Ffjord(
        num_in=num_classes,
        num_layers=4,
        hidden_size=64,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="autograd",
        solver="dopri5",
    )


def ffjord_l4_h64_sigmoid_rk4_autograd(num_classes):
    """Return a ffjord with 4 layers, 64 neurons, and sigmoid."""
    return Ffjord(
        num_in=num_classes,
        num_layers=4,
        hidden_size=64,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 20),
        sensitivity="autograd",
        solver="rk4",
    )


def ffjord_l4_h64_sigmoid_rk4_adjoint(num_classes):
    """Return a ffjord with 4 layers, 64 neurons, and sigmoid."""
    return Ffjord(
        num_in=num_classes,
        num_layers=4,
        hidden_size=64,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 20),
        sensitivity="adjoint",
        solver="rk4",
    )


def ffjord_l4_h64_sigmoid_euler_autograd(num_classes):
    """Return a ffjord with 4 layers, 64 neurons, and sigmoid."""
    return Ffjord(
        num_in=num_classes,
        num_layers=4,
        hidden_size=64,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 100),
        sensitivity="autograd",
        solver="euler",
    )


def ffjord_l4_h64_sigmoid_euler_adjoint(num_classes):
    """Return a ffjord with 4 layers, 64 neurons, and sigmoid."""
    return Ffjord(
        num_in=num_classes,
        num_layers=4,
        hidden_size=64,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 100),
        sensitivity="adjoint",
        solver="euler",
    )


def ffjord_l4_h64_sigmoid_da(num_classes):
    """Return a ffjord with 4 layers, 64 neurons, and sigmoid."""
    return Ffjord(
        num_in=num_classes,
        num_layers=4,
        hidden_size=64,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def ffjord_l4_h64_sigmoid_da_autograd(num_classes):
    """Return a ffjord with 4 layers, 64 neurons, and sigmoid."""
    return Ffjord(
        num_in=num_classes,
        num_layers=4,
        hidden_size=64,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="autograd",
        solver="dopri5",
    )


def ffjord_l4_h64_softplus_da(num_classes):
    """Return a ffjord with 4 layers, 64 neurons, and softplus."""
    return Ffjord(
        num_in=num_classes,
        num_layers=4,
        hidden_size=64,
        module_activate=nn.Softplus,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def ffjord_l4_h64_tanh_da(num_classes):
    """Return a ffjord with 4 layers, 64 neurons, and tanh."""
    return Ffjord(
        num_in=num_classes,
        num_layers=4,
        hidden_size=64,
        module_activate=nn.Tanh,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def ffjord_l4_h64_relu_da(num_classes):
    """Return a ffjord with 4 layers, 64 neurons, and relu."""
    return Ffjord(
        num_in=num_classes,
        num_layers=4,
        hidden_size=64,
        module_activate=nn.ReLU,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def ffjord_l8_h64_sigmoid_da(num_classes):
    """Return a ffjord with 8 layers, 64 neurons, and sigmoid."""
    return Ffjord(
        num_in=num_classes,
        num_layers=8,
        hidden_size=64,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def ffjord_l8_h37_sigmoid_da(num_classes):
    """Return a ffjord with 8 layers, 37 neurons, and sigmoid."""
    return Ffjord(
        num_in=num_classes,
        num_layers=8,
        hidden_size=37,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def ffjord_l8_h18_sigmoid_da(num_classes):
    """Return a ffjord with 8 layers, 18 neurons, and sigmoid."""
    return Ffjord(
        num_in=num_classes,
        num_layers=8,
        hidden_size=18,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def ffjord_l8_h10_sigmoid_da(num_classes):
    """Return a ffjord with 8 layers, 10 neurons, and sigmoid."""
    return Ffjord(
        num_in=num_classes,
        num_layers=8,
        hidden_size=10,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def ffjord_l6_h45_sigmoid_da(num_classes):
    """Return a ffjord with 6 layers, 45 neurons, and sigmoid."""
    return Ffjord(
        num_in=num_classes,
        num_layers=6,
        hidden_size=45,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def ffjord_l6_h22_sigmoid_da(num_classes):
    """Return a ffjord with 6 layers, 22 neurons, and sigmoid."""
    return Ffjord(
        num_in=num_classes,
        num_layers=6,
        hidden_size=22,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def ffjord_l6_h12_sigmoid_da(num_classes):
    """Return a ffjord with 6 layers, 12 neurons, and sigmoid."""
    return Ffjord(
        num_in=num_classes,
        num_layers=6,
        hidden_size=12,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def ffjord_l4_h128_sigmoid_da(num_classes):
    """Return a ffjord with 4 layers, 128 neurons, and sigmoid."""
    return Ffjord(
        num_in=num_classes,
        num_layers=4,
        hidden_size=128,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def ffjord_l4_h30_sigmoid_da(num_classes):
    """Return a ffjord with 4 layers, 30 neurons, and sigmoid."""
    return Ffjord(
        num_in=num_classes,
        num_layers=4,
        hidden_size=30,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def ffjord_l4_h17_sigmoid_da(num_classes):
    """Return a ffjord with 4 layers, 17 neurons, and sigmoid."""
    return Ffjord(
        num_in=num_classes,
        num_layers=4,
        hidden_size=17,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def ffjord_l3_h90_sigmoid_da(num_classes):
    """Return a ffjord with 3 layers, 90 neurons, and sigmoid."""
    return Ffjord(
        num_in=num_classes,
        num_layers=3,
        hidden_size=90,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def ffjord_l3_h43_sigmoid_da(num_classes):
    """Return a ffjord with 3 layers, 43 neurons, and sigmoid."""
    return Ffjord(
        num_in=num_classes,
        num_layers=3,
        hidden_size=43,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def ffjord_l3_h23_sigmoid_da(num_classes):
    """Return a ffjord with 3 layers, 23 neurons, and sigmoid."""
    return Ffjord(
        num_in=num_classes,
        num_layers=3,
        hidden_size=23,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def ffjord_l2_h1700_sigmoid_da(num_classes):
    """Return a ffjord with 2 layers, 1700 neurons, and sigmoid."""
    return Ffjord(
        num_in=num_classes,
        num_layers=2,
        hidden_size=1700,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def ffjord_l2_h400_sigmoid_da(num_classes):
    """Return a ffjord with 2 layers, 400 neurons, and sigmoid."""
    return Ffjord(
        num_in=num_classes,
        num_layers=2,
        hidden_size=400,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def ffjord_l2_h128_sigmoid_da(num_classes):
    """Return a ffjord with 2 layers, 128 neurons, and sigmoid."""
    return Ffjord(
        num_in=num_classes,
        num_layers=2,
        hidden_size=128,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def ffjord_l2_h128_sigmoid_da_autograd(num_classes):
    """Return a ffjord with 2 layers, 128 neurons, and sigmoid."""
    return Ffjord(
        num_in=num_classes,
        num_layers=2,
        hidden_size=128,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="autograd",
        solver="dopri5",
    )


def ffjord_l2_h64_sigmoid_da(num_classes):
    """Return a ffjord with 2 layers, 64 neurons, and sigmoid."""
    return Ffjord(
        num_in=num_classes,
        num_layers=2,
        hidden_size=64,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )
