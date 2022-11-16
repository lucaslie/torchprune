"""Module containing various FFjord NODE configurations with torchdyn lib."""

import torch
import torch.nn as nn
from torchdyn.models import autograd_trace

from .ffjord import Ffjord


class VanillaCNF(Ffjord):
    """Neural ODEs for CNFs via brute-force trace estimator."""

    @property
    def trace_estimator(self):
        """Return the desired trace estimator."""
        return autograd_trace


def cnf_l4_h64_sigmoid(num_classes):
    """Return a brute-force CNF with 4 layers, 64 neurons, and sigmoid."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=4,
        hidden_size=64,
        module_activate=nn.Sigmoid,
    )


def cnf_l4_h64_softplus(num_classes):
    """Return a brute-force CNF with 4 layers, 64 neurons, and softplus."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=4,
        hidden_size=64,
        module_activate=nn.Softplus,
    )


def cnf_l4_h64_tanh(num_classes):
    """Return a brute-force CNF with 4 layers, 64 neurons, and tanh."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=4,
        hidden_size=64,
        module_activate=nn.Tanh,
    )


def cnf_l4_h64_relu(num_classes):
    """Return a brute-force CNF with 4 layers, 64 neurons, and relu."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=4,
        hidden_size=64,
        module_activate=nn.ReLU,
    )


def cnf_l8_h64_sigmoid(num_classes):
    """Return a brute-force CNF with 8 layers, 64 neurons, and sigmoid."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=8,
        hidden_size=64,
        module_activate=nn.Sigmoid,
    )


def cnf_l2_h128_sigmoid(num_classes):
    """Return a brute-force CNF with 2 layers, 128 neurons, and sigmoid."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=2,
        hidden_size=128,
        module_activate=nn.Sigmoid,
    )


def cnf_l2_h64_sigmoid(num_classes):
    """Return a brute-force CNF with 2 layers, 64 neurons, and sigmoid."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=2,
        hidden_size=64,
        module_activate=nn.Sigmoid,
    )


def cnf_l4_h64_sigmoid_dopri_adjoint(num_classes):
    """Return a brute-force CNF with 4 layers, 64 neurons, and sigmoid."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=4,
        hidden_size=64,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def cnf_l4_h64_sigmoid_dopri_autograd(num_classes):
    """Return a brute-force CNF with 4 layers, 64 neurons, and sigmoid."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=4,
        hidden_size=64,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="autograd",
        solver="dopri5",
    )


def cnf_l4_h64_sigmoid_rk4_autograd(num_classes):
    """Return a brute-force CNF with 4 layers, 64 neurons, and sigmoid."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=4,
        hidden_size=64,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 20),
        sensitivity="autograd",
        solver="rk4",
    )


def cnf_l4_h64_sigmoid_rk4_adjoint(num_classes):
    """Return a brute-force CNF with 4 layers, 64 neurons, and sigmoid."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=4,
        hidden_size=64,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 20),
        sensitivity="adjoint",
        solver="rk4",
    )


def cnf_l4_h64_sigmoid_euler_autograd(num_classes):
    """Return a brute-force CNF with 4 layers, 64 neurons, and sigmoid."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=4,
        hidden_size=64,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 100),
        sensitivity="autograd",
        solver="euler",
    )


def cnf_l4_h64_sigmoid_euler_adjoint(num_classes):
    """Return a brute-force CNF with 4 layers, 64 neurons, and sigmoid."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=4,
        hidden_size=64,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 100),
        sensitivity="adjoint",
        solver="euler",
    )


def cnf_l4_h64_sigmoid_da(num_classes):
    """Return a brute-force CNF with 4 layers, 64 neurons, and sigmoid."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=4,
        hidden_size=64,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def cnf_l4_h64_softplus_da(num_classes):
    """Return a brute-force CNF with 4 layers, 64 neurons, and softplus."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=4,
        hidden_size=64,
        module_activate=nn.Softplus,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def cnf_l4_h64_tanh_da(num_classes):
    """Return a brute-force CNF with 4 layers, 64 neurons, and tanh."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=4,
        hidden_size=64,
        module_activate=nn.Tanh,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def cnf_l4_h64_relu_da(num_classes):
    """Return a brute-force CNF with 4 layers, 64 neurons, and relu."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=4,
        hidden_size=64,
        module_activate=nn.ReLU,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def cnf_l8_h64_sigmoid_da(num_classes):
    """Return a brute-force CNF with 8 layers, 64 neurons, and sigmoid."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=8,
        hidden_size=64,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def cnf_l8_h37_sigmoid_da(num_classes):
    """Return a brute-force CNF with 8 layers, 37 neurons, and sigmoid."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=8,
        hidden_size=37,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def cnf_l8_h18_sigmoid_da(num_classes):
    """Return a brute-force CNF with 8 layers, 18 neurons, and sigmoid."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=8,
        hidden_size=18,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def cnf_l8_h10_sigmoid_da(num_classes):
    """Return a brute-force CNF with 8 layers, 10 neurons, and sigmoid."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=8,
        hidden_size=10,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def cnf_l6_h45_sigmoid_da(num_classes):
    """Return a brute-force CNF with 6 layers, 45 neurons, and sigmoid."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=6,
        hidden_size=45,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def cnf_l6_h22_sigmoid_da(num_classes):
    """Return a brute-force CNF with 6 layers, 22 neurons, and sigmoid."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=6,
        hidden_size=22,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def cnf_l6_h12_sigmoid_da(num_classes):
    """Return a brute-force CNF with 6 layers, 12 neurons, and sigmoid."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=6,
        hidden_size=12,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def cnf_l4_h128_sigmoid_da(num_classes):
    """Return a brute-force CNF with 4 layers, 128 neurons, and sigmoid."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=4,
        hidden_size=128,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def cnf_l4_h30_sigmoid_da(num_classes):
    """Return a brute-force CNF with 4 layers, 30 neurons, and sigmoid."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=4,
        hidden_size=30,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def cnf_l4_h17_sigmoid_da(num_classes):
    """Return a brute-force CNF with 4 layers, 17 neurons, and sigmoid."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=4,
        hidden_size=17,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def cnf_l3_h90_sigmoid_da(num_classes):
    """Return a brute-force CNF with 3 layers, 90 neurons, and sigmoid."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=3,
        hidden_size=90,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def cnf_l3_h43_sigmoid_da(num_classes):
    """Return a brute-force CNF with 3 layers, 43 neurons, and sigmoid."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=3,
        hidden_size=43,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def cnf_l3_h23_sigmoid_da(num_classes):
    """Return a brute-force CNF with 3 layers, 23 neurons, and sigmoid."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=3,
        hidden_size=23,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def cnf_l2_h1700_sigmoid_da(num_classes):
    """Return a brute-force CNF with 2 layers, 1700 neurons, and sigmoid."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=2,
        hidden_size=1700,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def cnf_l2_h400_sigmoid_da(num_classes):
    """Return a brute-force CNF with 2 layers, 400 neurons, and sigmoid."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=2,
        hidden_size=400,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def cnf_l2_h128_sigmoid_da(num_classes):
    """Return a brute-force CNF with 2 layers, 128 neurons, and sigmoid."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=2,
        hidden_size=128,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def cnf_l2_h64_sigmoid_da(num_classes):
    """Return a brute-force CNF with 2 layers, 64 neurons, and sigmoid."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=2,
        hidden_size=64,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def cnf_l4_h64_sigmoid_da_high_tol(num_classes):
    """Return a brute-force CNF with 4 layers, 64 neurons, and sigmoid."""
    return VanillaCNF(
        num_in=num_classes,
        num_layers=4,
        hidden_size=64,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
        atol=1e-4,
        rtol=1e-4,
    )
