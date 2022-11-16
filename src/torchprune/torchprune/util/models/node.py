"""Module containing various Neural ODE configurations."""

import torch
import torch.nn as nn
from .ffjord import NeuralODEClassic


class Node(nn.Module):
    """Neural ODE for classification."""

    def __init__(
        self,
        num_in,
        num_out,
        num_layers,
        hidden_size,
        module_activate,
        s_span=torch.linspace(0, 1, 20),
        sensitivity="autograd",
        solver="rk4",
        atol=1e-4,
        rtol=1e-4,
    ):
        """Initialize a neural ode with the desired parameterization."""
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
        layers.append(nn.Linear(hidden_size, num_out))

        # wrap in sequential module
        self.f_forward = nn.Sequential(*layers)

        # wrap in torchdyn NeuralODE
        self.model = NeuralODEClassic(
            self.f_forward,
            s_span=s_span,
            sensitivity=sensitivity,
            solver=solver,
            atol=atol,
            rtol=rtol,
        )

    def forward(self, x):
        """Forward by passing it on to NeuralODE."""
        return self.model(x)


def node_l2_h64_tanh(num_classes):
    """Return a classification node with 2 layers, 64 neurons, and tanh."""
    return Node(
        num_in=num_classes,
        num_out=num_classes,
        num_layers=2,
        hidden_size=64,
        module_activate=nn.Tanh,
    )


def node_l2_h64_softplus(num_classes):
    """Return a classification node with 2 layers, 64 neurons, and softplus."""
    return Node(
        num_in=num_classes,
        num_out=num_classes,
        num_layers=2,
        hidden_size=64,
        module_activate=nn.Softplus,
    )


def node_l2_h64_sigmoid(num_classes):
    """Return a classification node with 2 layers, 64 neurons, and sigmoid."""
    return Node(
        num_in=num_classes,
        num_out=num_classes,
        num_layers=2,
        hidden_size=64,
        module_activate=nn.Sigmoid,
    )


def node_l2_h64_relu(num_classes):
    """Return a classification node with 2 layers, 64 neurons, and relu."""
    return Node(
        num_in=num_classes,
        num_out=num_classes,
        num_layers=2,
        hidden_size=64,
        module_activate=nn.ReLU,
    )


def node_l4_h32_tanh(num_classes):
    """Return a classification node with 4 layers, 32 neurons, and tanh."""
    return Node(
        num_in=num_classes,
        num_out=num_classes,
        num_layers=4,
        hidden_size=32,
        module_activate=nn.Tanh,
    )


def node_l2_h32_tanh(num_classes):
    """Return a classification node with 2 layers, 32 neurons, and tanh."""
    return Node(
        num_in=num_classes,
        num_out=num_classes,
        num_layers=2,
        hidden_size=32,
        module_activate=nn.Tanh,
    )


def node_l2_h128_tanh(num_classes):
    """Return a classification node with 2 layers, 128 neurons, and tanh."""
    return Node(
        num_in=num_classes,
        num_out=num_classes,
        num_layers=2,
        hidden_size=128,
        module_activate=nn.Tanh,
    )


def node_l4_h128_tanh(num_classes):
    """Return a classification node with 4 layers, 128 neurons, and tanh."""
    return Node(
        num_in=num_classes,
        num_out=num_classes,
        num_layers=4,
        hidden_size=128,
        module_activate=nn.Tanh,
    )


def node_l4_h32_tanh_dopri_adjoint(num_classes):
    """Return a classification node with 4 layers, 32 neurons, and tanh.

    We also modify solver options for this one
    """
    return Node(
        num_in=num_classes,
        num_out=num_classes,
        num_layers=4,
        hidden_size=32,
        module_activate=nn.Tanh,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def node_l4_h32_tanh_dopri_autograd(num_classes):
    """Return a classification node with 4 layers, 32 neurons, and tanh.

    We also modify solver options for this one
    """
    return Node(
        num_in=num_classes,
        num_out=num_classes,
        num_layers=4,
        hidden_size=32,
        module_activate=nn.Tanh,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="autograd",
        solver="dopri5",
    )


def node_l4_h32_tanh_rk4_adjoint(num_classes):
    """Return a classification node with 4 layers, 32 neurons, and tanh.

    We also modify solver options for this one
    """
    return Node(
        num_in=num_classes,
        num_out=num_classes,
        num_layers=4,
        hidden_size=32,
        module_activate=nn.Tanh,
        s_span=torch.linspace(0, 1, 20),
        sensitivity="adjoint",
        solver="rk4",
    )


def node_l4_h32_tanh_rk4_autograd(num_classes):
    """Return a classification node with 4 layers, 32 neurons, and tanh.

    We also modify solver options for this one
    """
    return Node(
        num_in=num_classes,
        num_out=num_classes,
        num_layers=4,
        hidden_size=32,
        module_activate=nn.Tanh,
        s_span=torch.linspace(0, 1, 20),
        sensitivity="autograd",
        solver="rk4",
    )


def node_l4_h32_tanh_euler_adjoint(num_classes):
    """Return a classification node with 4 layers, 32 neurons, and tanh.

    We also modify solver options for this one
    """
    return Node(
        num_in=num_classes,
        num_out=num_classes,
        num_layers=4,
        hidden_size=32,
        module_activate=nn.Tanh,
        s_span=torch.linspace(0, 1, 80),
        sensitivity="adjoint",
        solver="euler",
    )


def node_l4_h32_tanh_euler_autograd(num_classes):
    """Return a classification node with 4 layers, 32 neurons, and tanh.

    We also modify solver options for this one
    """
    return Node(
        num_in=num_classes,
        num_out=num_classes,
        num_layers=4,
        hidden_size=32,
        module_activate=nn.Tanh,
        s_span=torch.linspace(0, 1, 80),
        sensitivity="autograd",
        solver="euler",
    )


def node_l2_h64_tanh_da(num_classes):
    """Return a classification node with 2 layers, 64 neurons, and tanh."""
    return Node(
        num_in=num_classes,
        num_out=num_classes,
        num_layers=2,
        hidden_size=64,
        module_activate=nn.Tanh,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def node_l2_h64_softplus_da(num_classes):
    """Return a classification node with 2 layers, 64 neurons, and softplus."""
    return Node(
        num_in=num_classes,
        num_out=num_classes,
        num_layers=2,
        hidden_size=64,
        module_activate=nn.Softplus,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def node_l2_h64_sigmoid_da(num_classes):
    """Return a classification node with 2 layers, 64 neurons, and sigmoid."""
    return Node(
        num_in=num_classes,
        num_out=num_classes,
        num_layers=2,
        hidden_size=64,
        module_activate=nn.Sigmoid,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def node_l2_h64_relu_da(num_classes):
    """Return a classification node with 2 layers, 64 neurons, and relu."""
    return Node(
        num_in=num_classes,
        num_out=num_classes,
        num_layers=2,
        hidden_size=64,
        module_activate=nn.ReLU,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def node_l2_h3_tanh_da(num_classes):
    """Return a classification node with 2 layers, 3 neurons, and tanh."""
    return Node(
        num_in=num_classes,
        num_out=num_classes,
        num_layers=2,
        hidden_size=3,
        module_activate=nn.Tanh,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def node_l2_h32_tanh_da(num_classes):
    """Return a classification node with 2 layers, 32 neurons, and tanh."""
    return Node(
        num_in=num_classes,
        num_out=num_classes,
        num_layers=2,
        hidden_size=32,
        module_activate=nn.Tanh,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def node_l2_h128_tanh_da(num_classes):
    """Return a classification node with 2 layers, 128 neurons, and tanh."""
    return Node(
        num_in=num_classes,
        num_out=num_classes,
        num_layers=2,
        hidden_size=128,
        module_activate=nn.Tanh,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def node_l4_h32_tanh_da(num_classes):
    """Return a classification node with 4 layers, 32 neurons, and tanh."""
    return Node(
        num_in=num_classes,
        num_out=num_classes,
        num_layers=4,
        hidden_size=32,
        module_activate=nn.Tanh,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )


def node_l4_h128_tanh_da(num_classes):
    """Return a classification node with 4 layers, 128 neurons, and tanh."""
    return Node(
        num_in=num_classes,
        num_out=num_classes,
        num_layers=4,
        hidden_size=128,
        module_activate=nn.Tanh,
        s_span=torch.linspace(0, 1, 2),
        sensitivity="adjoint",
        solver="dopri5",
    )
