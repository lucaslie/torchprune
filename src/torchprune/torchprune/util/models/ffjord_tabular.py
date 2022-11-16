"""Module containing various FFjord NODE configurations from original code."""

import torch
import torch.nn as nn
from ..external.ffjord import train_misc


class FfjordTabular(nn.Module):
    """A wrapper class for FFJORD models in the tabular data setting."""

    def __init__(
        self,
        model,
        output_size,
        hdim_factor,
        regularization_fns,
        regularization_coeffs,
    ):
        """Initialize with the original model."""
        super().__init__()
        self.model = model
        self._num_weights_threshold = output_size * hdim_factor
        self._regularization_fns = regularization_fns
        self._regularization_coeffs = regularization_coeffs

    def forward(self, x):
        """Run vanilla forward and some extra args for loss computation."""
        # do a forward pass over the network
        zero = torch.zeros(x.shape[0], 1).to(x)
        z_out, delta_logp = self.model(x, zero)
        output = {"out": z_out, "delta_logp": delta_logp}

        # add regularizer loss to output
        if len(self._regularization_coeffs) > 0:
            reg_states = train_misc.get_regularization(
                self.model, self._regularization_coeffs
            )
            output["reg_loss"] = sum(
                reg_state * coeff
                for reg_state, coeff in zip(
                    reg_states, self._regularization_coeffs
                )
                if coeff != 0
            )

        # return output dictionary
        return output

    def is_compressible(self, module):
        """Return True if the provided module is compressible."""
        return module.weight.numel() > self._num_weights_threshold


class FfjordTabularConfig:
    """A class containing the configurations for FFJORD tabular models."""

    @property
    def layer_type(self):
        """Return layer_type.

        Check out external/ffjord/train_tabular.py for more info.
        """
        return "concatsquash"

    @property
    def hdim_factor(self):
        """Return hdim_factor.

        Check out external/ffjord/train_tabular.py for more info.
        """
        return self._hdim_factor

    @property
    def nhidden(self):
        """Return nhidden.

        Check out external/ffjord/train_tabular.py for more info.
        """
        return self._nhidden

    @property
    def dims(self):
        """Return dims.

        Check out external/ffjord/train_tabular.py for more info.
        """
        return "-".join(
            [str(self.hdim_factor * self._output_size)] * self.nhidden
        )

    @property
    def num_blocks(self):
        """Return num_blocks.

        Check out external/ffjord/train_tabular.py for more info.
        """
        return self._num_blocks

    @property
    def time_length(self):
        """Return time_length.

        Check out external/ffjord/train_tabular.py for more info.
        """
        return 1.0

    @property
    def train_T(self):  # pylint: disable=C0103
        """Return train_T.

        Check out external/ffjord/train_tabular.py for more info.
        """
        return True

    @property
    def divergence_fn(self):
        """Return divergence_fn.

        Check out external/ffjord/train_tabular.py for more info.
        """
        return "approximate"

    @property
    def nonlinearity(self):
        """Return nonlinearity.

        Check out external/ffjord/train_tabular.py for more info.
        """
        return self._nonlinearity

    @property
    def solver(self):
        """Return solver.

        Check out external/ffjord/train_tabular.py for more info.
        """
        return self._solver

    @property
    def atol(self):
        """Return atol.

        Check out external/ffjord/train_tabular.py for more info.
        """
        return self._atol

    @property
    def rtol(self):
        """Return rtol.

        Check out external/ffjord/train_tabular.py for more info.
        """
        return self._rtol

    @property
    def step_size(self):
        """Return step_size.

        Check out external/ffjord/train_tabular.py for more info.
        """
        return None

    @property
    def test_solver(self):
        """Return test_solver.

        Check out external/ffjord/train_tabular.py for more info.
        """
        return None

    @property
    def test_atol(self):
        """Return test_atol.

        Check out external/ffjord/train_tabular.py for more info.
        """
        return None

    @property
    def test_rtol(self):
        """Return test_rtol.

        Check out external/ffjord/train_tabular.py for more info.
        """
        return None

    @property
    def residual(self):
        """Return residual.

        Check out external/ffjord/train_tabular.py for more info.
        """
        return False

    @property
    def rademacher(self):
        """Return rademacher.

        Check out external/ffjord/train_tabular.py for more info.
        """
        return False

    @property
    def batch_norm(self):
        """Return batch_norm.

        Check out external/ffjord/train_tabular.py for more info.
        """
        return False

    @property
    def bn_lag(self):
        """Return bn_lag.

        Check out external/ffjord/train_tabular.py for more info.
        """
        return 0.0

    @property
    def l1int(self):
        """Return l1int.

        Check out external/ffjord/train_tabular.py for more info.
        """
        return None

    @property
    def l2int(self):
        """Return l2int.

        Check out external/ffjord/train_tabular.py for more info.
        """
        return None

    @property
    def dl2int(self):
        """Return dl2int.

        Check out external/ffjord/train_tabular.py for more info.
        """
        return None

    @property
    def JFrobint(self):  # pylint: disable=C0103
        """Return JFrobint.

        Check out external/ffjord/train_tabular.py for more info.
        """
        return None

    @property
    def JdiagFrobint(self):  # pylint: disable=C0103
        """Return JdiagFrobint.

        Check out external/ffjord/train_tabular.py for more info.
        """
        return None

    @property
    def JoffdiagFrobint(self):  # pylint: disable=C0103
        """Return JoffdiagFrobint.

        Check out external/ffjord/train_tabular.py for more info.
        """
        return None

    def __init__(
        self,
        output_size,
        hdim_factor=10,
        nhidden=1,
        num_blocks=1,
        nonlinearity="softplus",
        solver="dopri5",
        atol=1e-8,
        rtol=1e-6,
    ):
        """Initialize with the variable properties."""
        self._output_size = output_size
        self._hdim_factor = hdim_factor
        self._nhidden = nhidden
        self._num_blocks = num_blocks
        self._nonlinearity = nonlinearity
        self._solver = solver
        self._atol = atol
        self._rtol = rtol

    def get_model(self):
        """Construct and return model."""
        (
            regularization_fns,
            regularization_coeffs,
        ) = train_misc.create_regularization_fns(self)
        model = train_misc.build_model_tabular(
            self, self._output_size, regularization_fns
        )

        # wrap model into our tabular Ffjord model for correct output dict
        return FfjordTabular(
            model,
            self._output_size,
            self.hdim_factor,
            regularization_fns,
            regularization_coeffs,
        )


def ffjord_l3_hm10_f5_tanh(num_classes):
    """Return a ffjord with 3 layers, hidden factor 10, 5 flows, tanh.

    This is the standard configuration for the tabular POWER dataset.
    """
    config = FfjordTabularConfig(
        output_size=num_classes,
        nhidden=3,
        hdim_factor=10,
        num_blocks=5,
        nonlinearity="tanh",
    )
    return config.get_model()


# I think the factor is always hdim_factor * output_size


def ffjord_l3_hm20_f5_tanh(num_classes):
    """Return a ffjord with 3 layers, hidden factor 20, 5 flows, tanh.

    This is the standard configuration for the tabular GAS dataset.
    """
    config = FfjordTabularConfig(
        output_size=num_classes,
        nhidden=3,
        hdim_factor=20,
        num_blocks=5,
        nonlinearity="tanh",
    )
    return config.get_model()


def ffjord_l2_hm10_f10_softplus(num_classes):
    """Return a ffjord with 2 layers, hidden factor 10, 10 flows, softplus.

    This is the standard configuration for the tabular HEPMASS dataset.
    """
    config = FfjordTabularConfig(
        output_size=num_classes,
        nhidden=2,
        hdim_factor=10,
        num_blocks=10,
        nonlinearity="softplus",
    )
    return config.get_model()


def ffjord_l2_hm20_f1_softplus(num_classes):
    """Return a ffjord with 2 layers, hidden factor 20, 1 flow steps, softplus.

    This is the standard configuration for the tabular MINIBOONE dataset.
    """
    config = FfjordTabularConfig(
        output_size=num_classes,
        nhidden=2,
        hdim_factor=20,
        num_blocks=1,
        nonlinearity="softplus",
    )
    return config.get_model()


def ffjord_l3_hm20_f2_softplus(num_classes):
    """Return a ffjord with 3 layers, hidden factor 20, 2 flow steps, softplus.

    This is the standard configuration for the tabular BSDS300 dataset.
    """
    config = FfjordTabularConfig(
        output_size=num_classes,
        nhidden=3,
        hdim_factor=20,
        num_blocks=2,
        nonlinearity="softplus",
    )
    return config.get_model()
