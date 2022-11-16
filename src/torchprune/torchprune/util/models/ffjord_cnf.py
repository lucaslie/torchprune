"""Module containing various FFjord CNF configurations from original code."""

import torch
import torch.nn as nn
from ..external.ffjord import train_misc
from ..external.ffjord.lib import layers
from ..external.ffjord.lib import odenvp
from ..external.ffjord.lib import multiscale_parallel


class FfjordCNF(nn.Module):
    """A wrapper class for FFJORD models in the CNF (images) setting."""

    def __init__(self, model, regularization_fns, regularization_coeffs):
        """Initialize with the original model."""
        super().__init__()
        self.model = model
        self._regularization_fns = regularization_fns
        self._regularization_coeffs = regularization_coeffs

    def forward(self, x):
        """Run vanilla forward and some extra args for loss computation."""
        # do a forward pass over the network
        zero = torch.zeros(x.shape[0], 1).to(x)
        z_out, delta_logp = self.model(x, zero)
        output = {
            "out": z_out,
            "delta_logp": delta_logp,
            "nelement": x.nelement(),
        }

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


class FfjordCNFConfig:
    """A class containing the required configurations for FFJORD models."""

    @property
    def dims(self):
        """Return dims.

        Check out external/ffjord/train_cnf.py for more info.
        """
        return self._dims

    @property
    def strides(self):
        """Return strides.

        Check out external/ffjord/train_cnf.py for more info.
        """
        return self._strides

    @property
    def num_blocks(self):
        """Return num_blocks.

        Check out external/ffjord/train_cnf.py for more info.
        """
        return self._num_blocks

    @property
    def conv(self):
        """Return conv.

        Check out external/ffjord/train_cnf.py for more info.
        """
        return True

    @property
    def layer_type(self):
        """Return layer_type.

        Check out external/ffjord/train_cnf.py for more info.
        """
        return "concat"

    @property
    def divergence_fn(self):
        """Return divergence_fn.

        Check out external/ffjord/train_cnf.py for more info.
        """
        return "approximate"

    @property
    def nonlinearity(self):
        """Return nonlinearity.

        Check out external/ffjord/train_cnf.py for more info.
        """
        return self._nonlinearity

    @property
    def solver(self):
        """Return solver.

        Check out external/ffjord/train_cnf.py for more info.
        """
        return self._solver

    @property
    def atol(self):
        """Return atol.

        Check out external/ffjord/train_cnf.py for more info.
        """
        return self._atol

    @property
    def rtol(self):
        """Return rtol.

        Check out external/ffjord/train_cnf.py for more info.
        """
        return self._rtol

    @property
    def step_size(self):
        """Return step_size.

        Check out external/ffjord/train_cnf.py for more info.
        """
        return None

    @property
    def test_solver(self):
        """Return test_solver.

        Check out external/ffjord/train_cnf.py for more info.
        """
        return None

    @property
    def test_atol(self):
        """Return test_atol.

        Check out external/ffjord/train_cnf.py for more info.
        """
        return None

    @property
    def test_rtol(self):
        """Return test_rtol.

        Check out external/ffjord/train_cnf.py for more info.
        """
        return None

    @property
    def imagesize(self):
        """Return imagesize.

        Check out external/ffjord/train_cnf.py for more info.
        """
        return None

    @property
    def alpha(self):
        """Return alpha.

        Check out external/ffjord/train_cnf.py for more info.
        """
        return 1e-6

    @property
    def time_length(self):
        """Return time_length.

        Check out external/ffjord/train_cnf.py for more info.
        """
        return 1.0

    @property
    def train_T(self):  # pylint: disable=C0103
        """Return train_T.

        Check out external/ffjord/train_cnf.py for more info.
        """
        return True

    @property
    def batch_size(self):
        """Return batch_size.

        Check out external/ffjord/train_cnf.py for more info.

        This must be hard-coded here unfortunately.
        """
        return 200

    @property
    def batch_norm(self):
        """Return batch_norm.

        Check out external/ffjord/train_cnf.py for more info.
        """
        return False

    @property
    def residual(self):
        """Return residual.

        Check out external/ffjord/train_cnf.py for more info.
        """
        return False

    @property
    def autoencode(self):
        """Return autoencode.

        Check out external/ffjord/train_cnf.py for more info.
        """
        return False

    @property
    def rademacher(self):
        """Return rademacher.

        Check out external/ffjord/train_cnf.py for more info.
        """
        return False

    @property
    def multiscale(self):
        """Return multiscale.

        Check out external/ffjord/train_cnf.py for more info.
        """
        return True

    @property
    def parallel(self):
        """Return parallel.

        Check out external/ffjord/train_cnf.py for more info.
        """
        return False

    @property
    def l1int(self):
        """Return l1int.

        Check out external/ffjord/train_cnf.py for more info.
        """
        return None

    @property
    def l2int(self):
        """Return l2int.

        Check out external/ffjord/train_cnf.py for more info.
        """
        return None

    @property
    def dl2int(self):
        """Return dl2int.

        Check out external/ffjord/train_cnf.py for more info.
        """
        return None

    @property
    def JFrobint(self):  # pylint: disable=C0103
        """Return JFrobint.

        Check out external/ffjord/train_cnf.py for more info.
        """
        return None

    @property
    def JdiagFrobint(self):  # pylint: disable=C0103
        """Return JdiagFrobint.

        Check out external/ffjord/train_cnf.py for more info.
        """
        return None

    @property
    def JoffdiagFrobint(self):  # pylint: disable=C0103
        """Return JoffdiagFrobint.

        Check out external/ffjord/train_cnf.py for more info.
        """
        return None

    def __init__(
        self,
        output_size,
        data_shape,
        dims="64,64,64",
        strides="1,1,1,1",
        num_blocks=2,
        nonlinearity="softplus",
        solver="dopri5",
        atol=1e-5,
        rtol=1e-5,
    ):
        """Initialize with the variable properties."""
        self._output_size = output_size
        self._data_shape = data_shape
        self._dims = dims
        self._strides = strides
        self._num_blocks = num_blocks
        self._nonlinearity = nonlinearity
        self._solver = solver
        self._atol = atol
        self._rtol = rtol

    def _create_model(self, args, data_shape, regularization_fns):
        """Create model simulating the cnf function."""
        hidden_dims = tuple(map(int, args.dims.split(",")))
        strides = tuple(map(int, args.strides.split(",")))

        if args.multiscale:
            model = odenvp.ODENVP(
                (args.batch_size, *data_shape),
                n_blocks=args.num_blocks,
                intermediate_dims=hidden_dims,
                nonlinearity=args.nonlinearity,
                alpha=args.alpha,
                cnf_kwargs={
                    "T": args.time_length,
                    "train_T": args.train_T,
                    "regularization_fns": regularization_fns,
                },
            )
        elif args.parallel:
            model = multiscale_parallel.MultiscaleParallelCNF(
                (args.batch_size, *data_shape),
                n_blocks=args.num_blocks,
                intermediate_dims=hidden_dims,
                alpha=args.alpha,
                time_length=args.time_length,
            )
        else:
            if args.autoencode:

                def build_cnf():
                    autoencoder_diffeq = layers.AutoencoderDiffEqNet(
                        hidden_dims=hidden_dims,
                        input_shape=data_shape,
                        strides=strides,
                        conv=args.conv,
                        layer_type=args.layer_type,
                        nonlinearity=args.nonlinearity,
                    )
                    odefunc = layers.AutoencoderODEfunc(
                        autoencoder_diffeq=autoencoder_diffeq,
                        divergence_fn=args.divergence_fn,
                        residual=args.residual,
                        rademacher=args.rademacher,
                    )
                    cnf = layers.CNF(
                        odefunc=odefunc,
                        T=args.time_length,
                        regularization_fns=regularization_fns,
                        solver=args.solver,
                    )
                    return cnf

            else:

                def build_cnf():
                    diffeq = layers.ODEnet(
                        hidden_dims=hidden_dims,
                        input_shape=data_shape,
                        strides=strides,
                        conv=args.conv,
                        layer_type=args.layer_type,
                        nonlinearity=args.nonlinearity,
                    )
                    odefunc = layers.ODEfunc(
                        diffeq=diffeq,
                        divergence_fn=args.divergence_fn,
                        residual=args.residual,
                        rademacher=args.rademacher,
                    )
                    cnf = layers.CNF(
                        odefunc=odefunc,
                        T=args.time_length,
                        train_T=args.train_T,
                        regularization_fns=regularization_fns,
                        solver=args.solver,
                    )
                    return cnf

            chain = (
                [layers.LogitTransform(alpha=args.alpha)]
                if args.alpha > 0
                else [layers.ZeroMeanTransform()]
            )
            chain = chain + [build_cnf() for _ in range(args.num_blocks)]
            if args.batch_norm:
                chain.append(layers.MovingBatchNorm2d(data_shape[0]))
            model = layers.SequentialFlow(chain)
        return model

    def get_model(self):
        """Construct and return model."""
        (
            regularization_fns,
            regularization_coeffs,
        ) = train_misc.create_regularization_fns(self)
        model = self._create_model(self, self._data_shape, regularization_fns)

        # wrap model into our tabular Ffjord model for correct output dict
        return FfjordCNF(model, regularization_fns, regularization_coeffs)


def ffjord_multiscale_cifar(num_classes):
    """Return the FFJORD multiscale architecture for CIFAR10."""
    config = FfjordCNFConfig(output_size=num_classes, data_shape=(3, 32, 32))
    return config.get_model()


def ffjord_multiscale_mnist(num_classes, **kwargs):
    """Return the FFJORD multiscale architecture for MNIST."""
    config = FfjordCNFConfig(output_size=num_classes, data_shape=(1, 28, 28))
    return config.get_model()
