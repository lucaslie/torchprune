"""Module containing base class for the allocators."""
from abc import ABC, abstractmethod
from scipy import optimize
import torch.nn as nn
import torch


class BaseAllocator(nn.Module, ABC):
    """Base Alloctor implementation containing basic functionality.

    NAMING CONVENTION FOR CHILD CLASSES:
        * {D, H, R, U} = {'Deterministic', 'Hybrid', 'Random', 'Uniform'}
        * {W, F} = {'Weight-based', 'Filter-based'}
        * {S, T, None} = {'Sensitivity-based', 'Thresholding-based', 'Generic'}

    """

    def __init__(self, **kwargs):
        """Initialize with a flexible dictionary of kwargs."""
        super().__init__()
        self.register_buffer("_allocation", torch.Tensor())
        self._offsets = {}
        self._last_ratio = None

    def _extract(self, arr, offsets, ell):
        """Extract layer quantities from flat array."""
        arr_layer = arr[offsets[ell][0] : offsets[ell][1]]
        arr_layer = arr_layer.view(2, -1)
        return arr_layer

    def _allocate_budget(self, keep_ratio, size_original):
        """Allocate the budget with this internal function."""
        # check if we need to resolve the problem at first
        if self._last_ratio == keep_ratio:
            return
        self._last_ratio = keep_ratio

        # solve the problem if not
        self._allocate_method(keep_ratio, size_original)

    @abstractmethod
    def _allocate_method(self, keep_ratio, size_original):
        """Allocate the samples."""
        raise NotImplementedError

    def get_num_samples(self, layer):
        """Get the number of samples for a particular layer index."""
        return self._extract(self._allocation, self._offsets, layer)

    def get_num_features(self, tensor, dim):
        """Get the number of non-zero features in the given dimension."""
        dims_to_sum = [i for i in range(tensor.dim()) if i is not dim]
        return (torch.abs(tensor).sum(dims_to_sum) != 0).sum()

    def forward(self, x):
        """Fake forward function since it is inheriting from nn.Module."""


class BaseFilterAllocator(BaseAllocator):
    """The base allocator for filter pruning methods."""

    def __init__(self, net, out_mode, **kwargs):
        """Initialize a filter-pruning allocator with flexible kwargs."""
        super().__init__(**kwargs)

        self._use_flops = False
        self.size = net.size()
        self.flops = net.flops()

        # keep track of _arg_opt
        self._arg_opt = None

        modules = net.compressible_layers
        self._num_layers = net.num_compressible_layers
        self.register_buffer("_kernel_size", torch.Tensor())
        self.register_buffer("_num_patches", torch.Tensor())
        self._in_features = torch.zeros(
            self._num_layers, dtype=torch.int, device=modules[0].weight.device
        )
        self._out_features = torch.zeros_like(self._in_features)
        self._kernel_size = torch.zeros_like(self._in_features)
        self._num_patches = torch.zeros_like(self._in_features)
        self._num_dependencies = torch.zeros_like(self._in_features)

        # 1.) out_mode == True:  sensitivity is w.r.t. output features.
        # 2.) out_mode == False: sensitivity is w.r.t. input features.
        # This has implications on what and how we prune
        self._out_mode = out_mode

        for ell, module in enumerate(modules):
            self._in_features[ell] = self.get_num_features(module.weight, 1)
            self._out_features[ell] = self.get_num_features(module.weight, 0)
            self._kernel_size[ell] = module.weight[0, 0].numel()
            self._num_patches[ell] = net.num_patches[ell]
            if self._out_mode:
                self._num_dependencies[ell] = len(net.dependencies_out[module])
            else:
                self._num_dependencies[ell] = len(net.dependencies_in[module])

    # estimate number of resulting weights
    def _get_size(self, out_features, in_features):
        if self._use_flops:
            num_patches = self._num_patches
        else:
            num_patches = torch.ones_like(self._num_patches)

        for ell in range(self._num_layers):
            # check for special cases in which case we cannot assume to be
            # able to "propagate" pruning
            if self._num_dependencies[ell] != 1:
                if self._out_mode:
                    in_features[ell] = self._in_features[ell]
                else:
                    out_features[ell] = self._out_features[ell]

        size_total = (
            in_features * out_features * self._kernel_size * num_patches
        ).sum()
        return size_total

    def get_num_samples(self, layer):
        """Get the number of samples for a particular layer index."""
        return self._allocation[layer]

    def _allocate_method(self, keep_ratio, size_original):
        # compute available budget in terms of #weights
        if self._use_flops:
            raise NotImplementedError
            budget = int(keep_ratio * float(self.flops))
        else:
            budget = int(keep_ratio * float(size_original))

        # set up bisection method
        arg_min, arg_max = self._get_boundaries(keep_ratio)

        def f_opt(arg):
            out_features, in_features = self._get_proposed_num_features(arg)
            size_resulting = self._get_size(out_features, in_features)
            return budget - size_resulting

        # solve with bisection method and get resulting feature allocation
        f_value_min = f_opt(arg_min)
        f_value_max = f_opt(arg_max)
        if f_value_min > 0 and f_value_max > 0:
            arg_opt = arg_min if f_value_min < f_value_max else arg_max
            print(
                "no bisection possible. argMin: {}, minF: {}, "
                "argMax: {}, maxF: {}".format(
                    arg_min, f_value_min, arg_max, f_value_max
                )
            )
        else:
            arg_opt = optimize.bisect(
                f_opt, arg_min, arg_max, maxiter=1000, xtol=10e-250
            )
        out_features, in_features = self._get_proposed_num_features(arg_opt)

        # store allocation
        if self._out_mode:
            self._allocation = out_features
        else:
            self._allocation = in_features

        # keep track of _arg_opt as well
        self._arg_opt = arg_opt

    @abstractmethod
    def _get_boundaries(self, keep_ratio):
        raise NotImplementedError

    @abstractmethod
    def _get_proposed_num_features(self, arg):
        raise NotImplementedError

    def _extract(self, arr, offsets, ell):
        arr_layer = arr[offsets[ell][0] : offsets[ell][1]]
        return arr_layer
