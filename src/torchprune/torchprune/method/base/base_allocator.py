"""Module containing base class for the allocators."""
from abc import ABC, abstractmethod
from scipy import optimize
import torch.nn as nn
import torch


class BaseAllocator(nn.Module, ABC):
    """Base Alloctor implementation containing basic functionality."""

    def __init__(self, **kwargs):
        """Initialize with a flexible dictionary of kwargs."""
        super().__init__()
        self.register_buffer("_allocation", torch.Tensor())
        self._offsets = {}
        self._last_budget = None

    def _extract(self, arr, offsets, ell):
        """Extract layer quantities from flat array."""
        arr_layer = arr[offsets[ell][0] : offsets[ell][1]]
        arr_layer = arr_layer.view(2, -1)
        return arr_layer

    def allocate_budget(self, budget):
        """Allocate the budget with this internal function."""
        # make sure budget is int
        budget = int(budget)

        # check if we need to resolve the problem at first
        if self._last_budget == budget:
            return
        self._last_budget = budget

        # solve the problem if not
        self._allocate_method(budget)

    @abstractmethod
    def _allocate_method(self, budget):
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

        self.size = net.size()
        self.flops = net.flops()

        # keep track of _arg_opt
        self._arg_opt = None

        modules = net.compressible_layers
        self._num_layers = net.num_compressible_layers
        self.register_buffer("_in_features", None)
        self.register_buffer("_out_features", None)
        self.register_buffer("_kernel_size", None)
        self.register_buffer("_kernel_shapes", None)
        self.register_buffer("_num_patches", None)
        self._in_features = torch.zeros(
            self._num_layers, dtype=torch.long, device=modules[0].weight.device
        )
        self._out_features = torch.zeros_like(self._in_features)
        self._kernel_size = torch.zeros_like(self._in_features)
        self._kernel_shapes = torch.vstack(
            [torch.ones_like(self._in_features) for _ in range(2)]
        ).t()
        self._num_patches = torch.zeros_like(self._in_features)

        # 1.) out_mode == True:  sensitivity is w.r.t. output features.
        # 2.) out_mode == False: sensitivity is w.r.t. input features.
        # This has implications on what and how we prune
        self._out_mode = out_mode

        for ell, module in enumerate(modules):
            weight = module.weight
            self._in_features[ell] = self.get_num_features(weight, 1)
            self._out_features[ell] = self.get_num_features(weight, 0)
            self._kernel_size[ell] = weight[0, 0].numel()
            if weight.dim() > 2:
                self._kernel_shapes[ell] = torch.tensor(weight.shape[2:])
            self._num_patches[ell] = net.num_patches[ell]

    # estimate number of resulting weights
    def _get_size(self, out_features, in_features):
        size_total = (in_features * out_features * self._kernel_size).sum()
        return size_total

    def get_num_samples(self, layer):
        """Get the number of samples for a particular layer index."""
        return self._allocation[layer]

    def _get_resulting_size(self, arg):
        """Get resulting size for some arg."""
        out_features, in_features = self._get_proposed_num_features(arg)
        return self._get_size(out_features, in_features)

    def _allocate_method(self, budget, disp=False):
        # set up bisection method
        arg_min, arg_max = self._get_boundaries()

        def f_opt(arg):
            size_resulting = self._get_resulting_size(arg)
            return budget - size_resulting

        # solve with bisection method and get resulting feature allocation
        f_value_min = f_opt(arg_min)
        f_value_max = f_opt(arg_max)
        if f_value_min.sign() == f_value_max.sign():
            arg_opt, f_value_opt = (
                (arg_min, f_value_min)
                if abs(f_value_min) < abs(f_value_max)
                else (arg_max, f_value_max)
            )
            error_msg = (
                "no bisection possible"
                f"; argMin: {arg_min}, minF: {f_value_min}"
                f"; argMax: {arg_max}, maxF: {f_value_max}"
            )
            print(error_msg)
            if disp and abs(f_value_opt) / budget > 0.005:
                raise ValueError(error_msg)
        else:
            arg_opt = optimize.brentq(
                f_opt, arg_min, arg_max, maxiter=1000, xtol=10e-250, disp=False
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
    def _get_boundaries(self):
        raise NotImplementedError

    @abstractmethod
    def _get_proposed_num_features(self, arg):
        raise NotImplementedError

    def _extract(self, arr, offsets, ell):
        arr_layer = arr[offsets[ell][0] : offsets[ell][1]]
        return arr_layer
