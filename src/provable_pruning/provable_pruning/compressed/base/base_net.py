"""Implementation for abstract base class for any compressed network.

DESIGN PHILOSOPHY OF THE COMPRESSED NET CLASSES:

1.) All functions are implemented in a modular, hierarchical Base classes
2.) Actual classes just define properties and inject various functionality by
    deriving from all required base classes
3.) Overwrite small functions if deemed absolutely necessary
"""

import copy
import time
from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseCompressedNet(ABC, nn.Module):
    """The simplest API for a compression algorithms and the resulting net."""

    def __init__(self, original_net):
        """Initialize the compression with a uncompressed network."""
        super().__init__()

        # this is a hack so that pytorch doesn't recognize original_net as a
        # module and thus does not change its device if we call the ".to()#
        # function
        self.original_net = [original_net]
        self.compressed_net = copy.deepcopy(original_net)
        self.layers = range(self.original_net[0].num_compressible_layers)

        # reset members
        self.reset()

    @property
    def name(self):
        """Return the name of the compression method."""
        return self.compressed_net.name + "_" + self._get_name()

    def reset(self):
        """Reset the sparsity mask, thus deleting the compression."""
        param_iter = self.compressed_net.compressible_layers.parameters()
        for idx, _ in enumerate(param_iter):
            key = f"param{idx}"
            if key in self._buffers:
                del self._buffers[key]

    def forward(self, x):
        """Do the classic forward on the compressed net."""
        return self.compressed_net(x)

    def register_sparsity_pattern(self):
        """Register the sparsity pattern for the compression."""
        self.compressed_net.register_sparsity_pattern()

    def enforce_sparsity(self):
        """Enforce the sparsity by applying the sparsity masks."""
        self.compressed_net.enforce_sparsity()

    def replace_parameters(self, other_net):
        """Replace the parameters with the parameters of some other net."""
        self.compressed_net.load_state_dict(
            other_net.state_dict(), strict=False
        )

    def size(self):
        """Return total number of nonzero parameters in the compressed_net."""
        return self.compressed_net.size()

    def flops(self):
        """Return total number of flops in the compressed_net."""
        return self.compressed_net.flops()

    @property
    @abstractmethod
    def deterministic(self):
        """Indicate whether compression method is deterministic."""
        raise NotImplementedError

    @property
    @abstractmethod
    def retrainable(self):
        """Indicate whether we can retrain after applying this method."""
        raise NotImplementedError

    @abstractmethod
    def compress(self, keep_ratio, from_original=True, initialize=True):
        """Execute the compression step.

        This function should return the per-layer budget in terms of weights.
        """
        raise NotImplementedError


class CompressedNet(BaseCompressedNet):
    """A more advanced interface for compression algorithms.

    This interface requires more methods to implement but in turn already
    contains a lot of the commonalities among many of the compression
    algorithms.

    """

    def __init__(self, original_net):
        """Initialize the compression with a uncompressed network."""
        # declare a few members
        self.allocator = None
        self.pruners = None

        # call the super initializer
        super().__init__(original_net)

    def reset(self):
        """Reset the compresseion and all the associated modules."""
        super().reset()

        # we require an allocator for the network
        del self.allocator
        self.allocator = nn.Module()

        # and pruners for each layer
        del self.pruners
        self.pruners = nn.ModuleList()

    def _initialize_compression(self):
        """Initialize the compression.

        We initialize our compression scheme by doing a forward pass through
        the layers and compute properties within each layer (e.g. sensitivity).

        """
        # keep timing of total computation time
        time_total = 0.0

        self.reset()
        # doing some preprocessing (and potentially modifying compressed_net)
        time_pre = -time.time()
        self._start_preprocessing()
        time_pre += time.time()
        time_total += time_pre

        # obtain pruner for each layer
        time_l = 0
        for ell in self.layers:
            time_l -= time.time()
            self.pruners.append(self._get_pruner(ell))
            self.pruners[ell].compute_probabilities()
            time_l += time.time()

        # check total time for sparsifiers
        time_total += time_l

        # obtain allocator
        time_allocate = -time.time()
        self.allocator = self._get_allocator()
        time_allocate += time.time()
        time_total += time_allocate

        # finish up initialization
        time_post = -time.time()
        self._finish_preprocessing()
        time_post += time.time()
        time_total += time_post

        # print final stats
        print(
            " | ".join(
                [
                    f"Total time: {time_total:4.1f}s",
                    f"Preprocessing time: {time_pre:4.1f}s",
                    f"Sparsifier instantiation time: {time_l:4.1f}s",
                    f"Allocator instantiation time: {time_allocate:4.1f}s",
                    f"Clean up time: {time_post:4.1f}s",
                ]
            )
        )

    def compress(self, keep_ratio, from_original=True, initialize=True):
        """Execute the compression step.

        This is performed in a backwards manner to account for changes in the
        network in the layers following the current layer.

        """
        if from_original:
            self.replace_parameters(self.original_net[0])
        if initialize:
            self._initialize_compression()

        # compute budget
        size = self.original_net[0].size()
        compressible_size = self.original_net[0].compressible_size()
        total_budget = int((keep_ratio - 1) * float(size) + compressible_size)
        total_budget = min(total_budget, compressible_size)

        # compute "effective" sample ratio for compressible part
        kr_compressible = float(total_budget) / float(compressible_size)

        # allocate with "effective" sample ratio
        self.allocator._allocate_budget(kr_compressible, size)

        # loop through the layers in reverse to compress
        for ell in reversed(self.layers):
            # get the pruner and compute probabilities
            pruner = self.pruners[ell]

            # get the sparsifier from a pruner
            sparsifier = self._get_sparsifier(pruner)

            # generate sparsification
            size_pruned = self.allocator.get_num_samples(ell)
            num_samples = pruner.prune(size_pruned)
            weight_hat = self._generate_sparsification(
                num_samples, ell, sparsifier
            )

            if isinstance(weight_hat, tuple):
                # set compression
                self._set_compression(ell, weight_hat[0], weight_hat[1])
            else:
                self._set_compression(ell, weight_hat)

        # "spread" compression across layers for full compression potential
        for ell in self.layers:
            self._propagate_compression(ell)

        # keep track of layer budget (nonzero weights per layer)
        budget_per_layer = [
            module.weight.nonzero().shape[0]
            for module in self.compressed_net.compressible_layers
        ]

        # return stats about compression here
        return budget_per_layer

    def _generate_sparsification(self, num_samples, ell, sparsifier):
        """Generate sparsification for given sample size and layer."""
        return sparsifier.sparsify(num_samples)

    @abstractmethod
    def _set_compression(self, ell, weight_hat, bias=None):
        """Set the compression for the particular layer."""
        raise NotImplementedError

    @abstractmethod
    def _propagate_compression(self, ell):
        """Propagate the compression to other layers for max sparsity.

        This function works differently depending on FilterNet or WeightNet,
        and is thus implemented by the respective child classes.

        """
        raise NotImplementedError

    @abstractmethod
    def _start_preprocessing(self):
        """Execute this at the beginning of _initialize_compression."""
        raise NotImplementedError

    @abstractmethod
    def _get_sparsifier(self, pruner):
        """Get sparsifier corresponding to desired pruner."""
        return NotImplementedError

    @abstractmethod
    def _get_pruner(self, ell):
        """Create and return pruner to parent class (generic interface)."""
        return NotImplementedError

    @abstractmethod
    def _get_allocator(self):
        """Create and return allocator to parent class (generic interface)."""
        return NotImplementedError

    @abstractmethod
    def _finish_preprocessing(self):
        """Finish preprocessing at the end of initialization phase."""
        raise NotImplementedError


class WeightNet(CompressedNet, ABC):
    """This class implements the additional functions for weight pruning.

    This is a good API for weight-based compression/pruning method that follows
    the above API of a CompressedNet.

    """

    def _set_compression(self, ell, W_hat, bias=None):
        module = self.compressed_net.compressible_layers[ell]
        module.weight.data = W_hat
        if bias is not None:
            module.bias.data = bias

    def _propagate_compression(self, ell):
        pass


class FilterNet(CompressedNet, ABC):
    """This class implements the additional functions for filter pruning.

    This is a good API for filter-based compression/pruning method that follows
    the above API of a CompressedNet.

    In particular, this class takes care of "propagating" structured sparsity
    to other layers and contains code for doing that for some of the common
    architectures like ResNet. Structured pruning / Filter pruning can operate
    in two modes.

    These are the potential modes:
    1.) "out" (out_mode==True):
        In this case, we prune dim 0 of current layer (except for last layer!).
        This implies we can also prune dim 1 of next layer.
    2.) "in" (out_mode == False):
        In this case, we prune dim 1 of current layer. This implies we can
        also prune dim 0 of previous layer.
    dim 0: out features
    dim 1: in features

    """

    def _set_compression(self, ell, W_hat, bias=None):
        # short hand
        num_layers = self.compressed_net.num_compressible_layers

        # get the module
        module = self.compressed_net.compressible_layers[ell]

        # sparsify module (cannot do last layer in out_mode ...)
        if not (self.out_mode and ell + 1 >= num_layers):
            module.weight.data.copy_(W_hat)
            if bias is not None:
                module.bias.data.copy_(bias)

    def _propagate_compression(self, ell):
        # get short references to the modules and tensors
        module = self.compressed_net.compressible_layers[ell]
        module_original = self.original_net[0].compressible_layers[ell]
        weight_original = module_original.weight.data
        weight_hat = module.weight.data

        # in out_mode we have to recreate the compression for some layers
        dep_compression = self.compressed_net.compression_source_out[module]
        if self.out_mode and dep_compression is not None:
            dcw = dep_compression.weight
            maskc = dcw.view(dcw.shape[0], -1).sum(1) == 0.0
            weight_hat.copy_(weight_original)
            weight_hat[maskc] = 0.0

        # cannot compress everything in ThiNet
        if (
            self.__class__.__name__ == "ThiNet"
            and self.compressed_net.thi_compressible[module] is False
        ):
            weight_hat.copy_(weight_original)

        # retrieve dependencies
        if self.out_mode:
            deps = self.compressed_net.dependencies_out[module]
            mask = torch.ones(weight_hat.shape[1]).to(weight_hat.device).bool()
        else:
            deps = self.compressed_net.dependencies_in[module]
            mask = torch.ones(weight_hat.shape[0]).to(weight_hat.device).bool()

        # (no dependency means everything is important!!)
        if len(deps) == 0:
            return

        # don't prune out channels in ThiNet if "complicated"
        if (
            self.__class__.__name__ == "ThiNet"
            and self.compressed_net.thi_propagatable[module] is False
        ):
            return

        # construct mask from dependencies
        for dep in deps:
            # check what dep we get
            if isinstance(dep, nn.Module):
                dep_weight = dep.weight
                idx = None
            else:
                dep_weight = dep[0].weight
                idx = dep[1]

            # create mask for this dependency
            if self.out_mode:
                maskd = dep_weight.view(dep_weight.shape[0], -1).sum(1) == 0.0
            else:
                maskd = (
                    dep_weight.view(
                        dep_weight.shape[0], dep_weight.shape[1], -1
                    )
                    .sum(0)
                    .sum(-1)
                )
                maskd = maskd == 0.0

            # combine current mask with full mask
            if idx is None:
                mask = mask & maskd
            elif self.out_mode:
                mask[idx[0] : idx[1]] = maskd
            else:
                mask = mask & maskd[idx[0] : idx[1]]

        # add additional 0's if we had dependencies!
        if self.out_mode:
            weight_hat[:, mask] = 0.0
        else:
            weight_hat[mask] = 0.0

    @property
    @abstractmethod
    def out_mode(self):
        """Return boolean to check whether it's out_mode or inMode."""
        raise NotImplementedError
