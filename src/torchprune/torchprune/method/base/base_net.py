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
import warnings

from scipy import optimize
import torch
import torch.nn as nn

from ...util import tensor


class BaseCompressedNet(ABC, nn.Module):
    """The simplest API for a compression algorithms and the resulting net."""

    @property
    def layers(self):
        """Return iterator for layers."""
        return range(self.compressed_net.num_compressible_layers)

    def __init__(self, original_net):
        """Initialize the compression with a uncompressed network."""
        super().__init__()

        # this is a hack so that pytorch doesn't recognize original_net as a
        # module and thus does not change its device if we call the ".to()#
        # function
        self.original_net = [original_net]
        self.compressed_net = copy.deepcopy(original_net)

        # reset members
        self.reset()

    @property
    def name(self):
        """Return the name of the compression method."""
        return self.compressed_net.name + "_" + self._get_name()

    def _prepare_state_dict_loading(self, state_dict):
        """Prepare compressed net for loading a state_dict.

        Some networks may dynamically change the state_dict() depending on the
        keep_ratio. If we load a checkpoint we need to ensure that the network
        has the proper state_dict(). Otherwise loading fails.

        Args:
            keep_ratio (float): desired keep ratio for the compressed net
        """
        pass

    def _process_loaded_state_dict(self):
        """Process state_dict() that was loaded from checkpoint.

        Some networks may dynamically change the state_dict() after pruning.
        If we load a checkpoint we need to ensure that the network
        is properly compressed.
        """
        pass

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Wrap state_dict loader with pre- and postprocessing step."""
        self._prepare_state_dict_loading(state_dict)
        super().load_state_dict(state_dict, *args, **kwargs)
        self._process_loaded_state_dict()

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

    def __init__(self, original_net, loader_s, loss_handle):
        """Initialize the compression with a uncompressed network."""
        # declare a few members
        self.allocator = None
        self.pruners = None

        # call the super initializer
        super().__init__(original_net)

        # store data loader and loss handle in order to be able to do inference
        # internally if needed.
        self._loader_s = loader_s
        self._loss_handle = loss_handle

    def reset(self):
        """Reset the compresseion and all the associated modules."""
        super().reset()

        # we require an allocator for the network
        del self.allocator
        self.allocator = nn.Module()

        # and pruners for each layer
        del self.pruners
        self.pruners = nn.ModuleList()

    def _prepare_compression(self):
        """Prepare compression step.

        This function is called independent of the type of compression right at
        the beginning of the compression.
        """

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

        Since we might not get the keep_ratio right, we will try to repeat the
        compression until we get it approximately right. This is cheap since
        the expensive part of pruning is the initialization where we compute
        the sensitivities but we can re-use that.
        """
        # do an initial pre-processing step
        self._prepare_compression()

        # check if we start from original, otherwise we need to remember
        # weights!
        if from_original:
            self.replace_parameters(self.original_net[0])
            backup_net = self.original_net[0]
        else:
            backup_net = copy.deepcopy(self.compressed_net)

        # initialize as usual
        if initialize:
            self._initialize_compression()

        # boundaries for binary search over potential keep_ratios
        kr_current = self.compressed_net.size() / self.original_net[0].size()
        kr_min = 0.4 * keep_ratio
        kr_max = max(keep_ratio, 0.999 * kr_current)

        # wrapper for root finding and look-up to speed it up.
        f_opt_lookup = {}

        def _f_opt(kr_compress):
            # check for look-up
            if kr_compress in f_opt_lookup:
                return f_opt_lookup[kr_compress]

            # compress
            b_per_layer = self._compress_once(kr_compress, backup_net)

            # check resulting keep ratio
            kr_actual = (
                self.compressed_net.size() / self.original_net[0].size()
            )
            kr_diff = kr_actual - keep_ratio
            print(f"Current diff in keep ratio is: {kr_diff * 100.0:.2f}%")

            # set to zero if we are already close and stop
            if abs(kr_diff) < 0.005 * keep_ratio:
                kr_diff = 0.0

            # store look-up
            f_opt_lookup[kr_compress] = (kr_diff, b_per_layer)

            return f_opt_lookup[kr_compress]

        # some times the keep ratio is pretty accurate
        # so let's try with the correct keep ratio first
        try:
            # we can either run right away or update the boundaries for the
            # binary search to make it faster.
            kr_diff_nominal, b_per_layer = _f_opt(keep_ratio)
            if kr_diff_nominal == 0.0:
                return b_per_layer
            elif kr_diff_nominal > 0.0:
                kr_max = keep_ratio
            else:
                kr_min = keep_ratio

        except (ValueError, RuntimeError):
            pass

        # run the root search
        # if it fails we simply pick the best value from the look-up table
        try:
            kr_opt = optimize.brentq(
                lambda kr: _f_opt(kr)[0],
                kr_min,
                kr_max,
                maxiter=20,
                xtol=5e-3,
                rtol=5e-3,
                disp=True,
            )
        except (ValueError, RuntimeError):
            kr_diff_opt = float("inf")
            kr_opt = None
            for kr_compress, kr_diff_b_per_layer in f_opt_lookup.items():
                kr_diff = kr_diff_b_per_layer[0]
                if abs(kr_diff) < abs(kr_diff_opt):
                    kr_diff_opt = kr_diff
                    kr_opt = kr_compress
            print(
                "Cannot approximate keep ratio. "
                f"Picking best available keep ratio {kr_opt * 100.0:.2f}% "
                f"with actual diff {kr_diff_opt * 100.0:.2f}%."
            )

        # now run the compression one final time
        return self._compress_once(kr_opt, backup_net)

    def _compress_once(self, keep_ratio, start_net):
        """Execute the compression step starting from given parameters."""
        # replace parameters first
        self.replace_parameters(start_net)

        # compute "available" budget to achieve desired overall keep ratio
        compressible_size = start_net.compressible_size()
        uncompressible_size = start_net.size() - compressible_size
        budget_total = int(keep_ratio * float(self.original_net[0].size()))
        budget_available = budget_total - uncompressible_size

        # some sanity checks for the budget
        budget_available = min(budget_available, compressible_size)
        budget_available = max(0, budget_available)

        # allocate with "available" budget
        self.allocator.allocate_budget(budget_available)

        # loop through the layers in reverse to compress
        for ell in reversed(self.layers):
            # get the pruner and compute probabilities
            pruner = self.pruners[ell]

            # get the sparsifier from a pruner
            sparsifier = self._get_sparsifier(pruner)

            # generate sparsification
            size_pruned = self.allocator.get_num_samples(ell)
            num_samples = pruner.prune(size_pruned)
            weight_hat = sparsifier.sparsify(num_samples)

            if isinstance(weight_hat, tuple):
                # set compression
                self._set_compression(ell, weight_hat[0], weight_hat[1])
            else:
                self._set_compression(ell, weight_hat)

        # "spread" compression across layers for full compression potential
        self._propagate_compression()

        # keep track of layer budget (nonzero weights per layer)
        budget_per_layer = [
            (module.weight != 0.0).sum().item()
            for module in self.compressed_net.compressible_layers
        ]

        # return stats about compression here
        return budget_per_layer

    @abstractmethod
    def _set_compression(self, ell, weight_hat, bias=None):
        """Set the compression for the particular layer."""
        raise NotImplementedError

    def _propagate_compression(self):
        """Propagate the compression to other layers for max sparsity.

        Here we apply a simple heuristic that is true for any kind of pruning:
        * If the gradient of some parameters is consistently 0 across multiple
          batches of data, we can safely prune that parameter as well.
        * e.g.: a bias parameter of a channel that gets never used.

        Depending on FilterNet or WeightNet, we might apply additional
        heuristics to propagate the compression.
        """

        def _zero_grad(net):
            """Set gradients of all parameters back to None."""
            for param in net.parameters():
                param.grad = None

        # zero-out gradients
        _zero_grad(self.compressed_net)

        # get the device
        device = self.compressed_net.compressible_layers[0].weight.device

        # make sure we are in eval mode (avoid updates to BN, etc...)
        is_training_mode = self.compressed_net.training
        self.compressed_net.eval()

        # do a couple of forward+backward passes
        at_least_one_batch = False
        with torch.enable_grad():
            for images, targets in self._loader_s:
                if len(images) < 2:
                    continue
                at_least_one_batch = True
                images = tensor.to(images, device, non_blocking=True)
                targets = tensor.to(targets, device, non_blocking=True)
                outs = self.compressed_net(images)
                loss = self._loss_handle(outs, targets)
                loss.backward()
        assert at_least_one_batch, "No batch with more than one data point!"

        # post-process gradients to set respective weights to zero
        some_grad_none = False
        with torch.no_grad():
            for param in self._parameters_for_grad_prune():
                grad = param.grad
                if grad is None:
                    some_grad_none = True
                    continue

                # mask anything at machine precision or below.
                prune_mask = self._get_prune_mask_from_grad(grad)
                param.masked_fill_(prune_mask, 0.0)

        # issue warning in case some gradients were None
        if some_grad_none:
            warnings.warn(
                "Some parameters did not received gradients"
                " while propagating compression!"
            )

        # zero-out gradients one more time at the end
        _zero_grad(self.compressed_net)

        # revert back to training mode if it was in training mode before
        self.compressed_net.train(is_training_mode)

    @abstractmethod
    def _get_prune_mask_from_grad(self, grad):
        """Get the pruning mask based on the gradients."""

    @abstractmethod
    def _parameters_for_grad_prune(self):
        """Yield params where grad-based pruning heuristic is applicable."""

    @abstractmethod
    def _start_preprocessing(self):
        """Execute this at the beginning of _initialize_compression."""
        raise NotImplementedError

    @abstractmethod
    def _get_sparsifier(self, pruner):
        """Get sparsifier corresponding to desired pruner."""
        raise NotImplementedError

    @abstractmethod
    def _get_pruner(self, ell):
        """Create and return pruner to parent class (generic interface)."""
        raise NotImplementedError

    @abstractmethod
    def _get_allocator(self):
        """Create and return allocator to parent class (generic interface)."""
        raise NotImplementedError

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

    def _get_prune_mask_from_grad(self, grad):
        """Get the pruning mask based on the gradients."""
        return grad.abs() == 0.0

    def _parameters_for_grad_prune(self):
        """Yield params where grad-based pruning heuristic is applicable."""
        yield from self.compressed_net.parameters_without_embedding()


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

    def _get_prune_mask_from_grad(self, grad):
        """Get the pruning mask based on the gradients.

        Prune mask should only consider filter pruning since it is supposed to
        be structured sparsity only!
        """
        out_is_const = grad.view(grad.shape[0], -1).abs().sum(-1) == 0.0
        mask = out_is_const.view((-1,) + (1,) * (grad.dim() - 1))
        return mask

    def _parameters_for_grad_prune(self):
        """Yield params where grad-based pruning heuristic is applicable.

        Grad-based prune heuristic should only consider weights of actual
        compressible layers to ensure structured sparsity.
        """
        for module in self.compressed_net.compressible_layers:
            yield module.weight

    def _propagate_compression(self):
        """Do channel and filter propagation.

        Channel propagation:
        * When channels are zero, the gradient of the corresponding filter
          should be zero.
        * We thus do a couple of forward+backward passes.
        * We then check what filter gradients are zero and mask those filters.
        * --> this is already taken care of with a call to
              super()._propagate_compression()

        Filter propagation:
        * When filters are zero, corresponding channels should be const (not
          zero though because of bias or batchnorm e.g.)
        * We do a couple forward passes with hooks attached to each layer
        * With those hooks we keep track whether filter-wise layer input is
          const.
        * The filters of the corresponding const input channels can be set to 0
        """
        layers = self.compressed_net.compressible_layers
        device = layers[0].weight.device

        # pre-allocate in channel const masks for filter propagation
        in_channel_is_const = [
            torch.ones(mod.weight.shape[1], device=device, dtype=torch.bool)
            for mod in layers
        ]

        # hook-prototype
        def _hook(mod, inputs, outputs, in_is_const):
            ins = inputs[0]
            # flatten all batch dimensions first if it's linear...
            if isinstance(mod, nn.Linear):
                ins = tensor.flatten_all_but_last(ins)
            ins = ins.view(*ins.shape[:2], -1)
            # this trick only works for batch_size > 1 !!
            assert len(ins) > 1, "Min 2 data points for const check needed"
            in_is_const &= ((ins[:1] - ins).abs() == 0.0).all(2).all(0)

        # attach all hooks now
        hook_handles = []
        for mod, in_is_const in zip(layers, in_channel_is_const):
            handle = mod.register_forward_hook(
                lambda mod, ins, outs, is_const=in_is_const: _hook(
                    mod, ins, outs, is_const
                )
            )
            hook_handles.append(handle)

        # do a call to parent function with forward/backward passes
        # This will also zero out parameters that have zero gradient, which
        # will take care of propagation of channel compression
        super()._propagate_compression()

        # disable hooks now
        for hook in hook_handles:
            hook.remove()

        # propagate filter compression now based on values from hooks
        with torch.no_grad():
            for mod, in_is_const in zip(layers, in_channel_is_const):
                mod.weight[:, in_is_const] = 0.0

    @property
    @abstractmethod
    def out_mode(self):
        """Return boolean to check whether it's out_mode or inMode."""
        raise NotImplementedError
