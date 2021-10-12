"""Module containing the base net for decomposition implementations."""

from abc import abstractmethod, ABC
import torch
import torch.nn as nn

from ..base import CompressedNet, TensorPruner2
from .base_decompose_sparsifier import GroupedDecomposeSparsifier
from . import base_decompose_util as modutil


class BaseDecomposeNet(CompressedNet, ABC):
    """Decomposition-based compression for any weight layer."""

    @property
    @abstractmethod
    def _allocator_type(self):
        """Get allocator type."""

    @property
    @abstractmethod
    def _sparsifier_type(self):
        """Get sparsifier type."""

    @property
    @abstractmethod
    def _k_split(self):
        """Get number of k splits in each layer."""

    @property
    def deterministic(self):
        """Return the indicator for deterministic compression."""
        # it is technically not deterministic since the method to approximate
        # the optimal solution might *not* be
        return True

    @property
    def retrainable(self):
        """Return the indicator whether we can retrain afterwards."""
        return True

    def _prepare_state_dict_loading(self, state_dict):
        """Prepare compressed net for loading a state_dict."""
        # reset pruning parameters
        self.reset()

        # reset state_dict ...
        self._remove_projection()

        def _get_module_name(module):
            """Look up module name."""
            for k, mod in self.compressed_net.torchnet.named_modules():
                if mod is module:
                    return f"compressed_net.torchnet.{k}"
            return None

        # set fake compression so that self.state_dict() has correct keys
        for ell in self.layers:
            # extract module
            module = self.compressed_net.compressible_layers[ell]
            mod_name = _get_module_name(module)

            # check if this module needs to replaced with projected module
            if f"{mod_name}.weight" in state_dict:
                # double-check if we need to set bias
                bias_key = f"{mod_name}.bias"
                if bias_key in state_dict:
                    device = module.weight.device
                    module.bias = nn.Parameter(state_dict[bias_key].to(device))
                else:
                    module.bias = None
                # then move on.
                continue

            # get number of groups
            num_groups = state_dict[f"{mod_name}.groups_enc"].item()

            # get scheme
            try:
                scheme_val = state_dict[f"{mod_name}.scheme_value"].item()
                scheme = modutil.FoldScheme(scheme_val)
            except KeyError:
                scheme = modutil.FoldScheme.get_default()

            # check that groups agree now.
            try:
                assert module.groups == 1, "Groups need to be exactly 1!"
            except AttributeError:
                pass

            # fake sparsify with simplest sparsifier
            rank_j = 1
            pruner = BaseDecomposeNet._get_pruner(self, ell)
            sparsifier = GroupedDecomposeSparsifier(pruner)
            weights_hat = sparsifier.sparsify(
                torch.tensor([rank_j, num_groups, scheme.value])
            )
            # check if we also need to set bias
            bias_key = f"{mod_name}.decoding.bias"
            if f"{mod_name}.decoding.bias" in state_dict:
                bias = state_dict[bias_key]
            else:
                module.bias = None
                bias = None

            # set total compression now
            self._set_compression(ell, weights_hat, bias=bias)

        # now "propagate" compression to nethandle
        self._propagate_compression(cache_etas=False)

        # go through all keys and make sure dimensions correspond to the ones
        # from the state_dict
        for key, dict_param in state_dict.items():
            if key not in self.state_dict():
                continue
            curr_param = modutil.get_attr(self, key.split("."))
            if curr_param.shape != dict_param.shape:
                new_param = torch.zeros_like(dict_param).to(curr_param.device)
                if isinstance(curr_param, nn.Parameter):
                    new_param = nn.Parameter(new_param)
                # Set the new parameter now
                modutil.set_attr(self, key.split("."), new_param)

    def _process_loaded_state_dict(self):
        """Process state_dict() that was loaded from checkpoint."""
        for mod in self.compressed_net.compressible_layers:
            # set in, out channels/features correctly ...
            f_name = "features" if isinstance(mod, nn.Linear) else "channels"
            groups = mod.groups if hasattr(mod, "groups") else 1
            setattr(mod, f"in_{f_name}", mod.weight.shape[1] * groups)
            setattr(mod, f"out_{f_name}", mod.weight.shape[0])

        # propagate compression one final time to ensure everything is correct
        self._propagate_compression()

    def replace_parameters(self, other_net):
        """Reset the compression and the underlying decomposition."""
        # delete decomposition and reset forward functions.
        self._remove_projection()

        # now replace parameters ...
        super().replace_parameters(other_net)

    def _remove_projection(self):
        """Undo projection step-by-step.

        The network is still low-rank and we re-compose the low-rank parameters
        into one weight tensor.
        """
        just_undone = True
        while just_undone:
            just_undone = False
            # we go through modules one-by-one
            torchnet = self.compressed_net.torchnet
            for name, module in torchnet.named_modules():
                if isinstance(module, modutil.ProjectedModule):
                    module_original = module.get_original_module()
                    modutil.set_attr(
                        torchnet, name.split("."), module_original
                    )
                    just_undone = True
                    # break after one change so that named_modules() is never
                    # wrong ...
                    break

        # now "propagate" removed compression to nethandle
        self._propagate_compression()

    def _prepare_compression(self):
        """Prepare compression step by removing projection."""
        self._remove_projection()

    def _start_preprocessing(self):
        pass

    def _finish_preprocessing(self):
        pass

    def _get_allocator(self):
        return self._allocator_type(self.compressed_net, self._k_split)

    def _get_pruner(self, ell):
        """Return a fake pruner since we don't need a pruner."""
        module = self.compressed_net.compressible_layers[ell]
        return TensorPruner2(module.weight, module.weight.data.abs())

    def _get_sparsifier(self, pruner):
        return self._sparsifier_type(pruner)

    def _set_compression(self, ell, weight_hat, bias=None):
        """Set the compression by replacing the module."""
        # check if we should even compress...
        if len(weight_hat) < 1:
            return

        # retrieve module
        module = self.compressed_net.compressible_layers[ell]

        # set bias first if provided
        if bias is not None:
            module.bias = nn.Parameter(bias)

        # get projected module
        if isinstance(module, nn.Linear):
            module_projected = modutil.ProjectedLinear(module, weight_hat)
        elif isinstance(module, nn.Conv2d):
            module_projected = modutil.ProjectedConv2d(module, weight_hat)
        else:
            raise AssertionError("Unsupported module.")

        # set projected module
        for name, net_module in self.compressed_net.torchnet.named_modules():
            if module is net_module:
                modutil.set_attr(
                    self.compressed_net.torchnet,
                    name.split("."),
                    module_projected,
                )
                break

    def _propagate_compression(self, cache_etas=True):
        # re-register compressible layers
        self.compressed_net.register_compressible_layers()

        # re-cache etas now
        if cache_etas:
            self.compressed_net.cache_etas()

    def _get_prune_mask_from_grad(self, grad):
        """Get the pruning mask based on the gradients."""
        raise RuntimeError("This should not be called for decomposition nets.")

    def _parameters_for_grad_prune(self):
        """Yield params where grad-based pruning heuristic is applicable."""
        raise RuntimeError("This should not be called for decomposition nets.")
