"""A generic wrapper for any pytorch module with improved functionality."""
import torch.nn as nn


class NetHandle(nn.Module):
    """Wrapper class to handle all network specific stuff.

    This class takes "generic" neural networks (e.g. downloaded resnets or
    our self-configured NNs) and adds additional methods on top to enhance
    the usability.
    """

    @property
    def num_compressible_layers(self):
        """Return the number of compressible layers."""
        return len(self.compressible_layers)

    def __init__(self, torchnet, name=None):
        """Initialize the class from a vanilla pytorch net.

        Args:
            torchnet (nn.Module): any pytorch module/network
            name (str, optional): name of the net. Defaults to None.

        """
        super(NetHandle, self).__init__()
        self.torchnet = torchnet

        if name is None:
            self.name = type(torchnet).__name__
        else:
            self.name = name

        self.compressible_layers = nn.ModuleList()
        self.num_weights = []
        self.num_patches = []
        self.num_etas = []

        # keep one x around to store patches if not there
        self._x_for_patches = None

        # register compressible layers now
        self.register_compressible_layers()

    def reset(self):
        """Reset stats of the network."""
        self.compressible_layers = nn.ModuleList()
        self.num_weights = []
        self.num_patches = []
        self.num_etas = []

    def register_compressible_layers(self):
        """Register compressible layers now."""
        # reset stats
        self.reset()

        # then register linear and conv layers.
        for module in self.torchnet.modules():
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                continue
            if hasattr(module, "groups") and module.groups > 1:
                continue
            if hasattr(self.torchnet, "is_compressible") and not self.torchnet.is_compressible(
                module
            ):
                continue
            self.compressible_layers.append(module)
            self.num_weights.append(module.weight.data.numel())

    def size(self):
        """Get total number of parameters in network."""
        nonzeros = 0
        for param in self.parameters():
            if param is not None:
                nonzeros += (param != 0.0).sum().item()
        return nonzeros

    def flops(self):
        """Get total number of FLOPs in network."""
        flops = 0
        if len(self.num_patches) == self.num_compressible_layers:
            for ell, module in enumerate(self.compressible_layers):
                flops += (module.weight != 0.0).sum().item() * self.num_patches[ell]
        return flops

    def compressible_size(self):
        """Get total number of parameters that are compressible."""
        nonzeros = 0
        for module in self.compressible_layers:
            nonzeros += (module.weight != 0.0).sum().item()
        return nonzeros

    def forward(self, x):
        """Pytorch forward function."""
        # have to use forward() at least once to cache etas
        if len(self.num_patches) == 0:
            self.cache_etas(x)
        # return regular output
        return self.torchnet(x)

    def parameters_without_embedding(self):
        """Yield all parameters but skip embedding parameters."""
        # collect embedding parameters first
        params_embedding = []
        for module in self.torchnet.modules():
            if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
                for param in module.parameters():
                    params_embedding.append(param)

        # yield parameters but skip over embedding parameters
        for param in self.parameters():
            if any(param is p_e for p_e in params_embedding):
                continue
            yield param

    def register_sparsity_pattern(self):
        """Register the sparsity mask in each layer."""
        param_iter = self.compressible_layers.parameters()
        for idx, param in enumerate(param_iter):
            name = f"param{idx}"
            # TODO: remove byte() once NCCL for multi-GPU training supports
            # bools. For more information, see:
            # https://github.com/pytorch/pytorch/issues/24137
            pattern = (param.data == 0.0).byte()
            if hasattr(self, name):
                setattr(self, name, pattern)
            else:
                self.register_buffer(name, pattern)

    def enforce_sparsity(self):
        """Enforce the stored sparsity mask in each layer."""
        param_iter = self.compressible_layers.parameters()
        for idx, param in enumerate(param_iter):
            key = f"param{idx}"
            # TODO: remove bool when above is fixed!
            param.data.masked_fill_(getattr(self, key).bool(), 0.0)

    def cache_etas(self, x=None):
        """Cache etas and patches to store."""
        # check for x to use
        if self._x_for_patches is None:
            if x is None:
                return
            else:
                self._register_x_for_patches(x)

        # storage for etas and num_patches
        num_etas = {mod: 0 for mod in self.compressible_layers}
        num_patches = {mod: 0 for mod in self.compressible_layers}

        # define a hook function to store values
        def hook_fun(module, input, output):
            if isinstance(module, nn.Linear):
                num_etas_this = output.data.shape[-1]
                num_patches_this = output[0][..., 0].numel()
            else:
                num_etas_this = output.data[0].numel()
                num_patches_this = output.data[0, 0].numel()

            # now add to dictionary
            num_etas[module] = max(num_etas[module], num_etas_this)
            num_patches[module] += num_patches_this

        # attach hooks
        handles = []
        for module in self.compressible_layers:
            handles.append(module.register_forward_hook(hook_fun))

        # call forward function of torchnet (self(x) would be an infinite loop)
        self.torchnet(self._get_x_for_patches())

        # remove hooks
        for handle in handles:
            handle.remove()

        # process etas and patches
        self.num_etas = []
        self.num_patches = []
        for module in self.compressible_layers:
            self.num_etas.append(num_etas[module])
            self.num_patches.append(num_patches[module])

    def _get_x_for_patches(self):
        """Put together x for patches."""
        if isinstance(self._x_for_patches, dict):
            x_for_patches = {}
            for key_x, key_self in self._x_for_patches.items():
                x_for_patches[key_x] = getattr(self, key_self)
            return x_for_patches
        else:
            return self._x_for_patches

    def _register_x_for_patches(self, x):
        """Register x as buffer where x could be Tensor or dict."""
        # delete x for patches to start fresh
        del self._x_for_patches

        def _register_buffer(key_add, x_value):
            """Register with additional key add and return key."""
            key_full = "_x_for_patches"
            if key_add is not None:
                key_full += f"_{key_add}"
            self.register_buffer(key_full, x_value[:1].detach().clone())
            return key_full

        if isinstance(x, dict):
            self._x_for_patches = {}
            for x_key, x_tensor in x.items():
                key_full = _register_buffer(x_key, x_tensor)
                self._x_for_patches[x_key] = key_full
        else:
            _register_buffer(None, x)
