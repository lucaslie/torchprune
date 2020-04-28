"""A generic wrapper for any pytorch module with improved functionality."""
import torch.nn as nn


class NetHandle(nn.Module):
    """Wrapper class to handle all network specific stuff.

    This class takes "generic" neural networks (e.g. downloaded resnets or
    our self-configured NNs) and adds additional methods on top to enhance
    the usability.
    """

    def __init__(self, torchnet, name=None):
        """Initialize the class from a vanilla pytorch net.

        Args:
            torchnet (nn.Module): any pytorch module/network
            name (str, optional): name of the net. Defaults to None.

        """
        super(NetHandle, self).__init__()
        self.torchnet = torchnet

        self.name = name

        self.compressible_layers = []
        self.num_weights = []
        self.num_patches = []
        self.num_etas = []

        # dependencies
        # dict of lists containing dependencies_in
        self.dependencies_in = {}
        self.dependencies_out = {}
        self.compression_source_out = {}

        # this is for ThiNet
        # one dependency for out channels --> propagate
        # one dependency for in channels --> prune
        self.thi_compressible = {}
        self.thi_propagatable = {}

        if "ResNet" in torchnet._get_name():
            self._init_resnet()
        elif torchnet._get_name() == "DenseNet":
            self._init_densenet()
        else:
            self._init_sequential()

        self.num_compressible_layers = len(self.compressible_layers)

        # convert list to nn.Sequential
        self.compressible_layers = nn.Sequential(*self.compressible_layers)

    def size(self):
        """Get total number of parameters in network."""
        nonzeros = 0
        for param in self.parameters():
            if param is not None:
                nonzeros += param.nonzero().shape[0]
        return nonzeros

    def flops(self):
        """Get total number of FLOPs in network."""
        flops = 0
        if len(self.num_patches) == self.num_compressible_layers:
            for ell, module in enumerate(self.compressible_layers):
                flops += (
                    module.weight.nonzero().shape[0] * self.num_patches[ell]
                )
        return flops

    def compressible_size(self):
        """Get total number of parameters that are compressible."""
        nonzeros = 0
        for module in self.compressible_layers:
            nonzeros += module.weight.nonzero().shape[0]
        return nonzeros

    def forward(self, x):
        """Pytorch forward function."""
        # have to use forward() at least once to cache etas
        if len(self.num_patches) == 0:
            self._cache_etas(x)
        # return regular output
        return self.torchnet(x)

    def register_sparsity_pattern(self):
        """Register the sparsity mask in each layer."""
        param_iter = self.compressible_layers.parameters()
        for idx, param in enumerate(param_iter):
            name = f"param{idx}"
            # TODO: remove byte() once NCCL for multi-GPU training supports
            # bools. For more information, see:
            # https://github.com/pytorch/pytorch/issues/24137
            pattern = (param.data == 0.0).byte()
            if name in self._buffers:
                self._buffers[name] = pattern
            else:
                self.register_buffer(name, pattern)

    def enforce_sparsity(self):
        """Enforce the stored sparsity mask in each layer."""
        param_iter = self.compressible_layers.parameters()
        for idx, param in enumerate(param_iter):
            key = f"param{idx}"
            # TODO: remove bool when above is fixed!
            param.data.masked_fill_(self._buffers[key].bool(), 0.0)

    def _cache_etas(self, x):
        """Cache etas and patches to store."""
        # storage for etas and num_patches
        self.num_etas = []
        self.num_patches = []

        # define a hook function to store values
        def hook_fun(module, input, output):
            if isinstance(module, nn.Linear):
                self.num_etas.append(output.data.shape[-1])
                self.num_patches.append(1)
            else:
                self.num_etas.append(output.data[0].numel())
                self.num_patches.append(output.data[0, 0].numel())

        # attach hooks
        handles = []
        for module in self.compressible_layers:
            handles.append(module.register_forward_hook(hook_fun))

        # call forward function of torchnet (self(x) would be an infinite loop)
        self.torchnet(x)

        # remove hooks
        for handle in handles:
            handle.remove()

    def _init_sequential(self):
        """Initialize a network that is purely sequential."""
        for module in self.torchnet.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                self.append_layer(module)

        num_compressible_layers = len(self.compressible_layers)
        for ell, module in enumerate(self.compressible_layers):
            # check out dependencies
            if ell - 1 >= 0:
                self.dependencies_out[module] = [
                    self.compressible_layers[ell - 1]
                ]
            else:
                self.dependencies_out[module] = []

            # check in dependencies
            if ell + 1 < num_compressible_layers:
                self.dependencies_in[module] = [
                    self.compressible_layers[ell + 1]
                ]
            else:
                self.dependencies_in[module] = []

            #  no compression dependencies
            self.compression_source_out[module] = None

            # always thi compressible/propagatable
            self.thi_compressible[module] = True
            self.thi_propagatable[module] = True

    def append_layer(self, module):
        """Append current module to the list of compressible layers."""
        self.compressible_layers.append(module)
        self.num_weights.append(module.weight.data.numel())

    def _init_resnet(self):
        """Initialize a Resnet-like network.

        we build a dependency map for ResNets; In Resnets we cannot simply
        cancel the outFeatures of the next layer, but have to consider when
        the feature is used later in the skip connections.
        Resnets consist of conv1, layer1-3, fc
        Each layeri contains multiple blocks (conv1 + conv2 (+conv3) (
        +downsample)). For each block:
            * conv1 only needs to compute features of the corresponding conv2
            * conv2 (or conv3) needs to compute all features of subsequent
              conv1's until there is a downsample (which features we also need)
            * downsample is like a conv2 ( we need all features)

        So for each type of conv we store a dependency pointer:
            * conv1: simply a pointer to conv2
            * conv2 (or conv3): a list index. The dependencies_in are all list
            items from list index until we hit a downsample in the list
            * downsample: like conv2

        If there is a conv3 in the block, conv3 behaves like conv2,
        and conv2 behaves like conv1.
        """
        resnet = self.torchnet
        layers = [resnet.layer1, resnet.layer2, resnet.layer3]
        if hasattr(resnet, "layer4"):
            layers.append(resnet.layer4)

        dep_in_list = []
        dep_in_lookup = {}
        dep_out = {}
        comp_source = {}

        self.append_layer(resnet.conv1)
        dep_in_lookup[resnet.conv1] = len(dep_in_list)
        dep_out[resnet.conv1] = None
        comp_source[resnet.conv1] = None

        # should be thi compressible
        self.thi_compressible[resnet.conv1] = False
        self.thi_propagatable[resnet.conv1] = False

        # last downsample layer
        last_down = resnet.conv1

        # build up lists
        for layer in layers:
            for block in layer:

                # store conv1 and its dependencies
                self.append_layer(block.conv1)
                dep_in_list.append((True, block.conv1))
                # depInList.append((True, str(l+1)+'.'+str(b)+'.1'))
                dep_in_lookup[block.conv1] = block.conv2
                # depInLookup[block.conv1] = str(l+1)+'.'+str(b)+'.2'
                dep_out[block.conv1] = self.compressible_layers[-2]
                comp_source[block.conv1] = None

                # has more in dependency, thus not compressible
                self.thi_compressible[block.conv1] = False
                self.thi_propagatable[block.conv1] = True

                # store conv2 + dependencies_in (and conv3 if it exists)
                self.compressible_layers.append(block.conv2)
                dep_out[block.conv2] = block.conv1
                if block._get_name() == "Bottleneck":
                    self.append_layer(block.conv3)
                    dep_in_lookup[block.conv2] = block.conv3
                    final_block = block.conv3
                    dep_out[block.conv3] = block.conv2
                    comp_source[block.conv2] = None

                    # definitely thi compressible in this case
                    self.thi_compressible[block.conv2] = True
                    self.thi_propagatable[block.conv2] = True
                else:
                    final_block = block.conv2

                # final block is also thi compressible
                self.thi_compressible[final_block] = True
                self.thi_propagatable[final_block] = False

                # take care of downsample if it exists
                if block.downsample is not None:
                    self.append_layer(block.downsample[0])
                    dep_in_list.append((False, block.downsample[0]))
                    # depInList.append((False, str(l+1)+'.'+str(b)+'.d'))
                    dep_in_lookup[block.downsample[0]] = len(dep_in_list)
                    dep_out[block.downsample[0]] = dep_out[block.conv1]
                    comp_source[block.downsample[0]] = None
                    last_down = block.downsample[0]

                    # nothing
                    self.thi_compressible[block.downsample[0]] = False
                    self.thi_propagatable[block.downsample[0]] = False

                # only now set dependency of final block (depends on
                # downsample existence)
                comp_source[final_block] = last_down
                dep_in_lookup[final_block] = len(dep_in_list)

        self.append_layer(resnet.fc)
        dep_in_list.append((False, resnet.fc))
        dep_in_lookup[resnet.fc] = None
        dep_out[resnet.fc] = self.compressible_layers[-2]
        comp_source[resnet.fc] = None

        # no
        self.thi_compressible[resnet.fc] = False
        self.thi_propagatable[resnet.fc] = False

        # set compression sources for the modules
        self.compression_source_out = comp_source

        # finally convert it into the desired format
        for module in self.compressible_layers:
            dependency_in = []
            lookup = dep_in_lookup[module]
            if isinstance(lookup, int):
                idx = lookup
                more_dependencies = True
                while more_dependencies:
                    dependency_in.append(dep_in_list[idx][1])
                    more_dependencies = dep_in_list[idx][0]
                    idx += 1
            elif lookup is not None:
                dependency_in.append(lookup)

            self.dependencies_in[module] = dependency_in

            if dep_out[module] is None:
                self.dependencies_out[module] = []
            else:
                self.dependencies_out[module] = [dep_out[module]]

    def _init_densenet(self):
        """Initialize a Densenet-like network."""
        densenet = self.torchnet
        layers = [densenet.dense1, densenet.dense2, densenet.dense3]
        trans = [densenet.trans1.conv1, densenet.trans2.conv1, densenet.fc]
        dep_in_list = []
        dep_in_lookup = {}
        dep_out_list = []
        dep_out_lookup = {}

        # store
        self.append_layer(densenet.conv1)

        # in
        dep_in_lookup[densenet.conv1] = len(dep_in_list)

        # out
        dep_out_lookup[densenet.conv1] = None
        dep_out_list.append((False, densenet.conv1))

        # build up lists
        for ell, layer in enumerate(layers):
            # store dependencies for dense part
            for block in layer:

                # store conv1
                self.append_layer(block.conv1)

                # conv1: in
                dep_in_list.append((True, block.conv1))
                dep_in_lookup[block.conv1] = block.conv2

                # conv1: out
                dep_out_lookup[block.conv1] = len(dep_out_list)

                # conv2: storing
                self.compressible_layers.append(block.conv2)

                # conv2: in
                dep_in_lookup[block.conv2] = len(dep_in_list)

                # conv2: out
                dep_out_lookup[block.conv2] = block.conv1
                dep_out_list.append((True, block.conv2))

            # trans: storing
            self.append_layer(trans[ell])

            # trans: in
            dep_in_list.append((False, trans[ell]))
            dep_in_lookup[trans[ell]] = len(dep_in_list)

            # trans: out
            dep_out_lookup[trans[ell]] = len(dep_out_list)
            dep_out_list.append((False, trans[ell]))

        # fc: in (correction)
        dep_in_lookup[densenet.fc] = None

        # finally convert it into the desired format
        for module in self.compressible_layers:
            dependency_in = []
            lookup = dep_in_lookup[module]
            if isinstance(lookup, int):
                idx = lookup
                more_dependencies = True
                # we also need to store indices in this case which "slice" we
                # want
                i_finish = dep_in_list[idx][1].weight.shape[1]
                i_start = i_finish - module.weight.shape[0]
                while more_dependencies:
                    dependency_in.append(
                        (dep_in_list[idx][1], (i_start, i_finish))
                    )
                    more_dependencies = dep_in_list[idx][0]
                    idx += 1
            elif lookup is not None:
                dependency_in.append(lookup)

            self.dependencies_in[module] = dependency_in
            self.compression_source_out[module] = None

            # no strategy for DenseNet, but let's see.
            self.thi_compressible[module] = True
            self.thi_propagatable[module] = True

        # finally convert it into the desired format
        for module in self.compressible_layers:
            dependency_out = []
            lookup = dep_out_lookup[module]
            if isinstance(lookup, int):
                idx = lookup - 1
                more_dependencies = True
                i_start = module.weight.shape[1]
                while more_dependencies:
                    # we also need to store indices in this case which "slice"
                    # we want
                    i_finish = i_start
                    i_start = i_finish - dep_out_list[idx][1].weight.shape[0]
                    dependency_out.append(
                        (dep_out_list[idx][1], (i_start, i_finish))
                    )
                    more_dependencies = dep_out_list[idx][0]
                    idx -= 1
            elif lookup is not None:
                dependency_out.append(lookup)

            self.dependencies_out[module] = dependency_out
