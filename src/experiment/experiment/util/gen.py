"""A simple class that returns a random-initialized copy of a net."""
import torchprune.util.models as models
from torchprune.util.net import NetHandle


class NetGen(object):
    """This class retrieves the correct network for us."""

    def __init__(self, output_size, dataset, net_name, arch):
        """Initialize the class with the parameter dict."""
        # retrieve output size
        self.output_size = output_size
        self.dataset = dataset
        self.network_name = net_name
        self.arch = arch

    def __call__(self):
        """Return the network with the desired index."""
        return self.get_network()

    def get_network(self):
        """Return the network with the desired index."""
        # get right module and kwargs for generating the network
        if "ImageNet" in self.dataset:
            model_module = models.imagenet
        else:
            model_module = models

        kwargs = {"num_classes": self.output_size}
        if "MNIST" in self.dataset:
            kwargs["num_in_channels"] = 1

        # retrieve network
        net = getattr(model_module, self.arch)(**kwargs)

        # We don't need the logits ever (they appear in inception_v3)
        if hasattr(net, "aux_logits"):
            net.aux_logits = False
            net.AuxLogits = None

        # put in NetHandle
        net = NetHandle(net, self.network_name)
        net.eval()

        return net
