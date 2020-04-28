"""A simple class that returns a random-initialized copy of a net."""
import torchvision.models as imagenet
import provable_pruning.util.models as custom
from provable_pruning.util.net import NetHandle


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

    def get_network(self, pretrained=False):
        """Return the network with the desired index."""
        # retrieve network (pretrained for ImageNet and netIdx==0)
        if "ImageNet" in self.dataset:
            net = getattr(imagenet, self.arch)(
                num_classes=self.output_size, pretrained=pretrained
            )
            # to avoid flatten operation that is incompatible with the
            # _propagate_compression() function in BaseNet.py (doesn't alter
            # network though)
            if isinstance(net, imagenet.VGG):
                # net = self.adapt_vgg(net)
                net = custom.VGGWrapper(net)

        elif "MNIST" in self.dataset:
            net = getattr(custom, self.arch)(
                num_classes=self.output_size, num_in_channels=1
            )
        else:
            net = getattr(custom, self.arch)(num_classes=self.output_size)

        # We don't need the logits ever (they appear in inception_v3)
        if hasattr(net, "aux_logits"):
            net.aux_logits = False
            net.AuxLogits = None

        # put in NetHandle
        net = NetHandle(net, self.network_name)
        net.eval()

        return net
