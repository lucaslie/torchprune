# Provable Pruning of Neural Networks using Sensitivity
[Lucas Liebenwein*](http://www.mit.edu/~lucasl/), 
[Cenk Baykal*](http://www.mit.edu/~baykal/), 
[Igor Gilitschenski](https://www.gilitschenski.org/igor/), 
[Harry Lang](https://www.csail.mit.edu/person/harry-lang), 
[Dan Feldman](http://people.csail.mit.edu/dannyf/),
[Daniela Rus](http://danielarus.csail.mit.edu/)

Implementation of provable pruning using sensitivity as introduced in  [SiPPing
Neural Networks: Sensitivity-informed Provable Pruning of Neural Networks](https://arxiv.org/abs/1910.05422)
(weight pruning) and [Provable Filter Pruning for Efficient Neural
Networks](https://arxiv.org/abs/1911.07412) (filter pruning). These algorithm
rely on a notion of sensitivity (the product of the data and the weight) to
provably quantify the error introduced by pruning. 

This package contains our pruning methods as well as comparison pruning methods
mplemented in PyTorch.

## Setup
Run
```sh
pip install -e .
```
to install the `provable_pruning` package.

## Usage
To use this package follow the below workflow. In this example, we prune a
`Resnet20` with both our weight pruning method (`SiPP`) and filter pruning
method (`PFP`).
```python
# import the required packages
import torchvision
import torch
import provable_pruning as pp
import provable_pruning.util.net as util_net
import provable_pruning.util.models as util_models

# initialize the network and wrap it into the NetHandle class
net = util_models.resnet20()
net = util_net.NetHandle(net)

# initialize a data loader with a limited number of points
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        ),
    ]
)

testset = torchvision.datasets.CIFAR10(
    root="./local", train=False, download=True, transform=transform
)

size_s = 128
testset, set_s = torch.utils.data.random_split(
    testset, [len(testset) - size_s, size_s]
)

loader_s = torch.utils.data.DataLoader(set_s, batch_size=32, shuffle=False)

# Prune weights on the CPU
print("\n===========================")
print("Pruning weights with SiPP")
net_weight_pruned = pp.SiPPNet(net, loader_s)
net_weight_pruned.compress(keep_ratio=0.5)
print(
    f"The network has {net_weight_pruned.size()} parameters and "
    f"{net_weight_pruned.flops()} FLOPs left."
)
print("===========================")

# Prune filters on the GPU
print("\n===========================")
print("Pruning filters with PFP.")
net_filter_pruned = pp.PFPNet(net, loader_s)
net_filter_pruned.cuda()
net_filter_pruned.compress(keep_ratio=0.5)
net_filter_pruned.cpu()
print(
    f"The network has {net_filter_pruned.size()} parameters and "
    f"{net_filter_pruned.flops()} FLOPs left."
)
print("===========================")

# You can now retrain the network and repeat the procedure as desired...
```