# torchprune Package

This package contains implementations for various pruning methods to prune,
retrain, and test arbitrary neural networks on various data sets. Check out the
sections below to find out more about which pruning methods, networks, and data
sets are implemented. There is also descriptions on how to add your
implementations on top of the existing implementation. 

## Setup
Run
```bash
pip install -e .
```
to install the `torchprune` package.

Check out the [main README](../../README.md) for more info as well.

## Usage
To use this package follow the below workflow. In this example, we prune a
`Resnet20` with both our weight pruning method (`SiPP`) and filter pruning
method (`PFP`).
```python
# %% import the required packages
import os
import copy
import torch
import torchvision
import torchprune as tp

# %% initialize the network and wrap it into the NetHandle class
net_name = "resnet20_CIFAR10"
net = tp.util.models.resnet20()
net = tp.util.net.NetHandle(net, net_name)

# %% Setup some stats to track results and retrieve checkpoints
n_idx = 0  # network index 0
keep_ratio = 0.5  # Ratio of parameters to keep
s_idx = 0  # keep ratio's index
r_idx = 0  # repetition index

# %% initialize data loaders with a limited number of points
transform_train = [
    torchvision.transforms.Pad(4),
    torchvision.transforms.RandomCrop(32),
    torchvision.transforms.RandomHorizontalFlip(),
]
transform_static = [
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    ),
]


testset = torchvision.datasets.CIFAR10(
    root="./local",
    train=False,
    download=True,
    transform=tp.util.transforms.SmartCompose(transform_static),
)

trainset = torchvision.datasets.CIFAR10(
    root="./local",
    train=True,
    download=True,
    transform=tp.util.transforms.SmartCompose(
        transform_train + transform_static
    ),
)

size_s = 128
batch_size = 128
testset, set_s = torch.utils.data.random_split(
    testset, [len(testset) - size_s, size_s]
)

loader_s = torch.utils.data.DataLoader(set_s, batch_size=32, shuffle=False)
loader_test = torch.utils.data.DataLoader(
    testset, batch_size=batch_size, shuffle=False
)
loader_train = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=False
)

# %% Setup trainer
# Set up training parameters
train_params = {
    # any loss and corresponding kwargs for __init__ from tp.util.nn_loss
    "loss": "CrossEntropyLoss",
    "lossKwargs": {"reduction": "mean"},
    # exactly two metrics with __init__ kwargs from tp.util.metrics
    "metricsTest": [
        {"type": "TopK", "kwargs": {"topk": 1}},
        {"type": "TopK", "kwargs": {"topk": 5}},
    ],
    # any optimizer from torch.optim with corresponding __init__ kwargs
    "optimizer": "SGD",
    "optimizerKwargs": {
        "lr": 0.1,
        "weight_decay": 1.0e-4,
        "nesterov": False,
        "momentum": 0.9,
    },
    # batch size
    "batchSize": batch_size,
    # desired number of epochs
    "startEpoch": 0,
    "retrainStartEpoch": -1,
    "numEpochs": 5,  # 182
    # any desired combination of lr schedulers from tp.util.lr_scheduler
    "lrSchedulers": [
        {
            "type": "MultiStepLR",
            "stepKwargs": {"milestones": [91, 136]},
            "kwargs": {"gamma": 0.1},
        },
        {"type": "WarmupLR", "stepKwargs": {"warmup_epoch": 5}, "kwargs": {}},
    ],
    # output size of the network
    "outputSize": 10,
    # directory to store checkpoints
    "dir": os.path.realpath("./checkpoints"),
}

# Setup retraining parameters (just copy train-parameters)
retrain_params = copy.deepcopy(train_params)

# Setup trainer
trainer = tp.util.train.NetTrainer(
    train_params=train_params,
    retrain_params=retrain_params,
    train_loader=loader_train,
    test_loader=loader_test,
    valid_loader=loader_s,
    num_gpus=1,
)

# get a loss handle
loss_handle = trainer.get_loss_handle()

# %% Pre-train the network
trainer.train(net, n_idx)

# %% Prune weights on the CPU

print("\n===========================")
print("Pruning weights with SiPP")
net_weight_pruned = tp.SiPPNet(net, loader_s, loss_handle)
net_weight_pruned.compress(keep_ratio=keep_ratio)
print(
    f"The network has {net_weight_pruned.size()} parameters and "
    f"{net_weight_pruned.flops()} FLOPs left."
)
print("===========================")

# %% Prune filters on the GPU
print("\n===========================")
print("Pruning filters with PFP.")
net_filter_pruned = tp.PFPNet(net, loader_s, loss_handle)
net_filter_pruned.cuda()
net_filter_pruned.compress(keep_ratio=keep_ratio)
net_filter_pruned.cpu()
print(
    f"The network has {net_filter_pruned.size()} parameters and "
    f"{net_filter_pruned.flops()} FLOPs left."
)
print("===========================")

# %% Retrain the filter-pruned network now.

# Retrain the filter-pruned network now on the GPU
net_filter_pruned = net_filter_pruned.cuda()
trainer.retrain(net_filter_pruned, n_idx, keep_ratio, s_idx, r_idx)

# %% Test at the end
print("\nTesting on test data set:")
loss, acc1, acc5 = trainer.test(net_filter_pruned)
print(f"Loss: {loss:.4f}, Top-1 Acc: {acc1*100:.2f}%, Top-5: {acc5*100:.2f}%")

# Put back to CPU
net_filter_pruned = net_filter_pruned.cpu()
```

## Pruning methods
This package contains multiple pruning methods that are already implemented and
also abstract interfaces to add your own pruning methods. 

### Implemented pruning methods

#### **Provable Filter Pruning (PFP)** (Ours)
* Paper: 
  [Provable Filter Pruning for Efficient Neural
  Networks](https://arxiv.org/abs/1911.07412)
* Code: [torchprune/method/pfp](./torchprune/method/pfp)

#### **Sensitivity-informed Provable Pruning (SiPP)** (Ours)
* Paper: [SiPPing
  Neural Networks: Sensitivity-informed Provable Pruning of Neural
  Networks](https://arxiv.org/abs/1910.05422)
* Code: [torchprune/method/sipp](./torchprune/method/sipp)


#### **Fake-pruned ReferenceNet**
* Fake-pruned network that spoofs pruning and can be used to compare the
  unpruned network with pruned networks. It is also being recognized by 
  the codebase as "fake-pruned"
* Code: [torchprune/method/ref_net](./torchprune/method/ref)

#### **Norm-based matrix sampling** 
* Paper: 
  [Matrix Entry-wise Sampling: Simple is
  Best](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.297.576&rep=rep1&type=pdf)
* Code: [torchprune/method/norm](./torchprune/method/norm)

#### **Snip**
* Paper: [SNIP: Single-shot Network Pruning based on Connection
  Sensitivity](https://arxiv.org/abs/1810.02340)
* Code: [torchprune/method/snip](./torchprune/method/snip)

#### **ThiNet**
* Paper: [ThiNet: A Filter Level Pruning Method for Deep Neural Network
  Compression](https://openaccess.thecvf.com/content_iccv_2017/html/Luo_ThiNet_A_Filter_ICCV_2017_paper.html)
* Code: [torchprune/method/thi](./torchprune/method/thi)

#### **Filter Thresholding**
* Paper: [Pruning Filters for Efficient
  ConvNets](https://arxiv.org/abs/1608.08710)
* Code:
  [torchprune/method/thres_filter](./torchprune/method/thres_filter)

#### **L1-based Filter Thresholding**
* Paper: [Soft Filter Pruning for Accelerating Deep Convolutional Neural
  Networks](https://arxiv.org/abs/1808.06866) 
* Code:
  [torchprune/method/thres_filter](./torchprune/method/thres_filter)


#### **Weight Thresholding**
* Paper: [Deep Compression: Compressing Deep Neural Networks with Pruning,
  Trained Quantization and Huffman Coding](https://arxiv.org/abs/1510.00149)
* Code: [torchprune/method/thres_weight](./torchprune/method/thres_weight)


#### **Uniform Filter Pruning**
* Pruning filters uniformly at random in each layer
* Code: [torchprune/method/uni_filter](./torchprune/method/uni_filter)

#### **Uniform Weight Pruning**
* Pruning weights uniformly at random in each layer
* Code:
  [torchprune/method/uni_weight](./torchprune/method/uni_weight)
  


### Implementing your own pruning method
Check out [base_net.py](./torchprune/method/base/base_net.py) for the
two available abstract interfaces that can be used to derive your own pruning
method:
* `BaseCompressedNet`: simpler, with less structure and more freedom
* `CompressedNet`: more rigid but potentially less implementation effort. 
  
A simple, but complete pruning implementation to base your own implementation
on is the [thres_weight](./torchprune/method/thres_weight)
implementation. 

At the end don't forget to add an appropriate `__init__.py` just like
[here](./torchprune/method/thres_weight/__init__.py) and also modify
this [`__init__.py` file](./torchprune/method/__init__.py)
accordingly. 

## Networks
All models are implemented in
[torchprune/util/models](./torchprune/util/models) and you can
easily add your own networks as well. 

Check out the [`__init__.py`](./torchprune/util/models/__init__.py) how and
which models are exported. 

## Datasets
All datasets are implemented in
[torchprune/util/datasets](./torchprune/util/datasets) and you can
also add your own datasets. 

Some data sets require data to be downloaded first. Note that for most data
sets you have to specify two locations:
* `file_dir`: where to look for pre-downloaded datasets (usually
  `<top>/data/training`) for datasets which require manual download first.
* `root`: where to store data to be downloaded (usually `<top>/local`) or where
  to extract data to when the pre-downloaded dataset is located at `file_dir`.
`<top>` refers to the top-level directory of this repository.

More instructions for each data set below: 

#### **Cifar10/100**
* Code: pytorch implementation
* Description: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
* No action required. Everything will be downloaded to `root`.

#### **Cifar10.1**
* Code:
  [torchprune/util/datasets/cifar10.py](./torchprune/util/datasets/cifar10.py)
* Description: [Github](https://github.com/modestyachts/CIFAR-10.1)
* No action required. Everything will be downloaded to `root`.

#### **CIFAR-C**
* Code:
  [torchprune/util/datasets/cifar10.py](./torchprune/util/datasets/cifar10.py)
* Description: [Github](https://github.com/hendrycks/robustness)
* Naming convention of classes: `CIFAR10_C_<corruption>_<severity>`, where
  `<corruption>` is picked from
  [`CIFAR_10_C_VARIATIONS`](./torchprune/util/datasets/cifar10.py) and
  `<severity>` should be in `range(1,6)`. Note that these classes are
  dynamically created when the module is imported. 
* For test data: please download `CIFAR-10-C.tar` from
  [here](https://zenodo.org/record/2535967) and place it in `file_dir`.
* For train data: the train data will be generated on the fly and placed as
  `.tar.gz` file in `file_dir`. Note that the creation might take several
  hours. Subsequent calls should be quick though since everything will be
  pre-generated. 

#### **Driving**
Private data set, which is currently not available for public download.

#### **Glue**
* Code:
[torchprune/util/datasets/glue.py](./torchprune/util/datasets/glue.py)
* Description: the [GLUE Benchmark](https://gluebenchmark.com/) datasets
  implemented with [huggingface](https://huggingface.co/datasets/glue).
* No manual download required. Simply specify `root` directory for the 
download location. 
* **Note**: It has only been tested with our huggingface wrapper for 
[Bert models](src/torchprune/torchprune/util/models/bert.py).

#### **ImageNet**
* Code:
  [torchprune/util/datasets/imagenet.py](./torchprune/util/datasets/imagenet.py)
* Description: [http://image-net.org/](http://image-net.org/)
* Download original images from [here](http://image-net.org/download) or
  [here](https://www.kaggle.com/c/imagenet-object-localization-challenge/data).
  Place the `imagenet_object_localization.tar.gz` file into `file_dir`.
  Everything else is handled automatically.

#### **ImageNet-C**
* Code:
  [torchprune/util/datasets/imagenet_c.py](./torchprune/util/datasets/imagenet_c.py)
* Description: [Github](https://github.com/hendrycks/robustness)
* Please make sure that the `ImageNet` dataset is available first (see above)!
* Naming convention of classes: `ImageNet_C_<corruption>_<severity>`, where
  `<corruption>` is picked from
  [`IMAGENET_C_VARIATIONS`](./torchprune/util/datasets/imagenet_c.py) and
  `<severity>` should be in `range(1,6)`. Note that these classes are
  dynamically created when the module is imported. 
* For test data: please download the `.tar` file that contains your desired
  corruption from [here](https://zenodo.org/record/2235448) and place it in
  `file_dir`.
* For train data: the train data will be generated on the fly based on the
  nominal `ImageNet` dataset and placed as
  `.tar.gz` file in `file_dir`. Note that the creation might take several
  hours. Subsequent calls should be quick though since everything will be
  pre-generated. 

#### **ObjectNet**
* Code:
  [torchprune/util/datasets/objectnet.py](./torchprune/util/datasets/objectnet.py)
* Description: [https://objectnet.dev/](https://objectnet.dev/)
* Please download the _test dataset_ from
  [here](https://objectnet.dev/download.html) and put the `objectnet-1.0.zip`
  file into `file_dir`. Everything else 
  (including unzipping the password-protected file) with the right password is
  handled automatically. 
* Since ObjectNet is a test dataset only, this class uses the `ImageNet`
  dataset when `train=True` is specified. Make sure then that the `ImageNet`
  datset is available.

#### **Augmented Pascal VOC 2011 Segmentation Dataset**
* Code:
  [torchprune/util/datasets/voc.py](./torchprune/util/datasets/voc.py)
* Description: [VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and
  [Augmentation](http://home.bharathh.info/pubs/codes/SBD/download.html). 
* No download required
#### **Augmented Pascal VOC 2012 Segmentation Dataset**
* Code:
  [torchprune/util/datasets/voc.py](./torchprune/util/datasets/voc.py)
* Description: [VOC](http://host.robots.ox.ac.uk/pascal/VOC/) and
  [Augmentation](https://github.com/DrSleep/tensorflow-deeplab-resnet) 
* Please download the augmented training data from
  [here](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0)
  place the `SegmentationClassAug.zip` file into `file_dir`.
* Everything else is handled automatically.

#### **Corrupted PASCAL VOC Segmentation Datasets**
* Code:
  [torchprune/util/datasets/voc.py](./torchprune/util/datasets/voc.py)
* These datasets are based on the corruptions from `CIFAR10-C` and
  `ImageNet-C`.
* Naming convention: `VOCSegmentation<year>_<corruption>_<severity>`, where
  `<year>` is either `2011` or `2012`, `<corruption>` is from the list
  specified [here](./torchprune/util/datasets/voc.py), and `<severity>`
  is in `range(1,6)`.
* Note that the data is generated on the fly the first time this data set is
  called and the data is then placed in `file_dir`. It requires the respective
  nominal VOC Dataset to be functional. 
* Corruptions are implemented using [this code
  repository](https://github.com/bethgelab/imagecorruptions).

## Other utilities
* [`logging`](./torchprune/util/logging): logging utilities that allow 
  for logging to files and to a tensorboard. Logging training statistics
  is also supported.
* [`lr_scheduler.py`](./torchprune/util/lr_scheduler.py): A wrapper for
  pytorch-based learning rate schedulers and custom learning rate schedulers.

* [`metrics.py`](./torchprune/util/metrics.py): test metrics for NNs 
  like
  Top-1 accuracy or IoU.

* [`net.py`](./torchprune/util/net.py): A wrapper for any
  `torch.nn.Module` that register compressible module for further 
  processing by the pruning methods. 

* [`nn_loss.py`](./torchprune/util/nn_loss.py): loss functions for
  training, both custom and from pytorch.

* [`tensor.py`](./torchprune/util/tensor.py): some utilities for 
  tensors that generalize the corresponding functions in `torch`, e.g., 
  the custom `to()` function can handle both tensors and dictionary of 
  tensors as common in NLP networks.

* [`train.py`](./torchprune/util/train.py): A training utility for
  distributed training with logging and further customization options. 
  Class is initialized with parameters as specified in the
  [`experiment`](../experiment) package. 

* [`transforms.py`](./torchprune/util/transforms.py): A wrapper for
  pytorch vision transforms as well as additional custom transforms for
  datasets.