# Provable Filter Pruning for Efficient Neural Networks
[Lucas Liebenwein*](https://people.csail.mit.edu/lucasl/), 
[Cenk Baykal*](http://www.mit.edu/~baykal/), 
[Harry Lang](https://www.csail.mit.edu/person/harry-lang), 
[Dan Feldman](http://people.csail.mit.edu/dannyf/),
[Daniela Rus](http://danielarus.csail.mit.edu/)

Implementation of provable filter pruning using sensitivity as introduced in 
[Provable Filter Pruning for Efficient Neural
Networks](https://arxiv.org/abs/1911.07412). The algorithm
relies on a notion of sensitivity (the product of the data and the weight) to
provably quantify the error introduced by pruning.

***Equal contribution**

## Method
<p align="center">
  <img src="../../misc/imgs/pfp.png" width="100%">
</p>

### Sensitivity of a filter
The algorithm relies on a novel notion of filter sensitivity as saliency score
for weight parameters in the network to estimate their relative importance. The
filter sensitivity is a generalization of the weight sensitivity introduced in 
[SiPP](../sipp) that accounts for the filter having multiple weights and being
used in multiple places.
 
For illustrative purposes, note that in the simple case of a linear layer the
sensitivity of a single weight `w_ij` in layer `l` can be defined as the 
maximum relative contribution of the weight to the corresponding output neuron
over a small set of points `x \in S`:

<p align="center">
  <img src="../../misc/imgs/sensitivity.png" width="30%">
</p>

The weight hereby represents the edge connecting neuron `j` in layer `ell-1` to
neuron `i` in layer `l`. This notion can then be generalized to convolutional
layers, neurons, and filters among others as is shown in the paper. 

In the paper, we show how pruning filters according to (empirical) sensitivity
enables us to provably quantify the trade-off between the error and sparsity of
the resulting pruned neural network.

## Setup
Check out the main [README.md](../../README.md) and the respective packages for
more information on the code base. 

## Run experiments
The experiment configurations are located [here](./param). To reproduce the
experiments for a specific configuration, run: 
```bash
python -m experiment.main paper/pfp/param/cifar/resnet20.yaml
```

## Citation
Please cite the following paper when using our work.

### Paper link
[Provable Filter Pruning for Efficient Neural Networks](https://openreview.net/forum?id=BJxkOlSYDH)

### Bibtex
```
@inproceedings{
liebenwein2020provable,
title={Provable Filter Pruning for Efficient Neural Networks},
author={Lucas Liebenwein and Cenk Baykal and Harry Lang and Dan Feldman and Daniela Rus},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=BJxkOlSYDH}
}
```