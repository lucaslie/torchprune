# SiPPing Neural Networks: Sensitivity-informed Provable Pruning of Neural Networks
[Lucas Liebenwein*](http://www.mit.edu/~lucasl/), 
[Cenk Baykal*](http://www.mit.edu/~baykal/), 
[Igor Gilitschenski](https://www.gilitschenski.org/igor/), 
[Dan Feldman](http://people.csail.mit.edu/dannyf/),
[Daniela Rus](http://danielarus.csail.mit.edu/)

Implementation of provable pruning using sensitivity as introduced in  [SiPPing
Neural Networks: Sensitivity-informed Provable Pruning of Neural Networks](https://arxiv.org/abs/1910.05422)
(weight pruning).

***Equal contribution**

## Method
<p align="center">
  <img src="../../misc/imgs/sipp.png" width="100%">
</p>

### Sensitivity of a weight
The algorithm relies on a novel notion of weight sensitivity as saliency score
for weight parameters in the network to estimate their relative importance. 
In the simple case of a linear layer the sensitivity of a single weight `w_ij` 
in layer `l` can be defined as the maximum relative contribution of the weight
to the corresponding output neuron over a small set of points `x \in S`:

<p align="center">
  <img src="../../misc/imgs/sensitivity.png" width="30%">
</p>

The weight hereby represents the edge connecting neuron `j` in layer `ell-1` to
neuron `i` in layer `l`. This notion can  then be generalized to convolutional
layers, neurons, and filters among others as is shown in the respective papers. 

In the paper, we show how pruning according to (empirical) sensitivity
enables us to provably quantify the trade-off between the error and sparsity of
the resulting pruned neural network.

## Setup
Check out the main [README.md](../../README.md) and the respective packages for
more information on the code base. 

## Run experiments
The experiment configurations are located [here](./param). To reproduce the
experiments for a specific configuration, run: 
```bash
python -m experiment.main paper/sipp/cifar/cascade/resnet20.yaml
```

## Citations
Please cite our paper when using this codebase.
```
@article{baykal2019sipping,
  title={SiPPing Neural Networks: Sensitivity-informed Provable Pruning of Neural Networks},
  author={Baykal, Cenk and Liebenwein, Lucas and Gilitschenski, Igor and Feldman, Dan and Rus, Daniela},
  journal={arXiv preprint arXiv:1910.05422},
  year={2019}
}
```