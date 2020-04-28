# The experiment package for provable pruning of neural networks
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

This package contains the code required to reproduce our experiments.

## Setup
Make sure `provable_pruning` is installed. Then run
```sh
pip install -e .
```
to install the `experiment` package.

## Usage
Each experiment has its own unique parameter file located under 
`./experiment/experiment/param`.
In this example, we run the experiment for a `ResNet20` trained on `CIFAR10`.
```sh
python -m experiment.main cifar/resnet20.yaml
```

Check out the `param` folder for more available experiments. You can also 
configure and run your own experiments.

### Logging
Experiment progress is logged using tensorboard. To see the current progress,
simply start tensorboard from the log directory
```sh
tensorboard --logdir=./data/results
```
and follow the instructions to visualize the data.

### Results
At the end of the run, plots (`.pdf`) and the raw numpy data (`.npz`) is stored
under `./data/results`.