# The experiment package
This package contains the code required to reproduce our experiments. It is
essentially a wrapper for the `torchprune` package so that we can easily
specify and configure the experiment using a standard and easy-to-read `yaml`
file. 

It also contains additional utilities to log the experiments and produce the
desired data to interpret the outcome of experiments. In summary, an experiment
started with this package will: 
* prune+retrain the same network with different prune methods for multiple
  repetitions and various prune ratios as specified
* store all the raw data and the network files
* generates plots to interpret the results
* generate reports about the "prunability" of each network with each method.
* log data to tensorboard

## Setup
Make sure `torchprune` is installed. Then run
```bash
pip install -e .
```
to install the `experiment` package.

Check out the [main README](../../README.md) for more info as well.

## Usage
**Make sure to go to `<top>`** where `<top>` denotes the top-level directory of
this repository. 

Each experiment has its own unique parameter file.
In this example, we run a sample experiment for a `ResNet20` trained on
`CIFAR10`.
```bash
python -m experiment.main experiment/experiment/param/sample/resnet20.yaml
```
The code will check in two locations for the parameter file: 
1. relative path from current working directory
2. absolute path at [`experiment/param`](./experiment/param)

Therefore, we could have also started the above experiment as 
```bash
python -m experiment.main sample/resnet20.yaml
```

Check out the respective paper folders for more available experiments. You can
also configure and run your own experiments.

### Logging
Experiment progress is logged using tensorboard. To see the current progress,
simply start tensorboard from the log directory
```bash
tensorboard --logdir=./data/results
```
and follow the instructions to visualize the data.

### Results
At the end of the run, plots (`.pdf`) and the raw numpy data (`.npz`) is stored
under `./data/results`.

## Customization of parameter files

### Tutorial
Check out the [`sample` folder](./experiment/param/sample) for more examples of
parameter configurations. 

All possible parameter descriptions are provided at
[experiment/param/sample/tutorial.yaml](./experiment/param/sample/tutorial.yaml).

### Hyperparameters
You can also specify a sweep over hyperparameters in order to repeat the same 
experiment, but with one specific parameter in the parameter file modified for
each round of experiments. Check out
[experiment/param/sample/tutorial_hyperparameters.yaml](./experiment/param/sample/tutorial_hyperparameters.yaml)
for an example.

## Checkpoints 
The code uses frequent check pointing and can thus be interrupted at any given
time. When resumed it will look for results under the specified results
directory (usually `data/results`) and resume from there. Make sure you resume
with the _same_ parameter configuration, otherwise it will not recognize the
previous results to be compatible with the current run.

## Advanced experiment configurations

### Multi-GPU training
By default, the (re-)training will run across all available GPUs using
PyTorch's [Distributed Data
Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) 
toolbox. 

If you don't want to use all available GPUs, you should specify the GPUs via
the `CUDA_VISIBILE_DEVICES` environment variable. So for example, 
```bash
# using GPU 0 only
CUDA_VISIBLE_DEVICES=0 python -m experiment.main sample/resnet20.yaml
```
### Distributed experiments
The code also supports distributed experiments. Simply specify the total number
of workers and the id of the current worker with the `-j` and `-i` flag,
respectively, when starting the experiment. For example, to start one part of
the experiment on your first GPU and the second half of the experiment on the
second GPU start the experiment as follows: 
```bash
# starting worker 0 with GPU 0
CUDA_VISIBLE_DEVICES=0 python -m experiment.main sample/resnet20.yaml -j2 -i0

# starting worker 1 with GPU 1
CUDA_VISIBLE_DEVICES=1 python -m experiment.main sample/resnet20.yaml -j2 -i1
```
The workers will then split the workload w.r.t. the number of repetitions,
networks, and prune methods as specified in the parameter file.

Of course, you can also run your experiment distributed across multiple
machines in a scenario where the `/data` folder points to a network drive that
all machines have access to.

_Note that's why in most datasets in `torchprune.util.datasets` we
distinguish between `file_dir` and `root`. `file_dir` usually points to a
shared network drive (`data/training`) while `root` points to a local drive
(`./local`)._