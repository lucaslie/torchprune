"""Heatmap plots for input decision rationales across models.

Example usage:
python sis_backward_selection.py \
    --idx_experiment=0 \
    --num_images=250
"""
from __future__ import print_function

import os  # noqa
import sys
this_folder = os.path.split(os.path.abspath('__file__'))[0]  # noqa
src_folder = os.path.join(this_folder, 'provable_compression', 'src')  # noqa
os.chdir(src_folder)  # noqa
sys.path.insert(0, src_folder)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # noqa
# print(os.getcwd())  # noqa

from torchprune.util import get_parameters
from experiment import Logger, Evaluator
import torch
import copy
import numpy as np
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--idx_experiment', type=int, required=True)
parser.add_argument('--num_images', type=int, required=True, default=250)
args = parser.parse_args()
print(args)


# some parameters to configure the pruned networks
# parameters for running SIS

# NOTE: consider code down to line 131 for full extraction

# CHANGE IDX_EXPERIMENT TO DESIRED EXPERIMENT
# * range(0, 4)     --> resnet20
# * range(4, 7)     --> vgg16
# * range(8, 12)    --> resnet20_rewind
# * range(12, 16)   --> vgg16_rewind
#
# Within a range the following order of method applies:
# 0. WT     --> weight-pruning, magnitude-based
# 1. FT     --> filter-pruning, magnitude-based
# 2. SiPP   --> weight-pruning, data-informed
# 3. PFP    --> filter-pruning, data-informed
IDX_EXPERIMENT = args.idx_experiment

# DON'T CHANGE
IDX_REF = 0
IDX_UNCORRELATED = -1

# DON'T CHANGE
FILES = [
    "experiment/cifar/resnet20.yaml",
    "experiment/cifar/vgg16.yaml",
    "experiment/cifar/resnet20_rewind.yaml",
    "experiment/cifar/vgg16_rewind.yaml"
]
METHODS = ["ThresNet", "FilterThresNet", "SiPPNetStar", "PopNet"]
DESIRED_PR = [0.15, 0.46, 0.69, 0.80, 0.98]

# Put together the model description and file
assert(0 <= IDX_EXPERIMENT <= 15)

FILE = FILES[IDX_EXPERIMENT // len(METHODS)]

MODELS_DESCRIPTION = [{"method": "ReferenceNet", "pr": 0.0, "n_idx": 0}]
for pr in DESIRED_PR:
    MODELS_DESCRIPTION.append(
        {
            "method": METHODS[IDX_EXPERIMENT % len(METHODS)],
            "pr": pr,
            "n_idx": 0
        })
MODELS_DESCRIPTION.append({"method": "ReferenceNet", "pr": 0.0, "n_idx": 1})

# %%
# Run the compression experiments (or load results if available)
# Initialize the logger and get the parameters
param = next(get_parameters(FILE, 1, 0))
Logger().initialize_from_param(param)

# Initialize the evaluator
compressor = Evaluator()

# load stats into logger so we don't have to re-run the evaluations
# if that doesn't work because some parameters don't we have to re-run eval
try:
    Logger().load_global_state()
except ValueError:
    compressor.run()

# store mean and std dev for later
mean_c = np.asarray(param['datasets'][param['network']['dataset']]['mean'])[
    :, np.newaxis, np.newaxis]
std_c = np.asarray(param['datasets'][param['network']['dataset']]['std'])[
    :, np.newaxis, np.newaxis]

# device settings
torch.cuda.set_device("cuda:0")
device = torch.device("cuda:0")
device_storage = torch.device("cpu")

# %%
# Retrieve the models we want ...

# Generate all the models we like.
# get a list of models
models = [compressor.get_by_pr(**kwargs) for kwargs in MODELS_DESCRIPTION]

# get the prune ratios
PRUNE_RATIOS = [1 - model.size() / models[IDX_REF].size() for model in models]

# construct the legends
LEGENDS = [f"{param['network_names'][type(model).__name__]} (PR={pr:.2f})"
           for model, pr in zip(models, PRUNE_RATIOS)]
LEGENDS[IDX_REF] = "Unpruned network"
LEGENDS[IDX_UNCORRELATED] = "Separate network"

# get the standard plot color for each network
COLORS = [param['network_colors'][type(model).__name__] for model in models]
COLORS[IDX_UNCORRELATED] = "grey"

# store accuracy as well for reference
TEST_LOSS = []
ACCURACY_TOP1 = []
ACCURACY_TOP5 = []
for model in models:
    model.to(device)
    loss, acc1, acc5 = compressor._net_trainer.test(model)
    model.to(device_storage)
    TEST_LOSS.append(loss.item())
    ACCURACY_TOP1.append(acc1)
    ACCURACY_TOP5.append(acc5)

# Load datasets
loader_train, loader_val, loader_test = compressor.get_dataloader(
    "train", "valid", "test")


# %%
# create one big tensor of images for each set
def get_entire_dataset(dataloader):
    dataset = copy.deepcopy(dataloader.dataset)
    num_imgs = len(dataset)
    images = torch.zeros(size=(num_imgs,)+dataset[0][0].shape)
    labels = torch.zeros(dtype=int, size=(num_imgs,))

    for i in range(len(dataset)):
        images[i], labels[i] = dataset[i]
    return images, labels


data_train = get_entire_dataset(loader_train)
data_test = get_entire_dataset(loader_test)
data_val = get_entire_dataset(loader_val)



for m in models:
    m.to('cuda')
    m.eval()

# %%

os.chdir(this_folder)
import collections
import sis_util
from sufficient_input_subsets import sis
from tqdm import tqdm



OUT_BASEDIR = './sis_data/idx_experiment_%d/' % IDX_EXPERIMENT
print(OUT_BASEDIR)


# Run SIS backward selection on CIFAR test images and write to disk.

SIS_THRESHOLD = 0.0  # To capture the results of backward selection.
INITIAL_MASK = sis.make_empty_boolean_mask_broadcast_over_axis([3, 32, 32], 0)
FULLY_MASKED_IMAGE = np.zeros((3, 32, 32), dtype='float32')

for model_i in tqdm(range(len(models))):
    model = models[model_i]
    sis_out_dir = os.path.join(OUT_BASEDIR, 'model_%d' % model_i)
    print(sis_out_dir)
    if not os.path.exists(sis_out_dir):
        os.makedirs(sis_out_dir)

    for i in range(args.num_images):
        image = data_test[0][i]
        label = data_test[1][i]
        sis_filepath = os.path.join(sis_out_dir, 'test_%d.npz' % i)
        # If SIS file already exists, skip.
        if os.path.exists(sis_filepath):
            continue
        sis_result = sis_util.find_sis_on_input(
            model, image, INITIAL_MASK, FULLY_MASKED_IMAGE, SIS_THRESHOLD,
            add_softmax=True, batch_size=128)
        sis_util.save_sis_result(sis_result, sis_filepath)
