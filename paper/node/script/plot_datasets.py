# %% Setup script
import random
import os
import sys

import numpy as np
from IPython import get_ipython
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import torch

from torchprune.util import datasets

IN_JUPYTER = True
try:
    get_ipython().run_line_magic("matplotlib", "inline")  # show plots
except AttributeError:
    IN_JUPYTER = False

# switch to root folder for data
folder = os.path.abspath("")
if "paper/node/script" in folder:
    src_folder = os.path.join(folder, "../../..")
    os.chdir(src_folder)

# %% define plotting function
def plot_dataset(dset):
    # put them in a loader and retrieve batch
    loader = torch.utils.data.DataLoader(
        dataset=dset,
        batch_size=min(len(dset), 1000),
        num_workers=0,
        shuffle=False,
    )
    x_data, y_data = next(loader.__iter__())

    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(111)
    ax.scatter(x_data[:, 0], x_data[:, 1], s=1, c=y_data)


# %% go through datasets and plot each of them
dset_list = [
    # "ToyConcentric",
    # "ToyMoons",
    # "ToySpirals",
    # "ToySpirals2",
    # "ToyGaussians",
    # "ToyGaussiansSpiral",
    # "ToyDiffeqml",
    "Bsds300",
    "Hepmass",
    "Miniboone",
    "Power",
    "Gas",
]
dsets = []
for dset_name in dset_list:
    dsets.append(
        getattr(datasets, dset_name)(
            root="./local",
            file_dir="./data/training",
            download=True,
            train=False,
        )
    )
    plot_dataset(dsets[-1])
