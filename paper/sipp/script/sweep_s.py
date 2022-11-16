# To add a new markdown cell, type '# %% [markdown]'
# %% Set imports and working directory
from __future__ import print_function

import os
import sys
import copy
import re
import glob
from collections import OrderedDict

from IPython import get_ipython
import numpy as np
import experiment
import experiment.util as util
from matplotlib import ticker

# make sure matplotlib works if we are running the script as notebook
IN_JUPYTER = True
try:
    get_ipython().run_line_magic("matplotlib", "inline")
except AttributeError:
    IN_JUPYTER = False

# switch to root folder for data
if "paper/sipp/script" in os.path.abspath(""):
    os.chdir(os.path.abspath("../../.."))

# %% set parameters for testing
FILE = "paper/sipp/param/cifar/sweep/resnet20_sizes.yaml"
FILE = "paper/sipp/param/mnist/lenet300_sizes.yaml"
# FILE = "paper/sipp/param/mnist/lenet300_sizes2.yaml"

INLINE_PLOT = True
LEGEND_ON = True

# commensurate level for prune potential
COMM_LEVELS = [0.04]
# COMM_LEVELS = [0.01]

# folder for param/acc plot...
PLOT_FOLDER_SPECIAL = os.path.abspath(os.path.join("data/results/sipp_plots"))

# %% define functions
def get_results(file, logger):
    """Grab all the results according to the hyperparameter file."""
    results = []
    params = []
    deltas_s = []
    # fmt: off
    sizes_of_s = [35, 42, 49, 56, 70, 83, 97, 111, 125, 139, 152, 180, 208, 235, 263]
    # fmt: on
    # Loop through all experiments
    for param in util.file.get_parameters(file, 1, 0):
        # initialize logger and setup parameters
        logger.initialize_from_param(param)
        # run the experiment (only if necessary)
        try:
            state = logger.get_global_state()
            delta_s = param["coresets"]["deltaS"]
            # evaluator = experiment.Evaluator(logger)
            # size_of_s = len(evaluator.get_dataloader("s_set")[0].dataset)
        except ValueError as err:
            raise ValueError("Please compute global state first") from err
        # extract the results
        results.append(copy.deepcopy(state))
        params.append(copy.deepcopy(param))
        deltas_s.append(delta_s)
        # sizes_of_s.append(size_of_s)
    print(deltas_s)
    print(sizes_of_s)
    return (
        OrderedDict(zip(sizes_of_s, results)),
        OrderedDict(zip(sizes_of_s, params)),
    )


# do some plotting and analysis of the results now ...
def get_fig_name(title, tag):
    """Get the name of the figure with the title and tag."""
    fig_name = "_".join(re.split("/|-|_|,", title)).replace(" ", "")
    return f"{fig_name}_sweep_{tag}.pdf"


def extract_commensurate_size(stats, comm_level):
    """Compute prune potential for each result and return it."""
    # get the index closest to our desired comm_level
    c_idx = np.abs(np.array(stats[0]["commensurate"]) - comm_level).argmin()

    # pre-allocate results array
    # stats_all[0]['eBest']
    # has shape (len(commensurate), num_nets, num_rep, num_alg)
    _, num_nets, num_rep, num_alg = stats[0]["e_best"].shape
    num_sweeps = len(stats)
    size_comm = np.zeros((num_nets, num_sweeps, num_rep, num_alg))
    flops_comm = np.zeros_like(size_comm)
    e_comm = np.zeros_like(size_comm)
    e5_comm = np.zeros_like(size_comm)

    for i, stats_one in enumerate(stats):
        size_comm[:, i] = stats_one["siz_best"][c_idx]
        flops_comm[:, i] = stats_one["flo_best"][c_idx]
        e_comm[:, i] = stats_one["e_best"][c_idx]
        e5_comm[:, i] = stats_one["e5_best"][c_idx]

    return size_comm, flops_comm, e_comm, e5_comm


def plot_prune_potential(
    sizes_s,
    size_comm,
    idx_ref,
    legends,
    colors,
    title,
    plots_dir,
    plots_tag,
    comm_level,
    legend_on,
    folder_special,
):
    """Plot the prune potential for all methods."""

    sizes_s = np.broadcast_to(sizes_s[None, :, None, None], size_comm.shape)
    grapher_pp = util.grapher.Grapher(
        x_values=sizes_s,
        y_values=1.0 - size_comm,
        folder=plots_dir,
        file_name=get_fig_name(title, plots_tag),
        ref_idx=idx_ref,
        x_min=0,
        x_max=1000,
        legend=legends,
        colors=colors,
        xlabel="Size of S",
        ylabel=f"Prune Ratio, $\Delta\leq{comm_level * 100:.1f}\%$",
        title=title,
    )

    img_pp = grapher_pp.graph(
        show_ref=False,
        show_delta=False,
        percentage_x=False,
        percentage_y=True,
        remove_outlier=False,
        logplot=False,
        store=False,
    )

    # set axis limits
    img_pp.gca().set_xlim(120, 270)
    img_pp.gca().set_ylim(35, 72)

    # check for legend off
    if not legend_on:
        img_pp.gca().get_legend().remove()

    # then store it
    grapher_pp.store_plot()

    # and again in special folder
    grapher_pp._folder = folder_special
    grapher_pp.store_plot()

    return img_pp


def get_and_store_results(
    file, logger, comm_levels, legend_on, folder_special
):
    # get the results specified in the file (and hopefully pre-computed)
    results, params = get_results(file, logger)

    # reset stdout after our logger modifies it ...
    sys.stdout = sys.stdout._stdout_original

    # %% extract some additional information from the results
    results_one = list(results.values())[0]
    param_one = list(params.values())[0]
    train_dset = param_one["network"]["dataset"]

    labels_method = param_one["generated"]["network_names"]
    colors_method = param_one["generated"]["network_colors"]

    # some more stuff for plotting
    network_name = param_one["network"]["name"]
    title_pr = f"{network_name}, {train_dset}"
    if "rewind" in param_one["experiments"]["mode"]:
        title_pr += ", rewind"

    plots_dir = os.path.join(
        param_one["generated"]["resultsDir"], "plots", "sweep"
    )

    # get reference indices
    idx_ref_method = labels_method.index("ReferenceNet")

    # recall number of retraining as relative
    sizes_s = np.array(list(results.keys()))

    s_c_all, f_c_all, e_c_all, e5_c_all = (None,) * 4

    for i_c, comm_level in enumerate(comm_levels):
        # compute commensurate size for desired comm level for all results
        s_c, f_c, e_c, e5_c = extract_commensurate_size(
            [res["stats_comm"] for res in results.values()], comm_level
        )

        if s_c_all is None:
            s_c_all = np.zeros((len(comm_levels),) + s_c.shape)
            f_c_all = np.zeros_like(s_c_all)
            e_c_all = np.zeros_like(s_c_all)
            e5_c_all = np.zeros_like(s_c_all)

        # store info
        s_c_all[i_c] = s_c
        f_c_all[i_c] = f_c
        e_c_all[i_c] = e_c
        e5_c_all[i_c] = e5_c

        # now plot the commensurate size (prune potential)
        # plot a subset of the methods
        fig = plot_prune_potential(
            sizes_s=sizes_s,
            size_comm=s_c,
            idx_ref=idx_ref_method,
            legends=labels_method,
            colors=colors_method,
            title=title_pr,
            plots_dir=plots_dir,
            plots_tag=f"prune_pot_delta_{comm_level:.3f}",
            comm_level=comm_level,
            legend_on=legend_on,
            folder_special=folder_special,
        )

    print(f"PLOT DIR: {plots_dir}")
    return (
        param_one,
        {
            "names": results_one["names"],
            "sizes_s": sizes_s,
            "sizes": s_c_all,
            "flops": f_c_all,
            "e": e_c_all,
            "e5": e5_c_all,
        },
    )


# %% plot and store for all files now

# make sure matplotlib works correctly
IN_JUPYTER = True
try:
    if INLINE_PLOT:
        get_ipython().run_line_magic("matplotlib", "inline")
    else:
        get_ipython().run_line_magic("matplotlib", "agg")
except AttributeError:
    IN_JUPYTER = False

# get a logger
LOGGER = experiment.Logger()

# go through sweep
PARAM, STATS_COMM = get_and_store_results(
    FILE, LOGGER, COMM_LEVELS, LEGEND_ON, PLOT_FOLDER_SPECIAL
)
