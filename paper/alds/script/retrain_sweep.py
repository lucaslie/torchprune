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
if "paper/alds/script" in os.path.abspath(""):
    os.chdir(os.path.abspath("../../.."))

# %% [markdown]
# ## What is the optimal amount of retraining?
# Here, we do a hyperparameter sweep over potential values of retraining. In
# particular, we use "one-shot" learning rate rewinding.
# %% set parameters for testing
FOLDER = "paper/alds/param/cifar/retrainsweep"
LEGEND_ON = False
INLINE_PLOT = False
# commensurate level for prune potential
COMM_LEVELS = [0.00, 0.005, 0.01, 0.02, 0.03]

# desired table parameters
TABLE_BOLD_THRESHOLD = 0.005
TABLE_COMM_IDX = 2
TABLE_REL_RETRAIN = [0.0, 0.05, 0.1, 0.25, 0.5, 1.0]


# auto-discover files from folder without "common.yaml"
FILES = glob.glob(os.path.join(FOLDER, "[!common]*.yaml"))


def key_files(item):
    order = [
        "resnet20",
        "resnet56",
        "resnet110",
        "vgg16_bn",
        "densenet22",
        "wrn16_8",
        "resnet18",
        "resnet101",
        "wide_resnet50_2",
        "deeplabv3_resnet50",
    ]

    for i, net in enumerate(order):
        if net in item:
            return i
    return len(order)


# sort them manually according to order
FILES.sort(key=key_files)
# FILES = FILES[:1]

# folder for param/acc plot...
SPECIAL_TAG = "_".join(FOLDER.split("/")[-2:])
PLOT_FOLDER_SPECIAL = os.path.abspath(
    os.path.join("data/results/alds_plots", SPECIAL_TAG)
)


# %% define functions
def get_results(file, logger):
    """Grab all the results according to the hyperparameter file."""
    results = []
    params = []
    num_epochs_retraining = []
    # Loop through all experiments
    for param in util.file.get_parameters(file, 1, 0):
        # get number of retraining epochs
        n_e = param["generated"]["retraining"]["numEpochs"]
        n_e -= param["generated"]["retraining"]["startEpoch"]

        # initialize logger and setup parameters
        logger.initialize_from_param(param)
        # run the experiment (only if necessary)
        try:
            state = logger.get_global_state()
        except ValueError:
            experiment.Evaluator(logger).run()
            state = logger.get_global_state()
        # extract the results
        results.append(copy.deepcopy(state))
        params.append(copy.deepcopy(param))
        num_epochs_retraining.append(n_e)
    return (
        OrderedDict(zip(num_epochs_retraining, results)),
        OrderedDict(zip(num_epochs_retraining, params)),
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
    num_retrain,
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
    # remove zero retraining since we have log-scale
    valid = num_retrain != 0

    num_retrain = np.broadcast_to(
        num_retrain[None, :, None, None], size_comm.shape
    )
    num_retrain = num_retrain[:, valid]
    size_comm = size_comm[:, valid]
    grapher_pp = util.grapher.Grapher(
        x_values=num_retrain,
        y_values=1.0 - size_comm,
        folder=plots_dir,
        file_name=get_fig_name(title, plots_tag),
        ref_idx=idx_ref,
        x_min=0,
        x_max=1000,
        legend=legends,
        colors=colors,
        xlabel="Amount of Retraining",
        ylabel=f"Compression Ratio (Params)",
        title=f"{title}, $\delta={comm_level * 100:.1f}\%$",
    )

    img_pp = grapher_pp.graph(
        show_ref=False,
        show_delta=False,
        percentage_x=True,
        percentage_y=True,
        remove_outlier=False,
        logplot=True,
        store=False,
    )

    # flip x axis
    x_lim = img_pp.gca().get_xlim()
    img_pp.gca().set_xlim(x_lim[1], x_lim[0])

    # set nice y_lim as well
    img_pp.gca().set_ylim(-5, 97)

    # major locator
    img_pp.gca().xaxis.set_major_locator(ticker.LogLocator(subs=(1, 3)))

    # minor locator
    img_pp.gca().xaxis.grid(True, which="minor")
    img_pp.gca().xaxis.set_minor_locator(ticker.LogLocator(subs="auto"))

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
    num_retrain = np.array(list(results.keys()))
    num_retrain_rel = (
        num_retrain / param_one["generated"]["training"]["numEpochs"]
    )

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
            num_retrain=num_retrain_rel,
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
            "re_rel": num_retrain_rel,
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

# go through files
PARAMS = []
STATS_COMM = []
for file in FILES:
    param, stats_comm = get_and_store_results(
        file, LOGGER, COMM_LEVELS, LEGEND_ON, PLOT_FOLDER_SPECIAL
    )
    PARAMS.append(param)
    STATS_COMM.append(stats_comm)
print(f"SPECIAL FOLDER: {PLOT_FOLDER_SPECIAL}")

# %% now generate table


def compute_prune_potential(stats, re_levels, idx_comm):
    """Compute prune potential based on average."""
    # retrieve error and prune potential
    i_ref = stats["names"].index("ReferenceNet")
    e_delta = (
        stats["e"][idx_comm] - stats["e"][idx_comm][:, :, :, i_ref : i_ref + 1]
    )
    e5_delta = (
        stats["e5"][idx_comm]
        - stats["e5"][idx_comm][:, :, :, i_ref : i_ref + 1]
    )
    pp_param = 1.0 - stats["sizes"][idx_comm]
    pp_flops = 1.0 - stats["flops"][idx_comm]

    # average
    # shape (num_algorithms, num_sweeps)
    e_delta = np.mean(e_delta, axis=(0, 2)).T
    e5_delta = np.mean(e5_delta, axis=(0, 2)).T
    pp_param = np.mean(pp_param, axis=(0, 2)).T
    pp_flops = np.mean(pp_flops, axis=(0, 2)).T

    # shape (num_algorithms, num_re_levels)
    e_best = np.zeros((e_delta.shape[0], len(re_levels)))
    e5_best = np.zeros_like(e_best)
    pp_p_best = np.zeros_like(e_best)
    pp_f_best = np.zeros_like(e_best)

    for idx_m in range(e_delta.shape[0]):
        for idx_r, re_level in enumerate(re_levels):
            # find closest re_level
            idx_closest = np.argmin(np.abs(stats["re_rel"] - re_level))
            e_best[idx_m, idx_r] = e_delta[idx_m, idx_closest]
            e5_best[idx_m, idx_r] = e5_delta[idx_m, idx_closest]
            pp_p_best[idx_m, idx_r] = pp_param[idx_m, idx_closest]
            pp_f_best[idx_m, idx_r] = pp_flops[idx_m, idx_closest]

    return e_best, e5_best, pp_p_best, pp_f_best


def generate_table(
    param_all, stats_all, re_levels, idx_comm, thres_bold, math_sym=False
):
    """Generate the table now."""
    dataset = param_all[0]["network"]["dataset"]

    # check for top5
    top5 = True
    top1_str = "Top1"
    top5_str = "Top5"
    if "imagenet" in dataset.lower():
        top_str = "Top1/5"
    elif "voc" in dataset.lower():
        top_str = "IoU/Top1"
        top1_str = "IoU"
        top5_str = "Top1"
    else:
        top_str = "Top1"
        top5 = False

    # check number of methods in total across all params
    num_methods_all = sum(len(stats["names"]) - 1 for stats in stats_all)

    # start the table
    columns = "|c|c|c||" + "|".join(["ccc"] * len(re_levels)) + "|"
    cline = f"\\cline{{2-{3+3*len(re_levels)}}}"
    re_titles = [
        f"& \\multicolumn{{3}}{{c|}}{{$r={re*100:.0f}\\%\\,e$}}"
        for re in re_levels
    ]
    re_titles = "\n".join(re_titles)
    pp_titles = "\n".join([f"& {top_str} Acc. & CR-P & CR-F"] * len(re_levels))

    table = f"""\\begin{{tabular}}{{{columns}}}
\\hline
\\multirow{{{num_methods_all+2}}}{{*}}{{\\rotatebox{{90}}{{{dataset}}}}}
& \\multirow{{2}}{{*}}{{Model}}
& \\multirow{{2}}{{*}}{{\\shortstack{{Prune \\\\ Method}}}}
{re_titles} \\\\
& &
{pp_titles} \\\\ {cline}
"""

    # fill the table segments now
    table_segments = []
    for stats, param in zip(stats_all, param_all):
        # start table segment
        t_segment = ""

        # retrieve some info
        network = param["network"]["name"]
        num_methods = len(stats["names"]) - 1
        idx_ref = stats["names"].index("ReferenceNet")

        # get acc ref
        acc_ref = 1.0 - np.mean(stats["e"][idx_comm][:, :, :, idx_ref])
        acc5_ref = 1.0 - np.mean(stats["e5"][idx_comm][:, :, :, idx_ref])

        # generate e, e5, pp_p, pp_f in shape (num_algorithms, num_re_levels)
        e_delta, e5_delta, pp_p, pp_f = compute_prune_potential(
            stats, re_levels, idx_comm
        )

        # have a version of pp without ref ...
        pp_p_noref = np.delete(pp_p, idx_ref, axis=0)
        pp_f_noref = np.delete(pp_f, idx_ref, axis=0)

        # write nice network name
        nice_net_names = {
            "resnet20": "ResNet20",
            "resnet56": "ResNet56",
            "resnet110": "ResNet110",
            "vgg16_bn": "VGG16",
            "densenet22": "DenseNet22",
            "wrn16_8": "WRN16-8",
            "resnet18": "ResNet18",
            "resnet101": "ResNet101",
            "wide_resnet50_2": "WRN50-2",
            "deeplabv3_resnet50": "DeeplabV3-ResNet50",
        }

        if network in nice_net_names:
            network = nice_net_names[network]

        # network string with top1/top5 error
        network = f"{network} \\\\ \\\\ {top1_str}: {acc_ref*100.0:.2f}"
        if top5:
            network += f" \\\\ {top5_str}: {acc5_ref*100.0:.2f}"
        network = f"\\shortstack{{{network}}}"

        # write multi-row network name
        t_segment += f"& \\multirow{{{num_methods}}}{{*}}{{{network}}}\n"

        # now go through all methods
        first_method_added = False
        for idx_m, method in enumerate(stats["names"]):
            if "ReferenceNet" in method:
                continue
            if first_method_added:
                t_segment += "& "
            else:
                first_method_added = True

            # add method name now
            if method == "PP":
                t_segment += "& PP (Ours)"
            else:
                t_segment += f"& {method}"

            # go through all levels of delta now and fill in the data
            for idx_r, re_level in enumerate(re_levels):
                acc_delta = [-e_delta[idx_m, idx_r]]
                if top5:
                    acc_delta.append(-e5_delta[idx_m, idx_r])
                pp_param = pp_p[idx_m, idx_r]
                pp_flops = pp_f[idx_m, idx_r]

                def _check_best(pp_this, pp_no_ref):
                    if (
                        np.abs(pp_this - pp_no_ref[:, idx_r].max())
                        < thres_bold
                        and pp_this > 0.0
                    ):
                        return True
                    return False

                # check if that's either the best pp_param or pp_flops
                is_pp_p_best = _check_best(pp_param, pp_p_noref)
                is_pp_f_best = _check_best(pp_flops, pp_f_noref)

                if pp_param != 0.0:
                    acc_delta = [f"{delta*100.0:+.2f}" for delta in acc_delta]
                    acc_delta = "/".join(acc_delta)
                    pp_param = f"{pp_param*100.0:.2f}"
                    bold = "mathbf" if math_sym else "textbf"
                    if is_pp_p_best:
                        pp_param = f"\\{bold}{{{pp_param}}}"
                    pp_flops = f"{pp_flops*100.0:.2f}"
                    if is_pp_f_best:
                        pp_flops = f"\\{bold}{{{pp_flops}}}"
                    if math_sym:
                        acc_delta = f"${acc_delta}$"
                        pp_param = f"${pp_param}$"
                        pp_flops = f"${pp_flops}$"
                else:
                    acc_delta = " "
                    pp_param = " "
                    pp_flops = " "

                # add new stats to table
                t_segment += f"\n& {acc_delta} & {pp_param} & {pp_flops}"

            # at the end of the method we need to finish the line
            t_segment += " \\\\\n"

        # add t_segment now to list
        table_segments.append(t_segment)

    # add segments to table with cline joiner
    table += f"{cline}\n".join(table_segments)

    # finish the table
    table += """\\hline
\\end{tabular}
"""

    return table


TABLE = generate_table(
    PARAMS, STATS_COMM, TABLE_REL_RETRAIN, TABLE_COMM_IDX, TABLE_BOLD_THRESHOLD
)
with open(
    os.path.join(PLOT_FOLDER_SPECIAL, f"{SPECIAL_TAG}_table.tex"), "w"
) as t_file:
    t_file.write(TABLE)
