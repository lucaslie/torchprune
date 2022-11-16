"""View and plot Neural ODE sweep results."""
# %%
import os
import sys
import re
import warnings
import copy
import math
import yaml
import experiment
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from experiment.util.grapher import Grapher
from experiment.util.file import get_parameters, load_param_from_file

# change working directory to src
from IPython import get_ipython

# make sure it's using only GPU here...
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # noqa


# switch to root folder for data
folder = os.path.abspath("")
if "paper/node/script" in folder:
    src_folder = os.path.join(folder, "../../..")
    os.chdir(src_folder)

# add script path to sys path
sys.path.append("./paper/node/script")

# %% Define some parameters
FILE = "paper/node/param/toy/ffjord/gaussians/sweep_model_da.yaml"

INLINE_PLOT = False
USE_JPG = True

GEN_FIGS = False
GEN_ABS_FIGS = True
GEN_POT_FIGS = True
GEN_NODE_FIGS = True
REGEN_NODE_FIGS = True

COMM_LEVEL = 0.005
# fmt: off
FILTER_METHODS = [
    ["WT"],
    ["WT", "FT"],
    # ["SiPP", "PFP"],
    # ["WT", "SiPP"],
    # ["FT", "PFP"],
]
# fmt: on
IS_FFJORD = "ffjord" in FILE
if IS_FFJORD:
    import plots_cnf as plots
else:
    import plots2d as plots

# %% Some helpful functions
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        warnings.simplefilter("ignore")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        warnings.simplefilter("default")


def plot_abs_size_acc(
    logger, params, customizations, plots_dir, plot_loss, inline
):
    """Plot the absolute trade-off between # of parameters and accuracy."""

    # collect data function
    def _collect_size_err(param):
        # initialize logger and setup parameters
        with HiddenPrints():
            logger.initialize_from_param(param, setup_print=False)

        # compute absolute sizes
        sizes_abs = logger.sizes * logger.sizes_total[:, None, None, None]

        # get the desired score to plot
        if plot_loss:
            err_abs = copy.deepcopy(logger.loss)
        else:
            err_abs = copy.deepcopy(logger.error)

        # add the reference error and size at the beginning
        ref_idx = logger.names.index("ReferenceNet")
        sizes_abs = np.concatenate(
            (
                np.broadcast_to(
                    logger.sizes_total[:, None, None, None], sizes_abs.shape
                )[:, :1],
                sizes_abs,
            ),
            axis=1,
        )
        err_abs = np.concatenate(
            (
                np.repeat(
                    err_abs[:, 0:1, :, ref_idx : ref_idx + 1],
                    err_abs.shape[3],
                    axis=3,
                ),
                err_abs,
            ),
            axis=1,
        )

        return sizes_abs, err_abs

    # Loop through all experiments and collect data
    sizes_abs, err_abs = None, None
    for i, param in enumerate(params):
        sizes_abs_one, err_abs_one = _collect_size_err(param)
        if sizes_abs is None:
            sizes_abs = np.zeros(
                sizes_abs_one.shape + (len(params),), dtype=int
            )
            err_abs = copy.deepcopy(sizes_abs).astype(float)
        sizes_abs[:, :, :, :, i] = sizes_abs_one
        err_abs[:, :, :, :, i] = err_abs_one

    # get dataset now
    dataset = logger.dataset_test

    # plot per method now
    mcolor_list = list(mcolors.CSS4_COLORS.keys())
    custom_colors = [
        mcolor_list[hash(custom) % len(mcolor_list)]
        for custom in customizations
    ]
    for i_m, method in enumerate(logger.names):

        # graph the absolute trade-off
        grapher = Grapher(
            x_values=sizes_abs[:, :, :, i_m],
            y_values=err_abs[:, :, :, i_m],
            folder=os.path.join(plots_dir, "tradeoff"),
            file_name=f"err_{method}.pdf",
            ref_idx=0,
            x_min=0,
            x_max=1e20,
            legend=customizations,
            colors=custom_colors,
            xlabel="# of parameters",
            ylabel="Loss" if plot_loss else "Error",
            title=f"{method}, {dataset}",
        )
        grapher.graph(
            show_ref=True,
            show_delta=False,
            remove_outlier=True,
            store=False,
            percentage_y=not plot_loss,
            kwargs_legend={
                "loc": "upper left",
                "ncol": 1,
                "bbox_to_anchor": (1.05, 1.1),
            },
        )
        grapher.store_plot()
        if not inline:
            plt.close(grapher._figure)

    # plot per customization now
    for i_c, custom in enumerate(customizations):

        # graph the absolute trade-off
        grapher = Grapher(
            x_values=sizes_abs[:, :, :, :, i_c],
            y_values=err_abs[:, :, :, :, i_c],
            folder=os.path.join(plots_dir, "tradeoff"),
            file_name=f"err_{custom}.pdf",
            ref_idx=0,
            x_min=0,
            x_max=1e20,
            legend=copy.deepcopy(np.array(logger.names)).tolist(),
            colors=copy.deepcopy(np.array(logger._colors)).tolist(),
            xlabel="# of parameters",
            ylabel="Loss" if plot_loss else "Error",
            title=f"{custom}, {dataset}",
        )
        grapher.graph(
            show_ref=True,
            show_delta=False,
            remove_outlier=True,
            store=False,
            percentage_y=not plot_loss,
        )
        grapher.store_plot()
        if not inline:
            plt.close(grapher._figure)


def get_results(file, logger, regen_figs):
    """Grab all the results according to the file."""
    stats = []
    params = []
    # Loop through all experiments
    for param in get_parameters(file, 1, 0):
        # initialize logger and setup parameters
        with HiddenPrints():
            logger.initialize_from_param(param, setup_print=False)

        # print message if incomplete but don't stop
        if not logger.state_loaded:
            print("Grabbing incomplete results!")

        # compute the stats
        try:
            stats_one = logger.compute_stats(store_report=False)
        except ValueError as err:
            print(
                "Computing stats failed. Make sure that all partial results "
                "are stored as numpy, e.g., by running it in parallel."
            )
            raise err

        # extract the results
        stats.append(copy.deepcopy(stats_one))
        params.append(copy.deepcopy(param))

        # extract the plots and store them.
        if not regen_figs or not logger.state_loaded:
            continue
        try:
            with HiddenPrints():
                graphers = logger.generate_plots(store_figs=False)
                for grapher in graphers:
                    grapher.store_plot()
        except:
            print("Could not generate main graphs.")

    return stats, params


def extract_commensurate_size(stats, comm_level):
    """Compute prune potential for all experiments and return it."""
    # get the index closest to our desired comm_level
    c_idx = np.abs(np.array(stats[0]["commensurate"]) - comm_level).argmin()

    # pre-allocate results array
    # stats_all[0]['eBest']
    # has shape (len(commensurate), num_nets, num_rep, num_alg)
    _, num_nets, num_rep, num_alg = stats[0]["e_best"].shape
    num_exp = len(stats)
    size_comm = np.zeros((num_nets, num_exp, num_rep, num_alg))

    for i, stats_one in enumerate(stats):
        if stats_one is not None:
            size_comm[:, i, :, :] = stats_one["siz_best"][c_idx]

    return size_comm


def get_fig_name(title, tag, legends=[]):
    """Get the name of the figure with the title and tag."""
    fig_name = "_".join(re.split("/|-|_|,", title) + legends).replace(" ", "")
    return f"{fig_name}_prunepot_{tag}.pdf"


def plot_commensurate_size(
    size_comm,
    legends,
    colors,
    customizations,
    title,
    plots_dir,
    plots_tag,
    comm_level,
):
    """Plot the prune potential for all methods."""
    # get the x values
    x_val = np.arange(size_comm.shape[1], dtype=float)

    grapher_comm = Grapher(
        x_values=np.broadcast_to(x_val[None, :, None, None], size_comm.shape),
        y_values=1.0 - size_comm,
        folder=plots_dir,
        file_name=get_fig_name(title, plots_tag, legends),
        ref_idx=None,
        x_min=-1e10,
        x_max=1e10,
        legend=legends,
        colors=colors,
        xlabel=f"Prune Potential, $\delta={comm_level * 100:.1f}\%$",
        ylabel="Method",
        title=title,
    )

    with HiddenPrints():
        img_comm = grapher_comm.graph_histo(normalize=False, store=False)

    # set custom x ticks with labels
    img_comm.gca().set_xticks(x_val)
    img_comm.gca().set_xticklabels(customizations, rotation=75, fontsize=20)

    # then store it
    grapher_comm.store_plot()

    return img_comm


def generate_one_sweep(
    logger,
    params,
    plt_folder,
    n_idx,
    r_idx,
    method_name,
    customizations,
    inline,
    use_jpg,
    regen_figs,
):
    # get keep ratios and add zero pruning as well
    m_name_ref = "ReferenceNet"
    keep_ratios = params[0]["generated"]["keepRatios"]
    keep_ratios = np.concatenate(([1.0], keep_ratios))

    figsize = [6.4, 4.8]
    ncols = math.ceil(math.sqrt(len(params)))
    nrows = math.ceil(len(params) / ncols)
    figsize[0] *= ncols
    figsize[1] *= nrows

    for s_idx, keep_ratio in enumerate(keep_ratios):
        # only plot for one keep ratio for reference net.
        if m_name_ref in method_name and s_idx > 0:
            break

        # create folder
        os.makedirs(plt_folder, exist_ok=True)
        # generate figure name
        tag = f"i_{s_idx - 1}_p{keep_ratio:.3f}"
        if use_jpg:
            file_ending = ".jpg"
        else:
            file_ending = ".pdf"
        fig_file = os.path.join(plt_folder, tag + file_ending)

        # continue if exists and we shouldn't regen figures
        if not regen_figs and os.path.exists(fig_file):
            continue

        # initialize figure and layout
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=True,
            sharey=True,
            figsize=figsize,
            squeeze=False,
        )
        plt.style.use("default")
        plt.rcParams.update(
            {
                "xtick.labelsize": 16,
                "ytick.labelsize": 16,
            }
        )

        # go through each parameter config of sweep
        for axis, param, custom in zip(axes.flatten(), params, customizations):
            logger.initialize_from_param(param, setup_print=False)
            with HiddenPrints():
                # initialize evaluator and logger
                evaluator = experiment.Evaluator(logger)

                # get data loader
                loader_test = evaluator.get_dataloader("test")[0]

                # retrieve model and plot if it exists
                try:
                    net = evaluator.get_by_pr(
                        prune_ratio=1.0 - keep_ratio,
                        n_idx=n_idx,
                        r_idx=r_idx,
                        method=method_name if keep_ratio < 1.0 else m_name_ref,
                    ).compressed_net.cuda()
                except FileNotFoundError:
                    continue

                # set plot title
                axis.set_title(f"{custom}\n#p={int(net.size())}", fontsize=20)

            # setup and generate plots
            plots_kwargs = plots.prepare_data(net.torchnet, loader_test)
            plots.plot_for_sweep(axis=axis, **plots_kwargs)

        # store plot at the end now.
        fig.suptitle(f"{method_name}, {tag}", fontsize=24)
        fig.savefig(fig_file, bbox_inches="tight")
        if not inline:
            plt.close(fig)


def generate_sweepy_figures(
    logger, params, customizations, plot_dir, regen_figs, inline, use_jpg
):
    """Generate figures with view over sweep."""
    # extract all repetitions from here
    num_nets = params[0]["experiments"]["numNets"]
    num_repetitions = params[0]["experiments"]["numRepetitions"]
    method_names = params[0]["experiments"]["methods"]

    # loop through all repetitions and method names.
    for n_idx in range(num_nets):
        for r_idx in range(num_repetitions):
            for method_name in method_names:
                tag = "_".join([method_name, f"n{n_idx}", f"r{r_idx}"])
                plt_folder = os.path.join(plot_dir, tag)

                # print folder
                print(plt_folder)

                # now plot sequence of keep ratios
                generate_one_sweep(
                    logger,
                    params,
                    plt_folder,
                    n_idx,
                    r_idx,
                    method_name,
                    customizations,
                    inline,
                    use_jpg,
                    regen_figs,
                )

                # print finish message
                print("Done\n")


# %% Retrieve results
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
print(f"PARAM FILE: {FILE}")

# get the results specified in the file (and hopefully pre-computed)
STATS, PARAMS = get_results(FILE, LOGGER, GEN_FIGS)

# extract some other info from params
LABELS_METHOD = PARAMS[0]["generated"]["network_names"]
COLORS_METHOD = PARAMS[0]["generated"]["network_colors"]
NETWORK_NAME = PARAMS[0]["network"]["name"]
TRAIN_DSET = PARAMS[0]["network"]["dataset"]
TITLE_PR = f"{NETWORK_NAME}, {TRAIN_DSET}"
PLOTS_DIR = os.path.join(
    PARAMS[0]["generated"]["plotDir"],
    os.path.splitext(os.path.basename(FILE))[0],
)
CUSTOMIZATIONS = [
    ", ".join([f"{k}={v}" for k, v in custom["value"].items()])
    if isinstance(custom["value"], dict)
    else str(custom["value"])
    for custom in load_param_from_file(FILE)["customizations"]
]

# print plot folders for reference
for param, custom in zip(PARAMS, CUSTOMIZATIONS):
    print(f"Customizations: {custom}")
    print(f"Plot Folder: {param['generated']['plotDir']}\n")

# special folder as folder
print(f"Sweep plot folder: {PLOTS_DIR}")


# %% generate prune potential plot
if GEN_POT_FIGS:
    # compute commensurate size for desired comm level for all results
    SIZE_COMM = extract_commensurate_size(STATS, COMM_LEVEL)

    PLOTS_DIR_POT = os.path.join(PLOTS_DIR, "prune_pot")
    print(PLOTS_DIR_POT)

    # plot a subset of the methods
    for methods in FILTER_METHODS:
        try:
            idx_filt = [LABELS_METHOD.index(method) for method in methods]
        except ValueError:
            continue
        fig = plot_commensurate_size(
            size_comm=SIZE_COMM[:, :, :, idx_filt],
            legends=methods,
            colors=[COLORS_METHOD[i] for i in idx_filt],
            customizations=CUSTOMIZATIONS,
            title=TITLE_PR,
            plots_dir=PLOTS_DIR_POT,
            plots_tag="prune_pot",
            comm_level=COMM_LEVEL,
        )
    print("Done\n")

# %% Generate abs parameter-error trade-off figures
if GEN_ABS_FIGS:
    plot_abs_size_acc(
        LOGGER,
        PARAMS,
        CUSTOMIZATIONS,
        PLOTS_DIR,
        IS_FFJORD,
        INLINE_PLOT,
    )


# %% now generate the sweep plots
if GEN_NODE_FIGS:
    generate_sweepy_figures(
        LOGGER,
        PARAMS,
        CUSTOMIZATIONS,
        PLOTS_DIR,
        REGEN_NODE_FIGS,
        INLINE_PLOT,
        USE_JPG,
    )
