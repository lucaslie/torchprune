# %% Set imports and working directory
from __future__ import print_function

import os
import sys
import copy
import re
from collections import OrderedDict

from IPython import get_ipython
from matplotlib import cm
import matplotlib as mpl
import sklearn.linear_model
import numpy as np
import experiment
import experiment.util as util
from torchprune.util.datasets import cifar10

# make sure matplotlib works if we are running the script as notebook
IN_JUPYTER = True
try:
    get_ipython().run_line_magic("matplotlib", "inline")
except AttributeError:
    IN_JUPYTER = False

# switch to root folder for data
if "paper/lost/script" in os.path.abspath(""):
    os.chdir(os.path.abspath("../../.."))

# %% [markdown]
# ## Do compressed nets generalize?
# Here, we simply do a test to see if the relative performance of
# compressed networks remains the same under various circumstances. One way to
# test it is to see whether it performs the same on harder test sets as the
# original network.

# %% set parameters for testing
if IN_JUPYTER or len(sys.argv) < 2:
    FILE = "paper/lost/param/hyperparameters/cifar/resnet20.yaml"
else:
    FILE = sys.argv[1]

# standard severity level
SEVERITY_LEVEL = 3

# fmt: off
# method groups filter
FILTER_METHODS = [
    ["WT", "FT"],
    ["SiPP", "PFP"],
    ["WT", "SiPP"],
    ["FT", "PFP"],
]

# Individual PR curve filters
FILTER_DATA = {
    "WT": ["Jpeg", "Speckle", "Gauss"],
    "FT": ["Jpeg", "Speckle", "Gauss"],
}

# Excess error plot filters
FILTER_EXCESS = [
    ["WT", "SiPP", "FT", "PFP"],
    ["WT", "FT"],
    ["WT", "SiPP"],
    ["FT", "PFP"],
]

# filter for table data
FILTER_TABLE = [
    ["WT", "SiPP"],
    ["FT", "PFP"],
]
# fmt: on

# commensurate level for prune potential
COMM_LEVEL = 0.005


# %% obtain results
def get_results(file, logger, matches_exact, matches_partial):
    """Grab all the results according to the hyperparameter file."""
    results = []
    params = []
    labels = []
    # Loop through all experiments
    for param in util.file.get_parameters(file, 1, 0):
        # extract the legend (based on heuristic)
        label = param["generated"]["datasetTest"].split("_")
        if len(label) > 2:
            label = label[2:]
        label = "_".join(label)

        # check if label conforms to any of the provided filters
        if not (
            np.any([str(mtc) == label for mtc in matches_exact]).item()
            or np.any([str(mtc) in label for mtc in matches_partial]).item()
        ):
            continue

        # remove severity from label once identified
        label = label if "CIFAR10" in label else label.split("_")[0]

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
        labels.append(label)
    return OrderedDict(zip(labels, results)), OrderedDict(zip(labels, params))


# get a logger
LOGGER = experiment.Logger()

# get the results specified in the file (and hopefully pre-computed)
RESULTS_ORIG, PARAMS_ORIG = get_results(
    FILE,
    LOGGER,
    ["CIFAR10", "ObjectNet", "CIFAR10_1", "ImageNet", "VOCSegmentation2011"],
    [f"_{SEVERITY_LEVEL}"],
)

# reset stdout after our logger modifies it ...
sys.stdout = sys.stdout._stdout_original

# %% extract some additional information from the results
TRAIN_DSET = list(PARAMS_ORIG.values())[0]["network"]["dataset"]
IS_MIXED_TRAINING = "_Mix1" in TRAIN_DSET
ORIGINAL_DATA = "CIFAR10" if IS_MIXED_TRAINING else TRAIN_DSET
DATA_C_VARIATIONS = list(cifar10.CIFAR10_C_VARIATIONS.keys())
MIX1_C_VARIATIONS = cifar10.CIFAR10_C_Mix1.corruptions
MIX2_C_VARIATIONS = cifar10.CIFAR10_C_Mix2.corruptions

# now sort the results in first data sets including training, then test-only
TRAIN_SETS = [ORIGINAL_DATA]
TEST_SETS = []
if "ImageNet" in ORIGINAL_DATA:
    TEST_SETS.append("ObjectNet")
if "CIFAR" in ORIGINAL_DATA:
    TEST_SETS.append("CIFAR10_1")
if IS_MIXED_TRAINING:
    TRAIN_SETS.extend(MIX1_C_VARIATIONS)
    TEST_SETS.extend(MIX2_C_VARIATIONS)
else:
    TEST_SETS.extend(DATA_C_VARIATIONS)

RESULTS = OrderedDict()
PARAMS = OrderedDict()
for key in TRAIN_SETS + TEST_SETS:
    if key in RESULTS_ORIG:
        RESULTS[key] = RESULTS_ORIG[key]
        PARAMS[key] = PARAMS_ORIG[key]

LABEL_METRIC = LOGGER.names_metrics[0]
LABELS_METHOD = list(PARAMS.values())[0]["generated"]["network_names"]
COLORS_METHOD = list(PARAMS.values())[0]["generated"]["network_colors"]

# replace "SiPPDet" with "SiPP"
LABELS_METHOD = ["SiPP" if lab == "SiPPDet" else lab for lab in LABELS_METHOD]

# setup color map for data
REQUIRES_COLOR = DATA_C_VARIATIONS + [ORIGINAL_DATA, "CIFAR10_1", "ObjectNet"]
COLOR_MAP = [cm.tab20(i) for i in range(cm.tab20.N)]
COLORS_DATA = {
    key: COLOR_MAP[i * len(COLOR_MAP) // len(REQUIRES_COLOR)]
    for i, key in enumerate(REQUIRES_COLOR)
}


# insert original data set as well in data filters
for method in FILTER_DATA:
    FILTER_DATA[method].insert(0, ORIGINAL_DATA)

# some more stuff for plotting
NETWORK_NAME = list(PARAMS.values())[0]["network"]["name"]
TITLE_PR = f"{NETWORK_NAME}, {TRAIN_DSET}"
if "rewind" in list(PARAMS.values())[0]["experiments"]["mode"]:
    TITLE_PR += ", rewind"

PLOTS_DIR = os.path.join(LOGGER._results_dir, "plots", "Generalization")

# get reference indices
IDX_REF_METHOD = LABELS_METHOD.index("ReferenceNet")

# %%
# do some plotting and analysis of the results now ...
def get_fig_name(title, tag, legends=[]):
    """Get the name of the figure with the title and tag."""
    fig_name = "_".join(re.split("/|-|_|,", title) + legends).replace(" ", "")
    return f"{fig_name}_generalization_{tag}.pdf"


def plot_delta_acc(
    results_dict,
    colors,
    title,
    idx_ref,
    idx_method,
    use_err5,
    plot_prefix,
    plots_dir,
    plots_tag,
):
    # check for top5 or top1
    if use_err5:
        err_key = "error5"
    else:
        err_key = "error"

    # collect results together...
    sizes = [
        res_one["sizes"][:, :, :, idx_method]
        for res_one in results_dict.values()
    ]
    error = [
        res_one[err_key][:, :, :, idx_method]
        - res_one[err_key][:, :, :, idx_ref]
        for res_one in results_dict.values()
    ]
    legends = list(results_dict.keys())

    # insert zeros for fake reference
    sizes.append(np.zeros_like(sizes[0]))
    error.append(np.zeros_like(sizes[0]))
    legends.append("Unpruned network")

    # convert sizes and error to numpy
    sizes = np.stack(sizes, axis=-1)
    error = np.stack(error, axis=-1)

    grapher_error = util.grapher.Grapher(
        x_values=1.0 - sizes,
        y_values=1.0 - error,
        folder=plots_dir,
        file_name=get_fig_name(title, plots_tag, legends[:-1]),
        ref_idx=sizes.shape[-1] - 1,
        x_min=-10.0,
        x_max=10.0,
        legend=[
            "VOC2011" if name == "VOCSegmentation2011" else name
            for name in legends
        ],
        colors=colors,
        xlabel="Prune Ratio",
        ylabel=f"{plot_prefix} Test Accuracy",
        title=title.replace("VOCSegmentation2011", "VOC2011"),
    )

    img_err = grapher_error.graph(
        percentage_x=True,
        percentage_y=True,
        store=False,
        kwargs_legend={
            "bbox_to_anchor": (1.1, 0.75),
            "loc": "upper left",
            "ncol": 1,
            "borderaxespad": 0,
        },
    )

    grapher_error.store_plot()

    return img_err


def extract_commensurate_size(stats, comm_level):
    """Compute prune potential for all datasets and return it."""
    # get the index closest to our desired comm_level
    c_idx = np.abs(np.array(stats[0]["commensurate"]) - comm_level).argmin()

    # pre-allocate results array
    # stats_all[0]['eBest']
    # has shape (len(commensurate), num_nets, num_rep, num_alg)
    _, num_nets, num_rep, num_alg = stats[0]["e_best"].shape
    num_datasets = len(stats)
    size_comm = np.zeros((num_nets, num_datasets, num_rep, num_alg))

    for i, stats_one in enumerate(stats):
        size_comm[:, i, :, :] = stats_one["siz_best"][c_idx]

    return size_comm


def tabulate_prune_potential(
    size_comm_train, size_comm_test, f_extractors, network_name
):
    """Extract prune potential as table across test and train distributions."""
    # swap axes for easier iterator access
    size_comm_train = size_comm_train.swapaxes(0, -1)
    size_comm_test = size_comm_test.swapaxes(0, -1)

    # iterate through each method and produce desired result
    # 1st entry in table is methods, second is statistics
    table = [network_name]
    for sc_train, sc_test in zip(size_comm_train, size_comm_test):
        for f_extract in f_extractors:
            entry = []
            idx_smallest = []
            pp_smallest = 200.0
            for i, sc_tt in enumerate([sc_train, sc_test]):
                # sc has shape [num_datasets, num_reps, num_nets]
                # prune_pot has shape [num_reps, num_nets]
                prune_pot = f_extract(1 - sc_tt) * 100.0
                pp_mean, pp_std = np.mean(prune_pot), np.std(prune_pot)

                # keep track of max to "bold" later on...
                if pp_mean < pp_smallest:
                    idx_smallest = [i]
                    pp_smallest = pp_mean
                elif pp_mean == pp_smallest:
                    idx_smallest.append(i)

                entry.append(f"{pp_mean:.1f} $\pm$ {pp_std:.1f}")
            # bold the max ...
            entry = [
                f"\\textbf{{{ent}}}" if i in idx_smallest else ent
                for i, ent in enumerate(entry)
            ]
            entry.insert(0, "")
            table.append(" & ".join(entry).strip())
    table[-1] += " \\\\"
    return table


def plot_commensurate_size(
    size_comm,
    legends,
    colors,
    dataset_names,
    title,
    plots_dir,
    plots_tag,
    comm_level,
    split,
):
    """Plot the prune potential for all methods."""
    # get the x values
    x_val = np.arange(size_comm.shape[1], dtype=float)
    x_val[split:] += 1

    # construct the hatches
    hatches = ["//" if i < split else "\\" for i in range(len(x_val))]

    # make VOC slightly smaller ...
    dataset_names = [
        "VOC2011" if name == "VOCSegmentation2011" else name
        for name in dataset_names
    ]

    grapher_comm = util.grapher.Grapher(
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
        title=title.replace("VOCSegmentation2011", "VOC2011"),
        hatches=hatches,
    )

    img_comm = grapher_comm.graph_histo(normalize=False, store=False)

    # set custom x ticks with labels
    img_comm.gca().set_xticks(x_val)
    img_comm.gca().set_xticklabels(dataset_names, rotation=75, fontsize=30)

    # grab the rectangles to draw a legend for them
    rectangles = [tuple(), tuple()]
    for axis in img_comm.get_axes():
        child_last = None
        got_first = False
        for child in axis.get_children():
            if isinstance(child, mpl.patches.Rectangle) and not got_first:
                got_first = True
                rectangles[0] += (child,)
            elif got_first and not isinstance(child, mpl.patches.Rectangle):
                break
            child_last = child
        rectangles[1] += (child_last,)

    # customize legend
    img_comm.gca().legend(
        rectangles,
        ["Train", "Test"],
        title="Distribution:",
        bbox_to_anchor=(1.20, 1.35),
        loc="upper left",
        borderaxespad=0,
        handler_map={tuple: mpl.legend_handler.HandlerTuple(ndivide=None)},
    )

    # then store it
    grapher_comm.store_plot()

    return img_comm


def extract_excess_error(results, train_sets, test_sets):
    """Compute and return excess error.

    Excess error is defined as:
    (l(theta_hat, D') - l(theta_hat, D)) - (l(theta, D') - l(theta, D))
    """

    def mean_acc_over_corruptions(data_sets):
        """Compute test accuracy over all desired corruptions."""
        mean_error = []
        for key in data_sets:
            if key not in results:
                continue
            mean_error.append(results[key]["error"])
        mean_error = np.mean(mean_error, axis=0)
        return mean_error

    # extract delta in error between test and train corruptions
    error_train_c = mean_acc_over_corruptions(train_sets)
    error_test_c = mean_acc_over_corruptions(test_sets)
    excess_error = error_test_c - error_train_c

    # get prune ratios as well
    prune_ratios = 1.0 - results[test_sets[0]]["sizes"]

    return prune_ratios, excess_error


def plot_excess_error(
    prune_ratios,
    excess_error,
    idx_ref,
    legends,
    colors,
    title,
    plots_dir,
    plots_tag,
):
    """Plot prune ratios over excess error."""

    # plot now
    fig_name = get_fig_name(
        title,
        plots_tag,
        legends=[leg for i, leg in enumerate(legends) if i != idx_ref],
    )
    grapher_excess = util.grapher.Grapher(
        x_values=prune_ratios,
        y_values=excess_error,
        folder=plots_dir,
        file_name=fig_name,
        ref_idx=idx_ref,
        x_min=0.0,
        x_max=1.0,
        legend=legends,
        colors=colors,
        xlabel="Prune Ratio",
        ylabel="Excess Error",
        title=title.replace("VOCSegmentation2011", "VOC2011"),
    )

    fig_excess = grapher_excess.graph_regression(
        fit_intercept=False,
        percentage_x=True,
        percentage_y=True,
        store=False,
        kwargs_legend={
            "bbox_to_anchor": (1.10, 0.75),
            "loc": "upper left",
            "ncol": 1,
            "borderaxespad": 0,
        },
    )

    grapher_excess.store_plot()

    return fig_excess


# compute commensurate size for desired comm level for all results
SIZE_COMM = extract_commensurate_size(
    [res["stats_comm"] for res in RESULTS.values()], COMM_LEVEL
)


# extract avg and min prune potential for overparameterization
print(FILE + "\n")
for methods in FILTER_TABLE:
    idx_filt = [LABELS_METHOD.index(method) for method in methods]
    TABLE_PP = tabulate_prune_potential(
        SIZE_COMM[:, : len(TRAIN_SETS), :, idx_filt],
        SIZE_COMM[:, len(TRAIN_SETS) :, :, idx_filt],
        [lambda x: np.mean(x, axis=0), lambda x: np.min(x, axis=0)],
        NETWORK_NAME,
    )
    print(methods)
    print("Mean, min\n")
    for entry in TABLE_PP:
        print(entry)
    print("\n")

# compute excess error ...
PRUNE_RATIOS, EXCESS_ERROR = extract_excess_error(
    RESULTS, TRAIN_SETS, TEST_SETS
)

# also do some filtered plotting here...
for method, filters_data in FILTER_DATA.items():
    results_filtered = {
        key: RESULTS[key] for key in filters_data if key in RESULTS
    }
    fig = plot_delta_acc(
        results_dict=results_filtered,
        colors=[COLORS_DATA[key] for key in results_filtered],
        title=f"{method}, {TITLE_PR}",
        idx_ref=IDX_REF_METHOD,
        idx_method=LABELS_METHOD.index(method),
        use_err5=False,
        plot_prefix=LABEL_METRIC,
        plots_dir=PLOTS_DIR,
        plots_tag="prune_pot",
    )

# plot a subset of the methods
for methods in FILTER_METHODS:
    idx_filt = [LABELS_METHOD.index(method) for method in methods]
    fig = plot_commensurate_size(
        size_comm=SIZE_COMM[:, :, :, idx_filt],
        legends=methods,
        colors=[COLORS_METHOD[i] for i in idx_filt],
        dataset_names=list(RESULTS.keys()),
        title=TITLE_PR,
        plots_dir=PLOTS_DIR,
        plots_tag="prune_pot",
        comm_level=COMM_LEVEL,
        split=len(TRAIN_SETS),
    )

for methods in FILTER_EXCESS:
    idx_filt = [LABELS_METHOD.index(method) for method in methods]
    idx_filt.insert(0, IDX_REF_METHOD)
    fig = plot_excess_error(
        prune_ratios=PRUNE_RATIOS[:, :, :, idx_filt],
        excess_error=EXCESS_ERROR[:, :, :, idx_filt],
        idx_ref=0,  # this is zero since we inserted idx_ref at 0 position
        legends=[LABELS_METHOD[i] for i in idx_filt],
        colors=[COLORS_METHOD[i] for i in idx_filt],
        title=TITLE_PR,
        plots_dir=PLOTS_DIR,
        plots_tag="excess_err",
    )
