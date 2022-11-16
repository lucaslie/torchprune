# %% Change working directory from the workspace root to the ipynb file
# location. Turn this addition off with the DataScience.changeDirOnImportExport
# setting
# ms-python.python added
from __future__ import print_function

# make sure the setup is correct everywhere
import os
import copy
import sys
import glob

import numpy as np

# change working directory to src
from IPython import get_ipython
import experiment
from experiment.util.file import get_parameters

# make sure it's using only GPU here...
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # noqa

# switch to root folder for data
folder = os.path.abspath("")
if "paper/alds/script" in folder:
    src_folder = os.path.join(folder, "../../..")
    os.chdir(src_folder)

# %% Some parameters to retrieve results
# parameters for running the test
FOLDER = "paper/alds/param/cifar/retrain"
LEGEND_ON = False
INLINE_PLOT = False
if "prune" in FOLDER:
    TABLE_DELTA_LEVELS = [0.01, 0.05, 0.10, 0.20, 0.30]
else:
    TABLE_DELTA_LEVELS = [0.00, 0.005, 0.01, 0.02, 0.03]
TABLE_BOLD_THRESHOLD = 0.005

# auto-discover files from folder without "common.yaml"
FILES = glob.glob(os.path.join(FOLDER, "*[!common]*.yaml"))


def key_files(item):
    order = [
        "resnet20",
        "resnet56",
        "resnet110",
        "vgg16",
        "densenet22",
        "wrn16_8",
        "resnet18",
        "resnet101",
        "wide_resnet50_2",
        "mobilenet_v2",
        "deeplabv3_resnet50",
    ]

    for i, net in enumerate(order):
        if net in item:
            return i
    return len(order)


# sort them manually according to order
FILES.sort(key=key_files)
print(FILES)
# FILES = FILES[:1]

# folder for param/acc plot...
SPECIAL_TAG = "_".join(FOLDER.split("/")[-2:])
PLOT_FOLDER_SPECIAL = os.path.abspath(
    os.path.join("data/results/alds_plots", SPECIAL_TAG)
)

# %% Some helpful functions
def get_results(file, logger, legend_on):
    """Grab all the results according to the hyperparameter file."""
    results = []
    params = []
    labels = []
    graphers_all = []
    # Loop through all experiments
    for param in get_parameters(file, 1, 0):
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
        # extract the legend (based on heuristic)
        label = param["generated"]["datasetTest"].split("_")
        if len(label) > 2:
            label = label[2:]
        labels.append("_".join(label))
        # extract the plots
        graphers = logger.generate_plots(store_figs=False)

        # modify label of x-axis
        graphers[0]._figure.gca().set_xlabel("Compression Ratio (Parameters)")

        if "cifar/retraininit" in file:
            for i, grapher in enumerate(graphers[:6]):
                percentage_y = bool((i + 1) % 3)
                grapher.graph(
                    percentage_x=True,
                    percentage_y=percentage_y,
                    store=False,
                    show_ref=False,
                    show_delta=False,
                    remove_outlier=False,
                )
                if percentage_y:
                    grapher._figure.gca().set_ylim([86, 98])
        elif "cifar/prune" in file and "_plus" in file:
            graphers[0]._figure.gca().set_xlim([20, 65])
            graphers[0]._figure.gca().set_ylim([-61, 2])
        elif "cifar/prune" in file:
            graphers[0]._figure.gca().set_ylim([-87, 5])
        elif "imagenet/prune" in file and "_plus" in file:
            graphers[0]._figure.gca().set_xlim([39, 81])
            graphers[0]._figure.gca().set_ylim([-61, 2])
        elif "imagenet/prune" in file:
            graphers[0]._figure.gca().set_xlim([0, 87])
            graphers[0]._figure.gca().set_ylim([-87, 5])
        elif "imagenet/retrain/mobilenet_v2" in file:
            graphers[0]._figure.gca().set_ylim([-5, 0.5])
        elif "imagenet/retrain/" in file:
            graphers[0]._figure.gca().set_ylim([-3.5, 1.5])
        elif "imagenet/retraincascade" in file:
            # graphers[0]._figure.gca().set_xlim([-11, 2])
            graphers[0]._figure.gca().set_ylim([-2.5, 1])
        elif "imagenet/retrain" in file:
            graphers[0]._figure.gca().set_ylim([-11, 2])
        elif "cifar/retrainablation/" in file:
            graphers[0]._figure.gca().set_ylim([-3.2, 1.2])
        elif "cifar/retrain/densenet" in file:
            graphers[0]._figure.gca().set_xlim([14, 78])
            graphers[0]._figure.gca().set_ylim([-3.5, 1.5])
        elif "cifar/retrain/vgg" in file:
            graphers[0]._figure.gca().set_xlim([70, 97.5])
            graphers[0]._figure.gca().set_ylim([-3.5, 1.5])
        elif "cifar/retrain/wrn" in file:
            graphers[0]._figure.gca().set_xlim([85, 97.5])
            graphers[0]._figure.gca().set_ylim([-2.5, 0.0])
        elif "cifar/retrain/" in file:
            # graphers[0]._figure.gca().set_ylim([-11, 3])
            graphers[0]._figure.gca().set_ylim([-3.5, 1.5])
        elif "cifar/retrainlittle/" in file:
            graphers[0]._figure.gca().set_ylim([-3.5, 1.5])
        elif "cifar/retrain" in file:
            graphers[0]._figure.gca().set_ylim([-11, 2])
        elif "voc/prune" in file:
            graphers[0]._figure.gca().set_xlim([0, 90])
            graphers[0]._figure.gca().set_ylim([-87, 5])
        elif "voc/retrain" in file:
            graphers[0]._figure.gca().set_ylim([-3, 2])

        for grapher in graphers:
            legend = grapher._figure.gca().get_legend()
            if legend is not None:
                grapher._figure.gca().get_legend().remove()
                legend.set_bbox_to_anchor((1.1, 0.7))

        if legend_on:
            graphers[0].graph(
                percentage_x=True,
                percentage_y=True,
                store=False,
                kwargs_legend={
                    "loc": "upper left",
                    "ncol": 1,
                    "bbox_to_anchor": (1.1, 0.9),
                },
            )

        graphers_all.append(graphers)

    return results, params, labels, graphers_all


def get_and_store_results(file, logger, legend_on, folder_special):
    print(f"PARAM FILE: {file}")
    # get the results specified in the file (and hopefully pre-computed)
    results, params, _, graphers_all = get_results(file, logger, legend_on)
    # reset stdout after our logger modifies it ...
    sys.stdout = sys.stdout._stdout_original

    if len(graphers_all) > 1:
        raise ValueError("Not expecting multiple results per file")
    else:
        graphers = graphers_all[0]
        param = params[0]
        stats = results[0]

    for grapher in graphers:
        grapher.store_plot()
    print(f"PLOT FOLDER: {param['generated']['plotDir']}\n")

    # store the param/acc plots separately as well
    graphers[0]._folder = folder_special
    graphers[0].store_plot()

    return stats, param


def compute_prune_potential(stats, delta_levels):
    """Compute prune potential based on average."""
    # retrieve error and prune potential
    i_ref = stats["names"].index("ReferenceNet")
    e_delta = stats["error"] - stats["error"][:, :, :, i_ref : i_ref + 1]
    e5_delta = stats["error5"] - stats["error5"][:, :, :, i_ref : i_ref + 1]
    pp_param = 1.0 - stats["sizes"]
    pp_flops = 1.0 - stats["flops"]

    # average
    # shape (num_algorithms, num_intervals)
    e_delta = np.mean(e_delta, axis=(0, 2)).T
    e5_delta = np.mean(e5_delta, axis=(0, 2)).T
    pp_param = np.mean(pp_param, axis=(0, 2)).T
    pp_flops = np.mean(pp_flops, axis=(0, 2)).T

    # shape (num_algorithms, delta)
    e_best = np.zeros((e_delta.shape[0], len(delta_levels)))
    e5_best = np.zeros_like(e_best)
    pp_p_best = np.zeros_like(e_best)
    pp_f_best = np.zeros_like(e_best)

    for idx_m in range(e_delta.shape[0]):
        for idx_d in range(len(delta_levels)):
            e_valid = e_delta[idx_m] <= delta_levels[idx_d]

            if np.any(e_valid):
                i_biggest = np.argmax(pp_param[idx_m][e_valid])
                e_best[idx_m, idx_d] = e_delta[idx_m][e_valid][i_biggest]
                e5_best[idx_m, idx_d] = e5_delta[idx_m][e_valid][i_biggest]
                pp_p_best[idx_m, idx_d] = pp_param[idx_m][e_valid][i_biggest]
                pp_f_best[idx_m, idx_d] = pp_flops[idx_m][e_valid][i_biggest]

    return e_best, e5_best, pp_p_best, pp_f_best


def generate_table_entries(
    stats_all, param_all, delta_levels, thres_bold, math_sym=False
):
    """Generate all the table entries."""
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
    columns = "|c|c|c||" + "|".join(["ccc"] * len(delta_levels)) + "|"
    cline = f"\\cline{{2-{3+3*len(delta_levels)}}}"
    delta_titles = [
        f"& \\multicolumn{{3}}{{c|}}{{$\\delta={delta*100:.1f}\\%$}}"
        for delta in delta_levels
    ]
    delta_titles = "\n".join(delta_titles)
    pp_titles = "\n".join(
        [f"& {top_str} Acc. & CR-P & CR-F"] * len(delta_levels)
    )
    table = f"""\\begin{{tabular}}{{{columns}}}
\\hline
\\multirow{{{num_methods_all+2}}}{{*}}{{\\rotatebox{{90}}{{{dataset}}}}}
& \\multirow{{2}}{{*}}{{Model}}
& \\multirow{{2}}{{*}}{{\\shortstack{{Prune \\\\ Method}}}}
{delta_titles} \\\\
& &
{pp_titles} \\\\ {cline}
"""

    # fill the table segments now
    table_segments = []
    for stats, param in zip(stats_all, param_all):
        network = param["network"]["name"]
        num_methods = len(stats["names"]) - 1
        idx_ref = stats["names"].index("ReferenceNet")

        # retrieve prune potential
        e_delta, e5_delta, pp_p, pp_f = compute_prune_potential(
            stats, delta_levels
        )

        # retrieve reference accuracy
        acc_ref = 1.0 - np.mean(stats["error"][:, 0, :, idx_ref])
        acc5_ref = 1.0 - np.mean(stats["error5"][:, 0, :, idx_ref])

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
            "mobilenet_v2": "MobileNetV2",
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
        t_segment = f"& \\multirow{{{num_methods}}}{{*}}{{{network}}}\n"

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
            for idx_d, delta in enumerate(delta_levels):
                acc_delta = [-e_delta[idx_m, idx_d]]
                if top5:
                    acc_delta.append(-e5_delta[idx_m, idx_d])
                pp_param = pp_p[idx_m, idx_d]
                pp_flops = pp_f[idx_m, idx_d]

                def _check_best(pp_this, pp_no_ref):
                    if (
                        np.abs(pp_this - pp_no_ref[:, idx_d].max())
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


# %% Run through plots
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
STATS_ALL = []
PARAM_ALL = []
for file in FILES:
    stats, param = get_and_store_results(
        file, LOGGER, LEGEND_ON, PLOT_FOLDER_SPECIAL
    )
    STATS_ALL.append(stats)
    PARAM_ALL.append(param)

print(f"SPECIAL FOLDER: {PLOT_FOLDER_SPECIAL}")

# %% generate and write table
TABLE = generate_table_entries(
    STATS_ALL, PARAM_ALL, TABLE_DELTA_LEVELS, TABLE_BOLD_THRESHOLD
)
with open(
    os.path.join(PLOT_FOLDER_SPECIAL, f"{SPECIAL_TAG}_table.tex"), "w"
) as t_file:
    t_file.write(TABLE)
