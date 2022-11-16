# %%
# make sure the setup is correct everywhere
import os
import copy
import numpy as np

# change working directory to src
from IPython import get_ipython
import experiment
from experiment.util.file import get_parameters

# make sure it's using only GPU here...
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # noqa

# switch to root folder for data
folder = os.path.abspath("")
if "paper/sipp/script" in folder:
    src_folder = os.path.join(folder, "../../..")
    os.chdir(src_folder)

# %%
# parameters for running the test
FILE = "paper/sipp/param/mnist/fc_nettrim.yaml"
# FILE = "paper/sipp/param/mnist/lenet5.yaml"
FILE = "paper/sipp/param/cifar/cascade/wrn28_2.yaml"
INLINE_PLOT = True

# %% Manually recorded numbers

# RESULTS FROM NET-TRIM PAPER

# NET-TRIM for FC-Nettrim and LeNet5
# fmt: off
pr_nt = [44, 53.5, 59, 62, 66, 75, 80, 84.5, 88, 91, 93.5, 97, 99]
acc_nt = [98.7, 98.55, 98.6, 98.65, 98.62, 98.68, 98.62, 98.55, 98.45, 98.31, 98.07, 96.52, 0.0]
# fmt: on

pr_nt5 = [0.0, 76.25, 79.25, 87.5, 94.75, 96.4, 97.75, 98.15, 98.7, 98.75]
acc_nt5 = [99.46, 99.46, 99.48, 99.48, 99.43, 99.41, 99.38, 99.33, 99.21, 0.0]

# BAYESIAN COMPRESSION for FC-Nettrim and LeNet5
pr_bc = [39.0, 45.0, 47.0, 59.0, 74.2, 79.0, 83.0, 99.0]
acc_bc = [98.05, 98.05, 97.95, 97.75, 97.40, 97.15, 97.05, 0.0]

pr_bc5 = [0.0, 75.2, 78.33, 83.33, 93.2, 94.7, 99.5]
acc_bc5 = [99.3, 99.3, 99.15, 99.12, 98.66, 98.45, 0.0]

# DYNAMIC NETWORK SURGERY for FC-Nettrim and LeNet5
pr_dns = [51.5, 56.0, 57.5, 76.0, 88.0, 95.0, 97.7, 98.8, 99.5]
acc_dns = [98.60, 98.65, 98.62, 98.53, 98.60, 98.31, 96.68, 95.3, 0.0]

pr_dns5 = [0.0, 78.5, 84.7, 90.00, 93.1, 97.2, 97.95, 98.5, 99.5]
acc_dns5 = [99.51, 99.51, 99.48, 99.39, 99.41, 99.35, 99.1, 98.85, 0.0]

# RESULTS FROM DSR PAPER

# DSR (Dynamic Sparse Reparameterization) for WRN28-2
pr_dsr = [50.0, 60.0, 70.0, 80.0, 90.0]
acc_dsr = [94.7, 94.7, 94.52, 94.47, 93.65]

# DeepR for WRN28-2
pr_deepr = [50.0, 60.0, 70.0, 80.0, 90.0]
acc_deepr = [93.05, 92.83, 92.62, 92.45, 91.45]

# SET for WRN28-2
pr_set = [50.0, 60.0, 70.0, 80.0, 90.0]
acc_set = [94.72, 94.57, 94.38, 94.3, 93.3]

# "Compressed Sparse" for WRN2-2 (To prune, not to prune Zhu and Ghupta, 2017)
pr_tpntp = [50.0, 60.0, 70.0, 80.0, 90.0]
acc_tpntp = [94.53, 94.53, 94.53, 94.17, 93.8]

# %% make sure matplotlib works correctly
IN_JUPYTER = True
try:
    if INLINE_PLOT:
        get_ipython().run_line_magic("matplotlib", "inline")
    else:
        get_ipython().run_line_magic("matplotlib", "agg")
except AttributeError:
    IN_JUPYTER = False


# %% get results
def get_results(file, logger):
    """Grab all the results according to the hyperparameter file."""
    results = []
    params = []
    labels = []
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

    return results, params, labels, graphers


# get a logger
logger = experiment.Logger()

# get the results specified in the file (and hopefully pre-computed)
results, params, labels, graphers = get_results(FILE, logger)

error_res = results[0]["error"]
sizes_res = results[0]["sizes"]
names = results[0]["names"]

if "mnist/fc_nettrim" in FILE:
    legends = ["Net-Trim", "BC", "DNS"]
    colors = ["green", "black", "purple"]
    pr_new = [pr_nt, pr_bc, pr_dns]
    acc_new = [acc_nt, acc_bc, acc_dns]
elif "mnist/lenet5" in FILE:
    legends = ["Net-Trim", "BC", "DNS"]
    colors = ["green", "black", "purple"]
    pr_new = [pr_nt5, pr_bc5, pr_dns5]
    acc_new = [acc_nt5, acc_bc5, acc_dns5]
elif "cifar/cascade/wrn28_2" in FILE:
    legends = ["DSR", "DeepR", "SET", "TPNTP"]
    colors = ["green", "black", "purple", "magenta"]
    pr_new = [pr_dsr, pr_deepr, pr_set, pr_tpntp]
    acc_new = [acc_dsr, acc_deepr, acc_set, acc_tpntp]
else:
    raise ValueError("Please provide a valid parameter file")

prune_ratios = np.ones(sizes_res.shape[:3] + (len(legends),)) * 100.0
acc = np.zeros_like(prune_ratios)


# now store all the elements
for i, (pr_one, acc_one) in enumerate(zip(pr_new, acc_new)):
    # set values
    prune_ratios[:, : len(pr_one), :, i] = np.asarray(pr_one)[None, :, None]
    acc[:, : len(acc_one), :, i] = np.asarray(acc_one)[None, :, None]
    # set last value for remaining ...
    prune_ratios[:, len(pr_one) :, :, i] = pr_one[-1]
    acc[:, len(acc_one) :, :, i] = acc_one[-1]
# normalize them now
errors_manual = 1.0 - acc / 100.0
sizes_manual = 1.0 - prune_ratios / 100.0

# now we need to merge results

# remove other methods first
IDX_REMOVE = 4
errors_res = error_res[:, :, :, :IDX_REMOVE]
sizes_res = sizes_res[:, :, :, :IDX_REMOVE]
names = names[:IDX_REMOVE]

# merge results
legends_merged = names + legends
errors_merged = np.concatenate((errors_res, errors_manual), axis=-1)
sizes_merged = np.concatenate((sizes_res, sizes_manual), axis=-1)


# re-use grapher and store plot
grapher = graphers[0]
grapher._linestyles = ["-"] * len(legends_merged)
grapher._x_values = 1.0 - sizes_merged
grapher._y_values = 1.0 - errors_merged
grapher._legend = legends_merged
colors_merged = grapher._colors[:IDX_REMOVE]
colors_merged.extend(colors)
grapher._colors = colors_merged

grapher.graph(
    percentage_x=True,
    percentage_y=True,
    store=False,
    show_ref=False,
    show_delta=False,
    remove_outlier=False,
)
if "mnist/lenet5" in FILE:
    grapher._figure.gca().set_xlim([80, 100.0])
    grapher._figure.gca().set_ylim([97, 99.9])
elif "mnist/fc_nettrim" in FILE:
    grapher._figure.gca().set_xlim([69, 100.0])
    grapher._figure.gca().set_ylim([95, 99.5])
elif "wrn28_2" in FILE:
    grapher._figure.gca().set_xlim([70, 95])
    grapher._figure.gca().set_ylim([92, 95.2])

grapher.store_plot()