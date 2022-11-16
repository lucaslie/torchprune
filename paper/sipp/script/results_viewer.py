# %%
from __future__ import print_function

# make sure the setup is correct everywhere
import os
import copy

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
# parameter file to plot results
FILE = "paper/sipp/param/mnist/lenet5.yaml"
INLINE_PLOT = True


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

        if "imagenet" in file:
            graphers[0].graph(
                percentage_x=True,
                percentage_y=True,
                store=False,
                remove_outlier=False,
            )

        elif "_rand" in file or "retraininit" in file:
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
                    # grapher._figure.gca().set_xlim([50, 99])
                    grapher._figure.gca().set_ylim([80, 95])
        elif "mnist" in file:
            graphers[0].graph(
                percentage_x=True,
                percentage_y=True,
                store=False,
                show_ref=False,
                show_delta=False,
                remove_outlier=False,
            )
            graphers[0]._figure.gca().set_xlim([97, 100])
            graphers[0]._figure.gca().set_ylim([95, 99.5])

    return results, params, labels, graphers


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
logger = experiment.Logger()

# get the results specified in the file (and hopefully pre-computed)
results, params, labels, graphers = get_results(FILE, logger)

for grapher in graphers:
    grapher._figure.show()
    grapher.store_plot()
