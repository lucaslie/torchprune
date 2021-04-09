# To add a new markdown cell, type '# %% [markdown]'
# %% Change working directory from the workspace root to the ipynb file
# location. Turn this addition off with the DataScience.changeDirOnImportExport
# setting
# ms-python.python added
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

# make sure matplotlib works if we are running the script as notebook
in_jupyter = True
try:
    get_ipython().run_line_magic("matplotlib", "inline")
except AttributeError:
    in_jupyter = False

# switch to root folder for data
folder = os.path.abspath("")
if "paper/lost/script" in folder:
    src_folder = os.path.join(folder, "../../..")
    os.chdir(src_folder)

# %%
# parameters for running the test
FILE = "paper/lost/param/cifar/resnet56.yaml"


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

        if "_rand" in file or "retraininit" in file:
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

    return results, params, labels, graphers


# get a logger
logger = experiment.Logger()

# get the results specified in the file (and hopefully pre-computed)
results, params, labels, graphers = get_results(FILE, logger)

# %% show the results
for grapher in graphers:
    grapher._figure.show()
    grapher.store_plot()
