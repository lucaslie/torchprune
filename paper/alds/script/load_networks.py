# %%
from __future__ import print_function

# make sure the setup is correct everywhere
import os
import sys


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


# %% define a few parameters

# experiment file for which to retrieve networks
FILE = "paper/alds/param/cifar/retrain/resnet20.yaml"

# specific network to retrieve
# check out parameter file for available methods.
METHOD_REF = "ReferenceNet"  # unpruned network
METHOD = "ALDSNet"  # our method
TARGET_PR = 0.5  # get the target prune ratio
N_IDX = 0  # repetition index (0,1,2)


# %% make sure matplotlib works correctly
IN_JUPYTER = True
try:
    get_ipython().run_line_magic("matplotlib", "inline")
except AttributeError:
    IN_JUPYTER = False


# %% a couple of convenience functions
def get_results(file, logger):
    """Grab all the results according to the parameter file."""
    # Loop through all experiments
    for param in get_parameters(file, 1, 0):
        # initialize logger and setup parameters
        logger.initialize_from_param(param)
        # get evaluator
        evaluator = experiment.Evaluator(logger)
        # run the experiment (only if necessary)
        try:
            state = logger.get_global_state()
        except ValueError:
            evaluator.run()
            state = logger.get_global_state()

        # extract the results
        return evaluator, state


# %% initialize and get results
# get a logger and evaluator
LOGGER = experiment.Logger()

# get the results from the prune experiment
# you can check out the different stats from there
EVALUATOR, RESULTS = get_results(FILE, LOGGER)

# %% load a specific network and test it
# now try loading a network according to the desired parameters
NET_HANDLE = EVALUATOR.get_by_pr(
    prune_ratio=TARGET_PR, method=METHOD, n_idx=N_IDX
)
DEVICE = EVALUATOR._device

# retrieve the reference network as well and test it.
NET_ORIGINAL = EVALUATOR.get_by_pr(
    prune_ratio=TARGET_PR, method=METHOD_REF, n_idx=N_IDX
).compressed_net

# reset stdout after our logger modifies it so we can see printing
sys.stdout = sys.stdout._stdout_original

# %% test the networks and print accuracy
NET_HANDLE = NET_HANDLE.to(DEVICE)
LOSS, ACC1, ACC5 = EVALUATOR._net_trainer.test(NET_HANDLE)
NET_HANDLE.to("cpu")
print(
    "Pruned network: "
    f"size: {NET_HANDLE.size()}, loss: {LOSS:.3f}, "
    f"acc1: {ACC1:.3f}, acc5: {ACC5:.3f}"
)

NET_ORIGINAL.to(DEVICE)
LOSS_ORIG, ACC1_ORIG, ACC5_ORIG = EVALUATOR._net_trainer.test(NET_HANDLE)
NET_ORIGINAL.to("cpu")
print(
    "Original network: "
    f"size: {NET_ORIGINAL.size()}, loss: {LOSS_ORIG:.3f}, "
    f"acc1: {ACC1_ORIG:.3f}, acc5: {ACC5_ORIG:.3f}"
)