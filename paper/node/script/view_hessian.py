"""Analyze trained networks via Hessian."""
# %%
import argparse
import os
import warnings
import sys
import copy
import numpy as np
import torch
from torchprune.util.train import _get_loss_handle
from torchprune.util import models as tp_models
import experiment
from experiment.util.file import get_parameters

PARSER = argparse.ArgumentParser(
    description="Sparse Flow - Hessian Analysis",
)

PARSER.add_argument(
    "-p",
    "--param",
    type=str,
    default="paper/node/param/toy/ffjord/gaussians/l2_h128_sigmoid_da.yaml",
    dest="param_file",
    help="provide a parameter file",
)


# switch to root folder for data
FOLDER = os.path.abspath("")
if "paper/node/script" in FOLDER:
    SRC_FOLDER = os.path.join(FOLDER, "../../..")
    os.chdir(SRC_FOLDER)

# add script path to sys path
sys.path.append("./paper/node/script")

# import our custom pyhessian library
from sparsehessian import hessian


# %% Some stuff
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        warnings.simplefilter("ignore")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        warnings.simplefilter("default")


# retrieve file
ARGS = PARSER.parse_args()
FILE = ARGS.param_file


# %% Run the Hessian Stats
def _hessian_spectrum(dataset, criterion, net):
    """Return the Hutchison-based trace estimator of the Hessian."""
    param0 = next(net.parameters())
    if param0.numel() == 1:
        param0.requires_grad = False

    with torch.enable_grad():
        # get Hessian compute model
        hessian_comp = hessian(net, criterion, data=dataset, cuda=True)

        if False:
            # get trace
            return np.mean(hessian_comp.trace(maxIter=200, tol=1e-4))
        if False:
            # get top eigenvalue
            eigs, _ = hessian_comp.eigenvalues(maxIter=100, tol=1e-4)
            return eigs[-1]

        # get spectrum data
        eigs, _ = hessian_comp.density(iter=100, n_v=3)
    return np.asarray(eigs).mean(axis=0)


def get_spectrum_stats(spec_collection, loss_collection):
    """Return useful stats from spectrum."""
    spec_collect_filt = [spec[spec > 0] for spec in spec_collection]

    # compute spectral norm, largest eigenvalue
    spec_norm = np.max(spec_collection, axis=-1).mean()

    # compute trace, sum over all eigenvalues
    trace = np.mean([np.sum(spec) for spec in spec_collect_filt])

    # compute condition number, max/min eigenvalue
    cond_number = np.mean(
        [np.max(spec) / np.min(spec) for spec in spec_collect_filt]
    )

    # get average loss
    loss = np.mean(loss_collection)

    # print stats
    print(
        ", ".join(
            [
                f"NLL={loss:.5f}",
                f"lambda_max={spec_norm:.5f}",
                f"trace={trace:.5f}",
                f"kappa={cond_number:.5f}",
            ]
        )
    )


def get_bptt_net(net, param):
    """Return the same net with BPTT (autograd) instead of adjoint backprop."""
    net_name = param["network"]["name"]
    num_classes = param["network"]["outputSize"]

    net_bptt = getattr(tp_models, f"{net_name}_autograd")(num_classes)
    net_bptt.load_state_dict(net.state_dict())

    return net_bptt


def generate_hessian_stats(logger, param, data_size=0.1, num_reps=3):
    """Plot the distribution beautifully."""
    save_and_load = True

    with HiddenPrints():
        logger.initialize_from_param(param, setup_print=False)
        evaluator = experiment.Evaluator(logger)
        loader_train = evaluator.get_dataloader("train")[0]
        criterion = _get_loss_handle(evaluator._net_trainer.train_params)
        device = "cuda"
        print(logger._results_dir)

    # create huge tensor of the data
    dataset = loader_train.dataset
    inputs = torch.stack([data[0] for data in dataset]).to(device)
    targets = torch.tensor([data[1] for data in dataset]).to(device)

    # create a subset of the data as well
    indices = torch.randperm(len(inputs))[: int(data_size * len(inputs))]
    subset = (
        inputs[indices].detach().clone(),
        targets[indices].detach().clone(),
    )

    # store prune ratios and add zero prune ratio
    prune_ratios = 1 - np.array(evaluator._keep_ratios)
    prune_ratios = np.concatenate(([0.0], prune_ratios))

    # dictionary to store spectrum results
    hessian_tag = "hessian_spectrum"
    spectrum_results = {}

    # check and load if anything is already stored
    if save_and_load:
        spectrum_results.update(logger.load_custom_state(tag=hessian_tag))

    # check required number of reps
    num_nets = evaluator._num_nets
    num_reps_experiment = evaluator._num_repetitions
    num_reps_per_net = int(np.ceil(num_reps / num_nets))

    for method_name in evaluator._method_names:
        if "ReferenceNet" in method_name:
            continue
        print("")
        for s_idx, pr in enumerate(prune_ratios):
            # setup collection of hessian stats for this
            spectrum_collection = []
            loss_collection = []

            print_key = ", ".join(
                [method_name, f"pr_idx={s_idx}", f"PR={pr*100:5.1f}%"]
            )
            print(f"{print_key}: Estimating Hessian Spectrum")

            # check if we need to compute of these to know whether we save
            saving_required = False

            for n_idx in range(num_nets):
                for r_idx in range(num_reps_per_net):
                    key = "_".join(
                        map(
                            str,
                            [
                                n_idx,
                                r_idx,
                                s_idx,
                                int(pr * 10000),
                                method_name,
                            ],
                        )
                    )
                    hessian_key = f"{key}_hessian"
                    loss_key = f"{key}_loss"

                    # only re-compute hessian results if necessary
                    if hessian_key not in spectrum_results:
                        with HiddenPrints():
                            try:
                                if pr == 0.0:
                                    lookup_name = "ReferenceNet"
                                else:
                                    lookup_name = method_name
                                net = evaluator.get_by_pr(
                                    prune_ratio=pr,
                                    method=lookup_name,
                                    n_idx=n_idx,
                                    r_idx=r_idx % num_reps_experiment,
                                ).compressed_net.torchnet
                            except FileNotFoundError:
                                continue

                        # wrap net into net with autograd instead of adjoint
                        # torchdyn adjoint breaks create_graph=True in
                        # backwards pass, which you need for any kind of
                        # Hessian computation ...
                        net_bptt = get_bptt_net(net, param)

                        # generate spectrum
                        net_bptt = net_bptt.to(device)
                        spectrum = _hessian_spectrum(
                            subset, criterion, net_bptt
                        )

                        # get train loss
                        net = net.to(device)
                        loss = criterion(net(inputs), targets).item()

                        # update results and store again
                        spectrum_results[hessian_key] = copy.deepcopy(spectrum)
                        spectrum_results[loss_key] = copy.deepcopy(loss)

                        # finalize
                        del net, net_bptt, loss, spectrum
                        torch.cuda.empty_cache()

                        # recall to save later on
                        saving_required = True

                    # get stats and collect them together
                    spectrum = copy.deepcopy(spectrum_results[hessian_key])
                    loss = copy.deepcopy(spectrum_results[loss_key])

                    spectrum_collection.append(spectrum)
                    loss_collection.append(loss)

                    # store latest results
                    if save_and_load and saving_required:
                        logger.save_custom_state(spectrum_results, hessian_tag)
                        print("Hessian update saved")

            # process collected spectrums and losses
            if len(spectrum_collection) > 0:
                spectrum_collection_np = np.asarray(spectrum_collection)
                loss_collection_np = np.asarray(loss_collection)
                get_spectrum_stats(spectrum_collection_np, loss_collection_np)
            else:
                print("No networks available")
            print("")


def main(file):
    # get a logger and the parameters
    print(file)
    logger = experiment.Logger()
    param = next(get_parameters(file, 1, 0))
    generate_hessian_stats(logger, param)


if __name__ == "__main__":
    main(FILE)
