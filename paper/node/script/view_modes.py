"""Analyze CNFs via Mode Analysis."""
# %%
import argparse
import os
import warnings
import sys
import copy
import numpy as np
import torch
import experiment
from experiment.util.file import get_parameters
from torchprune.util.models import Ffjord, FfjordCNF

PARSER = argparse.ArgumentParser(
    description="Sparse Flow - Mode Analysis",
)

PARSER.add_argument(
    "-p",
    "--param",
    type=str,
    default="paper/node/param/toy/ffjord/gaussians/l4_h64_sigmoid_da.yaml",
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


# %% Main functions


def get_modes(dataset):
    """Retrieve the modes of the dataset."""
    inputs = torch.stack([data[0] for data in dataset])
    targets = torch.tensor([data[1] for data in dataset])

    # flatten inputs
    inputs = inputs.reshape(inputs.shape[0], -1)

    # unique labels
    labels = torch.unique(targets)

    # computes and corresponding covariance matrices
    modes = np.zeros((len(labels), inputs.shape[1]))
    covs = np.zeros((len(labels), inputs.shape[1], inputs.shape[1]))
    for i_lab, label in enumerate(labels):
        inputs_lab = inputs[targets == label]
        modes[i_lab] = torch.mean(inputs_lab, dim=0).cpu().numpy()
        covs[i_lab] = np.cov(inputs_lab.cpu().numpy().T)

    return modes, covs


def sample_torchdyn_ffjord(net, num_samples=20000):
    """Sample from a torchdyn network."""
    device = next(net.parameters()).device

    # extract ffjord model
    model = net.model

    # set s-span but keep old one around!
    s_span_backup = copy.deepcopy(model[1].s_span)
    model[1].s_span = torch.linspace(1, 0, 2).to(device)

    sample = net.prior.sample(torch.Size([num_samples])).to(device)
    with torch.no_grad():
        x_sampled = model(sample)

    # restore s-span
    model[1].s_span = s_span_backup

    return x_sampled[:, 1:]


def sample_ffjord_cnf(net, dataset, num_samples=2500):
    """Sample from ffjord ffjord cnf."""
    device = next(net.parameters()).device

    # extract ffjord model
    model = net.model

    # start prior
    data_shape = dataset[0][0].shape
    data_numel = dataset[0][0].numel()
    prior = torch.distributions.MultivariateNormal(
        torch.zeros(data_numel), torch.eye(data_numel)
    )

    # sample now from model
    batch_size = 1250
    samples_post_all = []
    for _ in range((num_samples - 1) // batch_size + 1):
        samples_prior = prior.sample((batch_size,))
        samples_prior = samples_prior.view(batch_size, *data_shape)
        with torch.no_grad():
            samples_post = model(samples_prior.to(device), reverse=True)
        samples_post = samples_post.view(batch_size, -1).detach().cpu()
        samples_post_all.append(samples_post)

    return torch.cat(samples_post_all)


def sample(net, dataset):
    """Sample from the network."""
    if isinstance(net, Ffjord):
        samples = sample_torchdyn_ffjord(net)
    elif isinstance(net, FfjordCNF):
        samples = sample_ffjord_cnf(net, dataset)
    else:
        raise NotImplementedError("Only works for torchdyn ffjord currently.")

    return samples.cpu().numpy()


def sample_and_compute_mode_distance(net, dataset, modes, covs):
    """Sample from the network and compute distance to each mode."""
    # let's sample first
    samples = sample(net, dataset)

    # now figure out squared distances of samples to modes as a multiplicative
    # factor of variance projected onto this direction from the cov-matrix
    # A little more explanation:
    # d^2 = "multiplies of variance" == "multiple of std.dev. squared"
    # x = sample
    # var_unnormed = (x - mode)' * Cov * (x-mode)
    # var = var_unnormed / ||x - mode||^2
    # d^2 = ||x - mode||^2 / var
    #     = ||x - mode||^4 / var_unnormed
    #
    # d   = ||x - mode||^2 / sqrt((x - mode)' * Cov * (x - mode))

    # now do the computations
    # modes.shape == num_modes x dim_state
    # shape = batch_size x num_modes x dim_state
    samples_centered = samples[:, None] - modes[None]

    # compute "unnormalized variance" using np.matmul broadcasting rules
    # covs.shape == num_modes x dim_state x dim_state
    # var_unnormed.shape == num_samples x num_modes
    var_unnormed = covs[None] @ samples_centered[..., None]
    var_unnormed = (samples_centered[:, :, None] @ var_unnormed)[:, :, 0, 0]

    # compute distance now
    # shape == num_samples x num_modes
    dist_unnormed = np.linalg.norm(samples_centered, ord=2, axis=-1)
    dist_normalized = dist_unnormed * dist_unnormed / np.sqrt(var_unnormed)

    return dist_normalized


def get_mode_stats(mode_distances):
    """Return stats about distance to nearest mode."""
    dist_checkers = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0]
    min_distances = np.min(mode_distances, axis=-1)
    num_modes = mode_distances.shape[-1]

    high_quality_ratio = [
        (min_distances <= dist).sum(axis=-1).mean() / mode_distances.shape[1]
        for dist in dist_checkers
    ]

    modes_captured = [
        np.any(mode_distances < dist, axis=1).sum(axis=1).mean()
        for dist in dist_checkers
    ]

    print(
        "Normalized std. dev.       : "
        + " | ".join(map(lambda x: f" {x:6.2f}", dist_checkers))
    )
    print(
        "High-quality samples       : "
        + " | ".join(map(lambda x: f"{x*100:6.2f}%", high_quality_ratio))
    )
    print(
        f"Modes captured (Total: {num_modes:3.0f}): "
        + " | ".join(map(lambda x: f" {x:6.2f}", modes_captured))
    )


def generate_mode_stats(logger, param, num_reps=15):
    """Plot the distribution beautifully."""
    # turn saving/loading on and off
    save_and_load = True

    # initialize experiment with logger and evaluator
    with HiddenPrints():
        logger.initialize_from_param(param, setup_print=False)
        evaluator = experiment.Evaluator(logger)
        loader_train = evaluator.get_dataloader("train")[0]
        dataset = loader_train.dataset
        print(logger._results_dir)

    # do cuda computations
    device = "cuda"

    # store prune ratios and add zero prune ratio
    prune_ratios = 1 - np.array(evaluator._keep_ratios)
    prune_ratios = np.concatenate(([0.0], prune_ratios))

    # dictionary to store mode results
    mode_tag = "mode_analysis"
    mode_results = {}

    # check and load if anything is already stored
    if save_and_load:
        mode_results.update(logger.load_custom_state(tag=mode_tag))

    # get mean and variance of each mode if not already pre-computed and save
    if "modes" in mode_results:
        modes, covs = mode_results["modes"], mode_results["covs"]
    else:
        modes, covs = get_modes(dataset)
        mode_results["modes"] = modes
        mode_results["covs"] = covs
        if save_and_load:
            logger.save_custom_state(mode_results, mode_tag)

    # check required number of reps
    num_nets = evaluator._num_nets
    num_reps_experiment = evaluator._num_repetitions
    num_reps_per_net = int(np.ceil(num_reps / num_nets))

    for method_name in evaluator._method_names:
        if "ReferenceNet" in method_name:
            continue
        print("")
        for s_idx, pr in enumerate(prune_ratios):
            # setup collection of mode distances for this run
            mode_dist_collected = []
            print_key = ", ".join(
                [method_name, f"pr_idx={s_idx}", f"PR={pr*100:5.1f}%"]
            )
            print(f"{print_key}: Estimating Mode distances")

            # check if we need to compute of these to know whether we save
            saving_required = False

            # compute mode_distances
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
                    mode_key = f"{key}_modes"

                    # only re-compute mode results if necessary
                    if mode_key not in mode_results:
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

                        # set and generate mode assignments for samples
                        net = net.to(device)
                        mode_distances = sample_and_compute_mode_distance(
                            net, dataset, modes, covs
                        )

                        # update results
                        mode_results[mode_key] = copy.deepcopy(mode_distances)

                        # finalize
                        del net, mode_distances

                        # recall to save later on
                        saving_required = True

                    # get stats and collect them together
                    mode_distances = copy.deepcopy(mode_results[mode_key])
                    mode_dist_collected.append(mode_distances)

            # store latest results
            if save_and_load and saving_required:
                logger.save_custom_state(mode_results, mode_tag)

            # process collected mode distances
            # shape = num_reps x num_samples x num_modes
            if len(mode_dist_collected) > 0:
                mode_dist_collected_np = np.asarray(mode_dist_collected)
                get_mode_stats(mode_dist_collected_np)
            else:
                print("No networks available")
            print("")


## %% Execute main
def main(file):
    # get a logger and the parameters
    print("\n")
    print(file)
    logger = experiment.Logger()
    param = next(get_parameters(file, 1, 0))
    generate_mode_stats(logger, param)


if __name__ == "__main__":
    main(FILE)
