# %% Setup script and imports
from __future__ import print_function

import os  # noqa
import sys  # noqa

# change working directory to src
from IPython import get_ipython
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib import cm
import experiment
import experiment.util as util
import torch
import torch.nn as nn
import copy
import itertools
import numpy as np
import yaml
import time
import re

# make sure matplotlib works if we are running the script as notebook
IN_JUPYTER = True
try:
    get_ipython().run_line_magic("matplotlib", "inline")
except AttributeError:
    IN_JUPYTER = False

# switch to root folder for data
SRC_FOLDER = os.path.abspath(
    os.path.join(os.path.split(__file__)[0], "../../..")
)
os.chdir(SRC_FOLDER)

# only one GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# %%
# Initialize with desired parameters
# pruned network parameters
if IN_JUPYTER or len(sys.argv) < 2:
    FILE = "paper/lost/param/cifar/resnet20.yaml"
else:
    FILE = sys.argv[1]

# %%
# Fixed noise parameters
# DON'T CHANGE
METHODS = ["WT", "FT", "SiPP", "PFP"]
IDX_REF = 0
IDX_UNCORRELATED = -1

MODELS_ALL_DESCRIPTION = [
    {"method": "ReferenceNet", "max_num": 1, "n_idx": 0},
    {"method": "PFPNet", "max_num": -1, "n_idx": 0},
    {"method": "SiPPNet", "max_num": -1, "n_idx": 0},
    {"method": "ThresNet", "max_num": -1, "n_idx": 0},
    {"method": "FilterThresNet", "max_num": -1, "n_idx": 0},
    {"method": "ReferenceNet", "max_num": 1, "n_idx": 1},
]

# desired PRs for diff plots
DESIRED_PR = [0.15, 0.46, 0.69, 0.80, 0.95]

# DON'T CHANGE
# noisy image parameters
NUM_IMAGES = 1000
NUM_SAMPLES = 100
EPS_MAX = 0.7
NUM_EPS = 15
NOISE = "uniform"

# DO NOT CHANGE
EPS_VALUES = np.linspace(0, EPS_MAX, NUM_EPS)

# generate noise tag
NOISE_TAG = (
    f"img{NUM_IMAGES}_samp{NUM_SAMPLES}_eps{10*EPS_MAX:02.0f}"
    f"_{NUM_EPS}_{NOISE}"
)

# %%
# Set some extra parameters
PARAM = next(util.file.get_parameters(FILE, 1, 0))

# store mean and std dev for later
if "ImageNet" in PARAM["network"]["dataset"]:
    MEAN_C = [0.485, 0.456, 0.406]
    STD_C = [0.229, 0.224, 0.225]
elif "CIFAR" in PARAM["network"]["dataset"]:
    MEAN_C = [0.4914, 0.4822, 0.4465]
    STD_C = [0.2023, 0.1994, 0.2010]
else:
    raise ValueError("Please adapt script to provide normalization of dset!")
MEAN_C = np.asarray(MEAN_C)[:, np.newaxis, np.newaxis]
STD_C = np.asarray(STD_C)[:, np.newaxis, np.newaxis]

# Now initialize the logger
LOGGER = experiment.Logger()
LOGGER.initialize_from_param(PARAM)

# Initialize the evaluator and run it (will return if nothing to compute)
COMPRESSOR = experiment.Evaluator(LOGGER)
COMPRESSOR.run()

# device settings
torch.cuda.set_device("cuda:0")
DEVICE = torch.device("cuda:0")
DEVICE_STORAGE = torch.device("cpu")

# Generate all the models we like.
# get a list of models
MODELS = [COMPRESSOR.get_all(**kw) for kw in MODELS_ALL_DESCRIPTION]
IDX_RANGES = []
NUM_TOTAL = 0
for model_array in MODELS:
    IDX_RANGES.append(np.arange(NUM_TOTAL, NUM_TOTAL + len(model_array)))
    NUM_TOTAL += len(model_array)
# flatten list ...
MODELS = list(itertools.chain.from_iterable(MODELS))

# make sure reference will return correct size...
MODELS[IDX_REF]._keep_ratio_latest = torch.tensor(1.0)
MODELS[IDX_UNCORRELATED]._keep_ratio_latest = torch.tensor(1.0)

# get the legends and prune ratios...
PRUNE_RATIOS = [1 - model.size() / MODELS[IDX_REF].size() for model in MODELS]
LEGENDS = [PARAM["network_names"][type(model).__name__] for model in MODELS]

LEGENDS[IDX_REF] = "Unpruned network"
LEGENDS[IDX_UNCORRELATED] = "Separate network"

# Replace "SiPPDet" with "SiPP"
LEGENDS = ["SiPP" if leg == "SiPPDet" else leg for leg in LEGENDS]

# also do colors ... .
COLORS = [PARAM["network_colors"][type(model).__name__] for model in MODELS]
COLORS[IDX_UNCORRELATED] = "grey"

# plots directory for later
PLOTS_DIR = os.path.join(LOGGER._results_dir, "plots", "Noise")

# Load datasets
LOADER_TEST = COMPRESSOR.get_dataloader("test")[0]

# generate random test loader from here
LOADER_TEST = torch.utils.data.DataLoader(
    dataset=LOADER_TEST.dataset,
    batch_size=LOADER_TEST.batch_size,
    num_workers=LOADER_TEST.num_workers,
    shuffle=True,
    pin_memory=True,
)


# Helper function that selects the probability for a single class, from the
# softmax output.
def make_f_hat(model, batch_size):
    def f_hat(ins):
        with torch.no_grad():
            if not isinstance(ins, torch.Tensor):
                ins = torch.tensor(ins)
            # big tensor containing all the predictions
            pred = None
            # list of batched views into pred
            pred_batched = None
            num_ins = ins.shape[0]
            for i, batch in enumerate(torch.split(ins, batch_size)):
                batch = batch.float().to(DEVICE)
                out = model(batch).detach().to(ins.device)
                if pred is None:
                    pred = torch.zeros((num_ins,) + out.shape[1:])
                    pred = pred.to(out.device)
                    pred_batched = torch.split(pred, batch_size)
                pred_batched[i].copy_(out)

        return pred

    return f_hat


# for each model store this slightly easier to handle function
F_HATS = [make_f_hat(model, batch_size=2048) for model in MODELS]


# %% A LOT OF HELPER FUNCTIONS
def gather_images(loader_test, num_images):
    """Gather a few images to use for generating samples."""
    imgs_all = torch.Tensor()
    labels_all = torch.LongTensor()
    for imgs, labels in loader_test:
        # now add to the images
        num_to_add = min(num_images - imgs_all.shape[0], imgs.shape[0])
        imgs_all = torch.cat((imgs_all, imgs[:num_to_add]), dim=0)
        labels_all = torch.cat((labels_all, labels[:num_to_add]), dim=0)

        if imgs_all.shape[0] >= num_images:
            break

    return imgs_all, labels_all


def generate_noisy_images(eps, imgs, num_samples, noise_type):
    noise_shape = (num_samples,) + imgs.shape
    if noise_type == "Gaussian":
        noise = torch.zeros(noise_shape, device=imgs.device)
        noise.normal_(mean=0, std=eps)
    elif noise_type == "uniform":
        noise = torch.rand(noise_shape, device=imgs.device)
    else:
        raise ValueError("Please specify valid noise type")

    # has shape (num_samples, ) + imgs.shape
    return imgs + 2 * eps * (noise - 0.5)


def generate_samples(f_hat, imgs_noisy):
    """Generate the desired samples for all noisy images for all f_hats."""
    batch_shape = imgs_noisy.shape[:-3]
    img_shape = imgs_noisy.shape[-3:]

    # we have to flatten batch dimensions into one big dim for inference
    out_noisy = f_hat(imgs_noisy.view((-1,) + img_shape)).to(imgs_noisy.device)

    # first dimension is batch size, rest is output size
    out_shape = out_noisy.shape[1:]
    samples = out_noisy.view(batch_shape + out_shape)

    # the full shape is (num_eps, num_images, num_samples, num_outputs)
    return samples


def get_all_samples():
    """Get everything we need for the next steps."""
    # gather images
    MODELS[IDX_REF].to(DEVICE)
    imgs, labels = gather_images(LOADER_TEST, NUM_IMAGES)
    imgs.to(DEVICE)
    labels.to(DEVICE)

    # generate noisy images
    # has temporary shape:
    # (num_eps, num_samples, num_images) + one_img.shape
    imgs_noisy = None
    for i, eps in enumerate(EPS_VALUES):
        t_one_eps = -time.time()

        imgs_noisy_i = generate_noisy_images(eps, imgs, NUM_SAMPLES, NOISE)
        if imgs_noisy is None:
            imgs_noisy = torch.zeros((len(EPS_VALUES),) + imgs_noisy_i.shape)
            imgs_noisy = imgs_noisy.to(imgs.device)
        imgs_noisy[i] = imgs_noisy_i
        t_one_eps += time.time()
        print(f"Finished eps value {i}, {eps} in {t_one_eps:.2f}sec")

    # switch axis to final shape
    # (num_eps, num_images, num_samples) + one_img.shape
    imgs_noisy = imgs_noisy.transpose(1, 2).contiguous()

    # move device back
    MODELS[IDX_REF].to(DEVICE_STORAGE)
    imgs.to(DEVICE_STORAGE)
    labels.to(DEVICE_STORAGE)

    # go through algorithms and store the results for each sample
    # shape at the end:
    # (num_f, num_eps, num_images, num_samples, num_outputs)
    samples = None
    for i, f_hat in enumerate(F_HATS):
        t_one_f = -time.time()
        MODELS[i].to(DEVICE)
        samples_i = generate_samples(f_hat, imgs_noisy)
        MODELS[i].to(DEVICE_STORAGE)
        if samples is None:
            samples = torch.zeros((len(F_HATS),) + samples_i.shape)
            samples = samples.to(imgs_noisy.device)
        samples[i] = samples_i
        t_one_f += time.time()
        print(f"Finished model {i} in {t_one_f:.2f}sec")

    return imgs, imgs_noisy, labels, samples


# Evaluate the samples now
def get_norm_diff(samples, samples_ref, norm=2, max_err=False):
    """Get a norm-based difference between ref samples and regular samples."""
    # compute the diff
    diff = samples - samples_ref

    # take the norm
    diff = torch.norm(diff, p=norm, dim=-1)

    if max_err:
        reduced = torch.max(diff, dim=-2)[0]
        std, mean = torch.std_mean(reduced, dim=-1)
    else:
        # mean over samples (i.e. "ROBUSTNESS" of an image)
        mean = torch.mean(diff, dim=2)

        # mean and std over images (mean "ROBUSTNESS")
        std, mean = torch.std_mean(mean, dim=2)

    return mean, std


def get_softmax_norm_diff(samples, samples_ref, norm=2, max_err=False):
    """Get a norm-based difference after applying softmax to output."""
    return get_norm_diff(
        samples.softmax(dim=-1), samples_ref.softmax(dim=-1), norm, max_err
    )


def get_matching(samples, labels, topk=1, max_err=False):
    """Get mean or max matching over images."""
    assert topk == 1

    # has shape (num_f, num_eps, num_images, num_samples)
    # NOTE: labels must be broadcastable to that ...
    idxs_top = samples.topk(topk, dim=-1)[1].squeeze(dim=-1)

    # compare matching labels
    matching = (idxs_top == labels).float()

    if max_err:
        # raise NotImplementedError
        num_samples = matching.shape[-2]
        percentile = 0.01
        # reduced = torch.min(matching, dim=-2)[0]
        reduced = torch.median(matching, dim=-2)[0]
        reduced = torch.sort(matching, dim=-2)[0][
            :, :, int(percentile * num_samples)
        ]
    else:
        # mean over images (i.e. "Test Accuracy" over noisy images )
        reduced = torch.mean(matching, dim=2)

    # has shape (num_f, num_eps, num_samples)
    return reduced


def get_agreement(samples, labels, topk=1, max_err=False):
    """Get Top-k agreement of samples on some samples."""
    reduced = get_matching(samples, labels, topk, max_err)

    # has shape (num_f, num_eps)
    std, mean = torch.std_mean(reduced, dim=-1)
    return mean, std


def get_overlap(samples, samples_ref, topk=1, max_err=False):
    # get top indices
    idxs_top_ref = samples_ref.topk(topk, dim=-1)[1].squeeze(dim=-1)

    return get_agreement(samples, idxs_top_ref, topk, max_err)


def get_accuracy(samples, labels, topk=1, max_err=False):
    """Get Top-k accuracy on samples."""
    return get_agreement(samples, labels.unsqueeze(-1), topk, max_err)


def get_commensurate_pr(samples, samples_ref, labels, idx_ranges, prs):
    """Get last PR for which we have commensurate acc."""
    # def get_comm_pr_one_range(m_range):
    def get_comm_pr_one_range(prs, samples, samples_ref, labels):
        # subsample everything ...
        # (num_prs, num_eps, num_images, num_samples, num_outputs)

        # now get accuracy diff (but not meaned over num_samples)
        # has shape (num_prs, num_eps, num_samples)
        accuracy = get_matching(samples, labels.unsqueeze(-1))
        accuracy_ref = get_matching(samples_ref, labels.unsqueeze(-1))
        accuracy_gain = accuracy - accuracy_ref

        # now check out commensurate level
        # has shape (num_eps, num_samples)
        comm_allowed = 0.005
        comm_level = torch.zeros_like(accuracy[0])

        for i, pr in enumerate(prs):
            # check what is allowed
            within_comm = accuracy_gain[i] + comm_allowed > 0.0

            # check whether it actually leads to higher pr as well
            should_update = within_comm & (comm_level < pr)

            # update the ones we should
            comm_level[should_update] = pr

        # has shape (num_eps, num_samples)
        return comm_level

    # gather comm levels now
    comm_levels = []

    for m_range in idx_ranges:
        prs_sub = prs[m_range]
        samples_sub = samples[m_range]
        comm_levels.append(
            get_comm_pr_one_range(prs_sub, samples_sub, samples_ref, labels)
        )

    # convert to torch
    # has shape (num_ranges, num_eps, num_samples)
    # return comm_levels
    comm_levels = torch.stack(comm_levels)

    # has shape (num_range, num_eps)
    std, mean = torch.std_mean(comm_levels, dim=-1, keepdim=True)
    mad, _ = torch.median(torch.abs(comm_levels - mean), dim=-1)
    mean = torch.squeeze(mean, -1)
    std = torch.squeeze(std, -1)
    return mean, mad


def get_all_stats(samples, samples_ref, labels, idx_ranges, prs):
    """Get all the stats we want to compute at the same time."""
    # check out a few options
    # each option has shape (num_f, num_eps)
    samples = samples.to(DEVICE)
    samples_ref = samples_ref.to(DEVICE)
    labels = labels.to(DEVICE)

    # compute all desired stats
    noise_stats = {
        "sm_2_diff": get_softmax_norm_diff(samples, samples_ref, norm=2),
        "agreement_overlap": get_overlap(samples, samples_ref),
        "accuracy": get_accuracy(samples, labels),
        "pr_commensurate": get_commensurate_pr(
            samples, samples_ref, labels, idx_ranges, prs
        ),
        "norm_2_diff": get_norm_diff(samples, samples_ref, norm=2),
        "norm_max_diff": get_norm_diff(
            samples, samples_ref, norm=float("inf")
        ),
        "sm_max_diff": get_softmax_norm_diff(
            samples, samples_ref, norm=float("inf")
        ),
        "agreement_overlap_min": get_overlap(
            samples, samples_ref, max_err=True
        ),
    }

    samples = samples.to(DEVICE_STORAGE)
    samples_ref = samples_ref.to(DEVICE_STORAGE)
    labels = labels.to(DEVICE_STORAGE)

    # switch to numpy at this point
    for k, v in noise_stats.items():
        noise_stats[k] = [v_one.cpu().numpy() for v_one in v]

    # now return them
    return noise_stats


def get_all_stats_batched(samples, samples_ref, labels, idx_ranges):
    """Do get_all_stats batch-wise."""
    # print stats computation time
    stats_time = -time.time()

    # DO THE BATCHES range-based!!!
    # now the stats are batched and we have them
    noise_stats_batched = []
    for idx_range in idx_ranges:
        batch = samples[idx_range]
        idx_ranges_fake = [np.arange(len(batch))]
        prs_fake = np.array(PRUNE_RATIOS)[idx_range]
        noise_stats_batched.append(
            get_all_stats(
                batch, samples_ref, labels, idx_ranges_fake, prs_fake
            )
        )

    # now de-batch them
    noise_stats = {}
    for k in noise_stats_batched[0]:
        noise_stats[k] = [
            np.concatenate([batch[k][i] for batch in noise_stats_batched])
            for i in range(2)
        ]

    # check final timing
    stats_time += time.time()
    print(f"Computing stats took {stats_time:.2f}s.")

    return noise_stats


# %% Do all the computations at once
# now get all the samples we need
NOISE_STATS = {}
IMGS_NOISY = None
try:
    NOISE_STATS.update(LOGGER.load_custom_state(NOISE_TAG))
except FileNotFoundError:
    # gather samples
    imgs, IMGS_NOISY, labels, samples = get_all_samples()
    samples_ref = samples[IDX_REF : IDX_REF + 1].clone().detach()

    # retrieve all the stats we want and save them
    NOISE_STATS.update(
        get_all_stats_batched(samples, samples_ref, labels, IDX_RANGES)
    )

    # store them
    LOGGER.save_custom_state(NOISE_STATS, NOISE_TAG)


# %% Do some plotting at the end
def get_fig_name(title, tag):
    """Get the name of the figure with the title and tag."""
    fig_name = "_".join(re.split("/|-|_|,", title)).replace(" ", "")
    return f"{fig_name}_noise_{tag}.pdf"


def plot_diff(
    diff_mean,
    diff_std,
    eps_values,
    legends,
    prune_ratios,
    title,
    ylabel,
    noise_type,
    name_filter,
    pr_filters,
    percentage_y,
    plots_dir,
    plots_tag,
):
    """Plot the mean differences with std dev."""

    # let's filter now with legends and prune_ratios
    filtered_name = np.array([name_filter in leg for leg in legends])

    # use those to determine best indices
    prune_ratios_filt = np.array(prune_ratios)[filtered_name]
    idxs_pr = [
        np.argmin(np.abs(prune_ratios_filt - pr_desired)).item()
        for pr_desired in pr_filters
    ]

    # full filter now
    filtered_pr = np.zeros_like(filtered_name)
    filtered_pr[np.where(filtered_name)[0][idxs_pr]] = True

    # make sure we also have reference networks included now
    filtered_ref = np.array(["network" in leg.lower() for leg in legends])

    legends_filtered = []
    colors_filtered = []
    ls_filtered = []
    for i in range(len(diff_mean)):
        if filtered_ref[i]:

            legend = legends[i].split()[0]
            color = "black" if "Unpruned" in legend else "magenta"
            ls = "--"
        elif filtered_pr[i]:
            legend = f"PR={prune_ratios[i]*100.0:.0f}%"
            color = cm.tab20(prune_ratios[i])
            ls = "-"
        else:
            continue
        legends_filtered.append(legend)
        colors_filtered.append(color)
        ls_filtered.append(ls)

    # convert data into the desired data format for the grapher and filter
    # (num_rep, num_y, num_rep, num_lines)
    diff = np.stack((diff_mean.T - diff_std.T, diff_mean.T + diff_std.T))
    diff = diff[:, :, None, np.logical_or(filtered_pr, filtered_ref)]

    # same for eps
    eps_values = np.broadcast_to(eps_values[None, :, None, None], diff.shape)

    grapher_diff = util.grapher.Grapher(
        x_values=eps_values,
        y_values=diff,
        folder=plots_dir,
        file_name=get_fig_name(title, plots_tag),
        ref_idx=None,
        x_min=-10.0,
        x_max=10.0,
        legend=legends_filtered,
        colors=colors_filtered,
        xlabel=f"{noise_type} noise level".capitalize(),
        ylabel=ylabel,
        title=title,
        linestyles=ls_filtered,
    )
    img_diff = grapher_diff.graph(
        percentage_y=percentage_y, remove_outlier=False, store=False
    )

    # customize legend
    img_diff.gca().legend(
        legends_filtered,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        ncol=1,
        borderaxespad=0,
    )

    # store it in the end
    grapher_diff.store_plot()

    return grapher_diff


def plot_comm_pr(
    pr_c_mean,
    pr_c_std,
    eps_values,
    legends,
    title,
    noise_type,
    colors,
    plots_dir,
    plots_tag,
):
    """Plot eps on x-axis and PR at commensurate accuracy."""
    # get the name based filter
    filtered_name = np.array(["network" not in leg for leg in legends])

    # convert data into the desired data format for the grapher and filter
    # (num_rep, num_y, num_rep, num_lines)
    pr_c = np.stack((pr_c_mean.T - pr_c_std.T, pr_c_mean.T + pr_c_std.T))
    pr_c = pr_c[:, :, None, filtered_name]

    # same for eps
    eps_values = np.broadcast_to(eps_values[None, :, None, None], pr_c.shape)

    # plot the data
    grapher_pp = util.grapher.Grapher(
        x_values=eps_values,
        y_values=pr_c,
        folder=plots_dir,
        file_name=get_fig_name(title, plots_tag),
        ref_idx=None,
        x_min=-10.0,
        x_max=10.0,
        legend=np.array(legends)[filtered_name],
        colors=np.array(colors)[filtered_name],
        xlabel=f"{noise_type} noise level".capitalize(),
        ylabel="Prune Potential, $\delta=0.5\%$",
        title=title,
    )
    plot_pp = grapher_pp.graph(
        percentage_y=True, remove_outlier=False, store=False
    )

    # customize legend to be single-column
    plot_pp.gca().legend(np.array(legends)[filtered_name], ncol=1)

    # set ylim to be always 0, 100
    plot_pp.gca().set_ylim([0, 100])

    # store it in the end
    grapher_pp.store_plot()

    return grapher_pp


def plot_noisy_images(imgs_noisy, eps_values, num_images=1, skip=3):
    """Plot a bunch of noisy image for each value of eps."""
    # pick a bunch of images with indices (let's pick first since it's random)
    num_imgs_total = len(imgs_noisy[0])
    idxs_imgs = torch.randint(num_imgs_total - 1, (num_images,))
    # idxs_imgs = good
    # print(idxs_imgs)
    imgs_to_plot = imgs_noisy[::skip, idxs_imgs, 0]
    eps_values = eps_values[::skip]
    num_eps = len(imgs_to_plot)

    # this is the grid to visualize the images
    plt.figure(figsize=(3 * num_eps, 3 * num_images))
    gs = plt.GridSpec(num_images, num_eps, wspace=0.1)

    def plot_image(ax, image):
        # unnormalize image
        image = np.asarray(image) * STD_C + MEAN_C
        image = np.clip(image, 0.0, 1.0)
        if image.shape[0] == 1:
            ax.imshow(image[0], cmap=plt.get_cmap("gray"))
        else:
            ax.imshow(np.moveaxis(image, 0, 2))
        ax.axis("off")

    for i, imgs_eps in enumerate(imgs_to_plot):
        for j, img in enumerate(imgs_eps):
            ax = plt.subplot(gs[j, i])
            plot_image(ax, imgs_to_plot[i, j].cpu())
            if j == 0:
                plt.title(f"eps={eps_values[i]:.2f}", size=25)


# desired title
TITLE = ", ".join([PARAM["network"]["name"], PARAM["network"]["dataset"]])
if "rewind" in PARAM["experiments"]["mode"]:
    TITLE += ", rewind"

# plot some noisy images if we can.
if IMGS_NOISY is not None:
    plot_noisy_images(IMGS_NOISY, EPS_VALUES, 2)


for method in METHODS:
    # Plot matching labels for each method
    fig = plot_diff(
        diff_mean=NOISE_STATS["agreement_overlap"][0],
        diff_std=NOISE_STATS["agreement_overlap"][1],
        eps_values=EPS_VALUES,
        legends=LEGENDS,
        prune_ratios=PRUNE_RATIOS,
        title=f"{TITLE}, {method}",
        ylabel="Matching labels",
        noise_type=NOISE,
        name_filter=method,
        pr_filters=DESIRED_PR,
        percentage_y=True,
        plots_dir=PLOTS_DIR,
        plots_tag="matching",
    )

    # Plot norm-based difference for each method
    fig = plot_diff(
        diff_mean=NOISE_STATS["sm_2_diff"][0],
        diff_std=NOISE_STATS["sm_2_diff"][1],
        eps_values=EPS_VALUES,
        legends=LEGENDS,
        prune_ratios=PRUNE_RATIOS,
        title=f"{TITLE}, {method}",
        ylabel="Softmax $\ell_2$-norm difference",
        noise_type=NOISE,
        name_filter=method,
        pr_filters=DESIRED_PR,
        percentage_y=False,
        plots_dir=PLOTS_DIR,
        plots_tag="diff",
    )

# Plot prune potential
fig = plot_comm_pr(
    pr_c_mean=NOISE_STATS["pr_commensurate"][0],
    pr_c_std=NOISE_STATS["pr_commensurate"][1],
    eps_values=EPS_VALUES,
    legends=[LEGENDS[idx_range[0]] for idx_range in IDX_RANGES],
    title=TITLE,
    noise_type=NOISE,
    colors=[COLORS[idx_range[0]] for idx_range in IDX_RANGES],
    plots_dir=PLOTS_DIR,
    plots_tag="prune_pot",
)
