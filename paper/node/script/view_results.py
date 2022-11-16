"""View and plot Neural ODE results."""
# %%
import os
import warnings
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from scipy import signal
import experiment
from experiment.util.file import get_parameters

# change working directory to src
from IPython import get_ipython

# make sure it's using only GPU here...
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # noqa


# switch to root folder for data
folder = os.path.abspath("")
if "paper/node/script" in folder:
    src_folder = os.path.join(folder, "../../..")
    os.chdir(src_folder)

# add script path to sys path
sys.path.append("./paper/node/script")

# %% Define some parameters
FILES = [
    # # TOY, VANILLA CNF GENERATIVE MODEL EXPERIMENTS
    # "paper/node/param/toy/ffjord/gaussians/vanilla_l2_h128.yaml",
    # "paper/node/param/toy/ffjord/gaussiansspiral/vanilla_l4_h64.yaml",
    # "paper/node/param/toy/ffjord/spirals/vanilla_l4_h64.yaml",
    # # TOY, GENERATIVE MODEL EXPERIMENTS
    # "paper/node/param/toy/ffjord/gaussians/l4_h64_sigmoid_da.yaml",
    # "paper/node/param/toy/ffjord/gaussians/l2_h128_sigmoid_da.yaml",
    # "paper/node/param/toy/ffjord/gaussiansspiral/l4_h64_sigmoid_da.yaml",
    # "paper/node/param/toy/ffjord/spirals/l4_h64_sigmoid_da.yaml",
    # #
    # # TOY, CLASSIFICATION EXPERIMENTS
    # "paper/node/param/toy/node/concentric/l2_h128_tanh_da.yaml",
    # "paper/node/param/toy/node/moons/l2_h3_tanh_da.yaml",
    # "paper/node/param/toy/node/moons/l2_h32_tanh_da.yaml",
    # "paper/node/param/toy/node/moons/l2_h64_tanh_da.yaml",
    # "paper/node/param/toy/node/moons/l2_h128_tanh_da.yaml",
    # "paper/node/param/toy/node/spirals/l2_h64_relu_da.yaml",
    # #
    # # TABULAR EXPERIMENTS
    # "paper/node/param/tabular/power/l3_hm10_f5_tanh.yaml",
    # "paper/node/param/tabular/gas/l3_hm20_f5_tanh.yaml",
    # "paper/node/param/tabular/hepmass/l2_hm10_f10_softplus.yaml",
    # "paper/node/param/tabular/miniboone/l2_hm20_f1_softplus.yaml",
    # "paper/node/param/tabular/bsds300/l3_hm20_f2_softplus.yaml",
    # #
    # # IMAGE EXPERIMENTS
    # "paper/node/param/cnf/mnist_multiscale.yaml",
    # "paper/node/param/cnf/cifar_multiscale.yaml",
]

PLOT_FILTERS = [
    ["WT", "FT"],
    ["WT"],
    # ["FT"],
]

STYLE_KWARGS = {
    "savgol_on": True,
    "savgol_mean": {"window_length": 3, "polyorder": 1},
    "savgol_std": {"window_length": 9, "polyorder": 1},
    "label": {"fontsize": 20},
    "tick": {"labelsize": 16},
    "xlim": [0, 85],
    "ylim": [1.5, 1.85],
    "legend": {
        "loc": "upper left",
        "bbox_to_anchor": (0.1, 1.3),
        "fontsize": 20,
    },
    "WT": {
        "plot": {"color": "darkblue", "ls": "-"},
        "fill": {"color": "lightskyblue", "alpha": 0.4},
    },
    "FT": {
        "plot": {"color": "darkgreen", "ls": "--"},
        "fill": {"color": "green", "alpha": 0.2},
    },
}

NUM_REP_LOSS = 12  # we want a total of 12 reps for the loss for better std dev

PLOT_FOLDER_SPECIAL = os.path.abspath("data/node/plots")
INLINE_PLOT = False

GEN_NODE_FIGS = False
REGEN_NODE_FIGS = False
GEN_ALL_NODE_FIGS = False
REGEN_FIGS = False

GEN_PAPER_FIGS_LOSS = True
GEN_PAPER_FIGS_DISTRIBUTION = True


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


def generate_node_figs(logger, cnf_plots=False):
    """Generate and store the Neural ODE figures for each model."""
    with HiddenPrints():
        evaluator = experiment.Evaluator(logger)
        loader_test = evaluator.get_dataloader("test")[0]

    if cnf_plots:
        from plots_cnf import plot_all
    else:
        from plots2d import plot_all

    # for n_idx in range(evaluator._num_nets):
    for n_idx in range(1):
        for r_idx in range(evaluator._num_repetitions):
            for s_idx, keep_ratio in enumerate(evaluator._keep_ratios):
                for method_name in evaluator._method_names:
                    if "ReferenceNet" in method_name and s_idx > 0:
                        continue
                    tag = "_".join(
                        [
                            method_name,
                            f"n{n_idx}",
                            f"r{r_idx}",
                            f"i{s_idx}",
                            f"p{keep_ratio:.4f}",
                        ]
                    )
                    plt_folder = os.path.join(logger._plot_dir, "flow", tag)

                    if os.path.exists(plt_folder) and not REGEN_NODE_FIGS:
                        continue
                    with HiddenPrints():
                        try:
                            net = evaluator.get_by_pr(
                                prune_ratio=1.0 - keep_ratio,
                                method=method_name,
                                n_idx=n_idx,
                                r_idx=r_idx,
                            ).compressed_net.torchnet
                        except FileNotFoundError:
                            continue
                    print(plt_folder)
                    plot_all(
                        net,
                        loader_test,
                        plot_folder=plt_folder,
                        all_p=GEN_ALL_NODE_FIGS
                        or "ReferenceNet" in method_name,
                    )


def get_results(file, logger, gen_node, regen_figs):
    """Grab all the results according to the file."""
    results = []
    params = []
    labels = []
    graphers_all = []
    # Loop through all experiments
    for param in get_parameters(file, 1, 0):
        # initialize logger and setup parameters
        with HiddenPrints():
            logger.initialize_from_param(param, setup_print=False)

        # don't
        try:
            state = logger.get_global_state()
        except ValueError:
            print("Global state not computed, handle with care!")
            state = copy.deepcopy(logger._stats)

        # extract the results
        results.append(copy.deepcopy(state))
        params.append(copy.deepcopy(param))

        # extract the legend (based on heuristic)
        label = param["generated"]["datasetTest"].split("_")
        if len(label) > 2:
            label = label[2:]
        labels.append("_".join(label))

        # store custom plots for neural ode as well.
        # only do that for Toy Examples though ...
        if gen_node and "toy" in file:
            generate_node_figs(logger, cnf_plots="ffjord" in file)

        if not regen_figs or not logger.state_loaded:
            continue

        # extract the plots and store them.
        try:
            with HiddenPrints():
                graphers = logger.generate_plots(store_figs=False)
                for grapher in graphers:
                    grapher.store_plot()
                graphers_all.append(graphers)
        except:
            print("Could not generate main graphs.")
            graphers_all.append([])

    return results, params, labels, graphers_all


def get_and_store_results(file, logger, gen_node=False, regen_figs=False):
    print(f"PARAM FILE: {file}")
    # get the results specified in the file (and hopefully pre-computed)
    results, params, _, _ = get_results(file, logger, gen_node, regen_figs)

    for param in params:
        print(f"PLOT FOLDER: {param['generated']['plotDir']}\n")

    return results, params


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
STATS_ALL = []
PARAM_ALL = []
for file in FILES:
    STATS, PARAM = get_and_store_results(
        file, LOGGER, gen_node=GEN_NODE_FIGS, regen_figs=REGEN_FIGS
    )
    STATS_ALL.append(STATS)
    PARAM_ALL.append(PARAM)


# %% now re-plot the loss so it looks better with smoothing
def resample_loss(logger, i_gen=None):
    """Resample the loss from the networks and return results."""
    if i_gen is None:
        tag_gen = logger.dataset_test
    else:
        tag_gen = f"{logger.dataset_test}_regen_{i_gen}"

    # try loading the re-generated loss if it exists and is compatible
    # we should also check that we get the same valid sizes since they
    # additional data might have generated when less networks were available
    stats_new = logger.load_custom_state(tag_gen)
    if logger._check_compatibility(stats_new):
        mask_new = np.all(stats_new["sizes"] != 0.0, axis=(0, 2))
        mask_old = np.all(logger.sizes != 0.0, axis=(0, 2))
        if np.all(mask_new == mask_old):
            print("Loaded re-sampled stats")
            return stats_new

    print("Generating re-sampled stats.")

    with HiddenPrints():
        evaluator = experiment.Evaluator(logger)

    # store prune ratios and add zero prune ratio
    prune_ratios = 1 - np.array(evaluator._keep_ratios)
    prune_ratios = np.concatenate(([0.0], prune_ratios))

    for n_idx in range(evaluator._num_nets):
        for r_idx in range(evaluator._num_repetitions):
            for s_idx, keep_ratio in enumerate(evaluator._keep_ratios):
                for a_idx, method_name in enumerate(evaluator._method_names):
                    if "ReferenceNet" in method_name and s_idx > 0:
                        continue
                    with HiddenPrints():
                        try:
                            ffjord_net = evaluator.get_by_pr(
                                prune_ratio=1.0 - keep_ratio,
                                method=method_name,
                                n_idx=n_idx,
                                r_idx=r_idx,
                            )
                        except FileNotFoundError as f_e:
                            if "ReferenceNet" in method_name:
                                raise f_e
                            else:
                                continue
                    # now re-do the stats
                    logger.update_global_state(
                        n_idx=n_idx, s_idx=s_idx, r_idx=r_idx, a_idx=a_idx
                    )
                    evaluator._do_stats(ffjord_net.cuda())

    # store re-generated stats
    if tag_gen is not None:
        logger.save_custom_state(logger._stats, tag_gen)
        print("Saving re-generated data")

    return copy.deepcopy(logger._stats)


def format_as_str(num):
    if num / 1e9 > 1:
        factor, suffix = 1e9, "B"
    elif num / 1e6 > 1:
        factor, suffix = 1e6, "M"
    elif num / 1e3 > 1:
        factor, suffix = 1e3, "K"
    else:
        factor, suffix = 1e0, ""

    num_factored = num / factor
    if num_factored / 1e2 > 1:
        num_rounded = str(int(round(num_factored)))
    elif num_factored / 1e1 > 1:
        num_rounded = f"{num_factored:.1f}"
    else:
        num_rounded = f"{num_factored:.2f}"
    return f"{num_rounded}{suffix} % {num}"


def plot_loss(
    logger,
    param,
    stats,
    plot_filters,
    style_kwargs,
    plt_folder,
    num_rep,
    compression_rate=False,
    use_loss=True,
):
    """Plot everything starting from stats and param."""
    # get reference index and names
    idx_ref = stats["methods"].index("ReferenceNet")
    names = np.delete(stats["names"], idx_ref)

    # initialize logger to current parameters
    with HiddenPrints():
        logger.initialize_from_param(param, setup_print=False)

    def _extract_pr_loss(stats):
        # [num_nets, num_intervals, num_repetitions, num_algorithms]
        prune_ratios = 100.0 * (1.0 - stats["sizes"])
        # prune_ratios = 1.0 / stats["sizes"]
        if use_loss:
            loss = copy.deepcopy(stats["loss"])
        else:
            loss = copy.deepcopy(stats["error"]) * 100.0

        # add 0 prune ratio to data
        prune_ratios = np.pad(prune_ratios, [(0, 0), (1, 0), (0, 0), (0, 0)])
        prune_ratios[:, 0] = 0.0
        loss = np.pad(loss, [(0, 0), (1, 0), (0, 0), (0, 0)])
        loss[:, 0] = loss[:, 1, :, idx_ref : idx_ref + 1]

        # remove ref idx, shape=[num_nets, num_intervals, num_rep, num_alg - 1]
        prune_ratios = np.delete(prune_ratios, idx_ref, axis=3)
        loss = np.delete(loss, idx_ref, axis=3)

        return prune_ratios, loss

    # get pr and loss with re-sampling always
    prune_ratios, loss = None, None

    # re-generate loss until we have enough repetitions
    i_gen = 0
    while loss is None or loss[:, 0, :, 0].size < num_rep:
        print(f"\nResampling Loss, i_gen={i_gen}")
        if num_rep > 1:
            stats_new = resample_loss(logger, i_gen)
        else:
            stats_new = resample_loss(logger)
        pr_new, loss_new = _extract_pr_loss(stats_new)
        if prune_ratios is None:
            prune_ratios = pr_new
            loss = loss_new
        else:
            prune_ratios = np.concatenate((prune_ratios, pr_new), axis=2)
            loss = np.concatenate((loss, loss_new), axis=2)
        i_gen += 1

    def _extract_valid_pr_loss(idx_alg):
        """Extract valid PRs and losses for desired algorithm index."""
        # shape=[num_nets, num_intervals, num_rep, num_alg - 1]
        num_intervals = prune_ratios.shape[1]
        pr_m, l_m, l_std = [], [], []
        for i_pr in range(num_intervals):
            # extract raw PR, loss for desired algorithm and interval
            pr_one_i = prune_ratios[:, i_pr, :, idx_alg].flatten()
            loss_one_i = loss[:, i_pr, :, idx_alg].flatten()

            # determine valid entries/repetitions
            valid = pr_one_i != 100.0

            # don't add if nothing valid
            if sum(valid) < 1:
                continue

            # filter for valid entries
            pr_one_i = pr_one_i[valid]
            loss_one_i = loss_one_i[valid]

            # store stats
            pr_m.append(np.mean(pr_one_i))
            l_m.append(np.mean(loss_one_i))
            l_std.append(np.std(loss_one_i))

        pr_m, l_m, l_std = np.asarray([pr_m, l_m, l_std])
        return pr_m, l_m, l_std

    def _plot(filter, legend_on=True):
        fig = plt.figure()
        sns.set_theme()
        legends = []
        legends_lookup = {
            "WT": "Unstructured Pruning",
            "FT": "Structured Pruning",
        }

        for name in filter:
            # get right data
            idx = np.argwhere(names == name)
            if len(idx) != 1:
                continue
            idx = idx[0].item()

            # get valid PRs and loss
            pr, l_m, l_std = _extract_valid_pr_loss(idx)

            # collect names for legend
            legends.append(legends_lookup[name])

            # try some smoothing
            if style_kwargs["savgol_on"]:
                l_m_filt = signal.savgol_filter(
                    l_m, **style_kwargs["savgol_mean"]
                )
                l_std_filt = signal.savgol_filter(
                    l_std, **style_kwargs["savgol_std"]
                )
            else:
                l_m_filt = l_m
                l_std_filt = l_std

            # plot
            # fig.gca().plot(pr, l_m, color="red")
            fig.gca().plot(pr, l_m_filt, **style_kwargs[name]["plot"])
            fig.gca().fill_between(
                pr,
                l_m_filt - l_std_filt,
                l_m_filt + l_std_filt,
                **style_kwargs[name]["fill"],
            )

        # axis labels
        if compression_rate:
            fig.gca().set_xlabel("Compression Rate", **style_kwargs["label"])
        else:
            fig.gca().set_xlabel("Prune Ratio (%)", **style_kwargs["label"])
        if use_loss:
            fig.gca().set_ylabel("Loss (NLL)", **style_kwargs["label"])
        else:
            fig.gca().set_ylabel("Top-1 Error (%)", **style_kwargs["label"])
        # ticks
        fig.gca().tick_params(axis="both", **style_kwargs["tick"])

        # x limits and y limits
        fig.gca().set_xlim(style_kwargs["xlim"])
        fig.gca().set_ylim(style_kwargs["ylim"])

        # legend now
        if legend_on:
            fig.gca().legend(
                legends, ncol=len(legends), **style_kwargs["legend"]
            )

        # a few stylistic changes
        fig.gca().spines["top"].set_visible(False)
        fig.gca().spines["right"].set_visible(False)
        fig.set_tight_layout(True)

        return fig

    for filters in plot_filters:
        # check if all methods that filters wants exist
        if not all([filt in names for filt in filters]):
            print(filters)
            continue
        # generate and store figure
        fig = _plot(filters, legend_on=False)
        file_name = "_".join(filters) + ".pdf"
        file_name = os.path.join(plt_folder, file_name)
        os.makedirs(plt_folder, exist_ok=True)
        fig.savefig(file_name, bbox_inches="tight")

    # now also print data
    size_abs = np.mean(stats["sizes_total"])
    for idx_alg, name in enumerate(names):
        prs_one, losses_one, _ = _extract_valid_pr_loss(idx_alg)
        for pr, loss_one in zip(prs_one, losses_one):
            size_pruned = (1 - pr / 100.0) * size_abs
            print(
                f"Sparse Flows ({name}, PR={int(round(pr))}\\%) & "
                f"{loss_one:.2f} & {format_as_str(size_pruned)}"
            )


def plot_flow(logger, param, plt_folder, cnf_plots=True):
    """Plot the distribution beautifully."""
    if cnf_plots:
        import plots_cnf as plots
    else:
        import plots2d as plots

    print(f"PLOT FOLDER: {plt_folder}")

    def _plot_distribution(plots_kwargs, tag):
        # plots once with the default scatter plot
        fig = plt.figure(figsize=(5, 5))
        sns.set_style("ticks")
        axis = fig.gca()

        plots.plot_for_sweep(axis=axis, **plots_kwargs)

        if cnf_plots:
            axis.set_xlim([-2, 2])
            axis.set_ylim([-2, 2])
            axis.set_aspect("equal")
        else:
            axis.set_aspect(1.5)
        plt.axis("off")
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
        plt.tight_layout()

        # store first plot
        plt_folder_original = os.path.join(plt_folder, "distribution_original")
        file_name = os.path.join(plt_folder_original, tag + ".jpg")
        os.makedirs(plt_folder_original, exist_ok=True)
        fig.savefig(file_name, bbox_inches="tight", pad_inches=0)

        # now re-load plot and filter out light colors
        if cnf_plots:
            img = np.copy(np.asarray(Image.open(file_name)))
            if not (IN_JUPYTER and INLINE_PLOT):
                threshold = 150
            else:
                threshold = 200
            img[img > threshold] = 255

            # show filtered plot
            fig2 = plt.figure(figsize=(5, 5))
            plt.imshow(img)
            fig2.gca().set_aspect("equal")
            plt.axis("off")
            plt.tight_layout()

            # store filtered plot
            plt_folder_filtered = os.path.join(
                plt_folder, "distribution_filtered"
            )
            file_name2 = os.path.join(plt_folder_filtered, tag + ".jpg")
            os.makedirs(plt_folder_filtered, exist_ok=True)
            # Image.fromarray(img).save(file_name2)
            fig2.savefig(file_name2, bbox_inches="tight", pad_inches=0)

        if not (IN_JUPYTER and INLINE_PLOT):
            plt.close(fig)
            if cnf_plots:
                plt.close(fig2)

    def _plot_field(plots_kwargs, tag):
        fig = plt.figure(figsize=(5, 5))
        sns.set_style("ticks")
        axis = fig.gca()

        # PLOTTING CODE
        plots.plot_static_vector_field(axis=axis, **plots_kwargs)

        if cnf_plots:
            axis.set_xlim([-2, 2])
            axis.set_ylim([-2, 2])
            axis.set_aspect("equal")
        else:
            axis.set_aspect(1.5)
        plt.axis("off")
        axis.set_title(None)
        axis.get_xaxis().set_ticks([])
        axis.get_yaxis().set_ticks([])
        plt.tight_layout()

        # store plot
        plt_folder_field = os.path.join(plt_folder, "field")
        file_name = os.path.join(plt_folder_field, tag + ".jpg")
        os.makedirs(plt_folder_field, exist_ok=True)
        fig.savefig(file_name, bbox_inches="tight", pad_inches=0)

        if not (IN_JUPYTER and INLINE_PLOT):
            plt.close(fig)

    def _plot_trajectory(plots_kwargs, tag, labels=False):
        sns.set_context("paper", font_scale=1.5)
        fig = plt.figure(figsize=(5, 3.5))
        axis1 = fig.add_subplot(211)
        axis2 = fig.add_subplot(212)

        # PLOTTING CODE
        plots.plot_2D_depth_trajectory(
            axis1=axis1, axis2=axis2, **plots_kwargs
        )

        # axis limits
        xlim = [0, 1]
        if cnf_plots:
            xlim = xlim[::-1]
        axis1.set_xlim(xlim)
        axis2.set_xlim(xlim)

        ylim1 = axis1.get_ylim()
        ylim2 = axis2.get_ylim()
        ylim = np.maximum(ylim1, ylim2)
        axis1.set_ylim(ylim)
        axis2.set_ylim(ylim)

        # axis layout
        sns.despine(offset=10, trim=True)
        axis1.get_xaxis().set_ticks([])
        axis1.get_xaxis().set_visible(False)
        axis1.spines["bottom"].set_visible(False)
        fig.tight_layout()

        # store plot
        plt_folder_traj = os.path.join(plt_folder, "trajectory")
        file_name = os.path.join(plt_folder_traj, tag + ".jpg")
        os.makedirs(plt_folder_traj, exist_ok=True)
        fig.savefig(file_name, bbox_inches="tight", pad_inches=0)

        # setup labels as separate plot
        labels = labels and not cnf_plots
        if labels:
            from matplotlib.lines import Line2D

            legend_handles = {}
            for color, label in zip(["midnightblue", "darkorange"], [0, 1]):
                legend_handles[f"Class {label}"] = Line2D(
                    [0], [0], color=color, lw=1.5
                )
            fig_labels = plt.figure(figsize=(1, 1))
            fig_labels.gca().legend(
                list(legend_handles.values()), list(legend_handles.keys())
            )
            fig_labels.gca().set_axis_off()
            fig_labels.tight_layout()
            file_name_labels = "labels.pdf"
            file_name_labels = os.path.join(plt_folder_traj, file_name_labels)
            fig_labels.savefig(
                file_name_labels, bbox_inches="tight", pad_inches=0
            )

        # close figure
        if not (IN_JUPYTER and INLINE_PLOT):
            plt.close(fig)
            if labels:
                plt.close(fig_labels)

    with HiddenPrints():
        logger.initialize_from_param(param, setup_print=False)
        evaluator = experiment.Evaluator(logger)
        loader_test = evaluator.get_dataloader("test")[0]

    # store prune ratios and add zero prune ratio
    prune_ratios = 1 - np.array(evaluator._keep_ratios)
    prune_ratios = np.concatenate(([0.0], prune_ratios))

    for n_idx in range(evaluator._num_nets):
        for r_idx in range(evaluator._num_repetitions):
            for s_idx, pr in enumerate(prune_ratios):
                for method_name in evaluator._method_names:
                    if "ReferenceNet" in method_name:
                        continue
                    with HiddenPrints():
                        try:
                            if pr == 0.0:
                                lookup_name = "ReferenceNet"
                            else:
                                lookup_name = method_name
                            ffjord_net = evaluator.get_by_pr(
                                prune_ratio=pr,
                                method=lookup_name,
                                n_idx=n_idx,
                                r_idx=r_idx,
                            ).compressed_net.torchnet
                        except FileNotFoundError:
                            continue

                    tag = "_".join(
                        [
                            logger.names[logger.methods.index(method_name)],
                            f"n{n_idx}",
                            f"r{r_idx}",
                            f"i{s_idx:02d}",
                            f"p{int(pr*100):03d}",
                        ]
                    )

                    # setup and generate data, and plot
                    plots_kwargs = plots.prepare_data(
                        ffjord_net.cuda(), loader_test, n_samp=50000
                    )
                    _plot_distribution(plots_kwargs, tag)
                    try:
                        _plot_field(plots_kwargs, tag)
                    except ValueError:
                        pass
                    _plot_trajectory(
                        plots_kwargs,
                        tag,
                        labels=n_idx == 0 and r_idx == 0 and s_idx == 0,
                    )

                print(f"Done with pr={pr:.2f}, r_idx={r_idx}, n_idx={n_idx}")


for STATS, PARAMS in zip(STATS_ALL, PARAM_ALL):
    for STAT, PARAM in zip(STATS, PARAMS):
        NET_NAME = PARAM["generated"]["netName"]
        DSET = PARAM["generated"]["datasetTest"]
        NETWORK = PARAM["network"]["name"]
        PLT_FOLDER = os.path.join(PLOT_FOLDER_SPECIAL, DSET, NET_NAME)
        IS_CNF = "ffjord_" in NET_NAME or "cnf_" in NET_NAME
        if GEN_PAPER_FIGS_LOSS:
            FOLDER_LOSS = os.path.join(PLT_FOLDER, "loss")
            plot_loss(
                LOGGER,
                PARAM,
                STAT,
                PLOT_FILTERS,
                STYLE_KWARGS,
                FOLDER_LOSS,
                NUM_REP_LOSS,
                use_loss=IS_CNF,
            )
        if (
            GEN_PAPER_FIGS_DISTRIBUTION
            and "toy" in PARAM["network"]["dataset"].lower()
        ):
            plot_flow(LOGGER, PARAM, PLT_FOLDER, IS_CNF)
