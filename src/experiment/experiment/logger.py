"""A module that contains our custom Logger classes to keep track."""
import os
import pathlib
import copy
import zipfile
import numpy as np
from torch.utils import tensorboard as tb

import torchprune.util.logging as tp_logging
import torchprune.util.train as tp_train

from .util import file as util_file
from .util.grapher import Grapher


class Logger:
    """Our custom logger to keep track of experiments."""

    def __getattribute__(self, attr):
        """Get attribute including easy access to self._stats."""
        __dict__ = object.__getattribute__(self, "__dict__")
        if "_stats" in __dict__ and attr in __dict__["_stats"]:
            return __dict__["_stats"][attr]
        else:
            return object.__getattribute__(self, attr)

    def __setattr__(self, attr, value):
        """Set an attribute and set it self._stats if possible."""
        __dict__ = object.__getattribute__(self, "__dict__")
        if "_stats" in __dict__ and attr in __dict__["_stats"]:
            __dict__["_stats"].__setitem__(attr, value)
        else:
            object.__setattr__(self, attr, value)

    def __init__(self):
        """Create a summary writer logging to log_dir for each algorithm.

        Standard Layout for data is:
            * [numLayers, numNets, numIntervals, numRepetitions, numAlgorithms,
               extraDims]
            * numLayers and/or extraDims are dropped if not needed.

        """
        # Immediately define self._stats so __getattr__ doesn't complain
        self._stats = {}

        # store parameters
        self.param = None

        # get names and colors
        self._class_to_names = None
        self._colors = None
        self._name_ref = None
        self._color_ref = None
        self.names_metrics = None

        # for printing consistent strings
        self._last_print_name = ""

        # for file logger
        self._stdout_logger = None

        # for train logging
        self._train_logger = None

        # to see whether we have loaded the stats
        self.state_loaded = False

        # directory management
        self._results_dir = None
        self._stdout_dir = None
        self._reports_dir = None
        self._log_dir = None
        self._data_dir = None
        self._plot_dir = None

        # writer stuff
        self._writer = None
        self._writer_general = None
        self._to_idx = None

        # a few variables to keep track off
        self._samples = None

        # these are stats
        # We keep them as separate dict so we can easily store/retrieve them.
        self._stats = {
            "methods": None,
            "names": None,
            "time_tag": None,
            "error": None,
            "error_train": None,
            "error_test": None,
            "error5": None,
            "error5_train": None,
            "error5_test": None,
            "loss": None,
            "loss_train": None,
            "loss_test": None,
            "sizes": None,
            "flops": None,
            "samples_per_layer": None,
            "sizes_total": None,
            "flops_total": None,
            "stats_comm": None,
            "layout": "[num_layers, num_nets, num_intervals, num_repetitions, "
            "num_algorithms, extra_dims]",
            "dataset_train": None,
            "dataset_test": None,
        }

        # a few experiment stats
        self.responsibility = None
        self.global_tag = None
        self._num_repetitions = None
        self._num_algorithms = None
        self._num_intervals = None
        self._num_nets = None
        self._x_min = None
        self._x_max = None

        # worker info
        self._num_workers = None
        self._id_worker = None
        self._is_primary_worker = None
        self._is_collector_worker = None

        # running indices to keep track of experiment
        self.s_idx = None
        self.n_idx = None
        self.r_idx = None
        self.a_idx = None

    def initialize_from_param(self, param, setup_print=True):
        """Initialize the logger directly with a set of parameters."""
        # call __init__ to reset all attributes to None
        self.__init__()

        # set the mosek licence file path
        util_file.set_mosek_path()

        # copy a copy of the parameters
        self.param = param

        # data sets
        self.dataset_train = self.param["network"]["dataset"]
        self.dataset_test = self.param["generated"]["datasetTest"]

        # create results directory
        self.time_tag = self.param["generated"]["timeTag"]
        self._results_dir = self.param["generated"]["resultsDir"]

        # global tag for Tensorboard
        self.global_tag = param["generated"]["globalTag"]

        # directory management
        self._stdout_dir = param["generated"]["stdoutDir"]
        self._log_dir = param["generated"]["logDir"]
        self._data_dir = param["generated"]["dataDir"]
        self._reports_dir = param["generated"]["reportsDir"]
        self._plot_dir = param["generated"]["plotDir"]

        # Populate
        self._x_min = self.param["experiments"]["plotting"]["minVal"]
        self._x_max = self.param["experiments"]["plotting"]["maxVal"]

        # get names and colors
        self.names = self.param["generated"]["network_names"]
        self.methods = self.param["experiments"]["methods"]
        self._colors = self.param["generated"]["network_colors"]
        self._class_to_names = self.param["network_names"]
        self.names_metrics = [
            metric.short_name
            for metric in tp_train.get_test_metrics(
                param["generated"]["training"]
            )
        ]

        # a few stats to keep track off
        self._num_repetitions = self.param["experiments"]["numRepetitions"]
        self._num_algorithms = len(self.names)

        self._num_intervals = len(self.param["generated"]["keepRatios"])
        self._num_nets = self.param["experiments"]["numNets"]

        self._samples = np.array(self.param["generated"]["keepRatios"])

        self.responsibility = param["generated"]["responsibility"]

        # worker info
        self._num_workers = self.param["generated"]["numWorkers"]
        self._id_worker = self.param["generated"]["idWorker"]
        self._is_primary_worker = self.responsibility.flatten()[0]
        self._is_collector_worker = np.all(self.responsibility)

        # MAIN STATS  #
        # Dimension convention for main stats:
        # (numLayers, numNets, numIntervals, numRepetitions, numAlgorithms)
        # numLayers is dropped if not in use
        # These are the _stats that we compute during pruning.
        self.error = np.zeros(
            (
                self._num_nets,
                self._num_intervals,
                self._num_repetitions,
                self._num_algorithms,
            )
        )
        self.error_train = np.zeros(
            (
                self._num_nets,
                self._num_intervals,
                self._num_repetitions,
                self._num_algorithms,
                max(
                    self.param["generated"]["retraining"]["numEpochs"],
                    self.param["generated"]["training"]["numEpochs"],
                ),
            )
        )
        self.error_test = copy.deepcopy(self.error_train)

        self.error5 = copy.deepcopy(self.error)
        self.error5_train = copy.deepcopy(self.error_train)
        self.error5_test = copy.deepcopy(self.error_train)

        self.loss = copy.deepcopy(self.error)
        self.loss_train = copy.deepcopy(self.error_train)
        self.loss_test = copy.deepcopy(self.error_train)

        self.sizes = copy.deepcopy(self.error)
        self.flops = copy.deepcopy(self.error)

        self.samples_per_layer = -np.ones(
            (1, self._num_nets, self._num_intervals, 1, self._num_algorithms)
        )

        self.sizes_total = np.zeros(self._num_nets)
        self.flops_total = np.zeros(self._num_nets)
        # # # # # # # #

        # get's computed at the end during storage...
        self.stats_comm = {}

        # indices for experiments
        self.s_idx = 0
        self.n_idx = 0
        self.r_idx = 0
        self.a_idx = 0

        if not np.any(np.array(param["generated"]["responsibility"])):
            print("Skipping since no responsibilities")
            return

        # the old results directories from the generated parameters
        results_dir_prev = self.param["generated"]["resultsDirPrev"]

        # now either create the directory or load the results
        is_collective = False

        if results_dir_prev is None:
            os.makedirs(self._results_dir, exist_ok=True)
        else:
            tag = f"{self.global_tag}_{self.dataset_test}"
            data_dir = os.path.join(self._results_dir, "data")
            # go through all data files that match our tag
            # matches are either exact matches or match with _j*_i* for
            # workers
            matches = sorted(pathlib.Path(data_dir).glob(f"{tag}.npz"))
            matches += sorted(
                pathlib.Path(data_dir).glob(f"{tag}_j[0-9]*_i[0-9]*.npz")
            )
            for path in matches:
                # load the data
                data = self._load_custom_state(str(path.absolute()))
                # load the data into our actual state
                is_collective = self._load_partial_state(data)
                # if we had a collective load, everything is loaded!
                if is_collective:
                    break

        # create the directory (if not already existing)
        os.makedirs(self._stdout_dir, exist_ok=True)

        # setup logging (with convenience function)
        stdout_file = os.path.join(
            self._stdout_dir,
            f"experiment_{self.dataset_test}{self._get_worker_tag()}.log",
        )
        if setup_print:
            self._stdout_logger = tp_logging.setup_stdout(stdout_file)
        else:
            self._stdout_logger = None

        # setup train logger
        self._train_logger = tp_logging.TrainLogger(
            self._log_dir,
            stdout_file,
            self.global_tag,
            self._class_to_names,
        )

        # initialize writer
        self._writer = {}
        self._to_idx = {}
        for i, name in enumerate(self.names):
            self._writer[name] = tb.SummaryWriter(
                os.path.join(self._log_dir, name)
            )
            self._to_idx[name] = i

        # have one writer pertaining to general results...
        self._writer_general = tb.SummaryWriter(
            os.path.join(self._log_dir, "General")
        )

        # write parameters to tensorboard as text
        tp_logging.log_text(
            self._writer_general,
            "parameters",
            self.param["generated"]["paramMd"],
            0,
        )

        # Check for completeness of results to decide whether we are done
        self.state_loaded = self._check_completeness(
            self._stats, self.responsibility
        )
        # if we now have all the results but there was no collective load, we
        # once store the collected results, so that next time we can
        # collectively load everything.
        if self.state_loaded and not is_collective:
            print("Saving collected state.")
            self.save_global_state()

        # Save parameter file (as primary worker)
        if self._is_primary_worker:
            util_file.write_parameters(self.param, self._results_dir)

    def _get_worker_tag(self):
        """Get the worker tag ID info."""
        if self._is_collector_worker:
            return ""
        else:
            return f"_j{self._num_workers}_i{self._id_worker}"

    def _get_npz_filename(self, tag):
        """Get numpy file name for given custom partial tag."""
        return f"{self.global_tag}_{tag}{self._get_worker_tag()}.npz"

    def update_global_state(
        self, s_idx=None, n_idx=None, r_idx=None, a_idx=None
    ):
        """Update the experiment indices."""
        if s_idx is not None:
            self.s_idx = s_idx
        if n_idx is not None:
            self.n_idx = n_idx
        if r_idx is not None:
            self.r_idx = r_idx
        if a_idx is not None:
            self.a_idx = a_idx

    def _check_compatibility(self, data):
        """Check if _stats are compatible with data."""

        def _check():
            compatible = True

            # first check special _stats
            if "methods" in data:
                compatible &= (self.methods == data["methods"]).all()
            else:
                compatible &= (self.names == data["names"]).all()
            compatible &= (
                "dataset_test" in data
                and self.dataset_test == data["dataset_test"]
            )
            compatible &= self.layout == data["layout"]

            # check if sizes agree.
            compatible &= self.sizes.shape == data["sizes"].shape

            return compatible

        # checks may fail if data is not compatible, then return False
        try:
            return _check()
        except (ValueError, KeyError):
            return False

    def _check_completeness(self, data, resp=None):
        """Check if data contains sound and complete results."""
        # now we can construct the mask to update the data
        if resp is None:
            mask = np.ones(self.error.shape, dtype=np.bool)
        else:
            mask = np.broadcast_to(resp[:, np.newaxis], self.error.shape)
        return np.all(data["loss"][mask] != 0.0)

    def _load_partial_state(self, data_prev):
        """Load the partial state according to responsibility masks.

        return True iff we just loaded *everything* with this previous data.
        """
        # check compatibility
        if not self._check_compatibility(data_prev):
            return False

        # now we can construct the mask to update the data
        mask_this = np.array(self.param["generated"]["responsibility"])
        mask_this = np.broadcast_to(mask_this[:, np.newaxis], self.error.shape)
        mask_other = data_prev["loss"] != 0.0

        # only stats that are valid in both masks
        mask = mask_this & mask_other

        # check if this is a "collective" load
        is_collective = np.all(mask)

        # nothing to load if there is no overlap in responsibility
        if not np.any(mask):
            return is_collective

        # for total we need a different mask
        mask_total = np.any(mask, axis=(1, 2, 3))

        # list of parameters to update
        updatable = ["error", "loss", "sizes", "flops", "layer"]

        for key in data_prev:
            # check if relevant for partial loading.
            if not any(pattern in key.lower() for pattern in updatable):
                continue

            # check if key also exists in current stats
            if key not in self._stats:
                continue

            # for ref the mask is slightly different...
            if "total" in key.lower():
                mask_k = mask_total
            else:
                mask_k = mask

            # layer-wise only store that for repetition 0 but many layers
            if "layer" in key.lower():
                mask_k = np.broadcast_to(
                    mask_k[None, :, :, 0:1], data_prev[key].shape
                )

                # make sure it has the same number of layers
                if len(self._stats[key]) != len(data_prev[key]):
                    self._stats[key] = np.zeros_like(data_prev[key])

            # now update it.
            self._stats[key][mask_k] = data_prev[key][mask_k]

        # load comm report as well
        # (but only if it comes from a complete result set ...)
        if is_collective and "stats_comm" in data_prev:
            self.stats_comm = data_prev["stats_comm"]

        return is_collective

    def load_global_state(self):
        """Load the state of the logger from the numpy file."""
        # load data with appropriate file name ...
        data = self.load_custom_state(self.dataset_test)

        # check compatibility to load
        if not self._check_compatibility(data):
            raise ValueError(
                "The logger state you want to load does not "
                "agree with the current state."
            )

        # check completeness of results to load
        if not self._check_completeness(data):
            raise ValueError(
                "The results you want to load do not contain "
                "fully computed results."
            )

        # state is all we got in the stored file
        self._stats.update(data)

        # after this the state is loaded
        # (this is the only way to get state_loaded to be True!)
        self.state_loaded = True

    def load_custom_state(self, tag, data_dir=None):
        """Load and return the state stored under the desired tag."""
        # extract data to actual dictionary
        if data_dir is None:
            data_dir = self._data_dir
        filename = os.path.join(data_dir, self._get_npz_filename(tag))
        return self._load_custom_state(filename)

    def _load_custom_state(self, filename):
        """Load custom state simply based on filename."""
        data = {}
        try:
            data.update(np.load(filename, allow_pickle=True))
        except (zipfile.BadZipFile, OSError, EOFError):
            print("Pre-loading data failed due to broken zip-file.")

        # these are stored as 0d-array of type of 'object' --> extract item
        item_extraction = ["stats_comm", "dataset_test"]
        for key in item_extraction:
            if key in data:
                data[key] = np.array(data[key]).item()

        return data

    def get_global_state(self):
        """Get a dictionary containing all the computed stats."""
        # check that state is complete
        if not self.state_loaded:
            raise ValueError("The state is not fully loaded/computed yet.")

        return self._stats

    def save_global_state(self, fast_saving=False):
        """Save the global state and generate all the plots."""
        if self._is_collector_worker and not fast_saving:
            # create directories needed here.
            os.makedirs(self._plot_dir, exist_ok=True)

            # plot data and save plot
            # plotting should not throw error if it doesn't work
            try:
                self.generate_plots()
            except:  # noqa: E722
                print("Generating plots failed!")

            # store a few stats for usage
            self.stats_comm.update(self.compute_stats())

        # save data at the end
        os.makedirs(self._data_dir, exist_ok=True)
        self.save_custom_state(self._stats, self.dataset_test)

        # see if the stuff we store is also complete ...
        self.state_loaded = self._check_completeness(
            self._stats, self.responsibility
        )

        return self.state_loaded

    def save_custom_state(self, state_dict, tag):
        """Store custom dictionary as well."""
        # data management
        data_file_name = os.path.join(
            self._data_dir, self._get_npz_filename(tag)
        )

        # make sure directory exists
        os.makedirs(self._data_dir, exist_ok=True)

        # now save it
        np.savez(data_file_name, **state_dict)

    def set_total_size(self, size_total, flops_total):
        """Set the size and flops of the original network."""
        self.sizes_total[self.n_idx] = size_total
        self.flops_total[self.n_idx] = flops_total

    def compute_stats(self, store_report=True):
        """Compute stats for "commensurate" accuracy."""
        # reference error and reference loss
        ref_idx = self.names.index("ReferenceNet")
        error_ref = self.error[:, :, :, ref_idx : ref_idx + 1]
        error5_ref = self.error5[:, :, :, ref_idx : ref_idx + 1]
        loss_ref = self.loss[:, :, :, ref_idx : ref_idx + 1]

        # check that ref is complete
        resp_ref = np.zeros_like(self.responsibility)
        resp_ref[:, :, ref_idx] = self.responsibility[:, :, ref_idx]
        if not self._check_completeness(self._stats, resp_ref):
            raise ValueError("Valid ref data required.")

        # get the means
        e_mean_ref = error_ref.mean(0).mean(-2)[0].item()
        e5_mean_ref = error5_ref.mean(0).mean(-2)[0].item()
        lo_mean_ref = loss_ref.mean(0).mean(-2)[0].item()

        # get the total size
        size_total = self.sizes_total.mean(0).item()
        flops_total = self.flops_total.mean(0).item()

        # compute a error tensor for comparisons
        # --> incomplete values mapped to np.Inf
        error_comp = copy.deepcopy(self.error)

        # get best size level for desired commensurate level
        def _get_best(comm_level):
            # orig stats have dim (numNets, numIntervals, numRep, numAlg)
            # best stats have dim (numNets, numRep, numAlg)
            # start with unpruned performance as reference ...
            e_best = np.ones_like(self.error[:, 0]) * np.Inf
            e5_best = copy.deepcopy(e_best)
            lo_best = copy.deepcopy(e_best)
            siz_best = np.ones_like(e_best)
            flo_best = np.ones_like(e_best)

            # now loop through intervals and check out the best ...
            for i in range(self._num_intervals):
                # these ones satisfy the commensurate accuracy requirement
                good_error = error_ref[:, i] + comm_level >= error_comp[:, i]

                # now check for smaller size
                smaller = siz_best >= self.sizes[:, i]

                # we need gg and smaller to set it
                set_it = np.logical_and(good_error, smaller)

                # now actually set it
                e_best[set_it] = self.error[:, i][set_it]
                e5_best[set_it] = self.error5[:, i][set_it]
                lo_best[set_it] = self.loss[:, i][set_it]
                siz_best[set_it] = self.sizes[:, i][set_it]
                flo_best[set_it] = self.flops[:, i][set_it]

            return e_best, e5_best, lo_best, siz_best, flo_best

        # helper functions for markdown reports
        def _add_ref_report():
            return "\n\n".join(
                [
                    "# Reference Net",
                    "",
                    f"Network: {self.param['network']['name']}",
                    f"Dataset: {self.param['network']['dataset']}",
                    f"Test Dataset: {self.dataset_test}",
                    f"Original Size: {size_total:.2E} parameters",
                    f"Original FLOPs: {flops_total:.2E}",
                    f"Original {self.names_metrics[0]} Test Error: "
                    + f"{e_mean_ref * 100.0:4.2f}%",
                    f"Original {self.names_metrics[1]} Test Error: "
                    + f"{e5_mean_ref * 100.0:4.2f}%",
                    f"Original Test Loss: {lo_mean_ref:6.4f}",
                    "\n",
                ]
            )

        def _add_comm_header(comm_level):
            return f"## Commensurate Level {comm_level * 100.0:4.2f}%\n"

        def _add_comm_alg(a_idx, e_best, e5_best, lo_best, siz_best, flo_best):
            summary_str = self._gen_one_diag_str(
                error=100.0 * e_best[a_idx],
                error5=100.0 * e5_best[a_idx],
                loss=lo_best[a_idx],
                size=100.0 * siz_best[a_idx],
                size_abs=siz_best[a_idx] * size_total,
                flops=100.0 * flo_best[a_idx],
                flops_abs=flo_best[a_idx] * flops_total,
                prune_ratio=100.0 * (1 - siz_best[a_idx]),
                flops_ratio=100.0 * (1 - flo_best[a_idx]),
                error_diff=100.0 * (e_best[a_idx] - e_mean_ref),
                error5_diff=100.0 * (e5_best[a_idx] - e5_mean_ref),
            )
            summary_str = summary_str.replace("|", "\n\n")
            partial_str = f"### {self.names[a_idx]}\n\n"
            partial_str += summary_str + "\n\n"
            return partial_str

        def _finish_ref_report():
            return "\n"

        # GENERATE REPORT #####################################################
        # get possible commensurate levels
        commensurate = [
            -0.1,
            -0.05,
            -0.02,
            -0.01,
            -0.005,
            -0.0025,
            0.0,
            0.0025,
            0.005,
            0.01,
            0.02,
            0.05,
            0.1,
        ]

        # string for mark down report
        md_str = "# ACCURACY REPORT\n\n"

        # beginning
        md_str += _add_ref_report()

        # also store all the info ....
        # has shape (len(commensurate), numNets, numRep, numAlg)
        e_best = np.zeros(
            (
                len(commensurate),
                self._num_nets,
                self._num_repetitions,
                self._num_algorithms,
            )
        )
        e5_best = np.zeros_like(e_best)
        lo_best = np.zeros_like(e_best)
        siz_best = np.zeros_like(e_best)
        flo_best = np.zeros_like(e_best)

        # all comm levels and algorithms
        for i, comm in enumerate(commensurate):
            # compute the results
            (
                e_best[i],
                e5_best[i],
                lo_best[i],
                siz_best[i],
                flo_best[i],
            ) = _get_best(comm)

            # take the mean for the report
            e_best_i_m = e_best[i].mean(axis=(0, 1))
            e5_best_i_m = e5_best[i].mean(axis=(0, 1))
            lo_best_i_m = lo_best[i].mean(axis=(0, 1))
            siz_best_i_m = siz_best[i].mean(axis=(0, 1))
            flo_best_i_m = flo_best[i].mean(axis=(0, 1))

            # add to the report
            md_str += _add_comm_header(comm)
            for a_idx in range(self._num_algorithms):
                md_str += _add_comm_alg(
                    a_idx,
                    e_best_i_m,
                    e5_best_i_m,
                    lo_best_i_m,
                    siz_best_i_m,
                    flo_best_i_m,
                )

        # finish
        md_str += _finish_ref_report()

        # store
        if store_report:
            full_tag = self.global_tag + "/" + "Commensurate Report"
            full_name = f"report_{self.global_tag}_{self.dataset_test}.md"
            tp_logging.log_text(self._writer_general, full_tag, md_str, 0)
            os.makedirs(self._reports_dir, exist_ok=True)
            with open(os.path.join(self._reports_dir, full_name), "w") as mdf:
                mdf.write(md_str)

        # also return the results for later use
        return {
            "commensurate": commensurate,
            "e_best": e_best,
            "e5_best": e5_best,
            "lo_best": lo_best,
            "siz_best": siz_best,
            "flo_best": flo_best,
        }

    def generate_plots(self, store_figs=True):
        """Generate all the plots and save them."""
        # correct layout of samples_per_layer:
        # [numLayers, numNets, numIntervals, numRepetitions=1, numAlgorithms]
        # new layout of samplesPlot:
        # [numNets, numLayers, numIntervals, numAlgorithms]
        samples_plot = copy.deepcopy(self.samples_per_layer)
        samples_plot = np.squeeze(samples_plot, axis=3)
        samples_plot = np.swapaxes(samples_plot, 0, 1)

        # get keep_ratio per layer
        ref_idx = self.names.index("ReferenceNet")
        kr_per_layer = (
            self._samples[None, None, :, None]
            * samples_plot
            / samples_plot[:, :, :, ref_idx : ref_idx + 1]
        )

        # grapher stats
        num_layers = samples_plot.shape[1]
        num_layers = int(num_layers)
        layers = np.fromiter(range(num_layers), dtype=np.int) + 1

        # data for sample sizes per layer plot with standard convention
        layers = np.tile(layers, (self._num_algorithms, 1)).transpose()
        layers = layers[np.newaxis, :, np.newaxis, :]

        # grapher labels
        y_label_error = f"{self.names_metrics[0]} Test Accuracy"
        y_label_error5 = f"{self.names_metrics[1]} Test Accuracy"
        y_label_loss = "Test Loss"

        # grapher stuff
        legend = copy.deepcopy(np.array(self.names)).tolist()
        colors = copy.deepcopy(np.array(self._colors)).tolist()
        title = ", ".join(
            [
                self.param["network"]["name"],
                self.param["generated"]["datasetTest"].replace("_", "-"),
            ]
        )

        def _do_graphs(x_label, x_data, tag):
            x_min = 1.0 - min(self._x_max, max(self._samples))
            x_max = 1.0 - max(self._x_min, min(self._samples))

            def _flip_data(arr):
                return 1.0 - arr

            # modify the xData to represent Prune Ratio...
            x_data = _flip_data(x_data)

            # y values ...
            acc = _flip_data(self.error)
            acc5 = _flip_data(self.error5)

            # global tag with test dataset
            global_tag_test = f"{self.global_tag}_{self.dataset_test}"

            # grapher initialization + plotting
            grapher_error = Grapher(
                x_values=x_data,
                y_values=acc,
                folder=self._plot_dir,
                file_name=global_tag_test + "_acc_" + tag + ".pdf",
                ref_idx=ref_idx,
                x_min=x_min,
                x_max=x_max,
                legend=legend,
                colors=colors,
                xlabel=x_label,
                ylabel=y_label_error,
                title=title,
            )
            img_err = grapher_error.graph(
                percentage_x=True, percentage_y=True, store=store_figs
            )

            grapher_error5 = Grapher(
                x_values=x_data,
                y_values=acc5,
                folder=self._plot_dir,
                file_name=global_tag_test + "_acc5_" + tag + ".pdf",
                ref_idx=ref_idx,
                x_min=x_min,
                x_max=x_max,
                legend=legend,
                colors=colors,
                xlabel=x_label,
                ylabel=y_label_error5,
                title=title,
            )
            img_err5 = grapher_error5.graph(
                percentage_x=True, percentage_y=True, store=store_figs
            )

            grapher_loss = Grapher(
                x_values=x_data,
                y_values=self.loss,
                folder=self._plot_dir,
                file_name=global_tag_test + "_loss_" + tag + ".pdf",
                ref_idx=ref_idx,
                x_min=x_min,
                x_max=x_max,
                legend=legend,
                colors=colors,
                xlabel=x_label,
                ylabel=y_label_loss,
                title=title,
            )
            img_loss = grapher_loss.graph(percentage_x=True, store=store_figs)

            # also write images to Tensorboard
            if store_figs:
                self.log_image(
                    self._writer_general,
                    f"{self.dataset_test} Test {self.names_metrics[0]} {tag}",
                    img_err,
                    0,
                )
                self.log_image(
                    self._writer_general,
                    f"{self.dataset_test} Test {self.names_metrics[1]} {tag}",
                    img_err5,
                    0,
                )
                self.log_image(
                    self._writer_general,
                    self.dataset_test + "Test Loss" + tag,
                    img_loss,
                    0,
                )

            return grapher_error, grapher_error5, grapher_loss

        # keep a list of figures around
        graphers = []

        # do parameter and flop plots
        graphers.extend(_do_graphs("Pruned Parameters", self.sizes, "param"))
        graphers.extend(_do_graphs("Pruned FLOPs", self.flops, "flops"))

        # do some layer-wise graphs
        title_layer = ", ".join(
            [
                self.param["network"]["name"],
                self.param["network"]["dataset"].replace("_", "-"),
            ]
        )

        def _do_layer_graph(x_label, y_label, y_data, tag, ref_idx=None):
            grapher_layer = Grapher(
                x_values=layers,
                y_values=y_data,
                folder=self._plot_dir,
                file_name=self.global_tag + f"_{tag}.pdf",
                ref_idx=ref_idx,
                x_min=np.min(layers),
                x_max=np.max(layers),
                legend=legend,
                colors=colors,
                xlabel=x_label,
                ylabel=y_label,
                title=title_layer,
            )
            img_layer = grapher_layer.graph_histo(
                show_delta=ref_idx is not None, store=store_figs
            )

            if store_figs:
                self.log_image(self._writer_general, tag, img_layer, 0)

            return grapher_layer

        graphers.append(
            _do_layer_graph(
                "Budget Allocation over Layers",
                "Percentage of Budget",
                samples_plot,
                "samples",
            )
        )

        graphers.append(
            _do_layer_graph(
                "Prune Ratio per Layer",
                "Prune Ratio",
                1 - kr_per_layer,
                "layer_pr",
                ref_idx,
            )
        )

        return graphers

    def get_train_logger(self):
        """Get the tp_logging.TrainLogger instance."""
        return self._train_logger

    def log_scalar(
        self,
        writer,
        tag,
        value,
        step,
        add_n_idx=False,
        add_r_idx=False,
        add_s_idx=False,
    ):
        """Log a scalar variable (wrapper for util)."""
        tp_logging.log_scalar(
            writer=writer,
            global_tag=self.global_tag,
            tag=tag,
            value=value,
            step=step,
            n_idx=self.n_idx if add_n_idx else None,
            r_idx=self.r_idx if add_r_idx else None,
            s_idx=self.s_idx if add_s_idx else None,
        )

    def log_image(self, writer, tag, image, step):
        """Log one image (wrapper for util)."""
        # tag and log
        full_tag = self.global_tag + "/" + tag
        tp_logging.log_image(writer, full_tag, image, step)

    def store_test_stats(self, size, flops, test_handle):
        """Store the test results for the currently compressed network."""
        idx_tuple = np.s_[self.n_idx, self.s_idx, self.r_idx, self.a_idx]

        # store sizes and flops
        self.sizes[idx_tuple] = size
        self.flops[idx_tuple] = flops

        # check if we should also store full test stats now
        if not self._store_test_stats_now():
            return

        if self.a_idx == self.names.index("ReferenceNet"):
            idx_tuple = np.s_[self.n_idx, :, :, self.a_idx]

        # compute test results
        loss, acc1, acc5 = test_handle()
        loss = float(loss)
        err1 = 1.0 - acc1
        err5 = 1.0 - acc5

        # store final test results
        self.loss[idx_tuple] = loss
        self.error[idx_tuple] = err1
        self.error5[idx_tuple] = err5

    def _store_test_stats_now(self):
        """Check whether we need to store test stats in this iteration."""
        if self.a_idx == self.names.index("ReferenceNet") and (
            self.s_idx != 0 or self.r_idx != 0
        ):
            return False
        else:
            return True

    def store_training_stats(self, is_retraining=True):
        """Store the stats currently stored in the train logger."""
        # check if we can log to the correct algorithm
        if not is_retraining:
            a_idx = self.names.index("ReferenceNet")
        elif self._train_logger.name == self.names[self.a_idx]:
            a_idx = self.a_idx
        else:
            return

        def _store(arr_loss, arr_error, arr_error5, tracker):
            """Store the data from the tracker."""
            if not tracker.contains_data():
                return

            # retrieve data
            epochs, loss, acc1, acc5, _ = tracker.get()
            num_epochs = len(epochs)

            # don't store if it can't hold the values
            # pylint: disable=E1136
            if num_epochs > self.error_train.shape[-1]:
                return

            # now store it
            def _store_one(arr, val):
                if is_retraining:
                    arr[
                        self.n_idx, self.s_idx, self.r_idx, a_idx, :num_epochs
                    ] = val
                else:
                    arr[self.n_idx, :, :, a_idx, :num_epochs] = val[None, None]

            _store_one(arr_loss, loss)
            _store_one(arr_error, 1.0 - acc1)
            _store_one(arr_error5, 1.0 - acc5)

        # store full test statistics
        _store(
            self.loss_train,
            self.error_train,
            self.error5_train,
            self._train_logger.tracker_train,
        )
        _store(
            self.loss_test,
            self.error_test,
            self.error5_test,
            self._train_logger.tracker_test,
        )

    def run_diagnostics_init(self):
        """Initialize the diagnostics for the next run."""
        self.print_info("====================")
        self.print_info(
            f"Evaluating sample size {self._samples[self.s_idx] * 100.0:4.1f}%"
            f" for network version {self.n_idx}"
        )

    def run_diagnostics_update(self, t_compress, t_retrain, t_stats, is_rerun):
        """Store the error diagnostics of a particular compressed network."""
        # for quick plotting
        idx_tuple = (self.n_idx, self.s_idx, self.r_idx, self.a_idx)
        error_now = self.error[idx_tuple]
        error5_now = self.error5[idx_tuple]
        loss_now = self.loss[idx_tuple]
        sizes_now = self.sizes[idx_tuple]
        flops_now = self.flops[idx_tuple]

        # get the name and store diagnostics
        name = self.names[self.a_idx]

        self.print_net(
            self._gen_one_diag_str(
                error=error_now * 100,
                error5=error5_now * 100,
                loss=loss_now,
                size=sizes_now * 100,
                flops=flops_now * 100,
                t_compress=t_compress,
                t_retrain=t_retrain,
                t_stats=t_stats,
            )
        )

        # don't write to summary if it's a rerun to avoid squiggles in plot
        if not is_rerun:
            self.log_scalar(
                self._writer[name],
                f"{self.names_metrics[0]} Test Error param",
                error_now * 100,
                sizes_now * 1e4,
                add_n_idx=True,
                add_r_idx=True,
            )

            self.log_scalar(
                self._writer[name],
                f"{self.names_metrics[1]} Test Error param",
                error5_now * 100,
                sizes_now * 1e4,
                add_n_idx=True,
                add_r_idx=True,
            )

            self.log_scalar(
                self._writer[name],
                "Test Loss param",
                loss_now,
                sizes_now * 1e4,
                add_n_idx=True,
                add_r_idx=True,
            )

            self.log_scalar(
                self._writer[name],
                f"{self.names_metrics[0]} Test Error flops",
                error_now * 100,
                flops_now * 1e4,
                add_n_idx=True,
                add_r_idx=True,
            )

            self.log_scalar(
                self._writer[name],
                f"{self.names_metrics[1]} Test Error flops",
                error5_now * 100,
                flops_now * 1e4,
                add_n_idx=True,
                add_r_idx=True,
            )

            self.log_scalar(
                self._writer[name],
                "Test Loss flops",
                loss_now,
                flops_now * 1e4,
                add_n_idx=True,
                add_r_idx=True,
            )

    def run_diagnostics_finish(self, t_elapsed):
        """Finish the diagnostics for a single run with total time elapsed."""
        self.print_info(f"Total time elapsed: {t_elapsed / 60.0:3.2f}min.")
        self.print_info("====================")
        self.print_info(" ")
        return

    def _gen_one_diag_str(
        self,
        error=None,
        error_diff=None,
        error5=None,
        error5_diff=None,
        loss=None,
        size=None,
        size_abs=None,
        flops=None,
        flops_abs=None,
        prune_ratio=None,
        flops_ratio=None,
        t_compress=None,
        t_retrain=None,
        t_stats=None,
    ):
        """Generate a diagnostics string to print in standard format."""
        metrics = []
        names = self.names_metrics
        if error is not None:
            metrics.append(f"{names[0]} Test Error: {error:5.2f}%")
        if error_diff is not None:
            metrics.append(f"{names[0]} Test Error Diff: {error_diff:5.2f}%")
        if error5 is not None:
            metrics.append(f"{names[1]} Test Error: {error5:5.2f}%")
        if error5_diff is not None:
            metrics.append(f"{names[1]} Test Error Diff: {error5_diff:5.2f}%")
        if loss is not None:
            metrics.append(f"Test Loss: {loss:7.4f}")
        if size is not None:
            metrics.append(f"Parameters Retained: {size:5.1f}%")
        if size_abs is not None:
            metrics.append(f"Parameters Retained: {size_abs:.2E}")
        if prune_ratio is not None:
            metrics.append(f"Prune Ratio: {prune_ratio:5.2f}%")
        if flops is not None:
            metrics.append(f"FLOPS Retained: {flops:5.1f}%")
        if flops_abs is not None:
            metrics.append(f"FLOPS Retained: {flops_abs:.2E}")
        if flops_ratio is not None:
            metrics.append(f"FLOPS Ratio: {flops_ratio:5.2f}%")
        if t_compress is not None:
            metrics.append(f"Compress Time: {t_compress:5.1f}s")
        if t_retrain is not None:
            metrics.append(f"Retrain Time: {t_retrain:5.1f}s")
        if t_stats is not None:
            metrics.append(f"Test Time: {t_stats:5.1f}s")

        # join together " | "
        print_str = " | ".join(metrics)

        return print_str

    def sample_diagnostics(self, budget_per_layer):
        """Store diagnostics related to samples per layer in compression."""
        # only write during first repetition since it is always the same
        if self.r_idx:
            return

        # check for correct length for the first time here.
        if len(budget_per_layer) != len(self.samples_per_layer):
            self.samples_per_layer = -np.ones(
                (
                    len(budget_per_layer),
                    self._num_nets,
                    self._num_intervals,
                    1,
                    self._num_algorithms,
                )
            )

        # total budget
        total_budget = sum(budget_per_layer)

        for layer, total_per_layer in enumerate(budget_per_layer):
            self.samples_per_layer[
                layer, self.n_idx, self.s_idx, :, self.a_idx
            ] = total_per_layer

            try:
                value_to_log = float(total_per_layer) / float(total_budget)
            except ZeroDivisionError:
                value_to_log = 0

            self.log_scalar(
                self._writer[self.names[self.a_idx]],
                tag="Percentage of Sample Budget Per Layer",
                value=value_to_log * 100.0,
                step=layer + 1,
                add_n_idx=True,
                add_r_idx=True,
            )

    def print(self, print_str):
        """Print a message with the current net name as prefix to the msg."""
        self.print_net(print_str)

    def print_info(self, print_str):
        """Print a generic message w/o information about the current net."""
        self._print(print_str, "")

    def print_net(self, print_str):
        """Print a message with the current net name as prefix to the msg."""
        self._print(print_str, self.names[self.a_idx])

    def _print(self, print_str, name):
        """Print with _stdout_logger if not None."""
        if self._stdout_logger is None:
            print(print_str)
        else:
            self._stdout_logger.write(print_str, name)
