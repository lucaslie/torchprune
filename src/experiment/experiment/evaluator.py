"""Evaluator module that contains the maihn class to run evaluations."""
import time

import numpy as np
import torch
import yaml

import torchprune as tp
import torchprune.util.train as tp_train
import torchprune.util.tensor as tp_tensor
from .util.gen import NetGen
from .util.data import get_data_loader


def _ensure_full_init(func):
    """Ensure full initialization of the Evaluator with this decorator."""

    def ensured_init_func(self, *args, **kwargs):
        if not self._fully_initialized:
            self._full_init()
        return func(self, *args, **kwargs)

    return ensured_init_func


class Evaluator(object):
    """Evaluation class to compress networks according to a config file."""

    def __init__(self, logger):
        """Initialize the evaluator with a already initialized logger."""
        # store the logger
        self._logger = logger

        # no grad computations
        torch.set_grad_enabled(False)

        # get reference to parameters and responsibilities
        param = self._logger.param
        self._responsibility = self._logger.responsibility

        if not np.any(self._responsibility):
            return

        self._logger.print_info("Responsibility Mask: ")
        self._logger.print_info(self._responsibility)

        # MAGIC C CONSTANT (C = 3 for ICLR experiments)
        # Using a larger C because the *Star algorithms work
        self._c_constant = 3
        # better with more data points (i.e., larger S)
        # self._c_constant = 10

        # save name of the data set
        self._dataset_name = param["network"]["dataset"]
        self._dataset_test_name = param["generated"]["datasetTest"]

        self._batch_size = int(param["training"]["batchSize"])
        self._num_repetitions = param["experiments"]["numRepetitions"]

        # check for experiment mode
        self._mode = param["experiments"]["mode"]

        self._size_original = None
        self._flops_original = None

        # device to for storage
        self._device_storage = torch.device("cpu")

        # device for computations
        num_gpus = param["generated"]["numAvailableGPUs"]
        self._device = torch.device("cuda:0" if num_gpus > 0 else "cpu")
        self._logger.print_info(
            "Number of available GPUS: {}".format(num_gpus)
        )

        # generator, evaluator, trainer for networks
        self._get_net = NetGen(
            output_size=param["network"]["outputSize"],
            dataset=param["network"]["dataset"],
            net_name=param["generated"]["netName"],
            arch=param["network"]["name"],
        )

        # Loaders and trainer initialized later when needed
        self._loaders = None
        self._net_trainer = None

        self._logger.print_info(
            "Training Parameters:\n"
            f'{yaml.dump(param["generated"]["training"])}'
        )
        self._logger.print_info(
            "Retraining Parameters:\n"
            f'{yaml.dump(param["generated"]["retraining"])}'
        )

        # These are all parameters later set in _initialize_networks()
        self._net_ref = None  # reference net for plots
        self._method_names = param["experiments"]["methods"]
        self._compressed_nets = []

        # get network name to also save results under the same name
        self._net_name = param["generated"]["netName"]

        # have multiple differently trained nets
        self._num_nets = param["experiments"]["numNets"]

        # Failure probability for constructing S.
        self._delta = param["coresets"]["deltaS"]

        # extract keep and compress ratios
        self._keep_ratios = param["generated"]["keepRatios"]
        self._compress_ratios = param["generated"]["compressRatios"]

        # bool to check whether 'run()' was called at least once
        self._fully_compressed = False

        # bool to check whether '__initialize_compression()' was called once
        self._initialized_once = False

        # bool to check whether '_full_init()' was called once
        self._fully_initialized = False

        # print keep ratios
        self._logger.print_info(
            "Keep ratios are: "
            + ", ".join([f"{kr:.3f}" for kr in self._keep_ratios])
        )

    @_ensure_full_init
    def get_dataloader(self, *args):
        """Retrieve desired dataloaders according to types list.

        Arguments:
            *args {list} -- list of dataloader (subset of all_types)
                            (default: {()})

        Raises:
            ValueError: when requested type is not in all_types

        Returns:
            list -- list of loaders according to types

        """
        return [self._loaders[arg] for arg in args]

    def get_all(
        self,
        method="ReferenceNet",
        n_idx=0,
        r_idx=0,
        min_pr=0.0,
        max_pr=1.0,
        max_num=-1,
    ):
        """Get all networks of one method within a window of prune ratios.

        Args:
            method (str, optional): compression method.
                                    Defaults to 'ReferenceNet'.
            n_idx (int, optional): network index. Defaults to 0.
            r_idx (int, optional): repetition index. Defaults to 0.
            min_pr (float, optional): min allowed prune ratio. Defaults to 0.0.
            max_pr (float, optional): max allowed prune ratio. Defaults to 1.0.
            max_num (int, optional): max # of nets to return. Defaults to -1.

        Raises:
            ValueError: if desired method is not available

        Returns:
            list: all networks as specified

        """
        # empty list with all nets ...
        nets = []

        for keep_ratio in self._keep_ratios:
            prune_ratio = 1.0 - keep_ratio
            # check if pr is in desired range
            if prune_ratio > max_pr or prune_ratio < min_pr:
                continue

            # append to list
            nets.append(self.get_by_pr(prune_ratio, method, n_idx, r_idx))

        # finally return the list
        if max_num == -1 or max_num >= len(nets):
            return nets
        else:
            # get indices with np.linspace
            idxs = np.linspace(0, len(nets) - 1, max_num, dtype=np.int)
            # can only index with index list in numpy...
            return np.array(nets)[idxs].tolist()

    @_ensure_full_init
    def get_by_pr(self, prune_ratio, method="ReferenceNet", n_idx=0, r_idx=0):
        """Get network by specific prune ratio."""
        # check if algorithm is available
        if method not in self._method_names:
            raise ValueError(
                f"Desired algorithm '{method}' was not found in "
                f"param file. Choose from: {self._method_names}."
            )

        # check that we initialized at least once
        if not self._initialized_once:
            self._initialize_networks(0)

        # pick the closest prune ratio
        kr_desired = 1 - prune_ratio
        kr_all = np.array(self._keep_ratios)
        s_idx = np.abs(kr_desired - kr_all).argmin()
        kr_best = kr_all[s_idx]

        # get a copy of the compressed net
        compressed_net = self._call_net_constructor(method)

        # now retrieve the network
        self._net_trainer.retrieve(
            compressed_net, n_idx, True, kr_best, s_idx, r_idx
        )

        # return the network
        return compressed_net

    def run(self):
        """Run the desired compression algorithms.

        This function will loop through all the desired compression algorithms
        and compute the compression that is desired according to the param
        file that was passed to the logger.
        """
        if not np.any(self._responsibility):
            return

        if self._logger.state_loaded:
            self._logger.print_info("Everything pre-loaded and computed.")
            return

        # initialize logger state
        self._logger.update_global_state(0, 0, 0, 0)

        # check for training
        if self._mode == "train":
            self._run_training()
            return

        # do it for differently trained networks of the same architecture to be
        # more robust in results
        for n_idx in range(self._num_nets):
            # update state of logger
            self._logger.update_global_state(
                n_idx=n_idx, s_idx=0, r_idx=0, a_idx=0
            )

            start_time = time.time()

            # check if we are responsible for any iteration of this net idx ...
            if not self._responsibility[n_idx].sum():
                continue

            # Initialize current networks
            self._initialize_networks(n_idx)

            for r_idx in range(self._num_repetitions):
                # update state of logger
                self._logger.update_global_state(r_idx=r_idx)

                # For each sample size...
                for s_idx, keep_ratio in enumerate(self._keep_ratios):
                    # update logger state
                    self._logger.update_global_state(
                        s_idx=s_idx, r_idx=r_idx, a_idx=0
                    )

                    self._single_run(keep_ratio)

            # Delete current networks to avoid wrong access
            # Now we should always initialize the networks with an idx
            finish_time = (time.time() - start_time) / 60.0
            self._print_finish_msg(n_idx, finish_time)

        # tell logger to store stuff
        self._logger.save_global_state()

        # we have now compressed all networks
        self._fully_compressed = True

    def _full_init(self):
        """Initialize the expensive parts that we may not need."""
        if self._fully_initialized:
            return

        param = self._logger.param

        # Load the training, validation, test loader
        # We need to run through the network once to get etas. That's why we
        # are passing in a dummy generated network
        self._loaders = get_data_loader(
            param, self._get_net(), self._c_constant
        )

        # initialize the nettrainer
        self._net_trainer = tp_train.NetTrainer(
            param["generated"]["training"],
            param["generated"]["retraining"],
            self._loaders["train"],
            self._loaders["test"],
            self._loaders["val"],
            param["generated"]["numAvailableGPUs"],
            self._logger.get_train_logger(),
        )

        # now fully initialized and print loader message.
        self._fully_initialized = True
        self._print_loader_msg()

    @_ensure_full_init
    def _initialize_networks(self, n_idx):
        # delete reference to old networks
        self._net_ref = None
        self._compressed_nets = []
        self._size_original = None
        self._flops_original = None

        # print initialization message
        self._print_init_msg_1(n_idx)

        # get network from generator
        self._net_ref = self._get_net()
        self._net_ref.to(self._device)

        # cache etas and num patches of the net
        for (img, _) in self._loaders["s_set"]:
            img = tp_tensor.to(img, self._device)
            self._net_ref(img)
            break

        # train network
        self._logger.print_net("")
        self._net_trainer.train(self._net_ref, n_idx)

        # store stats after training with correct n_idx
        self._logger.store_training_stats(is_retraining=False)

        # print update
        loss, acc1, acc5 = self._net_trainer.test(self._net_ref)
        self._print_init_msg_2(loss, acc1 * 100.0, acc5 * 100.0)

        # store sizes
        self._size_original = self._net_ref.size()
        self._flops_original = self._net_ref.flops()
        self._logger.set_total_size(self._size_original, self._flops_original)

        # initialize compression unless we only do training
        if self._mode != "train":
            self._initialize_compressed_networks()

        # send net back to storage device
        self._net_ref.to(self._device_storage)

        # print conclusion of initialization
        self._print_init_msg_3()

        # we have now initialized at least once
        self._initialized_once = True

    def _initialize_compressed_networks(self):
        # Now constructing the compressed networks.
        for a_idx, method in enumerate(self._method_names):
            # update logger state
            self._logger.update_global_state(a_idx=a_idx)

            # construct compressed net
            compressed_net = self._call_net_constructor(method)

            # append to array
            self._compressed_nets.append(compressed_net)

    @_ensure_full_init
    def _call_net_constructor(self, method):
        kwargs = {"original_net": self._net_ref}
        method_class = getattr(tp, method)

        methods_with_args = [
            (
                (tp.BaseSensNet,),
                {"delta_failure": self._delta, "c_constant": self._c_constant},
            ),
            (
                (tp.CompressedNet,),
                {
                    "loader_s": self._loaders["s_set"],
                    "loss_handle": self._net_trainer.get_loss_handle(),
                },
            ),
        ]

        for methods, kwargs_partial in methods_with_args:
            if issubclass(method_class, methods):
                kwargs.update(kwargs_partial)

        return method_class(**kwargs)

    def _run_training(self):
        for n_idx in range(self._num_nets):
            self._logger.update_global_state(n_idx=n_idx)
            if not self._responsibility[n_idx, 0, 0]:
                continue
            self._initialize_networks(n_idx)

    @_ensure_full_init
    def _single_run(self, keep_ratio):

        # start message at logger
        self._logger.run_diagnostics_init()

        # recall current indices from logger
        n_idx = self._logger.n_idx
        r_idx = self._logger.r_idx

        time_elapsed = -time.time()

        # Generate compressions
        # go through all nets and compress them according to sample size
        for a_idx, compressed_net in enumerate(self._compressed_nets):
            # update state of logger
            self._logger.update_global_state(a_idx=a_idx)

            # print once with the algorithm id
            self._logger.print("Transitioning to next algorithm.")

            # check if we are responsible
            if not self._responsibility[n_idx, r_idx, a_idx]:
                print("Skipping since there is no responsibilities.")
                continue

            # do compression
            time_compress = -time.time()
            compressed_net.to(self._device)
            net_available = self._do_compression(compressed_net, keep_ratio)
            compressed_net.to(self._device_storage)
            time_compress += time.time()

            # eliminate loader_s during retraining (not pickeable)
            if hasattr(compressed_net, "loader_s"):
                compressed_net.loader_s = None

            # do actual retraining
            time_retrain = -time.time()
            self._do_retraining(compressed_net, keep_ratio)
            time_retrain += time.time()

            # bring loader_s back
            if hasattr(compressed_net, "loader_s"):
                compressed_net.loader_s = self._loaders["s_set"]

            # compute stats
            # do compression
            time_stats = -time.time()
            compressed_net.to(self._device)
            self._do_stats(compressed_net)
            compressed_net.to(self._device_storage)
            time_stats += time.time()

            # send update to logger
            self._logger.run_diagnostics_update(
                t_compress=time_compress,
                t_retrain=time_retrain,
                t_stats=time_stats,
                is_rerun=net_available,
            )

            # do a quick saving of current state
            self._logger.save_global_state(fast_saving=True)

        # Compute over all time elapsed
        time_elapsed += time.time()

        # send to logger
        self._logger.run_diagnostics_finish(t_elapsed=time_elapsed)

    @_ensure_full_init
    def _do_compression(self, compressed_net, keep_ratio):
        # get some indices
        n_idx = self._logger.n_idx
        a_idx = self._logger.a_idx
        s_idx = self._logger.s_idx
        r_idx = self._logger.r_idx

        # also get the first indices
        s_idx_first = 0
        r_indices = self._responsibility[n_idx, :, a_idx]
        r_idx_first = np.argmax(r_indices)  # argmax returns first max!

        # Whether we should reinitialize the network with the
        # original parameters. If we are not in cascade mode,
        # we should always do this. Otherwise, if we are cascading,
        # then we should only do it in the beginning of the first
        # iteration of the cascade.
        replace_params = (s_idx == 0) or ("cascade" not in self._mode)

        init_compression = (
            ((s_idx == s_idx_first) and (r_idx == r_idx_first))
            or "cascade" in self._mode
            or len(self._compress_ratios[keep_ratio]) > 0
        )

        net_available = self._net_trainer.is_available(
            compressed_net, n_idx, keep_ratio, s_idx, r_idx
        )

        # check if network is available, if so skip compression
        if net_available:
            self._net_trainer.load_compression(
                compressed_net, n_idx, keep_ratio, s_idx, r_idx, keep_ratio
            )
            self._logger.print(
                "Skipping compression since we can load network!"
            )
            return net_available

        # remember whether we have loaded the network already
        compression_is_loaded = False

        # helper function to compress or load
        def compress_or_load(compress_ratio, replace_params, init_compression):

            # use outer scope variable to keep track of load status
            nonlocal compression_is_loaded

            # potentially store budget_per_layer...
            budget_per_layer = None

            # check whether we can load the compression ...
            (
                found_net,
                found_correct_ratio,
            ) = self._net_trainer.load_compression(
                compressed_net, n_idx, keep_ratio, s_idx, r_idx, compress_ratio
            )
            # we have just found the correct compress ratio and loaded it
            if found_net and found_correct_ratio:
                compression_is_loaded = True
                print("Loaded current compression ratio from file!")
            # in this case, we have already loaded the checkpoint
            # or the checkpoint doesn't exist at all ... so we have to compress
            elif (found_net and compression_is_loaded) or not found_net:
                budget_per_layer = compressed_net.compress(
                    compress_ratio, replace_params, init_compression
                )

                # now store the actual checkpoint
                self._net_trainer.store_compression(
                    compressed_net,
                    n_idx,
                    keep_ratio,
                    s_idx,
                    r_idx,
                    compress_ratio,
                )

                # have to set it to true in case we haven't found a net
                # previously but afterward we will and then ratio will never
                # agree...
                compression_is_loaded = True
            else:
                print("Skipping compression ratio; loading later...")

            return budget_per_layer

        # do compression-only iterations
        for compress_ratio in self._compress_ratios[keep_ratio]:

            # in-between logging
            print(f"In-between compression with ratio {compress_ratio:.4f}.")

            # compress
            compress_or_load(compress_ratio, replace_params, init_compression)

            # shouldn't replace parameters anymore after the first
            # iteration
            replace_params = False

        # do the final compression that is used for retraining afterwards
        print(f"Final compression with ratio {keep_ratio:.4f}.")

        try:
            budget_per_layer = compress_or_load(
                keep_ratio, replace_params, init_compression
            )
        # we need to initialize compression once and this try-statement
        # is supposed to catch this condition in case we end up at some
        # iteration but still haven't initialized... (can happen in
        # 'retrain' mode when we delete some of the stored networks,
        # but not the others
        except AttributeError:
            budget_per_layer = compress_or_load(
                keep_ratio, replace_params, True
            )

        # log budget if we can
        if budget_per_layer is not None:
            self._logger.sample_diagnostics(budget_per_layer)

        return net_available

    @_ensure_full_init
    def _do_retraining(self, compressed_net, keep_ratio):
        # get indices
        n_idx = self._logger.n_idx
        s_idx = self._logger.s_idx
        r_idx = self._logger.r_idx

        # initialize from scratch if desired
        if (
            "random" in self._mode or "rewind" in self._mode
        ) and compressed_net.retrainable:
            # generate a new net
            net_replacement = self._get_net()

            # for rewind we need to retrieve the proper rewind checkpoint
            # (instead of purely random)
            if "rewind" in self._mode:
                found = self._net_trainer.load_rewind(net_replacement, n_idx)
                if not found:
                    raise FileNotFoundError("Need rewind checkpoint!!")

            # the compressed_net going into retraining has the
            # * sparsity mask from the current prune iteration
            # * weights/parameters from the net_replacement ...
            compressed_net.register_sparsity_pattern()
            compressed_net.replace_parameters(net_replacement)
            compressed_net.enforce_sparsity()

        # (re-)train if desired. For 'random' this actually training
        if "retrain" in self._mode or "cascade" in self._mode:
            self._net_trainer.retrain(
                compressed_net, n_idx, keep_ratio, s_idx, r_idx
            )
            # also store training stats
            self._logger.store_training_stats()

        # delete compression checkpoint at this point
        self._net_trainer.delete_compression(
            compressed_net, n_idx, keep_ratio, s_idx, r_idx
        )

    @_ensure_full_init
    def _do_stats(self, compressed_net):
        # compute relative sizes
        size_rel = compressed_net.size() / self._size_original
        flops_rel = compressed_net.flops() / self._flops_original

        # store them
        self._logger.store_test_stats(
            size_rel, flops_rel, lambda: self._net_trainer.test(compressed_net)
        )

    @_ensure_full_init
    def _print_loader_msg(self):
        """Print information about the loader."""
        self._logger.print_info("Dataloader Information:")
        for name, loader in self._loaders.items():
            self._logger.print_info(
                f"{name:5} data: # batches: {len(loader):3}, "
                f"# data points: {len(loader.dataset):3}, "
                f"# workers: {loader.num_workers}"
            )
        self._logger.print_info("")

    def _print_init_msg_1(self, idx):
        self._logger.print_info(
            f"""

************************************************
************************************************
************************************************

          INITIALIZING NEW ARCHITECTURE

NAME OF THE NETWORK: {self._net_name}
DATASET: {self._dataset_name}
TEST DATASET: {self._dataset_test_name}

VERSION OF THE NETWORK: {idx}

TRAINING/LOADING:"""
        )

    def _print_init_msg_2(self, loss, acc1, acc5):
        m_name = self._logger.names_metrics
        self._logger.print_info(
            f"""TRAINING/LOADING DONE!

TEST: LOSS: {loss:3.3f} | {m_name[0]}: {acc1:3.2f}% | {m_name[1]}: {acc5:3.2f}%

NUMBER OF COMPRESSIBLE LAYERS: {self._net_ref.num_compressible_layers}
NUMBER OF WEIGHTS/LAYER: {self._net_ref.num_weights}
"""
        )

    def _print_init_msg_3(self):
        self._logger.print_info(
            """

FINISHED INITIALIZING NEW ARCHITECTURE

************************************************
************************************************
************************************************


        """
        )

    def _print_finish_msg(self, idx, t_finish):
        self._logger.print_info(
            f"""

************************************************
************************************************
************************************************

           FINISHED WITH ARCHITECTURE

NAME OF THE NETWORK: {self._net_name}
DATASET: {self._dataset_name}
TEST DATASET: {self._dataset_test_name}

NUMBER OF COMPRESSIBLE LAYERS: {self._net_ref.num_compressible_layers}
NUMBER OF WEIGHTS/LAYER: {self._net_ref.num_weights}
VERSION OF THE NETWORK: {idx}

TIME SPENT: {t_finish:.1f} minutes

************************************************
************************************************
************************************************

        """
        )
