"""A trainer module that handles all the training/retraining for us."""
import os
import copy
import time
import warnings

import numpy as np
import torch.nn as nn
import torch
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from . import lr_scheduler, metrics, nn_loss, tensor, logging

__all__ = ["NetTrainer"]


class NetTrainer(object):
    """Convenient class to handle training and retraining."""

    def __init__(
        self,
        train_params,
        retrain_params,
        train_loader,
        test_loader,
        valid_loader,
        num_gpus=1,
        train_logger=None,
    ):
        """Initialize the trainer with the required parameters.

        Args:
            train_params (dict): parameters for training
            retrain_params (dict): parameters for retraining
            train_loader (torch dataloader): dataloader for training set
            test_loader (torch dataloader): dataloader for test set
            valid_loader (torch dataloader): dataloader for validation set
            num_gpus (int, optional): number of GPUs to use. Defaults to 1.
            train_logger (util.logging.TrainLogger, optional): logging module.
                Defaults to None.
        """
        # save the data loaders
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.valid_loader = valid_loader

        # have an instance of the logger ready
        if train_logger is None:
            self._train_logger = logging.TrainLogger()
        else:
            self._train_logger = train_logger

        # check for number of GPUs
        self.num_gpus = num_gpus

        # initialize other quantities if it is not a pretrained network
        # Hyper-parameters for training.
        self.train_params = copy.deepcopy(train_params)

        # Hyper-parameters for retraining (based on training parameters and
        # whatever is specified on top of that in the retraining category)
        self.retrain_params = copy.deepcopy(retrain_params)

    def train(self, net, n_idx):
        """Train a network with specified network id."""
        self._train_procedure(net, n_idx, False, None)

    def retrain(self, net, n_idx, keep_ratio, s_idx, r_idx):
        """Retrain a compressed network."""
        self._train_procedure(net, n_idx, True, keep_ratio, s_idx, r_idx)

    def is_available(self, net, n_idx, keep_ratio, s_idx, r_idx):
        """Check whether network can be loaded from file."""
        file_name = self._get_net_name(
            net, n_idx, True, keep_ratio, s_idx, r_idx, False
        )
        file_name_check = self._get_net_name(
            net, n_idx, True, keep_ratio, s_idx, r_idx, True
        )
        return os.path.isfile(file_name) or os.path.isfile(file_name_check)

    def _get_net_name(
        self,
        net,
        n_idx,
        retraining,
        keep_ratio,
        s_idx,
        r_idx,
        get_checkpoint,
        get_compression=False,
        get_rewind=False,
        get_best=False,
    ):
        """For saving/loading the current configuration."""
        # check epochs and save directory from parameters
        if retraining:
            num_epochs = self.retrain_params["numEpochs"]
            save_dir = self.retrain_params["dir"]
        else:
            num_epochs = self.train_params["numEpochs"]
            save_dir = self.train_params["dir"]

        net_filename = [net.name]
        net_filename.append(str(n_idx))

        if keep_ratio is not None and retraining:
            net_filename.append(f"p{int(keep_ratio * 100.0)}")

        if retraining:
            # repetition number
            net_filename.append(f"rep{r_idx}")
            # record number of retrain epochs
            net_filename.append(f"re{num_epochs}")
            # record iteration number in the cascade
            net_filename.append(f"i{s_idx}")

        if get_checkpoint:
            net_filename.append("checkpoint")
        elif get_compression:
            net_filename.append("compression")
        elif get_rewind:
            net_filename.append("rewind")
        elif get_best:
            net_filename.extend(["checkpoint", "best"])

        net_file = os.path.join(save_dir, "_".join(net_filename))
        return net_file

    def get_loss_handle(self, retraining=False):
        """Get a handle for the loss function."""
        if retraining:
            params = self.retrain_params
        else:
            params = self.train_params
        return _get_loss_handle(params)

    def test(self, net):
        """Test the network on the desired dataset."""
        params = self.train_params
        device = next(net.parameters()).device
        criterion = _get_loss_handle(params).to(device)
        metrics_test = get_test_metrics(params)

        return _test_one_epoch(
            self.test_loader, criterion, metrics_test, 0, device, net, "Test"
        )

    def retrieve(
        self, net, n_idx, retraining=False, keep_ratio=-1, s_idx=-1, r_idx=-1
    ):
        """Retrieve the desired retrained, compressed (or regular) network.

        Arguments:
            net {NetHandle or BaseCompressedNet} -- net to load state_dict into
            n_idx {int} -- network index

        Keyword Arguments:
            retraining {bool} -- retrieve retrained net (default: {False})
            keep_ratio {int} -- keep_ratio of retrained net (default: {-1})
            s_idx {int} -- keep_ratio index of retrained net (default: {-1})
            r_idx {int} -- repetition index of retrained net (default: {-1})

        """
        # generate file name and load it
        file_name_net = self._get_net_name(
            net, n_idx, retraining, keep_ratio, s_idx, r_idx, False
        )
        found_trained_net, _ = load_checkpoint(file_name_net, net)

        # either return it or raise not found error
        if found_trained_net:
            return net
        else:
            raise FileNotFoundError(f"Could not find {file_name_net}")

    def store_compression(
        self, net, n_idx, keep_ratio, s_idx, r_idx, compress_ratio
    ):
        """Store the desired compression (not retrained).

        Arguments:
            net {BaseCompressedNet} -- net to load state_dict into
            n_idx {int} -- network index
            retraining {bool} -- retrieve retrained net
            keep_ratio {float} -- keep_ratio
            s_idx {int} -- keep_ratio index
            r_idx {int} -- repetition index
            compress_ratio {float} -- current in-between compression ratio

        """
        # generate file name and store it
        file_name_net = self._get_net_name(
            net, n_idx, True, keep_ratio, s_idx, r_idx, False, True
        )

        save_checkpoint(file_name_net, net, compress_ratio)

    def load_compression(
        self, net, n_idx, keep_ratio, s_idx, r_idx, compress_ratio
    ):
        """Load the desired compression (not retrained).

        Arguments:
            net {BaseCompressedNet} -- compressed network to load state_dict
            n_idx {int} -- network index
            s_idx {int} -- keep_ratio index
            keep_ratio {int} -- keep_ratio
            r_idx {int} -- repetition index
            compress_ratio {float} -- current in-between compression ratio

        """
        # generate file name
        file_name_net = self._get_net_name(
            net, n_idx, True, keep_ratio, s_idx, r_idx, False, True
        )

        # check if we can load checkpoint
        found_net, compress_ratio_found = load_checkpoint(
            file_name_net, net, epoch_desired=compress_ratio
        )

        # check if it was found and loaded
        found_correct_ratio = (
            found_net and compress_ratio_found == compress_ratio
        )

        return found_net, found_correct_ratio

    def delete_compression(self, net, n_idx, keep_ratio, s_idx, r_idx):
        """Delete the compression checkpoint (not retrained).

        Arguments:
            net {BaseCompressedNet} -- compressed network to load state_dict
            n_idx {int} -- network index
            s_idx {int} -- keep_ratio index
            keep_ratio {int} -- keep_ratio
            r_idx {int} -- repetition index

        """
        # generate file name
        file_name_net = self._get_net_name(
            net, n_idx, True, keep_ratio, s_idx, r_idx, False, True
        )

        # delete the checkpoint if it exists as expected
        delete_checkpoint(file_name_net)

    def load_rewind(self, net, n_idx):
        """Load the rewind checkpoint (during retraining).

        Arguments:
            net {NetHandle} -- compressed network to load state_dict
            n_idx {int} -- network index

        """
        # generate file name
        file_name_net = self._get_net_name(
            net, n_idx, False, -1, -1, -1, False, False, True
        )

        # check if we can load checkpoint
        found_net, _ = load_checkpoint(file_name_net, net)
        return found_net

    def _train_procedure(
        self, net, n_idx, retraining, keep_ratio, s_idx=0, r_idx=0
    ):

        # the parameters
        steps_per_epoch = len(self.train_loader)
        params = self.retrain_params if retraining else self.train_params

        # check the file names we need
        file_name_net = self._get_net_name(
            net, n_idx, retraining, keep_ratio, s_idx, r_idx, False
        )
        file_name_check = self._get_net_name(
            net, n_idx, retraining, keep_ratio, s_idx, r_idx, True
        )
        file_name_rewind = self._get_net_name(
            net,
            n_idx,
            retraining,
            keep_ratio,
            s_idx,
            r_idx,
            False,
            False,
            True,
        )
        file_name_best = self._get_net_name(
            net,
            n_idx,
            retraining,
            keep_ratio,
            s_idx,
            r_idx,
            get_checkpoint=False,
            get_best=True,
        )

        # get test metrics assembled
        metrics_test = get_test_metrics(params)

        # set up the train logger
        # doing this before returning with pre-trained net is important so that
        # we don't have old data stored in the train logger.
        if self._train_logger is not None:
            self._train_logger.initialize(
                net_class_name=type(net).__name__,
                is_retraining=retraining,
                num_epochs=params["numEpochs"],
                steps_per_epoch=steps_per_epoch,
                early_stop_epoch=params["earlyStopEpoch"],
                metrics_test=metrics_test,
                n_idx=n_idx,
                r_idx=r_idx,
                s_idx=s_idx,
            )

        # check if network is already pretrained and done. then we can return
        found_trained_net, _ = load_checkpoint(
            file_name_net,
            net,
            train_logger=self._train_logger,
            loc=str(next(net.parameters()).device),
        )

        if found_trained_net:
            print("Loading pre-trained network...")
            return

        # retrieve net handle
        if hasattr(net, "compressed_net"):
            net_handle = net.compressed_net
        else:
            net_handle = net

        # enable grad computations
        torch.set_grad_enabled(True)

        # empty gpu cache to make sure everything is ready for retraining
        torch.cuda.empty_cache()

        # register sparsity pattern to before retraining
        if retraining:
            net_handle.register_sparsity_pattern()

        args = (
            self.num_gpus,
            self.train_loader.num_workers,
            net_handle,
            retraining,
            self.train_loader.dataset,
            self.valid_loader.dataset,
            self.train_loader.collate_fn,
            params,
            self._train_logger,
            file_name_check,
            file_name_rewind,
            file_name_best,
        )

        # setup torch.distributed and spawn processes
        if not retraining or net.retrainable:
            if self.num_gpus > 1:
                os.environ["MASTER_ADDR"] = "127.0.0.1"
                os.environ["MASTER_PORT"] = "12355"
                mp.spawn(train_with_worker, nprocs=self.num_gpus, args=args)
            else:
                train_with_worker(0, *args)

        # disable grad computations
        torch.set_grad_enabled(False)

        # load result into this net here
        load_checkpoint(
            file_name_check,
            net_handle,
            train_logger=self._train_logger,
            loc=str(next(net_handle.parameters()).device),
        )

        # then overwrite with early stopping checkpoint (net only, no logger!)
        found_best, epoch_best = load_checkpoint(
            file_name_best,
            net_handle,
            loc=str(next(net_handle.parameters()).device),
        )
        if found_best:
            print(f"Loaded early stopping checkpoint from epoch: {epoch_best}")

        # store full net as well
        save_checkpoint(
            file_name_net, net, params["numEpochs"], self._train_logger
        )

        # delete checkpoint to save storage
        delete_checkpoint(file_name_check)
        delete_checkpoint(file_name_best)


def train_with_worker(
    gpu_id,
    num_gpus,
    num_workers,
    net_handle,
    retraining,
    train_dataset,
    val_dataset,
    collate_fn,
    params,
    train_logger,
    file_name_checkpoint,
    file_name_rewind,
    file_name_best,
):
    """Worker to train network in distributed fashion."""
    # check if we have multiple gpus available in which case we are doing a
    # parellel run
    is_distributed = num_gpus > 1

    # check if we are doing cpu only
    is_cpu = num_gpus < 1

    # get world size
    world_size = max(1, num_gpus)

    # initialize process group
    if is_distributed:
        dist.init_process_group(
            backend="nccl", world_size=world_size, rank=gpu_id
        )

    # make sure device is set correctly
    if not is_cpu:
        torch.cuda.set_device(gpu_id)
    loc = "cpu" if is_cpu else "cuda:{}".format(gpu_id)
    worker_device = torch.device(loc)

    # put the net on the worker device
    net_handle.to(worker_device)

    if is_distributed and not is_cpu:
        net_handle = nn.SyncBatchNorm.convert_sync_batchnorm(net_handle)
        net_parallel = nn.parallel.DistributedDataParallel(
            net_handle, device_ids=[gpu_id], find_unused_parameters=True
        )
    else:
        net_parallel = net_handle

    # construct optimizer
    optimizer = getattr(torch.optim, params["optimizer"])(
        params=net_parallel.parameters(), **params["optimizerKwargs"]
    )

    # get loss handle
    criterion = _get_loss_handle(params).to(worker_device)

    # get test metrics
    metrics_test = get_test_metrics(params)

    # setup gradient scaler for AMP
    scaler = torch.cuda.amp.GradScaler(enabled=params["enableAMP"])

    # get a proper distributed train loader
    batch_size = int(params["batchSize"] / world_size)
    num_workers = int((num_workers + world_size - 1) / world_size)

    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=gpu_id
        )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        sampler=train_sampler,
    )

    # get a test loader as well
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # setup learning rate schedulers ("chainable")
    # do it once here so that schedulers have initial learning rate set!
    schedulers = _get_lr_schedulers(optimizer, len(train_loader), params)

    # Load the checkpoint if available
    found_checkpoint, start_epoch = load_checkpoint(
        file_name_checkpoint,
        net_handle,
        train_logger,
        optimizer,
        scaler,
        schedulers,
        loc,
    )

    # wait for all processes to load the checkpoint
    if is_distributed:
        dist.barrier()

    # this may be non-zero in the case of rewinding ...
    if not found_checkpoint:
        start_epoch = params["startEpoch"]

        # step through learning rate scheduler to get it to current start epoch
        # it will throw a warning since lr_scheduler is supposed to be called
        # after optimizer.step(). However, it is okay in this setting, e.g.,
        # for weight rewinding. It is the most stable approach to bringing the
        # up scheduler to the desired learning rate.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for epoch in range(start_epoch):
                for _ in range(len(train_loader)):
                    for scheduler in schedulers:
                        scheduler.step()

    # make it faster
    if not is_cpu:
        cudnn.benchmark = True

    # convenience function for storing check points
    def store_checkpoints(epoch):
        # only do it for primary GPUf
        if gpu_id != 0:
            return

        # save checkpoint at the end of every epoch with 0 worker
        save_checkpoint(
            file_name_checkpoint,
            net_handle,
            epoch,
            train_logger,
            optimizer,
            scaler,
            schedulers,
        )

        # check whether we should store rewind checkpoint
        if not retraining and params["retrainStartEpoch"] == epoch:
            save_checkpoint(
                file_name_rewind,
                net_handle,
                epoch,
                train_logger,
                optimizer,
                scaler,
                schedulers,
            )

        # potentially store early-stopping checkpoint
        # epoch convention is "epoch corresponds to beginning"
        # logger convention is "epoch corresponds to end"
        # hence we need the -1
        if train_logger is not None and train_logger.is_best_epoch(epoch - 1):
            save_checkpoint(file_name_best, net_handle, epoch)

    # do the distributed training
    t_training = -time.time()
    for epoch in range(start_epoch, params["numEpochs"]):

        # set the epoch for the train sampler
        if is_distributed:
            train_sampler.set_epoch(epoch)

        # store the checkpoint
        store_checkpoints(epoch)

        # train for one epoch
        _train_one_epoch(
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            schedulers=schedulers,
            epoch=epoch,
            worker_device=worker_device,
            net_parallel=net_parallel,
            scaler=scaler,
            enable_amp=params["enableAMP"],
            net_handle=net_handle if retraining else None,
            train_logger=train_logger if gpu_id == 0 else None,
        )

        # test after one epoch
        _test_one_epoch(
            loader=val_loader,
            criterion=criterion,
            metrics_test=metrics_test,
            epoch=epoch,
            device=worker_device,
            net=net_parallel,
            tag="Val",
            train_logger=train_logger if gpu_id == 0 else None,
        )

    # store final checkpoint
    store_checkpoints(params["numEpochs"])

    # move back into eval mode
    net_parallel.eval()

    t_training += time.time()
    if gpu_id == 0:
        print(f"Overall Training took: {t_training}s")

    # destroy process group at the end
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()


def _train_one_epoch(
    train_loader,
    optimizer,
    criterion,
    schedulers,
    epoch,
    worker_device,
    net_parallel,
    scaler,
    enable_amp,
    net_handle=None,
    train_logger=None,
):
    # keep track of timing
    t_total = -time.time()
    t_loading = 0.0
    t_optim = 0.0
    t_enforce = 0.0
    t_log = 0.0

    # switch to train mode
    net_parallel.train()

    # go through one epoch and train
    for i, (images, targets) in enumerate(train_loader):

        # convert to CUDA tensor if desired
        t_loading -= time.time()
        images = tensor.to(images, worker_device, non_blocking=True)
        targets = tensor.to(targets, worker_device, non_blocking=True)
        t_loading += time.time()

        # Forward + Backward + Optimize
        t_optim -= time.time()

        optimizer.zero_grad()  # zero the gradient buffer
        with torch.cuda.amp.autocast(enabled=enable_amp):
            outputs = net_parallel(images)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        t_optim += time.time()

        # step scheduler
        for scheduler in schedulers:
            scheduler.step()

        # we need to enforce sparsity
        t_enforce -= time.time()
        if net_handle is not None:
            net_handle.enforce_sparsity()
        t_enforce += time.time()

        # log the current loss
        t_log -= time.time()
        if train_logger is not None:
            train_logger.train_diagnostics(
                step=i,
                epoch=epoch,
                l_r=optimizer.param_groups[0]["lr"],
                loss=loss,
                targets=targets,
                outputs=outputs,
            )
        t_log += time.time()

    # store timing
    t_total += time.time()

    # print timing
    if train_logger is not None:
        train_logger.epoch_diagnostics(
            t_total=t_total,
            t_loading=t_loading,
            t_optim=t_optim,
            t_enforce=t_enforce,
            t_log=t_log,
        )


def _test_one_epoch(
    loader,
    criterion,
    metrics_test,
    epoch,
    device,
    net,
    tag,
    train_logger=None,
):
    """Test after one epoch."""
    acc1 = 0
    acc5 = 0
    loss = 0
    num_total = 0

    # switch to eval mode
    net.eval()

    with torch.no_grad():
        for images, targets in loader:
            # move to correct device
            images = tensor.to(images, device, non_blocking=True)
            targets = tensor.to(targets, device, non_blocking=True)

            # compute the output for this batch
            outputs = net(images)
            acc1_batch, acc5_batch = [
                met(outputs, targets) for met in metrics_test
            ]

            # keep track of stats
            loss += criterion(outputs, targets) * len(images)
            acc1 += acc1_batch * len(images)
            acc5 += acc5_batch * len(images)
            num_total += len(images)

    # normalize at the end
    acc1 /= num_total
    acc5 /= num_total
    loss /= num_total

    # make sure loss is also a regular float (not torch.Tensor)
    loss = float(loss)

    if train_logger is not None:
        train_logger.test_diagnostics(epoch, loss, acc1, acc5, tag)

    return loss, acc1, acc5


def _get_loss_handle(params):
    """Initialize and return the loss handle according to params."""
    return getattr(nn_loss, params["loss"])(**params["lossKwargs"])


def _get_lr_schedulers(optimizer, steps_per_epoch, params):
    """Initialize and return list with learning rate schedulers."""
    # get types, kwargs, and stepKwargs
    types = [sched["type"] for sched in params["lrSchedulers"]]
    kwargs = [sched["kwargs"] for sched in params["lrSchedulers"]]
    step_kwargs = [sched["stepKwargs"] for sched in params["lrSchedulers"]]

    # step_kwargs need to be multipied by steps_per_epoch
    # (they are given in epoch steps but we step schedulers per iteration)
    step_kwargs = [
        {k: (np.array(v) * steps_per_epoch).tolist() for k, v in s_kw.items()}
        for s_kw in step_kwargs
    ]

    return [
        getattr(lr_scheduler, type)(optimizer, last_epoch=-1, **kw, **s_kw)
        for type, kw, s_kw in zip(types, kwargs, step_kwargs)
    ]


def get_test_metrics(params):
    """Initialize and return list of test metrics."""
    assert len(params["metricsTest"]) == 2, "Exactly two metrics needed!"
    # if we classes kwargs, we need to set them to the current output size
    return [
        getattr(metrics, metric["type"])(
            params["outputSize"], **metric["kwargs"]
        )  # get each metric we need
        for metric in params["metricsTest"]
    ]


def save_checkpoint(
    file_name,
    net,
    epoch,
    train_logger=None,
    optimizer=None,
    scaler=None,
    schedulers=None,
):
    """Save checkpoint of network."""
    # Populate checkpoint
    checkpoint = {
        "net": net.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "schedulers": None
        if schedulers is None
        else [sched.state_dict() for sched in schedulers],
        "epoch": epoch,
        "tracker_train": None,
        "tracker_test": None,
    }
    if train_logger is not None:
        checkpoint["tracker_train"] = train_logger.tracker_train.get()
        checkpoint["tracker_test"] = train_logger.tracker_test.get()

    # create directory if it doesn't exist yet
    file_dir = os.path.split(file_name)[0]
    os.makedirs(file_dir, exist_ok=True)
    # save now
    torch.save(checkpoint, file_name)


def delete_checkpoint(file_name):
    """Delete existing checkpoint."""
    if os.path.isfile(file_name):
        os.remove(file_name)


def load_checkpoint(
    file_name,
    net,
    train_logger=None,
    optimizer=None,
    scaler=None,
    schedulers=None,
    loc=None,
    epoch_desired=None,
):
    """Load checkpoint into provided net and optimizer."""
    epoch = 0
    found_checkpoint = os.path.isfile(file_name)

    def _check_tracker(chkpt, key):
        return train_logger is not None and key in chkpt.keys() and chkpt[key]

    if found_checkpoint:
        chkpt = torch.load(file_name, map_location=loc)
        epoch = chkpt["epoch"]
        if epoch_desired is None or epoch_desired == epoch:
            # strict=False ignores keys in the checkpoint that don't exist in
            # the net
            net.load_state_dict(chkpt["net"], strict=False)
            if optimizer is not None:
                optimizer.load_state_dict(chkpt["optimizer"])
            if scaler is not None and "scaler" in chkpt:
                scaler.load_state_dict(chkpt["scaler"])
            if (
                schedulers is not None
                and "schedulers" in chkpt
                and chkpt["schedulers"] is not None
            ):
                for sched, ckpt_s in zip(schedulers, chkpt["schedulers"]):
                    sched.load_state_dict(ckpt_s)
            if _check_tracker(chkpt, "tracker_train"):
                train_logger.tracker_train.set(*chkpt["tracker_train"])
            if _check_tracker(chkpt, "tracker_test"):
                train_logger.tracker_test.set(*chkpt["tracker_test"])

    return found_checkpoint, epoch
