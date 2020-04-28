"""A trainer module that handles all the training/retraining for us."""
import os
import copy
import time

from numpy import clip
import torch.nn as nn
import torch
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch.distributed as dist

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

        net_file = os.path.join(save_dir, "_".join(net_filename))
        return net_file

    def get_loss_handle(self, retraining=False):
        """Get a handle for the loss function."""
        if retraining:
            params = self.retrain_params
        else:
            params = self.train_params
        return _get_loss_handle(params)

    def test(self, net, eval_loader=None):
        """Test the network on the desired dataset."""
        if eval_loader is None:
            eval_loader = self.test_loader

        device = next(net.parameters()).device
        criterion = self.get_loss_handle().to(device)

        return _test_one_epoch(eval_loader, criterion, 0, device, net)

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

        save_checkpoint(file_name_net, net, None, compress_ratio)

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
            net {BaseCompressedNet} -- compressed network to load state_dict
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

        # check if network is already pretrained and done. then we can return
        found_trained_net, _ = load_checkpoint(
            file_name_net, net, loc=str(next(net.parameters()).device)
        )

        # set up the train logger
        # doing this before returning with pre-trained net is important so that
        # we don't have old data stored in the train logger.
        if self._train_logger is not None:
            self._train_logger.initialize(
                net_class_name=type(net).__name__,
                is_retraining=retraining,
                num_epochs=params["numEpochs"],
                steps_per_epoch=steps_per_epoch,
                n_idx=n_idx,
                r_idx=r_idx,
                s_idx=s_idx,
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

        # setup torch.distributed and spawn processes
        num_workers = self.train_loader.num_workers // self.num_gpus

        # empty gpu cache to make sure everything is ready for retraining
        torch.cuda.empty_cache()

        # register sparsity pattern to before retraining
        if retraining:
            net_handle.register_sparsity_pattern()

        args = (
            self.num_gpus,
            num_workers,
            net_handle,
            retraining,
            self.train_loader.dataset,
            self.test_loader.dataset,
            params,
            self._train_logger,
            file_name_check,
            file_name_rewind,
        )
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
            loc=str(next(net_handle.parameters()).device),
        )

        # store full net as well
        save_checkpoint(file_name_net, net, None, params["numEpochs"])

        # delete checkpoint to save storage
        delete_checkpoint(file_name_check)


def train_with_worker(
    gpu_id,
    num_gpus,
    num_workers,
    net_handle,
    retraining,
    train_dataset,
    test_dataset,
    params,
    train_logger,
    file_name_checkpoint,
    file_name_rewind,
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
        net_parallel = nn.parallel.DistributedDataParallel(
            net_handle, device_ids=[gpu_id]
        )
    else:
        net_parallel = net_handle

    # construct optimizer
    optim_kwargs = {
        "params": net_parallel.parameters(),
        "lr": params["learningRate"],
        "weight_decay": params["weightDecay"],
    }
    if params["optimizer"] == "SGD":
        optim_kwargs.update(
            {"nesterov": params["nesterov"], "momentum": params["momentum"]}
        )
    optimizer = getattr(torch.optim, params["optimizer"])(**optim_kwargs)

    # get loss handle
    criterion = _get_loss_handle(params).to(worker_device)

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
        pin_memory=True,
        sampler=train_sampler,
    )

    # get a test loader as well
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Load the checkpoint if available
    found_checkpoint, start_epoch = load_checkpoint(
        file_name_checkpoint, net_handle, optimizer, loc
    )

    # this may be non-zero in the case of rewinding ...
    if not found_checkpoint:
        start_epoch = params["startEpoch"]

    # set up learning rate scheduler
    lr_scheduler = get_lr_scheduler(
        steps_per_epoch=len(train_loader),
        warmup=params["warmup"],
        cooldown=params["cooldown"],
        learning_rate=params["learningRate"],
        lr_milestones=params["lRmilestones"],
        lr_gamma=params["lRgamma"],
        momentum=params["momentum"],
        momentum_delta=params["momentumDelta"],
    )

    # make it faster
    if not is_cpu:
        cudnn.benchmark = True

    # switch to train mode
    net_parallel.train()

    # convenience function for storing check points
    def store_checkpoints(epoch):
        # save checkpoint at the end of every epoch with 0 worker
        if gpu_id == 0:
            save_checkpoint(file_name_checkpoint, net_handle, optimizer, epoch)

        # check whether we should store rewind checkpoint
        if (
            gpu_id == 0
            and not retraining
            and params["retrainStartEpoch"] == epoch
        ):
            save_checkpoint(file_name_rewind, net_handle, optimizer, epoch)

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
            epoch=epoch,
            lr_scheduler=lr_scheduler,
            worker_device=worker_device,
            net_parallel=net_parallel,
            net_handle=net_handle if retraining else None,
            train_logger=train_logger if gpu_id == 0 else None,
        )

        # test after one epoch
        if gpu_id == 0 and train_logger is not None:
            _test_one_epoch(
                loader=test_loader,
                criterion=criterion,
                epoch=epoch,
                device=worker_device,
                net=net_parallel,
                train_logger=train_logger,
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
        dist.destroy_process_group()


def _train_one_epoch(
    train_loader,
    optimizer,
    criterion,
    epoch,
    lr_scheduler,
    worker_device,
    net_parallel,
    net_handle=None,
    train_logger=None,
):
    # keep track of timing
    t_total = -time.time()
    t_loading = 0.0
    t_optim = 0.0
    t_enforce = 0.0
    t_log = 0.0

    # go through one epoch and train
    for i, (images, targets) in enumerate(train_loader):

        # adjust the learning rate
        lr_scheduler(optimizer, epoch, i)

        # convert to CUDA tensor if desired
        t_loading -= time.time()
        images = images.to(worker_device, non_blocking=True)
        targets = targets.to(worker_device, non_blocking=True)
        t_loading += time.time()

        # Forward + Backward + Optimize
        t_optim -= time.time()

        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net_parallel(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        t_optim += time.time()

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
                loss=loss,
                targets=targets,
                outputs=outputs,
                acc_handle=get_accuracies,
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


def _test_one_epoch(loader, criterion, epoch, device, net, train_logger=None):
    """Test after one epoch."""
    acc1 = 0
    acc5 = 0
    loss = 0
    num_total = 0

    with torch.no_grad():
        for images, targets in loader:
            # move to correct device
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # compute the output for this batch
            outputs = net(images)
            acc1_batch, acc5_batch = get_accuracies(
                outputs, targets, topk_values=(1, 5)
            )

            # keep track of stats
            loss += criterion(outputs, targets) * len(images)
            acc1 += acc1_batch * len(images)
            acc5 += acc5_batch * len(images)
            num_total += len(images)

    # normalize at the end
    acc1 /= num_total
    acc5 /= num_total
    loss /= num_total

    # make sure loss is also a regular float (not torch.Tensor)/s
    loss = float(loss)

    if train_logger is not None:
        train_logger.test_diagnostics(epoch, loss, acc1, acc5)

    return loss, acc1, acc5


def get_lr_scheduler(
    steps_per_epoch,
    warmup,
    cooldown,
    learning_rate,
    lr_milestones,
    lr_gamma,
    momentum,
    momentum_delta,
):
    """Return a function handle to adjust the learning rate.

    The function that is returned can adjust the learning rate based on the
    current epoch and the current step within the epoch.

    Args:
        steps_per_epoch (int): steps_per_epoch needed for warmup/cooldown
        warmup (int): warm up for how many epochs
        cooldown (int): cool down for how many epochs
        learning_rate (float): initial learning rate
        lr_milestones (list of ints): epoch milestones to decay learning rate
        lr_gamma (float): multiplicative factor for decaying learning rate
        momentum (float): nominal momentum for learning algorithm
        momentum_delta (float): difference in delta over epochs

    Returns:
        function: adjust_lr(optimizer, epoch, step) to adjust optimizer
                  parameters.

    """
    # prototype for function handle
    def adjust_lr(optimizer, epoch, step):
        # decay learning rate with milestones and gamma factor
        lr_ms = learning_rate * lr_gamma ** sum(
            [epoch >= milestone for milestone in lr_milestones]
        )

        # current total steps
        steps_total = epoch * steps_per_epoch + step
        steps_warmup = warmup * steps_per_epoch + 1e-10
        steps_cooldown = cooldown * steps_per_epoch + 1e-10

        # get warm up learning rate
        lr_warmup = learning_rate * clip(steps_total / steps_warmup, 0, 1)

        # warmup is upper bounded by lr_ms
        lr_warmup = min(lr_ms, lr_warmup)

        # get cooldown learning rate
        lr_cooldown = learning_rate * clip(
            1 - (steps_total - steps_warmup) / steps_cooldown, 0, 1
        )

        # cooldown is lower bounded by lr_ms
        lr_cooldown = max(lr_ms, lr_cooldown)

        # final lr depends on range
        lr_final = lr_warmup if epoch < warmup else lr_cooldown

        # adjust delta depending on whether we have warmup and cooldown
        if warmup > 0 and cooldown > 0:
            actual_delta = momentum_delta
        else:
            actual_delta = 0

        # warmup momentum
        mom_warmup = (momentum + actual_delta) - 2 * actual_delta * clip(
            steps_total / steps_warmup, 0, 1
        )

        # cooldown momentum
        mom_cooldown = (momentum - actual_delta) + 2 * actual_delta * clip(
            (steps_total - steps_warmup) / steps_cooldown, 0, 1
        )

        # final momentum depends on range
        mom = mom_warmup if epoch < warmup else mom_cooldown

        # update learning rate in parameter groups of the optimizer
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_final
            param_group["momentum"] = mom

    # return the function handle
    return adjust_lr


def _get_loss_handle(params):
    """Initialize and return the loss handle according to params."""
    return getattr(nn, params["loss"])(reduction="mean")


def get_accuracies(output, target, topk_values=(1,)):
    """Compute the accuracy over the k top predictions for desired k."""
    with torch.no_grad():
        maxk = max(topk_values)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for topk in topk_values:
            correct_k = correct[:topk].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size).item())
        return res


def save_checkpoint(file_name, net, optimizer, epoch):
    """Save checkpoint of network."""
    # Populate checkpoint
    checkpoint = {
        "net": net.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "epoch": epoch,
    }
    # create directory if it doesn't exist yet
    file_dir = os.path.split(file_name)[0]
    if not os.path.isdir(file_dir):
        os.makedirs(file_dir)
    # save now
    torch.save(checkpoint, file_name)


def delete_checkpoint(file_name):
    """Delete existing checkpoint."""
    if os.path.isfile(file_name):
        os.remove(file_name)


def load_checkpoint(
    file_name, net, optimizer=None, loc=None, epoch_desired=None
):
    """Load checkpoint into provided net and optimizer."""
    epoch = 0
    found_checkpoint = os.path.isfile(file_name)

    if found_checkpoint:
        checkpoint = torch.load(file_name, map_location=loc)
        epoch = checkpoint["epoch"]
        if epoch_desired is None or epoch_desired == epoch:
            # strict=False ignores keys in the checkpoint that don't exist in
            # the net
            net.load_state_dict(checkpoint["net"], strict=False)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])

    return found_checkpoint, epoch
