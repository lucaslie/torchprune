"""A module for all helper functions relevant to datasets."""
import os
import math
import multiprocessing
import warnings

import torchvision.transforms as transforms
import torch
import numpy as np

import provable_pruning.util.datasets as dsets
from .file import create_directory

PIN_MEMORY = True

__all__ = ["get_data_loader"]


def get_data_loader(param, net, c_constant):
    """Construct the five data loaders we need: train, valid, test, S, T.

    Arguments:
        param {dict} -- parameters from parameter file
        net {NetHandle} -- net we are trying to compress (cache etas this way)
        c_constant {float} -- C constant for size of S

    Returns:
        tuple of data loader -- order is train, valid, test, S, T

    """
    # grab a few parameters from the param dict
    dset_name = param["network"]["dataset"]
    dset_name_test = param["generated"]["datasetTest"]
    mean = param["datasets"][dset_name]["mean"]
    std_dev = param["datasets"][dset_name]["std"]
    data_dir = os.path.realpath(param["directories"]["training_data"])
    valid_ratio = param["datasets"]["validSize"]
    batch_size = param["generated"]["training"]["batchSize"]

    # have a list of basic transforms that needs to be applied regardless
    transforms_base = [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std_dev),
    ]

    # check training transforms from param file ...
    transform_train = []
    for transform in param["training"]["transformsTrain"]:
        transform_train.append(eval("transforms." + transform))

    # check test transforms from param file ...
    transform_test = []
    for transform in param["training"]["transformsTest"]:
        transform_test.append(eval("transforms." + transform))

    # add base transforms to the end
    transform_train.extend(transforms_base)
    transform_test.extend(transforms_base)

    # put them in a composer
    transform_test = transforms.Compose(transform_test)
    transform_train = transforms.Compose(transform_train)

    # construct data sets (always just download to 'local_data')
    # WHY? Cloud or not, 'local_data' will live on the fastest storage device
    # making subsequeny actions much faster...
    root = param["directories"]["local_data"]
    create_directory(root)

    def get_dset(transform, train):
        # setup standard kwargs for all data sets
        name = dset_name if train else dset_name_test
        kwargs_dset = {
            "root": root,
            "train": train,
            "download": True,
            "transform": transform,
        }
        # check if it's an instance of a DownlaodDataset...
        dset_class = getattr(dsets, name)
        if issubclass(dset_class, dsets.DownloadDataset):
            kwargs_dset["file_dir"] = data_dir
        # initialize and return instance
        return dset_class(**kwargs_dset)

    set_train = get_dset(transform_train, train=True)
    set_valid = get_dset(transform_test, train=True)
    set_test = get_dset(transform_test, train=False)

    # get train/valid split
    idx_train, idx_valid = _get_valid_split(
        data_dir, dset_name, len(set_train), valid_ratio, batch_size
    )

    # now split the data
    set_train = torch.utils.data.Subset(set_train, idx_train)
    set_valid = torch.utils.data.Subset(set_valid, idx_valid)

    # cache etas
    device = next(net.parameters()).device
    image = set_valid[0][0]
    image = image.to(device).unsqueeze(0)  # do simulate batch dimension ...
    net(image)
    eta = torch.sum(torch.Tensor(net.num_etas)).item()
    eta_star = torch.max(torch.Tensor(net.num_etas)).item()

    # Get the theoretical size of S.
    delta = param["coresets"]["deltaS"] / eta
    size_s = math.ceil(c_constant * math.log(8.0 * eta_star / delta))

    # Get the size of T. 'sizeOfT' from the param hereby represents the ratio
    # from the validation set we want to use
    size_t = int(math.ceil(param["coresets"]["sizeOfT"] * len(set_valid)))

    # Make sure the computed sizes are a multiple of the batch size
    size_s = _round_to_batch_size(size_s, batch_size)
    size_t = _round_to_batch_size(size_t, batch_size)

    # Further split validation set into valid set, S Set, T Set
    set_valid, set_s, set_t = torch.utils.data.random_split(
        set_valid, [len(set_valid) - size_s - size_t, size_s, size_t]
    )

    # create all the loaders now
    num_threads = multiprocessing.cpu_count()

    def get_dataloader(dataset, shuffle=False):
        """Construct data loader."""
        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_threads,
            shuffle=shuffle,
            pin_memory=PIN_MEMORY,
        )
        return loader

    loader_train = get_dataloader(set_train, True)
    loader_valid = get_dataloader(set_valid)
    loader_test = get_dataloader(set_test)
    loader_s = get_dataloader(set_s)
    loader_t = get_dataloader(set_t)

    return {
        "train": loader_train,
        "val": loader_valid,
        "test": loader_test,
        "s_set": loader_s,
        "t_set": loader_t,
    }


def _get_valid_split(data_dir, dset_name, dset_len, ratio_valid, batch_size):
    """Split a dataset into two datasets using index-based subsets."""
    file = os.path.join(data_dir, f"{dset_name}_validation.npz")

    # check if there is already a copy of indices saved with the same ratio
    # otherwise create new set of indices
    is_new_set = False
    if os.path.isfile(file):
        with np.load(file) as data:
            if data["ratioSet1"] == ratio_valid:
                idx_valid = data["set1Idx"]
                idx_train = data["set2Idx"]
            else:
                is_new_set = True
    else:
        is_new_set = True

    if is_new_set:
        warnings.warn("Creating a new split for the data set!!!", Warning)

        indices = list(range(dset_len))
        split = int(np.floor(ratio_valid * dset_len))

        # convert split to be a multiple of batch_size
        split = _round_to_batch_size(split, batch_size)

        # shuffle indices
        np.random.seed()
        np.random.shuffle(indices)
        idx_train, idx_valid = indices[split:], indices[:split]

        # make sure directory exists
        create_directory(data_dir)

        # save data
        np.savez(
            file, set2Idx=idx_train, set1Idx=idx_valid, ratioSet1=ratio_valid
        )

    return idx_train, idx_valid


def _round_to_batch_size(number, batch_size):
    """Round number to the neareast multiple of batch_size."""
    return int(batch_size * math.ceil(float(number) / batch_size))
