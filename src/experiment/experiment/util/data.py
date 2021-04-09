"""A module for all helper functions relevant to datasets."""
import os
import math
import warnings

import torch
import numpy as np
from transformers import default_data_collator

import torchprune.util.datasets as dsets
from torchprune.util import transforms
from torchprune.util import tensor

from .file import create_directory

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
    # set a few parameters
    dset_name = param["network"]["dataset"]
    dset_name_test = param["generated"]["datasetTest"]
    data_dir = os.path.realpath(param["directories"]["training_data"])
    valid_ratio = 0.1
    batch_size = param["generated"]["training"]["batchSize"]

    # generate transform lists now
    def _get_transforms(transforms_type):
        return [
            getattr(transforms, transform["type"])(**transform["kwargs"])
            for transform in param["training"][transforms_type]
        ]

    # get all the desired transforms
    transform_train = _get_transforms("transformsTrain")
    transform_test = _get_transforms("transformsTest")
    transform_end = _get_transforms("transformsFinal")

    # put them in a composer
    transform_test = transforms.SmartCompose(transform_test + transform_end)
    transform_train = transforms.SmartCompose(transform_train + transform_end)

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
        }

        # check if it's an instance of a DownloadDataset...
        dset_class = getattr(dsets, name)
        if issubclass(
            dset_class,
            (
                dsets.DownloadDataset,
                dsets.CIFAR10_C_MixBase,
                dsets.VOCSegmentation2011,
                dsets.VOCSegmentation2012,
            ),
        ):
            kwargs_dset["file_dir"] = data_dir

        # standard arguments on top
        if issubclass(dset_class, dsets.BaseGlue):
            kwargs_dset.update({"model_name": net.torchnet.model_name})
        else:
            kwargs_dset.update({"download": True, "transform": transform})

        # initialize and return instance
        return dset_class(**kwargs_dset)

    def get_dataloader(dataset, shuffle=False, b_size=batch_size):
        """Construct data loader."""
        # ensure that we don't parallelize in data loader with glue.
        # It does not play out well ...
        # (also no need to do that for other small-scale datasets)
        no_thread_classes = (
            dsets.MNIST,
            dsets.BaseGlue,
        )
        if isinstance(dataset, no_thread_classes):
            num_threads = 0
        else:
            num_threads = 4

        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=b_size,
            num_workers=num_threads,
            shuffle=shuffle,
            pin_memory=True,
            collate_fn=_glue_data_collator
            if isinstance(dataset, dsets.BaseGlue)
            else None,
        )
        return loader

    # get test data and loader
    set_test = get_dset(transform_test, train=False)
    loader_test = get_dataloader(set_test)

    # get train/validation dataset
    # we have to do a manual split since true test data labels are not public
    set_train = get_dset(transform_train, train=True)
    set_valid = get_dset(transform_test, train=True)

    # cache etas
    with torch.no_grad():
        device = next(net.parameters()).device
        for in_data, _ in loader_test:
            net(tensor.to(in_data, device))
            break
    eta = torch.sum(torch.Tensor(net.num_etas)).item()
    eta_star = torch.max(torch.Tensor(net.num_etas)).item()

    # Get the size of T. 'sizeOfT' from the param hereby represents the ratio
    # from the validation set we want to use
    size_t_rel = param["coresets"]["sizeOfT"]
    assert size_t_rel <= 0.3
    size_t = int(math.ceil(size_t_rel * len(set_valid)))

    # Get the theoretical size of S.
    delta = param["coresets"]["deltaS"] / eta
    size_s = math.ceil(c_constant * math.log(8.0 * eta_star / delta))

    # Make sure the computed sizes are a multiple of the batch size
    size_s = _round_to_batch_size(size_s, batch_size)
    size_t = _round_to_batch_size(size_t, batch_size)

    # Limit it according to size_t since size_t is based on the ratio and thus
    # never runs overboard
    size_s = min(size_s, size_t)

    # make sure that valid ratio is big enough for both valid set and s set
    valid_size = _round_to_batch_size(valid_ratio * len(set_train), batch_size)
    valid_size += max(0, size_s + batch_size - valid_size)
    valid_ratio = valid_size / len(set_train)

    # get train/valid split
    idx_train, idx_valid = _get_valid_split(
        data_dir, dset_name, len(set_train), valid_ratio, batch_size
    )

    # now split the data
    set_train = torch.utils.data.Subset(set_train, idx_train)
    set_valid = torch.utils.data.Subset(set_valid, idx_valid)

    # Further split validation set into valid set, S Set
    set_valid, set_s = torch.utils.data.random_split(
        set_valid, [len(set_valid) - size_s, size_s]
    )

    # create the remaining loaders now
    loader_train = get_dataloader(set_train, True)
    loader_valid = get_dataloader(set_valid)
    loader_s = get_dataloader(set_s, b_size=min(4, batch_size))

    return {
        "train": loader_train,
        "val": loader_valid,
        "test": loader_test,
        "s_set": loader_s,
    }


def _glue_data_collator(features):
    """Wrap the data collator when we return tuple from dataset."""
    # since we are returning two dicts from the dataset, we have to zip it
    inputs, labels = zip(*features)
    return default_data_collator(inputs), torch.tensor(labels)


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
