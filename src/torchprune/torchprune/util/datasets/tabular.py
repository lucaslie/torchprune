"""Tabular data sets for FFJORD experiments."""

from abc import ABC, abstractmethod
import numpy as np
import torch
from .dds import DownloadDataset
from ..external.ffjord import datasets as tabular_external


class BaseTabularDataset(DownloadDataset, ABC):
    """An abstract interface for tabular datasets."""

    def __init__(self, *args, **kwargs):
        """Initialize like parent but add train_split, val_split."""
        self._train_split, self._val_split = None, None
        super().__init__(*args, **kwargs)

    @property
    @abstractmethod
    def _dataset_name(self):
        """Return the name of the tabular dataset."""

    @property
    def _train_tar_file_name(self):
        return "data.tar.gz"

    @property
    def _test_tar_file_name(self):
        return self._train_tar_file_name

    @property
    def _train_dir(self):
        return "data/"

    @property
    def _test_dir(self):
        return self._train_dir

    def _get_train_data(self, download):
        dset = getattr(tabular_external, self._dataset_name)(self._data_path)
        val_trn_x = self._join_and_store_split(dset)
        data = torch.tensor(val_trn_x)
        return [(x_data, 0) for x_data in data]

    def _get_test_data(self, download):
        dset = getattr(tabular_external, self._dataset_name)(self._data_path)
        self._join_and_store_split(dset)
        return [(x_data, 0) for x_data in torch.tensor(dset.tst.x)]

    def _convert_to_pil(self, img):
        return img

    def _convert_target(self, target):
        return int(target)

    def _join_and_store_split(self, dset):
        """Join train and validation set but recall split."""
        val_trn_x = np.concatenate((dset.trn.x, dset.val.x))
        self._train_split = list(np.arange(len(dset.trn.x)))
        self._val_split = list(np.arange(len(dset.trn.x), len(val_trn_x)))
        return val_trn_x

    def get_valid_split(self):
        """Return indices corresponding to training and validation."""
        return self._train_split, self._val_split


class Bsds300(BaseTabularDataset):
    """The Bsds300 dataset."""

    @property
    def _dataset_name(self):
        return "BSDS300"


class Gas(BaseTabularDataset):
    """The Gas dataset."""

    @property
    def _dataset_name(self):
        return "GAS"


class Hepmass(BaseTabularDataset):
    """The Hepmass dataset."""

    @property
    def _dataset_name(self):
        return "HEPMASS"


class Miniboone(BaseTabularDataset):
    """The Miniboone dataset."""

    @property
    def _dataset_name(self):
        return "MINIBOONE"


class Power(BaseTabularDataset):
    """The Power dataset."""

    @property
    def _dataset_name(self):
        return "POWER"
