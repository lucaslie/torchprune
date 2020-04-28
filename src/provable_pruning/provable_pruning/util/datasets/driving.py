"""Module which contains the implementation of the deepknight dataset."""
import os

import torch
from PIL import Image
import numpy as np
import h5py

from .dds import DownloadDataset


class Driving(DownloadDataset):
    """Our driving dataset from deepknight."""

    class _H5Batches:
        def __init__(self, h5_root, train):
            # retrieve data
            h5_files = [
                "20190628-094335_blue_prius_devens_rightside.h5",
                "20190628-150233_blue_prius_devens_rightside.h5",
                "20190723-133449_blue_prius_devens_rightside.h5",
                "20190723-134708_blue_prius_devens_rightside.h5",
                "20190723-154501_blue_prius_devens_rightside.h5",
                "20190723-161821_blue_prius_devens_rightside.h5",
            ]

            h5_datas = [
                h5py.File(os.path.join(h5_root, h5_file), "r")
                for h5_file in h5_files
            ]

            # desired region of interest
            self._roi = [130, 80, 190, 320]

            # get training data (90%) or test data (10%)
            self._data = []
            self._labels = []
            for h5_data in h5_datas:
                split = int(0.9 * len(h5_data["camera_front"]))
                if train:
                    self._data.append(h5_data["camera_front"][:split])
                    self._labels.append(h5_data["inverse_r"][:split])
                else:
                    self._data.append(h5_data["camera_front"][split:])
                    self._labels.append(h5_data["inverse_r"][split:])

            # store the index transitions
            self._transitions = np.cumsum(
                [0] + [len(data) for data in self._data]
            )

        def __getitem__(self, index):
            """Get the item as desired."""
            set_index, data_index = self._split_index(index)

            img = self._data[set_index][data_index]
            target = self._labels[set_index][data_index]

            # doing this so that it is consistent with all other datasets
            # to return a PIL Image
            roi = self._roi

            return img[roi[0] : roi[2], roi[1] : roi[3]], target

        def __len__(self):
            # last transition is also the length
            return self._transitions[-1]

        def _split_index(self, index):
            """Return index of h5_batch and index within batch."""
            set_index = np.sum(index >= self._transitions) - 1
            data_index = index - self._transitions[set_index]

            return set_index, data_index

    @property
    def _train_tar_file_name(self):
        return "deepknight.tar.gz"

    @property
    def _test_tar_file_name(self):
        return self._train_tar_file_name

    @property
    def _train_dir(self):
        return "deepknight/devens_large"

    @property
    def _test_dir(self):
        return self._train_dir

    def _get_train_data(self, download):
        return self._H5Batches(self._data_path, self._train)

    def _get_test_data(self, download):
        return self._get_train_data(None)

    def _convert_to_pil(self, img):
        return Image.fromarray(img)

    def _convert_target(self, target):
        return torch.tensor(target)
