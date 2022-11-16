"""Base Class Implementation to download any dataset from our S3 bucket."""

import os
from abc import ABC, abstractmethod

import torch.utils.data as data
from .download import download_and_extract_archive


class DownloadDataset(data.Dataset, ABC):
    """Custom abstract class for that can download and maintain dataset."""

    @property
    @abstractmethod
    def _train_tar_file_name(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def _test_tar_file_name(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def _train_dir(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def _test_dir(self):
        raise NotImplementedError

    @property
    def _file_url(self):
        return f"file://{self._file_dir}/"

    @abstractmethod
    def _get_train_data(self, download):
        """Return an indexable object for training data points."""
        raise NotImplementedError

    @abstractmethod
    def _get_test_data(self, download):
        """Return an indexable object for training data points."""
        raise NotImplementedError

    @abstractmethod
    def _convert_to_pil(self, img):
        """Get the image and return the PIL version of it."""
        raise NotImplementedError

    @abstractmethod
    def _convert_target(self, target):
        """Convert target to correct format."""
        raise NotImplementedError

    def __init__(
        self,
        root,
        file_dir,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        """Initialize ImageNet dataset.

        Args:
            root (str): where to store the data set to be downloaded.
            file_dir (str): where to look for downloading data.
            train (bool, optional): train or test data set. Defaults to True.
            transform (torchvision.transforms, optional): set of transforms to
            apply to input data. Defaults to None.
            target_transform (torchvision.transforms, optional): set of
            transforms to apply to target data. Defaults to None.
            download (bool, optional): try downloading. Defaults to False.

        """
        # expand the root to the full path and ensure it exists
        self._root = os.path.realpath(root)
        os.makedirs(self._root, exist_ok=True)

        # expand data_dir to the full path and ensure it exists
        self._file_dir = os.path.realpath(file_dir)
        os.makedirs(self._file_dir, exist_ok=True)

        # check whether it's training
        self._train = train

        # store transforms
        self.transform = transform
        self.target_transform = target_transform

        # check path and tar file to either train or test data
        if self._train:
            self._data_path = os.path.join(self._root, self._train_dir)
        else:
            self._data_path = os.path.join(self._root, self._test_dir)

        if download:
            # download now if needed
            self._download()
        elif not os.path.exists(self._data_path):
            # check if path exists
            raise FileNotFoundError("Data not found, please set download=True")

        # get the data
        if self._train:
            self._data = self._get_train_data(download)
        else:
            self._data = self._get_test_data(download)

    def _download(self):
        """Download the data set from the cloud and return full file path."""
        # tar file name
        if self._train:
            tar_file = self._train_tar_file_name
        else:
            tar_file = self._test_tar_file_name

        def download_and_extract(url):
            """Use the torchvision fucntion to download and extract."""
            download_and_extract_archive(
                url=url + tar_file,
                download_root=self._root,
                filename=tar_file,
                extract_root=self._root,
                remove_finished=True,
            )

        # try downloading it from files URL.
        if tar_file is not None and not os.path.exists(self._data_path):
            download_and_extract(self._file_url)

    def __getitem__(self, index):
        """Return appropriate item."""
        img, target = self._data[index]

        # it might not be a PIL image yet ...
        img = self._convert_to_pil(img)

        # target might be weird, so convert it first
        target = self._convert_target(target)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """Get total number of data points."""
        return len(self._data)
