"""A wrapper module for torchdyn data sets with configurations."""
import os
import shutil
import warnings
from abc import ABC, abstractmethod
from urllib.request import URLError

import tarfile
import numpy as np
import torch
import torchdyn.datasets as dyn_data

from .dds import DownloadDataset


class BaseToyDataset(DownloadDataset, ABC):
    """An abstract interface for torchdyn toy datasets."""

    @property
    @abstractmethod
    def _dataset_type(self):
        """Return the type of toy dataset we want to get."""

    @property
    @abstractmethod
    def _n_samples(self):
        """Return number of samples we should generate."""

    @property
    @abstractmethod
    def _dataset_kwargs(self):
        """Return the kwargs to initialize the toy dataset."""

    @property
    def _dataset_tag(self):
        """Return the tag used to identify the files related to dataset."""
        return self._dataset_type

    @property
    def _train_tar_file_name(self):
        return f"torchdyn_toy_{self._dataset_tag}.tar.gz"

    @property
    def _test_tar_file_name(self):
        return self._train_tar_file_name

    @property
    def _train_dir(self):
        return f"torchdyn_toy_{self._dataset_tag}"

    @property
    def _test_dir(self):
        return self._train_dir

    def _get_train_data(self, download):
        x_data = np.load(os.path.join(self._data_path, "x_data_train.npy"))
        y_data = np.load(os.path.join(self._data_path, "y_data_train.npy"))
        return list(zip(x_data, y_data))

    def _get_test_data(self, download):
        x_data = np.load(os.path.join(self._data_path, "x_data_test.npy"))
        y_data = np.load(os.path.join(self._data_path, "y_data_test.npy"))
        return list(zip(x_data, y_data))

    def _convert_to_pil(self, img):
        return torch.tensor(img)

    def _convert_target(self, target):
        return int(target)

    def _download(self):
        """Download data set and generate first if necessary."""
        try:
            super()._download()
        except URLError:
            self._generate_data()
            super()._download()

    def _generate_data(self):
        """Generate and store data now."""
        # issue warning at the beginning to remind user of this change
        warnings.warn(f"Generating new data for {type(self)}.")

        def _sample_data(n_samples):
            toy_dset = dyn_data.ToyDataset()
            x_data, y_data = toy_dset.generate(
                n_samples=n_samples,
                dataset_type=self._dataset_type,
                **self._dataset_kwargs,
            )
            # check that y_data is not none
            if y_data is None:
                y_data = torch.zeros(len(x_data), dtype=torch.long)

            # normalize x data now
            x_data -= x_data.mean()
            x_data /= x_data.std()

            return x_data.cpu().numpy(), y_data.cpu().numpy()

        def _x_tmp_file(tag):
            return os.path.join("/tmp", f"{self._dataset_tag}_{tag}_x.npy")

        def _y_tmp_file(tag):
            return os.path.join("/tmp", f"{self._dataset_tag}_{tag}_y.npy")

        # generate and save train/test data
        tags_size = {"train": self._n_samples, "test": self._n_samples // 2}
        for tag, n_samples in tags_size.items():
            x_data, y_data = _sample_data(n_samples)
            np.save(_x_tmp_file(tag), x_data)
            np.save(_y_tmp_file(tag), y_data)

        # now store in tar file
        tar_file = os.path.join(self._file_dir, self._train_tar_file_name)
        tmp_tar_file = os.path.join("/tmp", self._train_tar_file_name)
        with tarfile.open(tmp_tar_file, "w:gz") as tar:
            for tag in tags_size:
                tar.add(
                    _x_tmp_file(tag),
                    arcname=os.path.join(self._train_dir, f"x_data_{tag}.npy"),
                )
                tar.add(
                    _y_tmp_file(tag),
                    arcname=os.path.join(self._train_dir, f"y_data_{tag}.npy"),
                )

        # move tar file to right location
        shutil.move(tmp_tar_file, tar_file)

        # print reminder at the end
        print(f"Generated new data for {type(self)}.")


class ToyConcentric(BaseToyDataset):
    """The concentric spheres dataset."""

    @property
    def _dataset_type(self):
        """Return the type of toy dataset we want to get."""
        return "spheres"

    @property
    def _n_samples(self):
        """Return number of samples we should generate."""
        return 1024

    @property
    def _dataset_kwargs(self):
        """Return the kwargs to initialize the toy dataset."""
        return {
            "dim": 2,
            "noise": 1e-1,
            "inner_radius": 0.75,
            "outer_radius": 1.5,
        }


class ToyMoons(BaseToyDataset):
    """The moons dataset."""

    @property
    def _dataset_type(self):
        """Return the type of toy dataset we want to get."""
        return "moons"

    @property
    def _n_samples(self):
        """Return number of samples we should generate."""
        return 1024

    @property
    def _dataset_kwargs(self):
        """Return the kwargs to initialize the toy dataset."""
        return {"noise": 1e-1}


class ToySpirals(BaseToyDataset):
    """The spirals dataset."""

    @property
    def _dataset_type(self):
        """Return the type of toy dataset we want to get."""
        return "spirals"

    @property
    def _n_samples(self):
        """Return number of samples we should generate."""
        return 1024

    @property
    def _dataset_kwargs(self):
        """Return the kwargs to initialize the toy dataset."""
        return {"noise": 0.9}


class ToySpirals2(BaseToyDataset):
    """The spirals dataset with more samples and less noise."""

    @property
    def _dataset_type(self):
        """Return the type of toy dataset we want to get."""
        return "spirals"

    @property
    def _dataset_tag(self):
        """Return the tag used to identify the files related to dataset."""
        return "spirals2"

    @property
    def _n_samples(self):
        """Return number of samples we should generate."""
        return int(2 ** 14)

    @property
    def _dataset_kwargs(self):
        """Return the kwargs to initialize the toy dataset."""
        return {"noise": 0.5}


class ToyGaussians(BaseToyDataset):
    """The moon dataset."""

    @property
    def _n_gaussians(self):
        return 6

    @property
    def _dataset_type(self):
        """Return the type of toy dataset we want to get."""
        return "gaussians"

    @property
    def _n_samples(self):
        """Return number of samples we should generate."""
        return 2 ** 14 // self._n_gaussians

    @property
    def _dataset_kwargs(self):
        """Return the kwargs to initialize the toy dataset."""
        return {
            "n_gaussians": self._n_gaussians,
            "std_gaussians": 0.5,
            "radius": 4,
            "dim": 2,
        }


class ToyGaussiansSpiral(BaseToyDataset):
    """The Gaussian spiral dataset."""

    @property
    def _n_gaussians(self):
        return 10

    @property
    def _dataset_type(self):
        """Return the type of toy dataset we want to get."""
        return "gaussians_spiral"

    @property
    def _n_samples(self):
        """Return number of samples we should generate."""
        return 2 ** 14 // self._n_gaussians

    @property
    def _dataset_kwargs(self):
        """Return the kwargs to initialize the toy dataset."""
        return {
            "n_gaussians": self._n_gaussians,
            "n_gaussians_per_loop": 6,
            "dim": 2,
            "radius_start": 4.0,
            "radius_end": 1.0,
            "std_gaussians_start": 0.3,
            "std_gaussians_end": 0.1,
        }


class ToyDiffeqml(BaseToyDataset):
    """The diffeqml dataset."""

    @property
    def _dataset_type(self):
        """Return the type of toy dataset we want to get."""
        return "diffeqml"

    @property
    def _n_samples(self):
        """Return number of samples we should generate."""
        return 2 ** 14

    @property
    def _dataset_kwargs(self):
        """Return the kwargs to initialize the toy dataset."""
        return {"noise": 5e-2}
