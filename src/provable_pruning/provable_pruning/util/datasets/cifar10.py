"""Module with custom CIFAR10 Variations (all C-Variations and CIFAR10.1)."""

import os
from abc import ABC, abstractmethod
import copy
import warnings

import tarfile
import numpy as np
from PIL import Image
from torchvision.datasets import CIFAR10

from .dds import DownloadDataset, URLError

# CIFAR10_C variations...
CIFAR10_C_VARIATIONS = {
    "Brightness": "brightness",
    "Contrast": "contrast",
    "Defocus": "defocus_blur",
    "Elastic": "elastic_transform",
    "Fog": "fog",
    "Frost": "frost",
    "Blur": "gaussian_blur",
    "Gauss": "gaussian_noise",
    "Glass": "glass_blur",
    "Impulse": "impulse_noise",
    "Jpeg": "jpeg_compression",
    "Motion": "motion_blur",
    "Pixel": "pixelate",
    "Sat": "saturate",
    "Shot": "shot_noise",
    "Snow": "snow",
    "Spatter": "spatter",
    "Speckle": "speckle_noise",
    "Zoom": "zoom_blur",
}
CIFAR10_C_SEVERITY_MIN = 1
CIFAR10_C_SEVERITY_MAX = 5

# modify keys to contain the full class names
# all severities...
CIFAR10_C_CLASSES = {
    f"CIFAR10_C_{suffix}": (corruption, -1)
    for suffix, corruption in CIFAR10_C_VARIATIONS.items()
}
# fixed severity...
CIFAR10_C_CLASSES.update(
    {
        f"CIFAR10_C_{suffix}_{int(severity)}": (corruption, severity)
        for suffix, corruption in CIFAR10_C_VARIATIONS.items()
        for severity in range(
            CIFAR10_C_SEVERITY_MIN, CIFAR10_C_SEVERITY_MAX + 1
        )
    }
)

# a few variations with partially random labels
CIFAR10_RAND_CLASSES = {
    f"CIFAR10Rand{rand_factor}": rand_factor for rand_factor in range(0, 25, 5)
}

# all the class names that are defined here and should be exported
__all__ = [
    "CIFAR10_1",
    "CIFAR10_C_CLASSES",
    *CIFAR10_C_CLASSES,
    "CIFAR10_RAND_CLASSES",
    *CIFAR10_RAND_CLASSES,
]


class CIFAR10Base(DownloadDataset, ABC):
    """Base class for CIFAR10 replacement test sets that are harder."""

    @property
    @abstractmethod
    def _img_file(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def _label_file(self):
        raise NotImplementedError

    @property
    def _train_tar_file_name(self):
        return None

    @property
    def _train_dir(self):
        return "."

    def _get_train_data(self, download):
        """Return an indexable object for training data points."""
        return CIFAR10(
            self._root, train=True, transform=None, download=download
        )

    def _get_test_data(self, download):
        """Return an indexable object for training data points."""
        imgs = np.load(os.path.join(self._data_path, self._img_file))
        labels = np.load(os.path.join(self._data_path, self._label_file))
        return list(zip(imgs, labels))

    def _convert_to_pil(self, img):
        """Get the image and return the PIL version of it."""
        if self._train:
            return img
        else:
            return Image.fromarray(img)

    def _convert_target(self, target):
        return int(target)


class CIFAR10_1(CIFAR10Base):  # pylint: disable=C0103
    """CIFAR10.1 dataset.

    This data set is a replacement test data set for CIFAR 10 that was
    generated the same way that the original data was generated, thus trying to
    imidate the exact same data distribution. More info see:

    * Code:
      https://github.com/modestyachts/CIFAR-10.1
    * Paper:
      https://arxiv.org/abs/1806.00451
    """

    @property
    def _test_tar_file_name(self):
        return "CIFAR_10_1_v6.tar.gz"

    @property
    def _test_dir(self):
        return "CIFAR_10_1_v6"

    @property
    def _img_file(self):
        return "cifar10.1_v6_data.npy"

    @property
    def _label_file(self):
        return "cifar10.1_v6_labels.npy"


class CIFAR10_C_Base(CIFAR10Base, ABC):  # pylint: disable=C0103
    """CIFAR10-C dataset.

    This dataset tries to find adversarial, but common, data transformations
    that can potentially cause the network to significantly drop in
    performance. Example transformation include weather transformation, etc...
    More infor see:

    * Code:
      https://github.com/hendrycks/robustness
    * Paper:
      https://arxiv.org/abs/1903.12261
    """

    @property
    def _test_tar_file_name(self):
        return "CIFAR-10-C.tar.gz"

    @property
    def _test_dir(self):
        return "CIFAR-10-C"

    @property
    def _img_file(self):
        return f"{self._corruption}.npy"

    @property
    def _label_file(self):
        return "labels.npy"

    @property
    def _num_img_per_severity(self):
        return int(1e4)

    @property
    def _severity_min(self):
        return CIFAR10_C_SEVERITY_MIN

    @property
    def _severity_max(self):
        return CIFAR10_C_SEVERITY_MAX

    @property
    @abstractmethod
    def _corruption(self):
        raise NotImplementedError

    @property
    def _severity(self):
        """-1 means all severity level."""
        return -1

    def _get_test_data(self, download):
        """Return an indexable object for training data points."""
        data = super()._get_test_data(download)
        if self._severity_min <= self._severity <= self._severity_max:
            idx_start = (self._severity - 1) * self._num_img_per_severity
            idx_end = self._severity * self._num_img_per_severity
            data = data[idx_start:idx_end]

        return data


def get_cifar_c_class(name, corruption, severity=-1):
    """Generate new CIFAR10_C class with custom corruption and name."""
    return type(
        name,
        (CIFAR10_C_Base,),
        {"_corruption": corruption, "_severity": severity},
    )


# loop through all names, generate classes, and add them to global() dict
for name, (corruption, severity) in CIFAR10_C_CLASSES.items():
    globals()[name] = get_cifar_c_class(name, corruption, severity)


class CIFAR10RandBase(DownloadDataset):
    """The base class for Cifar10 with random noise in labels.

    Following the double-descent paper, we randomly inject noise into the
    labels of the training data.
    """

    @property
    @abstractmethod
    def _prob_label_noise(self):
        raise NotImplementedError

    @property
    def _label_file(self):
        return f"CIFAR10_Rand_{self._prob_label_noise}_labels.npy"

    @property
    def _train_tar_file_name(self):
        return f"CIFAR10_Rand_{self._prob_label_noise}.tar.gz"

    @property
    def _test_tar_file_name(self):
        return None

    @property
    def _train_dir(self):
        return f"CIFAR10_Rand_{self._prob_label_noise}"

    @property
    def _test_dir(self):
        return "."

    def _download(self):
        """Download data set or generate noisy labels (issue warning!)."""
        try:
            super()._download()
        except URLError:
            self._generate_noisy_labels()
            super()._download()

    def _generate_noisy_labels(self):
        """Generate noisy labels but also issue warning!."""
        # get the original labels
        data_original = CIFAR10(
            self._root, train=True, transform=None, download=True
        )
        labels_original = np.array([data[1] for data in data_original])
        label_max = labels_original.max()

        # check where to add wrong labels
        modify_labels = np.random.rand(len(labels_original)) < (
            self._prob_label_noise / 100.0
        )

        # now create the noisy labels (uniform from wrong labels)
        labels_noisy = copy.deepcopy(labels_original)
        labels_noisy[modify_labels] = (
            labels_noisy[modify_labels]
            + np.random.randint(1, label_max, modify_labels.sum())
        ) % (label_max + 1)

        # store the result
        file_labels = os.path.join("/tmp", self._label_file)
        np.save(file_labels, labels_noisy)

        # store the tar version as well
        tar_file = os.path.join(self._file_dir, self._train_tar_file_name)
        with tarfile.open(tar_file, "w:gz") as tar:
            tar.add(
                file_labels,
                arcname=os.path.join(self._train_dir, self._label_file),
            )

        # issue warning
        warnings.warn(
            "Generated new noisy label file. This should be manually uploaded"
            "to the default download locations."
        )

    def _get_train_data(self, download):
        """Return an indexable object for training data points."""
        data_original = CIFAR10(
            self._root, train=True, transform=None, download=download
        )
        imgs = [data[0] for data in data_original]
        labels_noisy = np.load(os.path.join(self._data_path, self._label_file))
        return list(zip(imgs, labels_noisy))

    def _get_test_data(self, download):
        """Return an indexable object for training data points."""
        return CIFAR10(
            self._root, train=False, transform=None, download=download
        )

    def _convert_to_pil(self, img):
        """Get the image and return the PIL version of it."""
        return img

    def _convert_target(self, target):
        return int(target)


# loop through all names, generate classes, and add them to global() dict
for name, prob_noise in CIFAR10_RAND_CLASSES.items():
    globals()[name] = type(
        name, (CIFAR10RandBase,), {"_prob_label_noise": prob_noise},
    )
