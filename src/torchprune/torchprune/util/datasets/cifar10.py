"""Module with custom CIFAR10 Variations (all C-Variations and CIFAR10.1)."""

import os
from abc import ABC, abstractmethod
import copy
import warnings
from urllib.request import URLError

import tarfile
import numpy as np
from PIL import Image
import torch.utils.data as data
from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import download_url

from .dds import DownloadDataset

# fails to import inside docker, so guard it
try:
    import imagecorruptions
except ImportError:
    warnings.warn(
        "imagecorruptions module not available. Please run\n"
        "apt-get install libmagickwand-dev libgl1-mesa-glx\n"
        "to make it available."
    )

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
CIFAR10_C_SEVERITY_MAX = 6

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
        for severity in range(CIFAR10_C_SEVERITY_MIN, CIFAR10_C_SEVERITY_MAX)
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
    "CIFAR10_C_MixBase",
    "CIFAR10_C_Mix1",
    "CIFAR10_C_Mix2",
    "CIFAR10_RAND_CLASSES",
    *CIFAR10_RAND_CLASSES,
]


class CIFAR10_1(CIFAR10):  # pylint: disable=C0103
    """CIFAR10.1 dataset.

    This data set is a replacement test data set for CIFAR 10 that was
    generated the same way that the original data was generated, thus trying to
    imidate the exact same data distribution. More info see:

    * Code:
      https://github.com/modestyachts/CIFAR-10.1
    * Paper:
      https://arxiv.org/abs/1806.00451
    """

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        """Initialize like CIFAR and download CIFAR10.1 if it is test data."""
        super().__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        # only do it for test data
        if train:
            return

        # replace test data with CIFAR10.1 test data now.
        base_url = (
            "https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/"
        )
        file_data = "cifar10.1_v6_data.npy"
        file_labels = "cifar10.1_v6_labels.npy"

        # download if desired
        if download:
            download_url(base_url + file_data, root=self.root)
            download_url(base_url + file_labels, root=self.root)

        # now replace data set as planned.
        self.data = np.load(os.path.join(self.root, file_data))
        self.targets = np.load(os.path.join(self.root, file_labels)).tolist()


class CIFAR10_C_Base(DownloadDataset, ABC):  # pylint: disable=C0103
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
    @abstractmethod
    def _corruption(self):
        raise NotImplementedError

    @property
    def _severity(self):
        """-1 means all severity level."""
        return -1

    @property
    def _train_tar_file_name(self):
        return "CIFAR-10-C-train.tar.gz"

    @property
    def _test_tar_file_name(self):
        return "CIFAR-10-C.tar"

    @property
    def _train_dir(self):
        return "CIFAR-10-C-train"

    @property
    def _test_dir(self):
        return "CIFAR-10-C"

    @property
    def _num_img_per_severity(self):
        return int(1e4)

    @property
    def _severity_min(self):
        return CIFAR10_C_SEVERITY_MIN

    @property
    def _severity_max(self):
        return CIFAR10_C_SEVERITY_MAX

    def _get_train_data(self, download):
        """Return an indexable object for training data points."""
        if self._severity_min <= self._severity < self._severity_max:
            severity_list = [self._severity]
        else:
            severity_list = list(range(self._severity_min, self._severity_max))

        imgs = [
            np.load(
                os.path.join(self._data_path, f"{self._corruption}_{sev}.npy")
            )
            for sev in severity_list
        ]
        labels = [
            np.load(os.path.join(self._data_path, "labels.npy"))
            for sev in severity_list
        ]

        return list(zip(np.concatenate(imgs), np.concatenate(labels)))

    def _get_test_data(self, download):
        """Return an indexable object for training data points."""
        imgs = np.load(
            os.path.join(self._data_path, f"{self._corruption}.npy")
        )
        labels = np.load(os.path.join(self._data_path, "labels.npy"))
        data = list(zip(imgs, labels))
        if self._severity_min <= self._severity < self._severity_max:
            idx_start = (self._severity - 1) * self._num_img_per_severity
            idx_end = self._severity * self._num_img_per_severity
            data = data[idx_start:idx_end]

        return data

    def _convert_to_pil(self, img):
        """Get the image and return the PIL version of it."""
        return Image.fromarray(img)

    def _convert_target(self, target):
        return int(target)

    def _download(self):
        """Download data set or generate noisy labels (issue warning!)."""
        try:
            super()._download()
        except URLError:
            self._generate_corrupted_data()
            super()._download()

    def _generate_corrupted_data(self):
        """Generate corrupted data for all corruptions at the same time."""
        # can only do it for the training data.
        if not self._train:
            return
        # get the original data
        dset = CIFAR10(self._root, train=True, transform=None, download=True)

        # all severitities
        severities = range(CIFAR10_C_SEVERITY_MIN, CIFAR10_C_SEVERITY_MAX)

        # issue warning at the beginning to remind user of this change
        warnings.warn("Generating new train data for CIFAR10_C.")

        # now corrupt the training data and store it and tar it
        for corruption in CIFAR10_C_VARIATIONS.values():
            for severity in severities:
                dset_corrupt = [
                    (
                        getattr(imagecorruptions, corruption)(img, severity),
                        label,
                    )
                    for img, label in dset
                ]
                imgs_corrupt, labels = zip(*dset_corrupt)
                imgs_corrupt = [
                    np.array(img, dtype=np.uint8) for img in imgs_corrupt
                ]
                img_file = f"/tmp/{corruption}_{severity}.npy"
                np.save(img_file, np.uint8(imgs_corrupt))
                np.save("/tmp/labels.npy", np.uint8(labels))
                print(f"Just saved {img_file}")

        # now go on to store it as tar file
        tar_file = os.path.join(self._file_dir, self._train_tar_file_name)
        with tarfile.open(tar_file, "w:gz") as tar:
            for corruption in CIFAR10_C_VARIATIONS.values():
                for severity in severities:
                    img_file = f"{corruption}_{severity}.npy"
                    tar.add(
                        os.path.join("/tmp", img_file),
                        arcname=os.path.join(self._train_dir, img_file),
                    )
            tar.add(
                os.path.join("/tmp", "labels.npy"),
                arcname=os.path.join(self._train_dir, "labels.npy"),
            )

        # issue warning at the end to remind user of this change
        print("Generated new train data for CIFAR10_C.")


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


class CIFAR10_C_MixBase(data.Dataset, ABC):  # pylint: disable=C0103
    """The CIFAR10 dataset with random transforms to corrupt each image.

    The corruption are a subset of the CIFAR10_C corruptions for a random level
    of severity.

    Test images are not corrupted to avoid inconsistencies in testing.
    """

    corruptions = None

    def __init__(
        self,
        root,
        file_dir,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        """Init and just pass it on to the CIFAR10_C classes."""
        severities = range(CIFAR10_C_SEVERITY_MIN, CIFAR10_C_SEVERITY_MAX)
        if train:
            self._data = {
                cor: {
                    sev: globals()[f"CIFAR10_C_{cor}_{sev}"](
                        root=root,
                        file_dir=file_dir,
                        train=train,
                        transform=transform,
                        target_transform=target_transform,
                        download=download,
                    )
                    for sev in severities
                }
                for cor in self.corruptions
            }
        else:
            self._data = {}
        self._data_original = CIFAR10(
            root, train, transform, target_transform, download
        )
        self._train = train

    def __getitem__(self, index):
        """Pick a random corruption and pass it on to the dataset."""
        corruption = np.random.choice(self.corruptions)
        severity = np.random.randint(
            CIFAR10_C_SEVERITY_MIN - 1, CIFAR10_C_SEVERITY_MAX
        )

        if severity < CIFAR10_C_SEVERITY_MIN or not self._train:
            return self._data_original[index]
        else:
            return self._data[corruption][severity][index]

    def __len__(self):
        """Return length of original data set."""
        return len(self._data_original)


class CIFAR10_C_Mix1(CIFAR10_C_MixBase):  # pylint: disable=C0103
    """The CIFAR10 dataset with random transforms to corrupt each image."""

    # this list was constructed once at random
    corruptions = [
        "Contrast",
        "Elastic",
        "Blur",
        "Impulse",
        "Motion",
        "Pixel",
        "Shot",
        "Snow",
        "Zoom",
    ]


class CIFAR10_C_Mix2(CIFAR10_C_MixBase):  # pylint: disable=C0103
    """The CIFAR10 dataset with random transforms to corrupt each image."""

    # this list was constructed once at random
    corruptions = [
        "Brightness",
        "Defocus",
        "Fog",
        "Frost",
        "Gauss",
        "Glass",
        "Jpeg",
        "Sat",
        "Spatter",
        "Speckle",
    ]


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
        name,
        (CIFAR10RandBase,),
        {"_prob_label_noise": prob_noise},
    )
