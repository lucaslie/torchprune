"""Our custom ImageNet implementation compatible with our downloadable data."""

import os

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import pil_loader

from .dds import DownloadDataset


class ImageNet(DownloadDataset):
    """Custom class for ImageNet that can download and maintain dataset."""

    @property
    def _train_tar_file_name(self):
        return "imagenet_object_localization.tar.gz"

    @property
    def _test_tar_file_name(self):
        return self._train_tar_file_name

    @property
    def _train_dir(self):
        return "ILSVRC/Data/CLS-LOC/train"

    @property
    def _test_dir(self):
        return "ILSVRC/Data/CLS-LOC/val"

    @property
    def _valprep_file(self):
        """Return file that gives the class for each validation image.

        File is taken from:
        https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
        """
        valprep = os.path.join(__file__, "../imagenetval/valprep.sh")
        return os.path.realpath(valprep)

    def _get_train_data(self, download):
        """Return an indexable object for training data points."""
        return ImageFolder(root=self._data_path)

    def _get_test_data(self, download):
        """Return an indexable object for training data points."""
        # retrieve val image name to val target class map from valprep
        val_lookup = {}
        classes_lookup = {}
        with open(self._valprep_file, "r") as file:
            valprep = file.read().split("\t\n")
        for line in valprep:
            if "mv" not in line and ".JPEG" not in line:
                continue

            _, val_img_name, target_class = line[:-1].split(" ")

            # store hash map from img to target_class
            val_lookup[val_img_name] = target_class

            # store hash map to look up class keys
            classes_lookup[target_class] = None

        classes = list(classes_lookup.keys())
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        assert len(classes) == 1000

        files = list(val_lookup.keys())
        targets = [class_to_idx[val_lookup[file]] for file in files]

        return list(zip(files, targets))

    def _convert_to_pil(self, img):
        """Get the image and return the PIL version of it."""
        if self._train:
            return img
        else:
            return pil_loader(os.path.join(self._data_path, img))

    def _convert_target(self, target):
        return int(target)
