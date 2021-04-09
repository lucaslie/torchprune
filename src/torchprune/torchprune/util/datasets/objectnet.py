"""ObjectNet implementation based on ImageNet derivative."""
import os
import json
import subprocess

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url

from .imagenet import ImageNet


class ObjectNet(ImageNet):
    """ObjectNet with ImageNet only classes."""

    @property
    def _test_tar_file_name(self):
        return "objectnet-1.0.zip"

    @property
    def _test_dir(self):
        return "objectnet-1.0/images"

    def _convert_to_pil(self, img):
        img = super()._convert_to_pil(img)

        if self._train:
            return img
        else:
            border = 3
            return img.crop(
                (border, border, img.size[0] - border, img.size[1] - border)
            )

    def _get_test_data(self, download):
        # have pytorch's ImageFolder class analyze the directories
        o_dataset = ImageFolder(self._data_path)

        # get mappings folder
        mappings_folder = os.path.abspath(
            os.path.join(self._data_path, "../mappings")
        )

        # get ObjectNet label to ImageNet label mapping
        with open(
            os.path.join(mappings_folder, "objectnet_to_imagenet_1k.json")
        ) as file_handle:
            o_label_to_all_i_labels = json.load(file_handle)

        # now remove double i labels to avoid confusion
        o_label_to_i_labels = {
            o_label: all_i_label.split("; ")
            for o_label, all_i_label in o_label_to_all_i_labels.items()
        }

        # some in-between mappings ...
        o_folder_to_o_idx = o_dataset.class_to_idx
        with open(
            os.path.join(mappings_folder, "folder_to_objectnet_label.json")
        ) as file_handle:
            o_folder_o_label = json.load(file_handle)

        # now get mapping from o_label to o_idx
        o_label_to_o_idx = {
            o_label: o_folder_to_o_idx[o_folder]
            for o_folder, o_label in o_folder_o_label.items()
        }

        # some in-between mappings ...
        with open(
            os.path.join(mappings_folder, "pytorch_to_imagenet_2012_id.json")
        ) as file_handle:
            i_idx_to_i_line = json.load(file_handle)
        with open(
            os.path.join(mappings_folder, "imagenet_to_label_2012_v2")
        ) as file_handle:
            i_line_to_i_label = file_handle.readlines()

        i_line_to_i_label = {
            i_line: i_label[:-1]
            for i_line, i_label in enumerate(i_line_to_i_label)
        }

        # now get mapping from i_label to i_idx
        i_label_to_i_idx = {
            i_line_to_i_label[i_line]: int(i_idx)
            for i_idx, i_line in i_idx_to_i_line.items()
        }

        # now get the final mapping of interest!!!
        o_idx_to_i_idxs = {
            o_label_to_o_idx[o_label]: [
                i_label_to_i_idx[i_label] for i_label in i_labels
            ]
            for o_label, i_labels in o_label_to_i_labels.items()
        }

        # now get a list of files of interest
        overlapping_samples = []
        for filepath, o_idx in o_dataset.samples:
            if o_idx not in o_idx_to_i_idxs:
                continue
            rel_file = os.path.relpath(filepath, self._data_path)
            overlapping_samples.append((rel_file, o_idx_to_i_idxs[o_idx][0]))

        return overlapping_samples

    def _download(self):
        """Download the data set from the cloud and return full file path."""
        # same download for training, otherwise we have pw-protected zip.
        if self._train:
            return super()._download()

        # download first.
        download_url(
            url=self._file_url + self._test_tar_file_name,
            root=self._root,
            filename=self._test_tar_file_name,
        )

        # now extract and consider password protection.
        from_path = os.path.join(self._root, self._test_tar_file_name)

        # with zipfile.ZipFile(from_path, "r") as f_zip:
        #     f_zip.extractall(
        #         self._root, pwd=bytes("objectnetisatestset", "utf-8")
        #     )

        # unzip with bash's unzip: so much faster due to password protection
        subprocess.run(
            ["unzip", "-P", "objectnetisatestset", from_path, "-d", self._root]
        )

        os.remove(from_path)
