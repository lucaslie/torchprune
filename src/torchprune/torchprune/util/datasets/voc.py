"""Pascal VOC 2012 with augmented training data for segmentation."""

from abc import ABC, abstractmethod
import os
import shutil
import tarfile
import warnings
from urllib.request import URLError

from PIL import Image
import imagecorruptions
import numpy as np
import torchvision.datasets
from .download import download_and_extract_archive


# VOC_C variations...
VOC_C_VARIATIONS = {
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
VOC_C_SEVERITY_MIN = 1
VOC_C_SEVERITY_MAX = 6

# generate all class names for VOC2011_C variations
VOCSEGMENTATION2011_C_CLASSES = {
    f"VOCSegmentation2011_C_{suffix}_{int(severity)}": (corruption, severity)
    for suffix, corruption in VOC_C_VARIATIONS.items()
    for severity in range(VOC_C_SEVERITY_MIN, VOC_C_SEVERITY_MAX)
}

# generate all class names for VOC2012_C variations
VOCSEGMENTATION2012_C_CLASSES = {
    f"VOCSegmentation2012_C_{suffix}_{int(severity)}": (corruption, severity)
    for suffix, corruption in VOC_C_VARIATIONS.items()
    for severity in range(VOC_C_SEVERITY_MIN, VOC_C_SEVERITY_MAX)
}


__all__ = [
    "VOCSegmentation2011",
    "VOCSegmentation2012",
    *VOCSEGMENTATION2011_C_CLASSES,
    *VOCSEGMENTATION2012_C_CLASSES,
]


class VOCSegmentation2011(torchvision.datasets.SBDataset):
    """Pascal VOC 2011 with augmented training data according to SBD."""

    @property
    def _year(self):
        """Get the year (version) of the Pascal VOC dataset."""
        return "2011"

    def __init__(
        self,
        root,
        file_dir,
        train=True,
        transform=None,
        download=False,
    ):
        """Initialize like other data sets in our collection."""
        super_kwargs = {
            "root": root,
            "image_set": "train" if train else "val",
            "mode": "segmentation",
            "download": download,
            "transforms": transform,
        }
        if download:
            # when already downloaded shutil.move will throw error.
            try:
                super().__init__(**super_kwargs)
            except shutil.Error:
                super_kwargs["download"] = False
        if not super_kwargs["download"]:
            super().__init__(**super_kwargs)

        # remember some of the args
        self.file_dir = file_dir
        self.download = download


class VOCSegmentation2012(torchvision.datasets.VOCSegmentation):
    """VOCSegmentation with augmented training data.

    This is the standard VOC segmentation data set but training is augmented
    with the data from
    https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0

    To use this dataset you should download the zip-file from the dropbox link
    and place it in the folder "file_dir". The rest is handled automatically.
    """

    @property
    def _year(self):
        """Get the year (version) of the Pascal VOC dataset."""
        return "2012"

    def __init__(
        self,
        root,
        file_dir,
        train=True,
        transform=None,
        download=False,
    ):
        """Initiliaze like the other datasets in our collection."""
        self.images = None
        super().__init__(
            root=root,
            year="2012",
            image_set="train" if train else "val",
            download=download,
            transforms=transform,
        )

        # validation set is standard.
        if not train:
            return

        # that is where the masks are after downloading
        mask_dir = os.path.join(self.root, "SegmentationClassAug")

        # check for downloading
        if not os.path.isdir(mask_dir) and download:
            url_path = f"file://{os.path.realpath(file_dir)}"
            file_name = "SegmentationClassAug.zip"
            download_and_extract_archive(
                url=f"{url_path}/{file_name}",
                download_root=self.root,
                filename=file_name,
                extract_root=self.root,
                remove_finished=True,
            )

        # check if mask directory exists now.
        if not os.path.isdir(mask_dir):
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        # now get the splits for this file.
        split_f = os.path.join(__file__, "../vocaug/train_aug.txt")
        split_f = os.path.realpath(split_f)

        with open(os.path.join(split_f), "r") as file:
            file_names = [x.strip() for x in file.readlines()]

        # get the image directory
        image_dir = os.path.realpath(os.path.split(self.images[0])[0])

        # now reassign images and masks file list
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        assert len(self.images) == len(self.masks)

        # remember some of the args
        self.file_dir = file_dir
        self.download = download


class VOC_C_Base(ABC):  # pylint: disable=C0103
    # pylint: disable=E1101,E0203
    """This is a corrupted Pascal VOC dataset.

    It can be used in a multiple-inheritance scenario for both VOC2011 and
    VOC2012 (see below for usage).

    Corruptions are the same as from the ImageNet-C/CIFAR-C datasets but
    generalized to any dataset using the pip package "imagecorruptions as
    described here:

    https://github.com/bethgelab/imagecorruptions
    """

    @property
    @abstractmethod
    def _corruption(self):
        """Get the name of the corruption (function)."""

    @property
    @abstractmethod
    def _severity(self):
        """Get the severity level."""

    @property
    def _corruption_tag(self):
        return f"VOC{self._year}_{self._corruption}_{self._severity}"

    @property
    def _tar_file(self):
        return f"{self._corruption_tag}.tar.gz"

    @property
    def _tar_download_path(self):
        return os.path.realpath(os.path.join(self.file_dir, self._tar_file))

    @property
    def _corruption_img_dir(self):
        return os.path.realpath(os.path.join(self.root, self._corruption_tag))

    def __init__(self, *args, **kwargs):
        """Initialize it as if it was a regular VOCSegmentation2011 dataset.

        Then generate the corrupted images if necessary and point to those.
        """
        super().__init__(*args, **kwargs)

        def _download_corrupted_data():
            """Download and extract the data as is necessary."""
            download_and_extract_archive(
                url=f"file://{self._tar_download_path}",
                download_root=self.root,
                filename=self._tar_file,
                extract_root=self.root,
                remove_finished=True,
            )

        # check if we want/can download
        if not os.path.isdir(self._corruption_img_dir) and self.download:
            try:
                # check if downloadable already (or already downloaded).
                _download_corrupted_data()
            except URLError:
                # if not, generate data and then download it.
                self._generate_corrupted_data()
                _download_corrupted_data()

        # check if img directory exists now.
        if not os.path.isdir(self._corruption_img_dir):
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        # now assign images (files) to new location
        self.images = [
            os.path.join(self._corruption_img_dir, os.path.split(img)[1])
            for img in self.images
        ]
        assert len(self.images) == len(self.masks)

    def _generate_corrupted_data(self):
        """Generate the corrupted data and store everything as jpg."""
        warnings.warn(
            f"Generating new corrupted data set for {type(self).__name__}"
        )

        # temporarily disable transforms
        transforms_backup = self.transforms
        self.transforms = None

        # generate temporary folder
        tmp_folder = os.path.join("/tmp", self._corruption_tag)
        os.makedirs(tmp_folder, exist_ok=True)

        # start a tar file
        tar_temp = f"{self._tar_download_path}.tmp"
        with tarfile.open(tar_temp, "w:gz") as tar:
            # go through all images
            for i, (img, _) in enumerate(self):
                # corrupt image
                img_corrupt = getattr(imagecorruptions, self._corruption)(
                    img, self._severity
                )

                # convert to numpy and then to pil
                img_corrupt = np.array(img_corrupt).astype(np.uint8)
                img_corrupt = Image.fromarray(img_corrupt)

                # store is as jpeg in tmp folder
                f_name = os.path.split(self.images[i])[1]
                tmp_file = os.path.join(tmp_folder, f_name)
                img_corrupt.save(tmp_file, format="JPEG")

                # add to tar file
                tar.add(
                    tmp_file,
                    arcname=os.path.join(self._corruption_tag, f_name),
                )

        # re-enable transforms
        self.transforms = transforms_backup

        # clean up tmp folder
        shutil.rmtree(tmp_folder, ignore_errors=True)

        # move tar file to true location
        shutil.move(tar_temp, self._tar_download_path)

        print(f"Generated new corrupted data set for {type(self).__name__}")


# using the base class now generate the actual classes and add them to the
# global dict. This way when importing the module, these classes are
# dynamically generated and available.
# ... 2011 ...
for name, (corruption, severity) in VOCSEGMENTATION2011_C_CLASSES.items():
    globals()[name] = type(
        name,  # give it the name "name"
        (VOC_C_Base, VOCSegmentation2011),  # inherit from these classes
        {"_corruption": corruption, "_severity": severity},  # added attributes
    )
# ... 2012 ...
for name, (corruption, severity) in VOCSEGMENTATION2012_C_CLASSES.items():
    globals()[name] = type(
        name,  # give it the name "name"
        (VOC_C_Base, VOCSegmentation2012),  # inherit from these classes
        {"_corruption": corruption, "_severity": severity},  # added attributes
    )
