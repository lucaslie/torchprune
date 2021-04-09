"""Module with all ImageNet-C variations meta-programmed into it."""

import os
from abc import ABC, abstractmethod

from torchvision.datasets import ImageFolder

from .imagenet import ImageNet

# CIFAR10_C variations...
IMAGENET_C_VARIATIONS = {
    "Brightness": ("brightness", "weather"),
    "Contrast": ("contrast", "digital"),
    "Defocus": ("defocus_blur", "blur"),
    "Elastic": ("elastic_transform", "digital"),
    "Fog": ("fog", "weather"),
    "Frost": ("frost", "weather"),
    "Blur": ("gaussian_blur", "extra"),
    "Gauss": ("gaussian_noise", "noise"),
    "Glass": ("glass_blur", "blur"),
    "Impulse": ("impulse_noise", "noise"),
    "Jpeg": ("jpeg_compression", "digital"),
    "Motion": ("motion_blur", "blur"),
    "Pixel": ("pixelate", "digital"),
    "Sat": ("saturate", "extra"),
    "Shot": ("shot_noise", "noise"),
    "Snow": ("snow", "weather"),
    "Spatter": ("spatter", "extra"),
    "Speckle": ("speckle_noise", "extra"),
    "Zoom": ("zoom_blur", "blur"),
}
IMAGENET_C_SEVERITY_MIN = 1
IMAGENET_C_SEVERITY_MAX = 5

# modify keys to contain the full class names
IMAGENET_C_CLASSES = {
    f"ImageNet_C_{suffix}_{int(severity)}": (corruption, corrup_type, severity)
    for suffix, (corruption, corrup_type) in IMAGENET_C_VARIATIONS.items()
    for severity in range(IMAGENET_C_SEVERITY_MIN, IMAGENET_C_SEVERITY_MAX + 1)
}

__all__ = ["IMAGENET_C_CLASSES", *IMAGENET_C_CLASSES]


class ImageNet_C_Base(ImageNet, ABC):  # pylint: disable=C0103
    """ImageNet-C base class that inherits from ImageNet."""

    @property
    @abstractmethod
    def _corruption(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def _corruption_type(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def _severity(self):
        raise NotImplementedError

    @property
    def _test_tar_file_name(self):
        return f"{self._corruption_type}.tar"

    @property
    def _test_dir(self):
        return os.path.join(self._corruption, str(int(self._severity)))

    def _get_test_data(self, download):
        return ImageFolder(root=self._data_path)

    def _convert_to_pil(self, img):
        return img


def get_imagenet_c_class(name, corruption, corruption_type, severity):
    """Generate new ImageNet-C class with custom corruption and name."""
    return type(
        name,
        (ImageNet_C_Base,),
        {
            "_corruption": corruption,
            "_corruption_type": corruption_type,
            "_severity": severity,
        },
    )


# loop through all names, generate classes, and add them to global() dict
for name, (corruption, corrupt_type, severity) in IMAGENET_C_CLASSES.items():
    globals()[name] = get_imagenet_c_class(
        name, corruption, corrupt_type, severity
    )
