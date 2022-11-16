"""Module for all torchvision and custom transforms."""

import random
import numpy as np
from PIL import Image

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F

# wildcard import to have torchvision transforms available here.
from torchvision.transforms import *  # noqa: F403,F401


def _pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        o_width, o_height = img.size
        padh = size - o_height if o_height < size else 0
        padw = size - o_width if o_width < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class SmartCompose(T.Compose):
    """Compose that can handle both segmentation and regular transforms."""

    def __call__(self, *args):
        """Call all transforms considering tuple or non-tuple returns."""

        def _tuplify(xargs):
            if not isinstance(xargs, (tuple, list)):
                xargs = (xargs,)
            return tuple(xargs)

        def _identity(*args):
            if len(args) == 1:
                return args[0]
            else:
                return args

        # can handle tensor returns or tuple of tensor returns...
        for transform in self.transforms:
            args = transform(*_tuplify(args))

        return _identity(*_tuplify(args))


class SegRandomResize(object):
    """T.RandomResize for segmentation datasets."""

    def __init__(self, min_size, max_size=None):
        """Initialize with min and max desired size."""
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        """Call transform with image, target tuple and resize."""
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size, interpolation=Image.NEAREST)
        return image, target


class SegRandomHorizontalFlip(object):
    """T.RandomHorizontalFlip for segmentation datasets."""

    def __init__(self, flip_prob):
        """Initialize with flip probability."""
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        """Call transform with image, target tuple and randomly flip."""
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class SegRandomCrop(object):
    """T.RandomCrop for segmentation datasets."""

    def __init__(self, size):
        """Initialize with desired crop size."""
        self.size = size

    def __call__(self, image, target):
        """Call by padding or cropping to desird size."""
        image = _pad_if_smaller(image, self.size)
        target = _pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class SegCenterCrop(object):
    """T.CenterCrop for segmentation datasets."""

    def __init__(self, size):
        """Initialize with desired crop size."""
        self.size = size

    def __call__(self, image, target):
        """Center crop and return."""
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class SegToTensor(object):
    """T.ToTensor for segmentation dataset."""

    def __call__(self, image, target):
        """Call and return image, target tuple as tensors."""
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class SegNormalize(object):
    """T.Normalize for segmentation dataset."""

    def __init__(self, mean, std):
        """Initialize with desired normalization."""
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        """Normalize image but not target."""
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class RandomNoise(object):
    """Random uniform noise transformation."""

    def __init__(self, normalization):
        """Initialize with normalization constant."""
        self._normalization = normalization

    def __call__(self, image):
        """Return noise image from the current image."""
        noise = image.new().resize_as_(image).uniform_()
        image = image * self._normalization + noise
        image = image / (self._normalization + 1)
        return image
