"""Module implementing network compression with classic SVD."""

from abc import ABC

from ..base_decompose import (
    BaseDecomposeNet,
    DecomposeRankAllocator,
    GroupedDecomposeSparsifier,
)
from ..alds.alds_allocator import ALDSErrorAllocator

from .svd_allocator import (
    SVDNuclearEnergyAllocator,
    SVDFrobeniusEnergyAllocator,
)


class BaseSVDNet(BaseDecomposeNet, ABC):
    """Base class for SVD compression, i.e., ALDS compression with k=1."""

    @property
    def _sparsifier_type(self):
        """Get sparsifier type."""
        return GroupedDecomposeSparsifier

    @property
    def _k_split(self):
        """Get number of k splits in each layer."""
        return 1


class SVDNet(BaseSVDNet):
    # pylint: disable=C0301
    """Simple (HO-)SVD based on ALDS compression with k=1.

    This procedure follows the following setup:
    * low-rank decomposition based on scheme 0
    * constant relative rank reduction across layers

    This was used before among others in the following works:
    * https://arxiv.org/abs/1404.0736
    * http://openaccess.thecvf.com/content_ECCV_2018/html/Chong_Li_Constrained_Optimization_Based_ECCV_2018_paper.html
    * https://arxiv.org/abs/1703.09746
    * https://arxiv.org/abs/1812.02402
    """

    @property
    def _allocator_type(self):
        """Get allocator type."""
        return DecomposeRankAllocator


class SVDNuclearNet(BaseSVDNet):
    """SVD decomposition based on scheme 0 and const energy reduction.

    The procedure follows the following steps:
    * low-rank decomposition based on scheme 0.
    * constant energy reduction in each layer where "energy" is defined as the
      nuclear norm, i.e., sum of singular values.

    This was used in:
    * https://arxiv.org/abs/1711.02638
    """

    @property
    def _allocator_type(self):
        """Get allocator type."""
        return SVDNuclearEnergyAllocator


class SVDFrobeniusNet(BaseSVDNet):
    """SVD decomposition based on scheme 0 and const energy reduction.

    The procedure follows the following steps:
    * low-rank decomposition based on scheme 0.
    * constant energy reduction in each layer where "energy" is defined via the
      Frobenius norm, i.e., sum of squared singular values.

    This was used in:
    * https://arxiv.org/abs/1703.09746
    * https://arxiv.org/abs/1812.02402
    """

    @property
    def _allocator_type(self):
        """Get allocator type."""
        return SVDFrobeniusEnergyAllocator


class SVDErrorNet(SVDNet):
    """SVD decomposition based on scheme 0 and const rel error reduction.

    This corresponds to ALDSNetErrorOnly with k=1 (i.e. based on ours),
    i.e., it's a highly simplified version of ProjectiveNet.
    """

    @property
    def _allocator_type(self):
        """Get allocator type."""
        return ALDSErrorAllocator
