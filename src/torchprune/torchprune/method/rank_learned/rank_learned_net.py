"""Module implementing network compression with classic SVD."""

from .rank_learned_allocator import (
    LearnedRankAllocator,
    LearnedRankAllocatorScheme0,
)
from ..svd import BaseSVDNet


class LearnedRankNet(BaseSVDNet):
    # pylint: disable=C0301
    """SVD-based compression with learned ranks for each layer.

    This is simple procedure following the following setup:
    * low-rank decomposition based on scheme 1
    * learned ranks based on cost equation Eq (5) of cited paper below

    This was used before in the following work:
    * http://openaccess.thecvf.com/content_CVPR_2020/html/Idelbayev_Low-Rank_Compression_of_Neural_Nets_Learning_the_Rank_of_Each_CVPR_2020_paper.html
    """

    @property
    def _allocator_type(self):
        """Get allocator type."""
        return LearnedRankAllocator


class LearnedRankNetScheme0(BaseSVDNet):
    """SVD-based compression with learned ranks for each layer and scheme 0."""

    @property
    def _allocator_type(self):
        """Get allocator type."""
        return LearnedRankAllocatorScheme0
