"""Module containing the ALDS net implementations."""


from ..base_decompose import (
    BaseDecomposeNet,
    GroupedDecomposeSparsifier,
    DecomposeRankAllocator,
)

from .alds_allocator import (
    ALDSErrorAllocator,
    ALDSErrorIterativeAllocator,
    ALDSErrorIterativeAllocatorPlus,
    ALDSErrorKOnlyAllocator,
)


class ALDSNet(BaseDecomposeNet):
    """ALDS pruning with k-SVD and iterative error-based allocation."""

    @property
    def _allocator_type(self):
        """Get allocator type."""
        return ALDSErrorIterativeAllocator

    @property
    def _sparsifier_type(self):
        """Get sparsifier type."""
        return GroupedDecomposeSparsifier

    @property
    def _k_split(self):
        """Get number of k splits in each layer."""
        return 3


class ALDSNetPlus(ALDSNet):
    """k-SVD, iterative error-based allocation and schemes."""

    @property
    def _allocator_type(self):
        """Get allocator type."""
        return ALDSErrorIterativeAllocatorPlus


class ALDSNetOne(ALDSNet):
    """ALDSNetPlus with one-shot always."""

    def compress(self, keep_ratio, from_original, initialize):
        """Compress like parent but enforce one-shot."""
        return super().compress(
            keep_ratio,
            from_original=True,
            initialize=len(self.pruners) != len(self.layers),
        )


class ALDSNetOptK(ALDSNet):
    """ALDS with fixed per-layer prune ratio and optimal k."""

    @property
    def _allocator_type(self):
        """Get allocator type."""
        return ALDSErrorKOnlyAllocator


class ALDSNetSimple(ALDSNet):
    """ALDS with k-SVD and const rank layer allocation."""

    @property
    def _allocator_type(self):
        """Get allocator type."""
        return DecomposeRankAllocator


class ALDSNetSimple5(ALDSNetSimple):
    """Simple ALDS with k=3."""

    @property
    def _k_split(self):
        """Get number of k splits in each layer."""
        return 5


class ALDSNetErrorOnly(ALDSNet):
    """ALDS with k-SVD and error-based allocation."""

    @property
    def _allocator_type(self):
        """Get allocator type."""
        return ALDSErrorAllocator
