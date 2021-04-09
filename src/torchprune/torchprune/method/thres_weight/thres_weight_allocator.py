"""Module containing allocators for weight thresholding."""
from ..sipp.sipp_allocator import SiPPAllocator


class ThresAllocator(SiPPAllocator):
    """The allocator for weight thresholding."""

    def get_num_samples(self, layer):
        """Get the number of samples for a particular layer index."""
        num_samples = super().get_num_samples(layer)
        return num_samples.sum(dim=0)
