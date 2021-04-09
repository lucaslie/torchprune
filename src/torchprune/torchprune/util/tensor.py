"""Module containing a bunch of tensor utilities."""
import math
import torch


def to(tensors, *args, **kwargs):
    """Transfer tensors via their to-method and return them.

    We handle case where tensors are just one tensor or dict of tensors.
    """

    def _convert(tnsr):
        if isinstance(tnsr, torch.Tensor):
            tnsr = tnsr.to(*args, **kwargs)
        return tnsr

    if isinstance(tensors, dict):
        tensors_to = {}
        for key, tens in tensors.items():
            tensors_to[key] = _convert(tens)
    else:
        tensors_to = _convert(tensors)

    return tensors_to


def flatten_all_but_last(tensor):
    """Flatten all dimensions except last one and return tensor."""
    return tensor.view(tensor[..., 0].numel(), tensor.shape[-1])


def get_slice(tensors, slicer):
    """Return slice from tensors.

    We handle case where tensors are just one tensor or dict of tensors.
    """
    if isinstance(tensors, dict):
        tensors_sliced = {}
        for key, tens in tensors.items():
            tensors_sliced[key] = tens[slicer]
    else:
        tensors_sliced = tensors[slicer]

    return tensors_sliced


class MiniDataLoader:
    """A data loader iterator with mini-batches."""

    def __init__(self, dataloader, batch_size=1):
        """Initialize with the desired mini-batch size."""
        super().__init__()
        self._dataloader = dataloader
        self.batch_size = min(batch_size, dataloader.batch_size)

    def __len__(self):
        """Return approximate length of mini data loader."""
        return len(self._dataloader) * math.ceil(
            self._dataloader.batch_size / self.batch_size
        )

    def __iter__(self):
        """Initialize iterator with generator."""
        return self._gen()

    def _get_batch_len(self, batch):
        """Get the length of a batch tensor."""
        if isinstance(batch, dict):
            key = list(batch.keys())[0]
            return len(batch[key])
        return len(batch)

    def _gen(self):
        """Iterate with next and return mini-batch.

        We don't need to raise StopIteration since the original dataloader will
        do that for us.
        """
        for images, targets in self._dataloader:
            batch_len = self._get_batch_len(images)
            for i_mini in range(0, batch_len, self.batch_size):
                img_mini = get_slice(
                    images, slice(i_mini, i_mini + self.batch_size)
                )
                targets_mini = get_slice(
                    targets, slice(i_mini, i_mini + self.batch_size)
                )

                yield img_mini, targets_mini
