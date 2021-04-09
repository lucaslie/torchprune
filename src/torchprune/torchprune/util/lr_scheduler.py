"""Module containing all the desired learning rate schedulers.

These are all standard learning rate schedulers from pytoch and any additional
custom learning rate schedulers.

"""
# wild card import is okay here since we want the full functionaliyt of the
# torch lr_scheduler module available here.
from torch.optim.lr_scheduler import *  # noqa: F403,F401
from torch.optim.lr_scheduler import LambdaLR  # avoid linter error below.


class WarmupLR(LambdaLR):
    """Linear warmup learning rate based on the initial learning rate.

    Note that this scheduler is not chainable during the warmup period since it
    overwrites other changes to the learning rate schedule. Make sure when
    using it that other modifications to the learning rate schedule happen
    afterwards!
    """

    def __init__(self, optimizer, warmup_epoch, last_epoch=-1, verbose=False):
        """Initialize with desired number of warmup epochs."""
        self.warmup_epoch = warmup_epoch
        super().__init__(
            optimizer,
            lambda e: min(e / self.warmup_epoch, 1.0),
            last_epoch=last_epoch,
            verbose=verbose,
        )

    def step(self, epoch=None):
        """Only step here if we are still within warmup."""
        if self.last_epoch >= self.warmup_epoch:
            return
        super().step(epoch)


class PolyLR(LambdaLR):
    """Polynomial learning rate based on the initial learning rate.

    Note that this scheduler is not chainable while active since it
    overwrites other changes to the learning rate schedule.
    """

    def __init__(
        self, optimizer, max_epoch, last_epoch=-1, power=0.9, verbose=False
    ):
        """Initialize with desired power for multiplier and last epoch."""
        super().__init__(
            optimizer,
            lambda e: max(0.0, (1 - e / max_epoch)) ** power,
            last_epoch=last_epoch,
            verbose=verbose,
        )
