"""A logger that can be used during training to log to tensorboard."""
import os
import time
import math

from torch.utils import tensorboard as tb
from .tensorboard import log_scalar


class TrainLogger(object):
    """A logger class that is pickeable (used for training).

    A separate class for logging training. The logger cannot be pickled,
    thus cannot be used as argument for training with multiprocessing
    enabled. This class can be pickled and then initialized within the
    multiprocessing context.
    """

    def __init__(self, log_dir=None, global_tag=None, class_to_names=None):
        """Initialize the train logger.

        If the optional arguments are not supplied, the logger will print
        updates about the training progress but won't log it to tensorboard
        """
        self._global_tag = global_tag
        self._logdir = log_dir
        self._diagnostics_step = 20 if "imagenet" in self._logdir else 50
        self._class_to_names = class_to_names

        # some parameters initialized later with "initialize()"
        self.name = None
        self._writer = None
        self._is_retraining = None
        self._steps_per_epoch = None
        self._n_idx = None
        self._r_idx = None
        self._s_idx = None
        self._title = None
        self._t_last_print = None

        # templates for print strings
        self._progress_str = None
        self._test_str = None
        self._timing_str = None

        # some internal variables to keep track of test statistics
        self.train_epoch = []
        self.train_acc1 = []
        self.train_acc5 = []
        self.train_loss = []

        # some internal variables to keep track of test statistics
        self.test_epoch = []
        self.test_acc1 = []
        self.test_acc5 = []
        self.test_loss = []

    def _get_writer(self):
        """Get writer and initialize if possible."""
        if (
            self._writer is None
            and self._logdir is not None
            and self._global_tag is not None
            and self.name is not None
        ):
            self._writer = tb.SummaryWriter(
                os.path.join(self._logdir, self.name)
            )

        return self._writer

    def initialize(
        self,
        net_class_name,
        is_retraining,
        num_epochs,
        steps_per_epoch,
        n_idx=None,
        r_idx=None,
        s_idx=None,
    ):
        """Initialize the logger for the current (re-)training session."""
        # reset the writer
        self._writer = None

        # setup parameters
        if self._class_to_names is None:
            self.name = net_class_name
        else:
            self.name = self._class_to_names[net_class_name]

        self._steps_per_epoch = steps_per_epoch
        self._is_retraining = is_retraining
        self._n_idx = n_idx
        self._r_idx = r_idx
        self._s_idx = s_idx
        self._title = "Retraining" if self._is_retraining else "Training"
        self._t_last_print = time.time()

        # start arrays for training stats
        self.train_epoch = []
        self.train_acc1 = []
        self.train_acc5 = []
        self.train_loss = []

        # start arrays for test accuracies, etc...
        self.test_epoch = []
        self.test_acc1 = []
        self.test_acc5 = []
        self.test_loss = []

        # a few display lengths ....
        format_epoch = (
            "{:>" + str(int(math.ceil(math.log10(max(1, num_epochs))))) + "}"
        )
        format_steps = (
            "{:>"
            + str(int(math.ceil(math.log10(self._steps_per_epoch))))
            + "}"
        )

        # construct template for print string
        progress_metrics = [
            "Epoch [{}/{}]".format(format_epoch, num_epochs),
            "Step [{}/{}]".format(format_steps, self._steps_per_epoch),
            "Loss: {:2.6f}",
            "Top 1: {:2.2f}%",
            "Top 5: {:2.2f}%",
            "Elapsed Time: {:3.2f}s",
        ]
        self._progress_str = "{} Progress: ".format(self._title)
        self._progress_str += " | ".join(progress_metrics)

        # construct template for print string for testing
        test_metrics = [
            "Epoch [{}/{}]".format(format_epoch, num_epochs),
            "Loss: {:2.6f}",
            "Top 1: {:2.2f}%",
            "Top 5: {:2.2f}%",
        ]
        self._test_str = "{} Test Metrics: ".format(self._title)
        self._test_str += " | ".join(test_metrics)

        # construct template for timing string
        timing_metrics = [
            "Total: {:3.2f}s",
            "Loading: {:3.2f}s",
            "Optim: {:3.2f}s",
            "Enforce: {:3.2f}s",
            "Log: {:3.2f}s",
            "Remaining: {:3.2f}s",
        ]
        self._timing_str = "Timing: " + " | ".join(timing_metrics)

    def train_diagnostics(
        self, step, epoch, loss, targets, outputs, acc_handle,
    ):
        """Store and print main stats from (re-)training."""
        # return if it's not a diagnostics step
        if step % self._diagnostics_step:
            return

        # get the tensorboard writer
        writer = self._get_writer()

        # compute the accuracy from the acc handle
        acc1, acc5 = acc_handle(outputs, targets, topk_values=(1, 5))

        # current time
        t_elapsed = time.time() - self._t_last_print
        self._t_last_print = time.time()

        # print progress
        print(
            self._progress_str.format(
                epoch + 1, step, loss, acc1 * 100.0, acc5 * 100.0, t_elapsed
            )
        )

        # compute x axis in terms of epochs
        epoch_step = epoch * self._steps_per_epoch + step

        # store statistics
        self.train_epoch.append(epoch_step / self._steps_per_epoch)
        self.train_acc1.append(acc1)
        self.train_acc5.append(acc5)
        self.train_loss.append(float(loss))

        # logging for tensorboard
        if writer is not None:
            log_scalar(
                writer,
                self._global_tag,
                "{}Loss".format(self._title),
                loss,
                epoch_step,
                self._n_idx,
                self._r_idx,
                self._s_idx,
            )

            log_scalar(
                writer,
                self._global_tag,
                "{}Acc1".format(self._title),
                acc1,
                epoch_step,
                self._n_idx,
                self._r_idx,
                self._s_idx,
            )

            log_scalar(
                writer,
                self._global_tag,
                "{}Acc5".format(self._title),
                acc5,
                epoch_step,
                self._n_idx,
                self._r_idx,
                self._s_idx,
            )

    def test_diagnostics(self, epoch, loss, acc1, acc5):
        """Finish test statistics computations and store them."""
        # store statistics
        self.test_epoch.append(epoch)
        self.test_acc1.append(loss)
        self.test_acc5.append(acc1)
        self.test_loss.append(acc5)

        # get the writer
        writer = self._get_writer()

        # log the results
        if writer is not None:
            log_scalar(
                writer,
                self._global_tag,
                "{}TestLoss".format(self._title),
                self.test_loss[-1],
                self.test_epoch[-1],
                self._n_idx,
                self._r_idx,
                self._s_idx,
            )

            log_scalar(
                writer,
                self._global_tag,
                "{}TestAcc1".format(self._title),
                self.test_acc1[-1],
                self.test_epoch[-1],
                self._n_idx,
                self._r_idx,
                self._s_idx,
            )

            log_scalar(
                writer,
                self._global_tag,
                "{}TestAcc5".format(self._title),
                self.test_acc5[-1],
                self.test_epoch[-1],
                self._n_idx,
                self._r_idx,
                self._s_idx,
            )

        # print progress
        print(
            self._test_str.format(
                self.test_epoch[-1] + 1,
                self.test_loss[-1],
                self.test_acc1[-1] * 100.0,
                self.test_acc5[-1] * 100.0,
            )
        )

    def epoch_diagnostics(self, t_total, t_loading, t_optim, t_enforce, t_log):
        """Print diagnostics around the timing of one epoch."""
        t_remaining = t_total - sum([t_loading, t_optim, t_enforce, t_log])
        print(
            self._timing_str.format(
                t_total, t_loading, t_optim, t_enforce, t_log, t_remaining
            )
        )
