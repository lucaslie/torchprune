"""A logger that can be used during training to log to tensorboard."""
import os
import time
import math
import copy

import numpy as np
from torch.utils import tensorboard as tb

from .stdout import setup_stdout
from .tensorboard import log_scalar


class _StatTracker(object):
    """A class to track stats over multiple training epochs."""

    def __init__(self, num_epochs=0):
        """Initialize with the expected number of epochs."""
        # the statistics
        self._epochs = None
        self._loss = None
        self._acc1 = None
        self._acc5 = None

        # internal tracker for last epoch
        self._epoch = None

        # first epoch for which we consider early stopping
        self._early_stop_epoch = None

        self.reset(num_epochs)

    def reset(self, num_epochs, early_stop_epoch=0):
        """Reset the tracked statistics."""
        self._epochs = np.arange(num_epochs)
        self._loss = np.zeros(num_epochs)
        self._acc1 = np.zeros(num_epochs)
        self._acc5 = np.zeros(num_epochs)
        self._epoch = -1.0
        self._early_stop_epoch = early_stop_epoch

    def update(self, epoch, loss, acc1, acc5):
        """Update the stats with the latest results."""
        # get the index of the epoch
        idx_e = int(epoch)

        # check if we have entered a new epoch
        if idx_e != int(self._epoch):
            self._epoch = idx_e

        # update the results with a running average
        def _update_one_stat(stat, stat_update):
            weight_total = epoch - idx_e
            if weight_total > np.finfo(float).eps:
                alpha = (self._epoch - idx_e) / weight_total
            else:
                alpha = 0.0
            stat[idx_e] *= alpha
            stat[idx_e] += (1.0 - alpha) * stat_update

        _update_one_stat(self._loss, loss)
        _update_one_stat(self._acc1, acc1)
        _update_one_stat(self._acc5, acc5)

        # update epoch tracker
        self._epoch = epoch

    def contains_data(self):
        """Return an indicator whether any stats are stored."""
        return np.any(self._loss)

    def get(self):
        """Return the stats."""
        return (
            copy.deepcopy(self._epochs),
            copy.deepcopy(self._loss),
            copy.deepcopy(self._acc1),
            copy.deepcopy(self._acc5),
            copy.deepcopy(self._epoch),
        )

    def set(self, epochs, loss, acc1, acc5, epoch):
        """Set the stats."""
        self._epochs = copy.deepcopy(epochs)
        self._loss = copy.deepcopy(loss)
        self._acc1 = copy.deepcopy(acc1)
        self._acc5 = copy.deepcopy(acc5)
        self._epoch = copy.deepcopy(epoch)

    def get_best_acc1_epoch(self):
        """Get the epoch corresponding to lowest acc1."""
        e_valid = np.logical_and(
            self._loss != 0.0, self._epochs >= self._early_stop_epoch
        )

        # filter and inverse list
        acc1_candidates = self._acc1[e_valid][::-1]
        e_candidates = self._epochs[e_valid][::-1]

        # now return latest best epoch
        # (np returns first index, hence on inverse list)
        if len(acc1_candidates) > 0:
            return e_candidates[np.argmax(acc1_candidates)]
        return -2


class TrainLogger(object):
    """A logger class that is pickeable (used for training).

    A separate class for logging training. The logger cannot be pickled,
    thus cannot be used as argument for training with multiprocessing
    enabled. This class can be pickled and then initialized within the
    multiprocessing context.
    """

    def __init__(
        self,
        log_dir=None,
        stdout_file=None,
        global_tag=None,
        class_to_names=None,
    ):
        """Initialize the train logger.

        If the optional arguments are not supplied, the logger will print
        updates about the training progress but won't log it to tensorboard or
        log it to a file
        """
        self._global_tag = global_tag
        self._logdir = log_dir
        self._stdout_file = stdout_file
        self._stdout_init = False
        self._diagnostics_step_nominal = 50
        self._diagnostics_step = self._diagnostics_step_nominal
        self._class_to_names = class_to_names

        self._metrics_test = None

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

        # tracker for train and test statistics
        self.tracker_train = _StatTracker()
        self.tracker_test = _StatTracker()

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
        early_stop_epoch,
        metrics_test,
        n_idx=None,
        r_idx=None,
        s_idx=None,
    ):
        """Initialize the logger for the current (re-)training session."""
        # reset the writer and logger
        self._writer = None
        self._stdout_init = False

        # setup parameters
        if (
            self._class_to_names is None
            or net_class_name not in self._class_to_names
        ):
            self.name = net_class_name
        else:
            self.name = self._class_to_names[net_class_name]

        self._steps_per_epoch = steps_per_epoch
        self._diagnostics_step = min(
            self._diagnostics_step_nominal, self._steps_per_epoch // 10
        )  # print at least a 10th of the time
        self._diagnostics_step = max(1, self._diagnostics_step)
        self._is_retraining = is_retraining
        self._n_idx = n_idx
        self._r_idx = r_idx
        self._s_idx = s_idx
        self._title = "Retraining" if self._is_retraining else "Training"
        self._t_last_print = time.time()

        # reset trackers
        self.tracker_train.reset(num_epochs)
        self.tracker_test.reset(num_epochs, early_stop_epoch)

        # store new test metrics (want two exactly)
        self._metrics_test = [metrics_test[0], metrics_test[1]]

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
            self._metrics_test[0].short_name + ": {:2.2f}%",
            self._metrics_test[1].short_name + ": {:2.2f}%",
            "Elapsed Time: {:3.2f}s",
        ]
        self._progress_str = "{} Progress: ".format(self._title)
        self._progress_str += " | ".join(progress_metrics)

        # construct template for print string for testing
        test_metrics = [
            "Epoch [{}/{}]".format(format_epoch, num_epochs),
            "Loss: {:2.6f}",
            self._metrics_test[0].short_name + ": {:2.2f}%",
            self._metrics_test[1].short_name + ": {:2.2f}%",
            "Early-stop: {}",
        ]
        self._test_str = "{} {} Metrics: ".format(self._title, "{}")
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

    def train_diagnostics(self, step, epoch, l_r, loss, targets, outputs):
        """Store and print main stats from (re-)training."""
        # return if it's not a diagnostics step
        if step % self._diagnostics_step:
            return

        # get the tensorboard writer
        writer = self._get_writer()

        # compute the accuracy from the acc handle
        acc1, acc5 = [met(outputs, targets) for met in self._metrics_test]

        # current time
        t_elapsed = time.time() - self._t_last_print
        self._t_last_print = time.time()

        # print progress
        self._print(
            self._progress_str.format(
                epoch + 1, step, loss, acc1 * 100.0, acc5 * 100.0, t_elapsed
            )
        )

        # compute x axis in terms of total steps
        epoch_step = epoch * self._steps_per_epoch + step
        epoch_fraction = epoch_step / self._steps_per_epoch

        # store statistics
        self.tracker_train.update(epoch_fraction, float(loss), acc1, acc5)

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
                self._title + self._metrics_test[0].short_name,
                acc1,
                epoch_step,
                self._n_idx,
                self._r_idx,
                self._s_idx,
            )

            log_scalar(
                writer,
                self._global_tag,
                self._title + self._metrics_test[1].short_name,
                acc5,
                epoch_step,
                self._n_idx,
                self._r_idx,
                self._s_idx,
            )

            log_scalar(
                writer,
                self._global_tag,
                self._title + "LearningRate",
                l_r,
                epoch_step,
                self._n_idx,
                self._r_idx,
                self._s_idx,
            )

    def test_diagnostics(self, epoch, loss, acc1, acc5, tag):
        """Finish test statistics computations and store them."""
        # store statistics
        self.tracker_test.update(epoch, loss, acc1, acc5)

        # get the writer
        writer = self._get_writer()

        # log the results
        if writer is not None:
            log_scalar(
                writer,
                self._global_tag,
                f"{self._title}{tag}Loss",
                float(loss),
                epoch,
                self._n_idx,
                self._r_idx,
                self._s_idx,
            )

            log_scalar(
                writer,
                self._global_tag,
                f"{self._title}{tag}{self._metrics_test[0].short_name}",
                acc1,
                epoch,
                self._n_idx,
                self._r_idx,
                self._s_idx,
            )

            log_scalar(
                writer,
                self._global_tag,
                f"{self._title}{tag}{self._metrics_test[1].short_name}",
                acc5,
                epoch,
                self._n_idx,
                self._r_idx,
                self._s_idx,
            )

        # print progress
        self._print(
            self._test_str.format(
                tag,
                epoch + 1,
                loss,
                acc1 * 100.0,
                acc5 * 100.0,
                self.tracker_test.get_best_acc1_epoch() + 1,
            )
        )

    def epoch_diagnostics(self, t_total, t_loading, t_optim, t_enforce, t_log):
        """Print diagnostics around the timing of one epoch."""
        t_remaining = t_total - sum([t_loading, t_optim, t_enforce, t_log])
        self._print(
            self._timing_str.format(
                t_total, t_loading, t_optim, t_enforce, t_log, t_remaining
            )
        )

    def is_best_epoch(self, epoch):
        """Check if epoch is best epoch according to acc1 from test stats."""
        return epoch == self.tracker_test.get_best_acc1_epoch()

    def _print(self, value):
        """Print and ensure we are also printing to file."""
        if not self._stdout_init and self._stdout_file is not None:
            stdout = setup_stdout(self._stdout_file)
            stdout.write(" " * 200, name=self.name)
            self._stdout_init = True
        print(value)
