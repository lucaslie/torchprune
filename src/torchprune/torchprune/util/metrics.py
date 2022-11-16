"""Module containing a bunch of test metrics to use."""

from abc import ABC, abstractmethod
from scipy import stats
import torch

from .nn_loss import NLLPriorLoss, NLLNatsLoss, NLLBitsLoss


class AbstractMetric(ABC):
    """Functor template for metric."""

    def __init__(self, output_size):
        """Initialize with output size."""
        super().__init__()
        self.output_size = output_size

    @property
    @abstractmethod
    def name(self):
        """Get the display name of this metric."""

    @property
    @abstractmethod
    def short_name(self):
        """Get the short name of this metric."""

    @torch.no_grad()
    def __call__(self, output, target):
        """Return desired metric."""
        # check if output is dict
        if isinstance(output, dict):
            if "out" in output:
                # Segmentation networks and ffjord networks
                output = output["out"]
            elif "logits" in output:
                # BERT
                output = output["logits"]
        met = self._get_metric(output, target)
        if torch.isfinite(met):
            return met.item()
        return 0.0

    @abstractmethod
    def _get_metric(self, output, target):
        """Compute metric and return as 0d tensor."""


class TopK(AbstractMetric):
    """Functor for top-k accuracy."""

    def __init__(self, output_size, topk):
        """Initialize with desired value of top-k."""
        super().__init__(output_size)
        self.topk = min(topk, self.output_size)

    @property
    def name(self):
        """Get the display name of this metric."""
        return f"Top-{self.topk} Accuracy"

    @property
    def short_name(self):
        """Get the short name of this metric."""
        return f"Top{self.topk}"

    def _get_metric(self, output, target):
        """Return average top-k accuracy over batch."""
        num_classes = self.output_size

        # below we are assuming that output has shape [batches, num_classes]
        # and target has shape [batches, 1]
        # so let's bring it into that shape
        output = output.transpose(0, 1).reshape(output.shape[1], -1).t()
        target = target.squeeze().unsqueeze(1)
        target = target.transpose(0, 1).view(target.shape[1], -1).t()

        # ignore indices outside range of potential classes
        idx_valid = ((target >= 0) & (target < num_classes))[:, 0]
        output = output[idx_valid]
        target = target[idx_valid]

        # pred has shape [batches, topk]
        _, pred = output.topk(self.topk, 1, True, True)

        # check if any of the predictions corresponds to target now
        correct = pred == target

        # compute accuracy from here
        topk_acc = correct.sum() / correct.shape[0]
        return topk_acc


class ConfusionMetric(AbstractMetric):
    """Abstract functor for confusion matrix-based metrics."""

    def _get_metric(self, output, target):
        """Compute confusion matrix and then return child-depdendent metric."""
        num_classes = self.output_size

        pred = output.detach().argmax(dim=1).flatten()
        target = target.to(dtype=pred.dtype).flatten()

        idx_valid = (target >= 0) & (target < num_classes)
        inds = num_classes * target[idx_valid] + pred[idx_valid]
        conf = (
            torch.bincount(inds, minlength=num_classes ** 2)
            .flipud()
            .reshape(num_classes, num_classes)
            .float()
        )

        return self._compute_confusion_metric(conf)

    @abstractmethod
    def _compute_confusion_metric(self, conf):
        """Compute metric based on confusion metric."""


class IoU(ConfusionMetric):
    """Functor for intersection-over-union."""

    @property
    def name(self):
        """Get the display name of this metric."""
        return "IoU"

    @property
    def short_name(self):
        """Get the short name of this metric."""
        return "IoU"

    def _compute_confusion_metric(self, conf):
        """Return mean intersection over union (mIoU) over batch."""
        iou = torch.diag(conf) / (conf.sum(1) + conf.sum(0) - torch.diag(conf))
        return iou[torch.isfinite(iou)].mean()


class MCorr(ConfusionMetric):
    """Matthew's Correlation."""

    @property
    def name(self):
        """Get the display name of this metric."""
        return "Matthew's Correlation"

    @property
    def short_name(self):
        """Get the short name of this metric."""
        return "MCC"

    def _compute_confusion_metric(self, conf):
        """Compute Matthew's correlation factor."""
        # a bunch of stats from confusion matrix
        truly = conf.sum(0)  # number of true occurrences per class
        num_predicted = conf.sum(1)  # number of predictions per class
        correct = conf.diag().sum()  # total number of correct predictions
        total = conf.sum()  # total number of samples

        # compute mcc now
        mcc_num = correct * total - truly @ num_predicted
        mcc_denom = torch.sqrt(
            total * total - num_predicted @ num_predicted
        ) * torch.sqrt(total * total - truly @ truly)
        return mcc_num / mcc_denom


class F1(ConfusionMetric):
    """F1-Score functor."""

    def __init__(self, *args, **kwargs):
        """Initialize like parent but ensure binary classification task."""
        super().__init__(*args, **kwargs)
        assert self.output_size == 2, "Only binary classification have F1"

    @property
    def name(self):
        """Get the display name of this metric."""
        return "F1 Score"

    @property
    def short_name(self):
        """Get the short name of this metric."""
        return "F1"

    def _compute_confusion_metric(self, conf):
        """Compute and return F1-Score."""
        true_pos = conf[0, 0]
        false_pos = conf[0, 1]
        false_neg = conf[1, 0]
        return true_pos / (true_pos + 0.5 * (false_pos + false_neg))


class CorrelationCoefficientMetric(AbstractMetric):
    """Abstract functor interface for correlation coefficients from scipy."""

    @property
    @abstractmethod
    def _scipy_correlation_function(self):
        """Return relevant correlation function."""

    def _get_metric(self, output, target):
        """Compute Spearson metric and return as 0d tensor."""
        # flattened numpy array
        output = output.detach().cpu().flatten().numpy()
        target = target.detach().cpu().flatten().numpy()

        # spearman from scipy stats
        spearman, _ = self._scipy_correlation_function(output, target)

        return torch.tensor(spearman)


class SpearmanRank(CorrelationCoefficientMetric):
    """Spearman Rank Correlation Metric."""

    @property
    def name(self):
        """Get the display name of this metric."""
        return "Spearman's Rank Correlation Coefficient"

    @property
    def short_name(self):
        """Get the short name of this metric."""
        return "Spearman"

    @property
    def _scipy_correlation_function(self):
        """Return relevant correlation function."""
        return stats.spearmanr


class Pearson(CorrelationCoefficientMetric):
    """Pearson Correlation Metric."""

    @property
    def name(self):
        """Get the display name of this metric."""
        return "Pearson Correlation Coefficient"

    @property
    def short_name(self):
        """Get the short name of this metric."""
        return "Pearson"

    @property
    def _scipy_correlation_function(self):
        """Return relevant correlation function."""
        return stats.pearsonr


class Dummy(AbstractMetric):
    """A dummy metric that always returns 0."""

    @property
    def name(self):
        """Get the display name of this metric."""
        return "Dummy Metric"

    @property
    def short_name(self):
        """Get the short name of this metric."""
        return "Dummy"

    def _get_metric(self, output, target):
        """Compute metric and return as 0d tensor."""
        return torch.tensor(0.0)


class NLLPrior(AbstractMetric):
    """A wrapper for the NLLPriorLoss as metric."""

    @property
    def name(self):
        """Get the display name of this metric."""
        return "Negative Log Likelihood"

    @property
    def short_name(self):
        """Get the short name of this metric."""
        return "NLL"

    def _get_metric(self, output, target):
        """Compute metric and return as 0d tensor."""
        # since for the metric higher is better we negate the loss
        return -(NLLPriorLoss()(output["output"], target))

    def __call__(self, output, target):
        """Call metric like output but wrap output so we keep dictionary."""
        return super().__call__({"output": output}, target)


class NLLNats(NLLPrior):
    """A wrapper for the NLLNatsLoss as metric."""

    @property
    def name(self):
        """Get the display name of this metric."""
        return "Negative Log Probability (nats)"

    @property
    def short_name(self):
        """Get the short name of this metric."""
        return "Nats"

    def _get_metric(self, output, target):
        return -(NLLNatsLoss()(output["output"], target))


class NLLBits(NLLPrior):
    """A wrapper for the NLLBitsLoss as metric."""

    @property
    def name(self):
        """Get the display name of this metric."""
        return "Negative Log Probability (bits/dim)"

    @property
    def short_name(self):
        """Get the short name of this metric."""
        return "Bits"

    def _get_metric(self, output, target):
        return -(NLLBitsLoss()(output["output"], target))
