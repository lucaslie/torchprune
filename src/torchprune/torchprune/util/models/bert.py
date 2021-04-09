"""A module for BERT and BERT-like networks."""
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoConfig


class Bert(nn.Module):
    """The BERT architecture."""

    @property
    def model_name(self):
        """Return the model name for huggingface/transformers."""
        return self._model_name

    def __init__(self, num_classes, model_name):
        """Initialize the network with the desired number of classes."""
        super().__init__()

        self._model_name = model_name
        self._config = AutoConfig.from_pretrained(
            self.model_name, num_labels=num_classes
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, config=self._config
        )

    def forward(self, x_dict):
        """Forward via model itself."""
        return self._model(**x_dict)


def bert(num_classes, **kwargs):
    """Return an initialized instance of Bert."""
    return Bert(num_classes, "bert-base-cased")
