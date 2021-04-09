"""Module wrapper for GLUE data sets from huggingface/transformers."""

from abc import abstractmethod
import os
import transformers
import datasets
import torch.utils.data as data

GLUE_TASKS = {
    "cola": "GlueCoLA",
    "sst2": "GlueSST2",
    "mrpc": "GlueMRPC",
    "stsb": "GlueSTSB",
    "qqp": "GlueQQP",
    "mnli": "GlueMNLI",
    "qnli": "GlueQNLI",
    "rte": "GlueRTE",
    "wnli": "GlueWNLI",
}

TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


class BaseGlue(data.Dataset):
    """A base class for all Glue Benchmark datasets used within transformers.

    The setup was extracted from
    https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py

    using mostly standard arguments and hard-coding them here for simplicity.
    """

    @property
    @abstractmethod
    def task(self):
        """Get the task."""

    @property
    @abstractmethod
    def sentence_keys(self):
        """Get both sentence keys."""

    def __init__(self, root, model_name, train=True):
        """Initialize dataset and download it if necessary."""
        # setup correct mode
        mode = "train" if train else "validation"
        if self.task == "mnli" and mode == "validation":
            mode = "validation_matched"
        self._mode = mode

        # put root inside other directory for clean download
        root = os.path.join(root, "glue")

        # get the tokenizer and other data stats
        self._padding = "max_length"
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=root,
            use_fast=True,
            revision="main",
            use_auth_token=None,
        )
        self._max_seq_length = min(128, self.tokenizer.model_max_length)

        # load data set now
        self._data = datasets.load_dataset("glue", self.task, cache_dir=root)
        self.num_labels = (
            len(self._data["train"].features["label"].names)
            if self.task != "stsb"
            else 1
        )

        # pre-process data set and extract desired data split
        self._data = self._data.map(self._preprocess_function, batched=True)
        self._data = self._data[self._mode]

    def _preprocess_function(self, examples):
        # Tokenize the texts
        skey1, skey2 = self.sentence_keys
        args = (
            (examples[skey1],)
            if skey2 is None
            else (examples[skey1], examples[skey2])
        )
        result = self.tokenizer(
            *args,
            padding=self._padding,
            max_length=self._max_seq_length,
            truncation=True
        )

        return result

    def __getitem__(self, index):
        """Return appropriate item."""
        data_point = self._data[int(index)]
        if "idx" in data_point:
            data_point.pop("idx")
        return data_point, data_point["label"]

    def __len__(self):
        """Get total number of data points."""
        return len(self._data)


def _get_glue_class(name, task, sentence_keys):
    """Generate new GLUE class with proper tag."""
    return type(
        name,
        (BaseGlue,),
        {"task": task, "sentence_keys": sentence_keys},
    )


# loop through all names, generate classes, and add them to global() dict
for task, name in GLUE_TASKS.items():
    globals()[name] = _get_glue_class(name, task, TASK_TO_KEYS[task])
