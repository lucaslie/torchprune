"""Quick utitlity to run to download and store the ImageNet weights."""

from provable_pruning.util.train import NetTrainer, save_checkpoint
from .file import get_parameters
from .gen import NetGen

ALL_PARAM_FILE = [
    "imagenet/resnet18.yaml",
    "imagenet/resnet50.yaml",
    "imagenet/resnet101.yaml",
    "imagenet/vgg11.yaml",
    "imagenet/vgg16.yaml",
    "imagenet/wrn50_2.yaml",
]


def store_weights(param_file):
    """Store the weights for the network specified in the param file."""
    param = next(get_parameters(param_file, 1, 0))

    net_handle = NetGen(
        output_size=param["network"]["outputSize"],
        dataset=param["network"]["dataset"],
        net_name=param["generated"]["netName"],
        arch=param["network"]["name"],
    ).get_network(pretrained=True)

    trainer = NetTrainer(
        param["generated"]["training"],
        param["generated"]["retraining"],
        None,
        None,
        None,
    )

    file_name = trainer._get_net_name(
        net_handle, 0, False, None, None, None, False
    )
    print(file_name)
    save_checkpoint(
        file_name, net_handle, param["generated"]["training"]["numEpochs"]
    )


def main():
    """Call this function as main."""
    # go through the list of parameters we should download the networks for
    for param_file in ALL_PARAM_FILE:
        store_weights(param_file)


if __name__ == "__main__":
    main()
