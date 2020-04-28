"""A module for all helper functions pertaining to files and parameters."""
import subprocess
import copy
import os
import pathlib
import datetime
import re
import torch

import yaml
import numpy as np


# Frees memory on the GPU by killing all python processes
def free_gpu_memory():
    """Kill all python process on the GPU."""
    cmd = "nvidia-smi | awk '$5~\"python\" {print $3}' | xargs kill -9"
    try:
        subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        print("No python processes were running on the GPU")


def create_directory(path):
    """Create directory if it doesn't exist yet."""
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def write_parameters(param, dir, file="parameters.yaml"):
    """Write parameters to desired directory and file."""
    # don't dump generated parameters (pop them from a copy)
    param_original = copy.deepcopy(param)
    if "generated" in param_original:
        param_original.pop("generated")

    # don't store "ReferenceNet" as method
    param_original["experiments"]["methods"].remove("ReferenceNet")

    with open(os.path.join(dir, file), "w+") as ymlfile:
        yaml.dump(param_original, ymlfile)

    # also store a copy of some generated parameters ...
    param_generated = {}
    param_generated["numWorkers"] = param["generated"]["numWorkers"]
    param_generated["idWorker"] = param["generated"]["idWorker"]
    param_generated["responsibility"] = param["generated"][
        "responsibility"
    ].tolist()

    file_generated, file_ending = os.path.splitext(file)
    file_generated = f"{file_generated}_generated{file_ending}"

    with open(os.path.join(dir, file_generated), "w+") as ymlfile:
        yaml.dump(param_generated, ymlfile)


def set_mosek_path():
    """Set the MOSEK licence file path."""
    os.environ["MOSEKLM_LICENSE_FILE"] = os.path.realpath("../misc/mosek.lic")


def get_parameters(file, num_workers, id_worker):
    """Load parameters and generate additional parameters.

    Args:
        file (str): name of parameter file
        num_workers (int): total number of workers
        id_worker (int): id of this worker

    Returns:
        list: list of parameter dictionary with experiments to run.

    """
    # retrieve array of parameter files
    param_array, file_actual = _generate_file_param(file)

    # those should always be the same (doesn't make sense otherwise ...)
    num_nets = param_array[0]["experiments"]["numNets"]
    num_rep = param_array[0]["experiments"]["numRepetitions"]
    methods = param_array[0]["experiments"]["methods"]
    is_training = param_array[0]["experiments"]["mode"] == "train"

    num_custom = len(param_array)
    num_methods = len(methods)

    # check for responsibilities.
    if is_training:
        responsibilities = _check_responsibilities(
            num_custom, num_nets, 1, 1, num_workers, id_worker
        )
    else:
        responsibilities = _check_responsibilities(
            num_custom, num_nets, num_rep, num_methods, num_workers, id_worker
        )

    # check the parameters for soundness
    for param, responsibility in zip(param_array, responsibilities):
        # some asserts on the validity of the mode
        mode = param["experiments"]["mode"]
        assert (
            mode == "cascade"
            or mode == "retrain"
            or mode == "train"
            or mode == "cascade-random"
            or mode == "cascade-rewind"
        )

        # also check that these parameters always align...
        assert (
            num_nets == param["experiments"]["numNets"]
            and num_rep == param["experiments"]["numRepetitions"]
            and methods == param["experiments"]["methods"]
            and is_training == (mode == "train")
        )

    # generator-style function so that we generate remaining parameters on the
    # fly dynamically
    param_resp = zip(param_array, responsibilities)
    for id_custom, (param, responsibility) in enumerate(param_resp):
        # generate remaining set of parameters
        param["generated"] = _generate_remaining_param(
            param,
            responsibility,
            file_actual,
            num_custom,
            id_custom,
            num_workers,
            id_worker,
        )
        yield param


def _generate_file_param(file, preload_fixed=True):
    """Generate the parameter dictionary from the yaml file.

    We will fill up the fixed parameters with the fixed parameter files, but if
    the actual param file overwrites some of them, these will be preferred.

    Args:
        file (str): relative path under root/param to param file
        preload_fixed (bool, optional): load fixed param too. Defaults to True.

    Returns:
        dict: all parameters to load

    """

    def load_param_from_file(file, preload_fixed=True):
        """Load the parameters from file w/ fixed parameters if desired."""
        # parameter dictionary that will be built up
        param = {}

        # helper function to get parameters out of yaml files
        def update_param(file_name):
            with open(_get_full_param_path(file_name), "r") as ymlfile:
                param.update(yaml.full_load(ymlfile))

        # list of parameters files to load in right order
        if preload_fixed:
            param_files = [
                "fixed/parameters_data.yaml",
                "fixed/parameters_dir.yaml",
                "fixed/parameters_blacklist.yaml",
                "fixed/parameters_names.yaml",
                file,
            ]
        else:
            param_files = [
                file,
            ]

        # actually load the parameters (order is important here since we use
        # the built-in update function of dicts, which will overwrite existing
        # values.
        for param_file in param_files:
            update_param(param_file)

        # also check for training file in parameters
        if "training" in param and "file" in param["training"]:
            update_param(param["training"]["file"])

        return param

    # now build up the param array ...
    param_array = []

    # quickly load params from the file
    param_original = load_param_from_file(file, False)

    # this is a param file that contains multiple parameters
    if "file" in param_original and "customizations" in param_original:
        # get the actual vanilla parameters
        param_vanilla = load_param_from_file(
            param_original["file"], preload_fixed
        )
        # generate a list of customized parameters
        for customization in param_original["customizations"]:
            # get a copy of the vanilla parameters
            param_custom = copy.deepcopy(param_vanilla)

            # recurse into subdictionaries and modify desired key
            subdict = param_custom
            for key in customization["key"][:-1]:
                subdict = subdict[key]
            subdict[customization["key"][-1]] = customization["value"]

            # now append it to the array
            param_array.append(param_custom)

        # also modify actual file
        file_actual = param_original["file"]
    else:
        # simply put the one element in an array
        param_array.append(load_param_from_file(file, preload_fixed))
        file_actual = file

    # make sure we add "ReferenceNet" method everywhere
    for param in param_array:
        if "experiments" in param:
            param["experiments"]["methods"].insert(0, "ReferenceNet")

    return param_array, file_actual


def _generate_remaining_param(
    param, responsibility, file, num_custom, id_custom, num_workers, id_worker
):
    """Auto-generate the remaining parameters that are required.

    Args:
        param (dict): parameters loaded from file
        responsibility (np.array): responsibility mask for this worker
        file (str): name of file
        num_custom (int): number of customizations
        id_custom (int): ID of this customization in range(num_custom)
        num_workers (int): total number of workers to split
        id_worker (int): ID of this worker in range(num_workers)

    Returns:
        dict: generated parameters

    """
    generated = {}

    # assign responsibility right away
    generated["responsibility"] = responsibility

    # also assign number of customizations right away
    generated["numCustom"] = num_custom
    generated["idCustom"] = id_custom

    # different spacing options to determine keep ratios
    all_ratios = []
    for spacing_config in param["experiments"]["spacing"]:
        all_ratios.extend(_get_keep_ratios(spacing_config))

    # split and store
    generated["keepRatios"], generated["compressRatios"] = _split_keep_ratios(
        all_ratios,
        param["experiments"]["retrainIterations"],
        param["experiments"]["mode"],
    )

    # generate net name and store
    generated["netName"] = "_".join(
        [
            param["network"]["name"],
            param["network"]["dataset"],
            f"e{param['training']['numEpochs']}",
        ]
    )

    # also store the number of workers and the id
    generated["numWorkers"] = num_workers
    generated["idWorker"] = id_worker

    # check for test dataset
    if "datasetTest" in param["experiments"]:
        generated["datasetTest"] = param["experiments"]["datasetTest"]
    else:
        generated["datasetTest"] = param["network"]["dataset"]

    # make sure test set is derived from training data set, otherwise this is
    # not allowed
    assert param["network"]["dataset"] in generated["datasetTest"] or (
        param["network"]["dataset"] == "ImageNet"
        and generated["datasetTest"] == "ObjectNet"
    )

    # generate a markdown version of the param file
    generated["paramMd"] = _generate_param_markdown(param)

    # generate list of names and colors for these particular set of algorithms
    generated["network_names"] = [
        param["network_names"][key] for key in param["experiments"]["methods"]
    ]
    generated["network_colors"] = [
        param["network_colors"][key] for key in param["experiments"]["methods"]
    ]

    # generate training and retraining parameters from provided ones
    generated["training"] = copy.deepcopy(param["training"])
    generated["training"]["momentumDelta"] = 0.0
    generated["training"]["startEpoch"] = 0
    generated["training"]["loss"] = generated["training"]["loss"].replace(
        "nn.", ""
    )

    generated["retraining"] = copy.deepcopy(generated["training"])
    generated["retraining"].update(param["retraining"])
    generated["retraining"]["sensTracking"] = False

    # for rewind we might have a different start epoch in the retraining
    if "rewind" in param["experiments"]["mode"]:
        # adjust start based on training epochs and desired retrain epochs
        generated["retraining"]["startEpoch"] = (
            generated["training"]["numEpochs"]
            - generated["retraining"]["numEpochs"]
        )

        # total number of epochs is now the same as in training ...
        generated["retraining"]["numEpochs"] = generated["training"][
            "numEpochs"
        ]

        # also store that in train parameters so we can store rewind epoch
        generated["training"]["retrainStartEpoch"] = generated["retraining"][
            "startEpoch"
        ]
    else:
        generated["retraining"]["startEpoch"] = 0
        generated["training"]["retrainStartEpoch"] = -1  # invalid in this case

    # check for number of GPUs and apply linear scaling rule to training and
    # retraining parameters
    generated["numAvailableGPUs"] = torch.cuda.device_count()
    scaling_factor = (
        generated["numAvailableGPUs"] / param["training"]["numGPUs"]
    )

    # in case we want to do GPU-wise scaling of learning rate at some point...
    if False:
        scaling_factor = max(1.0, scaling_factor)  # don't scale down
    else:
        scaling_factor = 1.0

    generated["training"]["learningRate"] *= scaling_factor
    generated["training"]["batchSize"] = int(
        generated["training"]["batchSize"] * scaling_factor
    )
    generated["training"]["momentumDelta"] *= scaling_factor
    generated["retraining"]["learningRate"] *= scaling_factor
    generated["retraining"]["batchSize"] = int(
        generated["retraining"]["batchSize"] * scaling_factor
    )
    generated["retraining"]["momentumDelta"] *= scaling_factor

    # get the results parent directory
    parent_dir = os.path.join(
        param["directories"]["results"],
        param["network"]["dataset"].lower(),
        param["network"]["name"],
    )
    parent_dir = os.path.realpath(parent_dir)

    # generate the list of folders that have some parameter settings
    results_dirs_prev = _find_latest_results(param, generated, parent_dir)
    generated["resultsDirsPrev"] = results_dirs_prev

    # finally check for ability to repurpose directoreis
    repurpose_dir = len(results_dirs_prev) == 1 and results_dirs_prev[0][1]
    generated["repurposeDir"] = repurpose_dir
    if repurpose_dir:
        old_dir, _, _ = results_dirs_prev[0]
        parent_dir, time_tag = os.path.split(old_dir)
    else:
        time_tag = datetime.datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S_%f")

    # now set the directories.
    results_dir = os.path.join(parent_dir, time_tag)

    # store them
    generated["parentDir"] = parent_dir
    generated["timeTag"] = time_tag
    generated["resultsDir"] = results_dir

    # also define sub directories
    generated["logDir"] = os.path.join(generated["resultsDir"], "log")
    generated["dataDir"] = os.path.join(generated["resultsDir"], "data")
    generated["reportsDir"] = os.path.join(generated["resultsDir"], "reports")
    generated["plotDir"] = os.path.join(
        generated["resultsDir"], "plots", generated["datasetTest"]
    )

    # generate global tag for tensorboard
    generated["globalTag"] = "_".join(
        [
            generated["netName"],
            f"re{generated['retraining']['numEpochs']}",
            param["experiments"]["mode"],
            f"int{len(generated['keepRatios'])}",
        ]
    )

    # for rewind we want separate training directory, so we train for sure
    if "rewind" in param["experiments"]["mode"]:
        generated["training"]["dir"] = os.path.join(
            results_dir, "trained_networks"
        )
    else:
        generated["training"]["dir"] = os.path.realpath(
            param["directories"]["trained_networks"]
        )
    generated["retraining"]["dir"] = os.path.join(
        results_dir, "retrained_networks"
    )

    # return the generated parameters
    return generated


def _generate_param_markdown(param):
    """Generate a markdown compatible string from the parameters."""
    text = yaml.dump(param)
    text = re.sub("#.*?\n", "\n", text)
    text = text.replace("\n", "  \n")
    return text


def _get_keep_ratios(spacing_config):
    """Get the keep ratios for a given spacing configuration."""
    if spacing_config["type"] == "harmonic":
        # numIntervals is the number of intervals to go down to 0.5
        pow = np.log(2) / np.log(1 + spacing_config["numIntervals"])
        num_iters = max(1, int(spacing_config["minVal"] ** (-1.0 / pow)))
        keep_ratios = [(i + 1) ** -pow for i in range(1, num_iters + 1)]
    elif spacing_config["type"] == "harmonic2":
        # numIntervals is the number of intervals to go from max to min
        # value as everywhere
        r_min = spacing_config["minVal"]
        r_max = spacing_config["maxVal"]
        num_int = spacing_config["numIntervals"]
        pow = np.log(r_max / r_min) / np.log(num_int)
        keep_ratios = r_max * np.arange(1, num_int + 1) ** (-pow)
        keep_ratios = keep_ratios.tolist()
    elif spacing_config["type"] == "geometric":
        keep_ratios = np.geomspace(
            spacing_config["maxVal"],
            spacing_config["minVal"],
            spacing_config["numIntervals"],
        ).tolist()
    elif spacing_config["type"] == "cubic":
        r_min = spacing_config["minVal"]
        r_max = spacing_config["maxVal"]
        num_int = spacing_config["numIntervals"]
        keep_ratios = (
            r_min
            + (r_max - r_min) * (1 - np.arange(num_int) / (num_int - 1)) ** 3
        )
        keep_ratios = keep_ratios.tolist()
    elif spacing_config["type"] == "linear":
        keep_ratios = np.linspace(
            spacing_config["maxVal"],
            spacing_config["minVal"],
            spacing_config["numIntervals"],
        ).tolist()
    elif spacing_config["type"] == "manual":
        keep_ratios = spacing_config["values"]

    else:
        raise ValueError(
            "This spacing configuration is not implemented: "
            "{}".format(spacing_config["type"])
        )

    return keep_ratios


def _split_keep_ratios(all_ratios, iterations_retrain, mode):
    """Split ratios into pure compression and retraining ratios.

    Args:
        all_ratios (list): all ratios that we consider
        iterations_retrain (int): number of retrain iterations from there.
                                 -1 == using all for retraining.
        mode (str): type of experiment

    Returns:
        tuple: list with retraining ratios and list with compress ratios

    """
    if iterations_retrain < 1:
        iterations_retrain = len(all_ratios)

    splitted_ratios = np.array_split(np.array(all_ratios), iterations_retrain)

    retrain_ratios = []
    compress_ratios = {}

    for i, partial_ratios in enumerate(splitted_ratios):

        # last one in split becomes a keep ratio with compression and
        # retraining
        retrain_ratios.append(partial_ratios[-1])

        # the others become pure compression ratios
        # Note that if we are not in cascade mode we need to start from
        # scratch every time as so we need to gradually build up compress
        # ratios every time
        partial_ratios = partial_ratios[:-1].tolist()
        if "cascade" not in mode and i > 0:
            previous_ratios = compress_ratios[retrain_ratios[-2]]
            partial_ratios[0:0] = previous_ratios
        compress_ratios[retrain_ratios[-1]] = partial_ratios

    return retrain_ratios, compress_ratios


def _check_responsibilities(
    num_custom, num_nets, num_rep, num_methods, num_workers, id_worker
):
    """Split responsibilities based on number of workers and current id.

    Args:
        num_custom (int): number of customizations (i.e. different parameters)
        num_nets (int): number of networks to repeat experiment
        num_rep (int): number of repetitions per network
        num_methods (int): number of compression methods to test
        num_workers (int): total number of workers
        id_worker (int): ID of this worker in range(num_workers)

    Returns:
        np.array: boolean mask with the responsibilities

    """
    # compute total workload now
    total_workload = num_custom * num_nets * num_rep * num_methods

    # create 1d indices this worker is responsible for
    worker_responsibilities = [
        total_idx % num_workers == id_worker
        for total_idx in range(total_workload)
    ]

    # convert to 4d numpy array so we can conveniently index into it
    # doing it this ensures that if numNets == num_workers, each worker gets
    # one net (and thus we don't do training multiple times ...)
    # same for numCustom ...
    worker_responsibilities = np.array(worker_responsibilities)
    worker_responsibilities = np.reshape(
        worker_responsibilities, (num_methods, num_rep, num_nets, num_custom)
    )
    worker_responsibilities = worker_responsibilities.transpose()

    return worker_responsibilities


def _find_latest_results(param, generated, parent_dir):
    """Find the latest directory containing the same parameter configuration.

    Args:
        param (dict): dict of parameters that are loaded from the file
        generated (dict): generated parameters from param
        parent_dir (str): directory where to look for other results

    Returns:
        list: all sub-directories with compatible results

    """
    # take the current results directory without the time tag
    parent_dir = pathlib.Path(parent_dir)

    # ignore those fields
    blacklist = param["blacklist"]

    # recursively compares the values of (possibly nested) dictionaries with
    # respect to the keys that are *not* in the blacklist
    def equal_params(this_params, other_params):
        """Check if parameters match (except for blacklist)."""
        # check for empty/nonexisting dictionary
        if not other_params or other_params is None:
            return False

        # loop through key, value pairs and check
        for key in this_params:
            if key in blacklist:
                continue

            if key not in other_params:
                return False

            if isinstance(this_params[key], dict):
                if not equal_params(this_params[key], other_params[key]):
                    return False
            elif this_params[key] != other_params[key]:
                return False

        return True

    def compatible_responsibility(this_generated, other_generated_file):
        # now go through and check both compatibility and deletability
        compatible = None
        deleteable = None

        # get the responsibility mask
        this_res = this_generated["responsibility"]

        # also get some parameters from the other file
        if os.path.isfile(other_generated_file):
            # get generated parameters
            other_generated = _generate_file_param(
                other_generated_file, False
            )[0][0]
            # retrieve responsibilities ...
            other_res = np.array(other_generated["responsibility"])
            # check for num workers of other
            other_num_workers = other_generated["numWorkers"]
            other_id_worker = other_generated["idWorker"]
        else:
            # standard behavior for backwards compatibility
            other_res = np.zeros_like(this_res, dtype=np.bool)
            other_num_workers = -1
            other_id_worker = -1

        if this_generated["numWorkers"] == 1:
            # all good since there is only one worker in this case working on
            # the last customization
            compatible = True
            deleteable = True
        elif os.path.isfile(other_generated_file):
            if other_num_workers == 1:
                # in this case we can use the results, but not delete them
                compatible = True
                deleteable = False
            elif (
                np.array_equal(this_res, other_res)
                and other_num_workers == this_generated["numWorkers"]
                and other_id_worker == this_generated["idWorker"]
            ):
                # in this case the folder is a previous version of the current
                # worker
                compatible = True
                deleteable = True
            else:
                # in this case the folder is a previous version of a different
                # worker
                compatible = False
                deleteable = False
        else:
            # undetermined in this case
            # this is just for legacy support ...
            compatible = True
            deleteable = False

        # also check for overlapping responsibility now
        if not other_res.shape == this_res.shape:
            other_res = np.zeros_like(this_res, dtype=np.bool)

        return compatible, deleteable, other_res

    # check for matching params in all subdirectories (except latest)
    matches = []
    for path in sorted(parent_dir.glob("*[!latest]")):
        # retrieve other parameters and check for matches
        other_file = os.path.join(path, "parameters.yaml")
        other_generated_file = os.path.join(path, "parameters_generated.yaml")

        try:
            has_equal_params = equal_params(
                param, _generate_file_param(other_file)[0][0]
            )
        except (FileNotFoundError, TypeError):
            has_equal_params = False

        if not has_equal_params:
            continue

        # check compatible from responsibilities as well
        comp, delable, other_resp = compatible_responsibility(
            generated, other_generated_file
        )
        # only add to matches if compatible ...
        if comp:
            matches.append((path, delable, other_resp))

    return matches


def _get_full_param_path(file):
    """Get full path to parameter file."""
    # this would be the standard file location (from the package)
    file_std = os.path.abspath(os.path.join(__file__, "../../param", file))

    # this file would be the relative location
    file_custom = os.path.abspath(file)

    if os.path.isfile(file_custom):
        return file_custom
    if os.path.isfile(file_std):
        return file_std

    raise FileNotFoundError("This parameter file does not exist")
