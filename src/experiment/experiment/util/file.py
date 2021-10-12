"""A module for all helper functions pertaining to files and parameters."""
import copy
import os
import pathlib
import datetime
import time
import re
import torch

import yaml
import numpy as np
import matplotlib.colors as mcolors


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


def set_mosek_path():
    """Set the MOSEK licence file path."""
    # TODO: do this differently...
    os.environ["MOSEKLM_LICENSE_FILE"] = os.path.realpath("misc/mosek.lic")


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
    param_array = _generate_file_param(file)

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
            num_custom,
            id_custom,
            num_workers,
            id_worker,
        )
        yield param


def load_param_from_file(file):
    """Load the parameters from file w/ fixed parameters if desired."""
    # parameter dictionary that will be built up
    param = {}

    def update_param_key_from_file(param_to_update):
        """Update param (sub-) dict according to file."""
        if not (
            isinstance(param_to_update, dict) and "file" in param_to_update
        ):
            return

        # delete file key-word from parameters
        file_for_param = param_to_update["file"]
        del param_to_update["file"]

        # update with new parameters
        # do it recursively so that we can also update file param there
        param_current = copy.deepcopy(param_to_update)
        param_to_update.update(load_param_from_file(file_for_param))
        param_to_update.update(param_current)

    # load the parameters.
    with open(_get_full_param_path(file), "r") as ymlfile:
        param.update(yaml.full_load(ymlfile))

    # check if any configurations are specified using other param files
    update_param_key_from_file(param)

    # also recurse on dictionaries then, not on lists though...
    for key in param:
        update_param_key_from_file(param[key])

    return param


def _generate_file_param(file):
    """Generate the parameter dictionary from the yaml file.

    We will fill up the fixed parameters with the fixed parameter files, but if
    the actual param file overwrites some of them, these will be preferred.

    Args:
        file (str): relative path under root/param to param file

    Returns:
        dict: all parameters to load
    """
    # load params from the file (first default, then update with provided)
    param_original = load_param_from_file("default.yaml")
    param_original.update(load_param_from_file(file))

    # add reference method.
    if "experiments" in param_original:
        param_original["experiments"]["methods"].insert(0, "ReferenceNet")

    # now build up the param array with multiple customizations.
    param_array = []

    # this is a param file that contains multiple parameters
    if "customizations" in param_original:
        # pop customization from params
        customizations = param_original.pop("customizations")

        # generate a list of customized parameters
        for custom in customizations:
            # get a copy of the vanilla parameters
            param_custom = copy.deepcopy(param_original)

            # recurse into subdictionaries and modify desired key
            subdict = param_custom
            for key in custom["key"][:-1]:
                subdict = subdict[key]
            subdict[custom["key"][-1]] = custom["value"]

            # now append it to the array
            param_array.append(param_custom)
    else:
        # simply put the one element in an array
        param_array.append(param_original)

    return param_array


def _generate_remaining_param(
    param, responsibility, num_custom, id_custom, num_workers, id_worker
):
    """Auto-generate the remaining parameters that are required.

    Args:
        param (dict): parameters loaded from file
        responsibility (np.array): responsibility mask for this worker
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

    # generate a markdown version of the param file
    generated["paramMd"] = _generate_param_markdown(param)

    # generate list of names and colors for these particular set of algorithms
    generated["network_names"] = [
        param["network_names"][key] if key in param["network_names"] else key
        for key in param["experiments"]["methods"]
    ]
    mcolor_list = list(mcolors.CSS4_COLORS.keys())
    generated["network_colors"] = [
        param["network_colors"][key]
        if key in param["network_colors"]
        else mcolor_list[hash(key) % len(mcolor_list)]
        for key in param["experiments"]["methods"]
    ]

    # generate training and retraining parameters from provided ones
    generated["training"] = copy.deepcopy(param["training"])
    generated["training"]["startEpoch"] = 0
    generated["training"]["outputSize"] = param["network"]["outputSize"]
    if "earlyStopEpoch" not in generated["training"]:
        generated["training"]["earlyStopEpoch"] = generated["training"][
            "numEpochs"
        ]
    if "enableAMP" not in generated["training"]:
        generated["training"]["enableAMP"] = True

    if "testBatchSize" not in generated["training"]:
        generated["training"]["testBatchSize"] = generated["training"][
            "batchSize"
        ]

    generated["retraining"] = copy.deepcopy(generated["training"])
    generated["retraining"].update(param["retraining"])

    # same metrics for training and re-training
    generated["retraining"]["metricsTest"] = generated["training"][
        "metricsTest"
    ]

    # needed for rewind checkpoint
    if "rewind" in param["experiments"]["mode"]:
        generated["training"]["retrainStartEpoch"] = generated["retraining"][
            "startEpoch"
        ]
    else:
        generated["training"]["retrainStartEpoch"] = -1  # invalid in this case

    # check for number of available GPUs
    generated["numAvailableGPUs"] = torch.cuda.device_count()

    # get the results parent directory
    parent_dir = os.path.join(
        param["directories"]["results"],
        param["network"]["dataset"].lower(),
        param["network"]["name"],
    )
    parent_dir = os.path.realpath(parent_dir)

    # generate the list of folders that have some parameter settings
    results_dir_prev = _find_latest_results(param, generated, parent_dir)
    generated["resultsDirPrev"] = results_dir_prev

    # check if we re-purpose time tag now.
    if results_dir_prev is None:
        time_tag = datetime.datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S_%f")
    else:
        parent_dir, time_tag = os.path.split(results_dir_prev)

    # now set the directories.
    results_dir = os.path.join(parent_dir, time_tag)

    # store them
    generated["parentDir"] = parent_dir
    generated["timeTag"] = time_tag
    generated["resultsDir"] = results_dir

    # also define sub directories
    generated["stdoutDir"] = os.path.join(generated["resultsDir"], "stdout")
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

    # we can put trained networks inside results folder
    if (
        "rewind" in param["experiments"]["mode"]
        or param["directories"]["trained_networks"] is None
    ):
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
    # doing it this way ensures the following splits for responsibilities:
    # 1.) if num_custom == 1 && num_nets % num_workers == 0:
    #     workers get responsibilities for entire nets (mutually exclusive)
    # 1.) if num_custom != 1 && num_custom % num_workers == 0:
    #     workers get responsibilities for entire customization
    worker_responsibilities = np.array(worker_responsibilities)
    worker_responsibilities = np.reshape(
        worker_responsibilities,
        (num_methods, num_rep, num_nets, num_custom),
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
            # if both dictionaries are empty it's fine
            if other_params != this_params:
                return False

        # loop through key, value pairs and check
        for key in set(list(other_params.keys()) + list(this_params.keys())):
            if key in blacklist:
                continue

            if key not in other_params or key not in this_params:
                return False

            if isinstance(this_params[key], dict):
                if not equal_params(this_params[key], other_params[key]):
                    return False
            elif this_params[key] != other_params[key]:
                return False

        return True

    def get_matches():
        """Get all the matching resulting directories in the parent dir."""
        # check for matching params in all subdirectories (except latest)
        matches = []
        for path in sorted(parent_dir.glob("*[!latest]")):
            # retrieve other parameters and check for matches
            other_file = os.path.join(path, "parameters.yaml")

            try:
                has_equal_params = equal_params(
                    param, _generate_file_param(other_file)[0]
                )
            except (FileNotFoundError, TypeError):
                has_equal_params = False

            if has_equal_params:
                matches.append(path)

        return matches

    # only worker 0 can generate folder. So loop and wait for a while until
    # worker 0 has generated the matching directory.
    backoff_time = 1
    while backoff_time < 513:
        matches = get_matches()

        # at most one match should exist...
        if len(matches) > 1:
            raise ValueError(
                "Exactly one matching directory is expected. "
                "Please clean up results folder."
            )

        # we can return if we find a match
        if len(matches) == 1:
            return matches[0]

        # worker with first responsibility gets to return to generate folder.
        # this is usually either worker 0 or the first worker with
        # responsibilities when we have customizations ...
        if generated["responsibility"].flatten()[0]:
            return

        # sleep and wait a little longer to if we get a match
        print(f"Waiting {backoff_time}s until primary worker generates dir.")
        time.sleep(backoff_time)
        backoff_time *= 2

    # if at the end there is still no match, we should raise an error
    raise ValueError(
        "No matching results directory found; "
        "please initiate worker 0 which will initiate the results directory."
    )


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
