"""Main file/function to start compression experiments."""
import argparse
import os

from . import Logger, Evaluator
from .util.file import get_parameters

PARSER = argparse.ArgumentParser(
    description="BLG Compression",
)

PARSER.add_argument(
    "param_files",
    nargs=1,
    type=str,
    metavar="param_files",
    help="provide a space-separated list of parameter files",
)

PARSER.add_argument(
    "-workers",
    "-j",
    default=1,
    type=int,
    help="Specify number of workers working on this param file",
    dest="num_workers",
)

PARSER.add_argument(
    "--id",
    "-i",
    default=0,
    type=int,
    help="Worker id of this instance",
    dest="id",
)


def main():
    """Initialize and start compression experiment from command line."""
    # parser arguments
    args = PARSER.parse_args()

    # get param file
    file = args.param_files[0]

    # check for environment variables from DRL cluster and overwrite local
    # values here
    env_num_workers = os.environ.get("DRL_GPU_NUM_WORKERS")
    if env_num_workers is not None:
        args.num_workers = int(env_num_workers)

    env_worker_id = os.environ.get("DRL_GPU_WORKER_ID")
    if env_worker_id is not None:
        args.id = int(env_worker_id)

    print("Running experiments now for the folllowing parameter file:")
    print(file)
    print(f"Total # of workers: {args.num_workers}, this worker id: {args.id}")

    # Loop through all experiments
    for param in get_parameters(file, args.num_workers, args.id):
        # initialize a logger
        logger = Logger()
        logger.initialize_from_param(param)
        # run the experiment
        Evaluator(logger).run()


if __name__ == "__main__":
    main()
