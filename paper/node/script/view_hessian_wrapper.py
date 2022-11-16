"""Wrapper for Hessian since we keep running into """

import subprocess
import argparse


PARSER = argparse.ArgumentParser(
    description="Sparse Flow - Hessian Analysis",
)

PARSER.add_argument(
    "param_file",
    type=str,
    metavar="param_file",
    help="provide a parameter file",
)

# retrieve file
ARGS = PARSER.parse_args()
FILE = ARGS.param_file


def main(file):
    for _ in range(5000):
        ret_code = subprocess.run(
            ["python", "paper/node/script/view_hessian.py", "-p", file]
        ).returncode
        if ret_code:
            print("Catching CUDA-OOM and retrying.")
        else:
            print("Finished successfully without CUDA-OOM failure.")
            break


if __name__ == "__main__":
    main(FILE)
