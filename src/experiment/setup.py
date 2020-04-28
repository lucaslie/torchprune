"""The setup sript for the experiment package."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import setuptools


with open("README.md", "r") as fh:
    LONG_DESCRIPTION = fh.read()

setuptools.setup(
    name="pruning_experiments",
    version="1.0.0",
    author="The Provable Pruning Authors",
    author_email="lucasl@mit.edu",
    description="Experiments for provable pruning for neural networks",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    licence="MIT Licence",
    url="https://github.com/lucaslie/provable_pruning",
    package_data={"": ["*.yaml"]},
    install_requires=[
        "matplotlib",
        "numpy",
        "pyyaml",
        "seaborn",
        "torch",
        "torchvision",
        "tensorboard",
    ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.6",
)
