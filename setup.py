#!/usr/bin/env python
"""A setup module for feature-aggregation."""

from os import path
from setuptools import find_packages, setup


# Define constants that describe the package to PyPI
NAME = "feature-aggregation"
VERSION = "0.1"
DESCRIPTION = "Aggregate local features into global features"
with open("README.rst") as f:
    LONG_DESCRIPTION = f.read()
MAINTAINER = ("Despoina Paschalidou <paschalidoud@gmail.com>, "
              "Angelos Katharopoulos <katharas@gmail.com>")
MAINTAINER_EMAIL = "paschalidoud@gmail.com"
LICENSE = "MIT"

def setup_package():
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        license=LICENSE,
        classifiers=[
            "Intended Audience :: Science/Research",
            "Intended Audience :: Developers",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering",
            "Programming Language :: Python",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 2.7",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.4",
            "Programming Language :: Python :: 3.5",
        ],
        packages=find_packages(exclude=["docs", "tests"]),
        install_requires=["scikit-learn"]
    )

if __name__ == "__main__":
    setup_package()
