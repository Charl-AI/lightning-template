#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="project",
    version="0.0.0",
    description="Describe Your Cool Project",
    author="Charles Jones",
    author_email="",
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url="https://github.com/Charl-AI/PROJECT-NAME",
    install_requires=["pytorch-lightning"],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
