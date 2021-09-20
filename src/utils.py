from pathlib import Path, PurePath
import os

# This file assumes a structure of Project-Root/src/utils.py
# If this file is moved, the get_project_root functions
# will need to be updated.


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_project_root_name() -> Path:
    return os.path.basename(str(get_project_root()))
