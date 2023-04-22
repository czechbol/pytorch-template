#! /usr/bin/env python

import argparse
from pathlib import Path
from shutil import copytree, ignore_patterns


def new_project(script_parent: Path, target_dir: Path):
    """Initialize new pytorch project with the template files.

    Args:
        target_dir (Path): Target directory for the new project.
    """
    ignore = [
        ".git",
        "data",
        "saved",
        ".venv",
        "venv",
        "new_project.py",
        "__pycache__",
    ]
    copytree(script_parent, target_dir, ignore=ignore_patterns(*ignore))
    print("New project initialized at", target_dir.absolute().resolve())


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-d",
        "--destination",
        default="configs/mnist.json",
        type=str,
        help="Destination path for the new project",
        required=True,
    )
    parsed_args = args.parse_args()
    script_parent = Path(__file__).resolve().parent

    new_project(script_parent, Path(parsed_args.destination))
