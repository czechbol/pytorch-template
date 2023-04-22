from __future__ import annotations

import logging
import os
from argparse import Namespace
from datetime import datetime
from functools import partial, reduce
from operator import getitem
from pathlib import Path
from typing import Any, Optional

from logger import setup_logging
from utils import read_json, write_json


class ConfigParser:
    """Class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving
    and logging module."""

    def __init__(
        self,
        config: dict,
        resume: Optional[str] = None,
        modification: Optional[dict] = None,
        run_id: Optional[str] = None,
    ):
        """Inits ConfigParser

        Args:
            config (dict): Dict containing configurations, hyperparameters for training. contents of `config.json` file for example.
            resume (Optional[str]): String, path to the checkpoint being loaded.
            modification (Optional[dict]): Dict keychain:value, specifying position values to be replaced from config dict.
            run_id (Optional[str]): Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        """

        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config["trainer"]["save_dir"])

        exper_name = self.config["name"]
        if run_id is None:  # use timestamp as default run-id
            run_id = datetime.now().strftime(r"%m%d_%H%M%S")
        self._save_dir: Path = save_dir / "models" / exper_name / run_id
        self._log_dir: Path = save_dir / "log" / exper_name / run_id

        # make directory for saving checkpoints and log.
        exist_ok = run_id == ""
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # save updated config file to the checkpoint dir
        write_json(self.config, str(self.save_dir / "config.json"))

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

    @classmethod
    def from_args(cls, args: Namespace) -> ConfigParser:
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        if args.device != "":
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.resume != "":
            resume = args.resume
            resume_path = Path(resume)
            cfg_fname = str(resume_path.parent / "config.json")
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config != "", msg_no_cfg
            resume = None
            cfg_fname = args.config

        config = read_json(cfg_fname)
        if args.config and resume:
            # update new config for fine-tuning
            config.update(read_json(args.config))

        # parse custom cli options into dictionary
        modification = {}
        if not hasattr(args, "learning_rate") or args.learning_rate > 0:
            modification["learning_rate"] = args.learning_rate
        if args.batch_size > 0:
            modification["batch_size"] = args.batch_size
        return cls(config, resume, modification)

    def init_obj(self, name: str, module: object, *args, **kwargs) -> Any:
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]["type"]
        module_args = dict(self[name]["args"])
        assert all(
            [k not in module_args for k in kwargs]
        ), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs) -> partial:
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]["type"]
        module_args = dict(self[name]["args"])
        assert all(
            [k not in module_args for k in kwargs]
        ), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name) -> Any:
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2) -> logging.Logger:
        msg_verbosity = "verbosity option {} is invalid. Valid options are {}.".format(
            verbosity, self.log_levels.keys()
        )
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self) -> dict[str, Any]:
        return self._config

    @property
    def save_dir(self) -> Path:
        return self._save_dir

    @property
    def log_dir(self) -> Path:
        return self._log_dir


# helper functions to update config dict with custom cli options
def _update_config(
    config: dict[str, Any], modification: Optional[dict[str, Any]]
) -> dict:
    """Update config dict with custom cli options.

    Args:
        config (dict[str, Any]): The original config
        modification (Optional[dict[str, Any]]): The custom cli options to be set

    Returns:
        dict[str, Any]: The updated config
    """
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config


def _set_by_path(tree: dict, keys: str, value: Any):
    """Set a value in a nested object in tree by sequence of keys.

    Args:
        tree (dict): Dictionary to be updated.
        keys (str): Keys to be updated
        value (Any): Value to be set
    """
    key_list = keys.split(";")
    _get_by_path(tree, key_list[:-1])[keys[-1]] = value


def _get_by_path(tree: dict, keys: list) -> dict:
    """Access a nested object in tree by sequence of keys.

    Args:
        tree (dict): Dictionary to search
        keys (list): Keys to be searched

    Returns:
        dict: The nested object
    """
    return reduce(getitem, keys, tree)
