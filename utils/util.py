import json
from collections import OrderedDict
from itertools import repeat
from pathlib import Path
from typing import Any, List, Tuple, Type

import pandas as pd
import torch

from base.base_data_loader import BaseDataLoader


def ensure_dir(dirname: str):
    """Ensure the directory exists.

    Args:
        dirname (str): the given directory string path
    """
    dir_path = Path(dirname)
    if not dir_path.is_dir():
        dir_path.mkdir(parents=True, exist_ok=False)


def read_json(file_path: str) -> OrderedDict:
    """Read a json file and returns the resulting dictionary.

    Args:
        file_path (str): Path to the json file

    Returns:
        OrderedDict: The contents of theat file
    """
    with open(file_path, "rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content: Any, file_path: str):
    """Write an object to a json file.

    Args:
        content (Any): The object to be written
        file_path (str): Path to the json file
    """
    with open(file_path, "wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader: Type[BaseDataLoader]) -> Any:
    """Wrapper function for endless data loader.

    Yields:
        Any: The next batch of data
    """
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use: int) -> Tuple[torch.device, List[int]]:
    """Setup GPU device if available.

    Returns:
        Tuple[torch.device, List[int]]: Pytorch device and list of ID's to use for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))

    return device, list_ids


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        """Reset all tracked metrics."""
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
