from typing import Callable, Optional, Tuple, Type

import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """Base class for all data loaders."""

    def __init__(
        self,
        dataset: Type[Dataset],
        batch_size: int = 1,
        shuffle: bool = False,
        validation_split: float = 0.0,
        num_workers: int = 0,
        collate_fn: Callable = default_collate,
    ):
        """Init BaseDataLoader.

        Args:
            dataset (Type[Dataset]): Dataset from which to load the data.
            batch_size (int, optional): How many samples per batch to load.
                    Defaults to 1.
            shuffle (bool, optional): Set to True to have the data reshuffled at
                    every epoch. Defaults to False.
            validation_split (float, optional): The proportion of how to split the dataset into
                    training and validation sets. Either takes numbers from 0 to 1 (percentage split),
                    or an integer no larger than the dataset length. Defaults to 0.0.
            num_workers (int, optional): How many subprocesses to use for data loading.
                    0 means that the data will be loaded in the main process. Defaults to 0.
            collate_fn (Callable, optional): Merges a list of samples to form a mini-batch of Tensor(s).
                    Used when using batched loading from a map-style dataset. Defaults to default_collate.
        """
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        temp_sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "collate_fn": collate_fn,
            "num_workers": num_workers,
        }
        super().__init__(
            sampler=temp_sampler,
            dataset=dataset,
            batch_size=batch_size,
            shuffle=self.shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )

    def _split_sampler(
        self, split: float
    ) -> Tuple[Optional[SubsetRandomSampler], Optional[SubsetRandomSampler]]:
        """Split the dataset into training and validation sets.

        Args:
            split (float): numbers from 0 to 1 (percentage split) or an integer no larger than the dataset length

        Returns:
            Tuple[Optional[SubsetRandomSampler], Optional[SubsetRandomSampler]]: training and validation samplers
        """
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples, dtype=np.int64)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert (
                split < self.n_samples
            ), "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx.tolist())
        valid_sampler = SubsetRandomSampler(valid_idx.tolist())

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self) -> Optional[DataLoader]:
        """Return the validation set dataloader.

        Returns:
            Optional[DataLoader]: Resulting validation dataloader
        """
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
