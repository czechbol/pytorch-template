from torchvision import datasets, transforms

from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    """MNIST data loading demo using BaseDataLoader."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        shuffle: bool = True,
        validation_split: float = 0.0,
        num_workers: int = 1,
        training: bool = True,
    ):
        """Initialize a MnistDataLoader.

        Args:
            data_dir (str): Data directory.
            batch_size (int): Batch size.
            shuffle (bool, optional): Shuffle. Defaults to True.
            validation_split (float, optional): Validation split. Defaults to 0.0.
            num_workers (int, optional): Number of workers. Defaults to 1.
            training (bool, optional): Training. Defaults to True.
        """
        trsfm = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(
            self.data_dir, train=training, download=True, transform=trsfm
        )
        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )
