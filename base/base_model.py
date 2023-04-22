from abc import abstractmethod
from typing import Iterable

import numpy as np
import torch.nn as nn


class BaseModel(nn.Module):
    """Base class for all models."""

    @abstractmethod
    def forward(self, *inputs):
        """Forward pass logic.

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self) -> str:
        """Model prints with number of trainable parameters."""
        model_parameters: Iterable[nn.Parameter] = filter(
            lambda p: p.requires_grad, self.parameters()
        )
        params = sum([np.prod(p.size()).item() for p in model_parameters])
        return super().__str__() + f"\nTrainable parameters: {params}"
