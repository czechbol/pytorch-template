import torch.nn.functional as F
import torch


def nll_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative log likelihood loss.

    Args:
        output (torch.Tensor): Model output.
        target (torch.Tensor): Target.

    Returns:
        torch.Tensor: Result.
    """
    return F.nll_loss(output, target)
