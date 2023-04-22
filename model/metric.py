import torch


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Calculate the accuracy of the output.

    Args:
        output (torch.Tensor): Model output.
        target (torch.Tensor): Target.

    Returns:
        float: Result.
    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output: torch.Tensor, target: torch.Tensor, k: int = 3) -> float:
    """Top-k Accuracy classification score.

    This metric computes the number of times where the correct label is among
    the top `k` labels predicted (ranked by predicted scores).

    Args:
        output (torch.Tensor): Model output.
        target (torch.Tensor): Target.
        k (int, optional): Number of most likely outcomes considered to find the correct label. Defaults to 3.

    Returns:
        float: Result.
    """
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
