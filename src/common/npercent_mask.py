import torch
from torch import Tensor


def batchwise_npercent_mask(original: Tensor, percent: float) -> Tensor:
    """
    Given 1D or 2D tensor, create a mask tensor of the same shape where along each
    batch dimension, `percent` of the elements in the mask tensor are True.
    It is guaranteed that exactly `percent` of the elements in the mask tensor
    along the batch dimension are True.

    Args:
        original (Tensor): The original tensor to mask of shape (batch_size, seq_len)
        percent (float): The percentage of elements to mask

    Returns:
        Tensor: The mask tensor of shape (batch_size, seq_len)
    """
    batch_size, seq_len = original.shape

    num_masked = int(seq_len * percent)

    rand = torch.rand((batch_size, seq_len), device=original.device)
    mask = rand.argsort(dim=-1) < num_masked

    return mask
