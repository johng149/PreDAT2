import torch
from torch import Tensor


def logsumexp_infsafe(x: Tensor, dim: int) -> Tensor:
    # Solving nan issue when x contains -inf
    # See https://github.com/pytorch/pytorch/issues/31829
    # https://github.com/thu-coai/DA-Transformer/blob/main/fs_plugins/custom_ops/dag_loss.py
    m, _ = x.max(dim=dim, keepdim=True)
    mask = m == -float("inf")
    m = m.detach()
    s = (x - m.masked_fill_(mask, 0)).exp_().sum(dim=dim, keepdim=True)
    return s.masked_fill_(mask, 1).log_() + m.masked_fill_(mask, -float("inf"))
