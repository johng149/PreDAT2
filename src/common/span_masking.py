from torch import Tensor
from typing import Tuple
import torch


def span_masking(original: Tensor, mask: Tensor, span_idx: int) -> Tensor:
    """
    Mask the spans of `original` tensor, where spans are defined as
    any position where `mask` is True. Each span will be collapsed
    into a single token, which will be represented by `span_idx`.

    The `original` tensor and `mask` tensor must have the same shape
    and must be 1D tensors.

    @param original: The original tensor to mask
    @param mask: The mask tensor, where True values represent spans
    @param span_idx: The index to represent masked spans
    @return: The masked tensor

    For example:
    original = torch.tensor([1, 2, 3, 4, 5])
    mask     = torch.tensor([0, 1, 1, 0, 1]).bool()
    span_idx = -1
    masked = span_masking(original, mask, span_idx)

    The `masked` tensor will be:
    torch.tensor([1, -1, 4, -1])
    """
    original[mask] = span_idx
    prev = None
    elements = [prev := x for x in original if x != prev or x != span_idx]
    return torch.stack(elements)


def flow(
    original: Tensor, mask: Tensor, enc_span_idx: int, targ_span_idx: int
) -> Tuple[Tensor, Tensor]:
    """
    Given an `original` tensor and a `mask` tensor, this function
    will return two tensors: one tensor with the encoder spans
    masked and another tensor with the target spans masked.

    It is assumed that the given mask will be a 1D boolean tensor with the
    same shape as the `original` tensor. Also, the mask tensor should be
    True where the spans in the encoder should be masked and False otherwise.

    The target span tensor will be the inverse of the encoder span tensor.
    Any elements that are not masked in the encoder will be masked in the
    target tensor and vice versa.

    Also consecutive masked elements will be collapsed into a single token
    represented by `enc_span_idx` and `targ_span_idx` respectively.

    @param original: The original tensor to mask
    @param mask: The mask tensor, where True values represent spans
    @param enc_span_idx: The index to represent encoder masked spans
    @param targ_span_idx: The index to represent target masked spans
    @return: A tuple of two tensors, the encoder and target masked tensors

    For example:
    original = torch.tensor([1, 2, 3, 4, 5])
    mask     = torch.tensor([0, 1, 1, 0, 1]).bool()
    enc_span_idx = -1
    targ_span_idx = -2
    enc, targ = flow(original, mask, enc_span_idx, targ_span_idx)

    The `enc` tensor will be:
    torch.tensor([1, -1, 4, -1])

    The `targ` tensor will be:
    torch.tensor([-2, 2, 3, -2, 5])
    """
    target_mask = ~mask
    enc = span_masking(original.clone(), mask, enc_span_idx)
    targ = span_masking(original.clone(), target_mask, targ_span_idx)
    return enc, targ


import torch
from torch import Tensor
import random


def limit_percentage(mask, max_percentage):
    """
    Ensures that no more than `max_percentage` of the 1-D mask is True

    Args:
        - mask (torch.Tensor): 1-D tensor of booleans.
        - max_percentage (float): The maximum proportion of the mask that should be True.

    Returns:
        - mask (torch.Tensor): The mask with no more than `max_percentage` True values.
    """
    assert mask.dim() == 1, "Mask must be a 1-D tensor."
    assert 0 <= max_percentage <= 1, "Invalid max percentage."

    max_true = int(len(mask) * max_percentage)
    true_indices = torch.nonzero(mask).flatten()

    if len(true_indices) > max_true:
        excess_indices = true_indices[torch.randperm(len(true_indices))][max_true:]
        mask[excess_indices] = False

    return mask


def mask_span(
    tokens,
    max_num_spans: int = 6,
    max_span_fill: float = 0.8,
    min_num_spans: int = 0,
    min_span_fill: float = 0,
    hard_fill=True,
):
    """
    Creates a mask for the tokens, where spans of tokens are masked out. Elements in mask that
    are True indicate that the token should be masked out.

    Args:
        - tokens (torch.Tensor): 1-D tensor of tokens to mask out.
        - max_num_spans (int): The maximum number of spans to mask out.
        - max_span_fill (float): The maximum proportion of tokens to mask out in a span.
        - min_num_spans (int): The minimum number of spans to mask out.
        - min_span_fill (float): The minimum proportion of tokens to mask out in a span.
        - hard_fill (bool): If True, will ensure that no more than `max_span_fill` percent of the
            tokens are masked out. If False, the percentage of masked tokens may be higher.

    Returns:
        - mask (torch.Tensor): A mask of the same shape as `tokens` where True indicates that the
            token should be masked out.
    """
    assert tokens.dim() == 1, "Tokens must be a 1-D tensor."
    assert 0 <= min_span_fill <= max_span_fill <= 1, "Invalid span fill percentages."
    assert min_num_spans >= 0, "min_num_spans must be non-negative."
    assert max_num_spans >= min_num_spans, "max_num_spans must be >= min_num_spans."

    fill_percent = random.uniform(min_span_fill, max_span_fill)
    max_mask_len = int(len(tokens) * fill_percent)
    num_spans = random.randint(min_num_spans, max_num_spans)
    start_indices = torch.randint(0, max(1, len(tokens)), (num_spans,))
    span_lengths = torch.randint(1, max(2, max_mask_len + 1), (num_spans,))
    span_lengths = torch.min(span_lengths, len(tokens) - start_indices)
    indices = torch.arange(len(tokens))
    mask = torch.any(
        (indices >= start_indices[:, None])
        & (indices < start_indices[:, None] + span_lengths[:, None]),
        dim=0,
    )

    if hard_fill:
        mask = limit_percentage(mask, fill_percent)

    return mask
