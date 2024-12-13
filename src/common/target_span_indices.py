import torch
from torch import Tensor


def targetSpanIndices(targets: Tensor, is_span: Tensor, ratio: int):
    """
    Given a 1-D tensor of targets and a 1-D tensor of booleans that
    indicate if the respective target element is a span token,
    create a new tensor of the same shape as the targets tensor
    where every element that is a span token is replaced by a
    index that is `ratio` times the index of the span token in the
    original tensor otherwise the element is set to a value less
    than zero

    For example, if the targets tensor is [1, 2, 3, 4, 5] and the
    is_span tensor is [False, True, False, False, True] and the
    ratio is 3, then the resulting tensor is [-3, 3, -3, -3, 12]

    @param targets: (batch_size, seq_len) tensor
    @param is_span: (batch_size, seq_len) tensor of bools (usually by doing targets == span_token)
    @param ratio: int, the upsample ratio

    @return: (batch_size, seq_len) tensor of span token indices upsampled
    """
    assert targets.ndim == 1, "targets must be 1-D tensor"
    assert is_span.ndim == 1, "is_span must be 1-D tensor"
    assert (
        targets.shape == is_span.shape
    ), "targets and is_span must have the same shape"

    indices = torch.arange(len(targets))
    indices[~is_span] = -1
    indices = indices * ratio
    return indices
