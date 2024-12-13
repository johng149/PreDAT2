import torch
from torch import Tensor


def correct_dp_slice_for_spans(
    dp_slice: Tensor, target_span_indices: Tensor, iteration: int
) -> Tensor:
    """
    @param dp: (batch_size, num_vertices, 1) tensor of scores that is being consiered
        by the current iteration (position in target sequence)
    @param target_span_indices: (batch_size, seq_len) tensor of indices, if element
        is non-negative, it represents that position in target sequence span otherwise
        it is negative
    @param iteration: int, the current iteration in the target sequence

    @return dp_slice: (batch_size, num_vertices, 1) tensor of scores that guarantees
        that all paths considered go through the vertex specified by the target_span_indices
        if it is non-negative, otherwise scores of paths are not changed to -inf
    """
    res = dp_slice
    res = res.squeeze(2)

    # here we cannot simply use this res to update the dp table,
    # because this res update assumes that all paths are allowed,
    # however, if we are at a span, only the path that passes
    # through the span's specified vertex is allowed
    # to do this, we set the probability of all other vertices
    # to -inf, so that they are not considered in the max operation
    # in the next step. This guarantees in the next step, we only
    # consider paths that pass through the span's specified vertex
    curr_span_status = target_span_indices[:, iteration - 1]
    in_span = curr_span_status >= 0

    # we need to do a masked_fill here for curr_span_status, because
    # we use gather to get the current probability of the span vertex
    # in res, and gather doesn't support negative indices.
    # this also means the corresponding probability might not be useful
    # because if it is not a span then we don't enforce that paths
    # must pass through the span's vertex. We account for this later
    curr_span_status = curr_span_status.masked_fill(~in_span, 0).unsqueeze(1)

    selection = res.gather(1, curr_span_status)

    masked = torch.full_like(res, -float("inf"))
    masked.scatter_(1, curr_span_status, selection)

    in_span = in_span.unsqueeze(-1)
    in_span = in_span.expand_as(res)

    res = torch.where(in_span, masked, res)
    return res
