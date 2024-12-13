import torch
from torch import Tensor
from typing import Tuple
from src.common.dp_correction import correct_dp_slice_for_spans
from src.common.vector_gather import vector_gather


def findPathBatched(
    transition_matrix: Tensor,
    emission_probs: Tensor,
    target_seq: Tensor,
    target_span_indices: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    @param transition_matrix: (b, n, n) tensor, `n` is number of vertices
        the matrix should be in log-space, and describe an acyclic directed graph
    @param emission_probs: (b, n, m) tensor, where `n` is the number of vertices
        and `m` is the vocabulary size. The matrix should be in log-space
    @param target_seq: (b, seq_len) tensor
    @param target_span_indices: (b, seq_len) tensor, the indices of the target span
        for their respective elements in `target_seq`. If the element in `target_seq`
        is a span, then the corresponding element in `target_span_indices` should be
        non-negative, otherwise it should be negative.

    @return: (dp, backtrace) where `dp` is a (b, seq_len, n) tensor and `backtrace`
        is a (b, seq_len - 1, n) tensor. `dp` contains the log-probabilities of the
        most likely path to each vertex at each time step. `backtrace` contains
        the backtrace pointers for each vertex at each time step.

    Find the most likely path through the graph given the target sequence and
    the transition and emission probabilities. The graph is an acyclic directed
    graph, and the transition and emission probabilities are in log-space.

    The path must start at the first vertex and end at the last vertex with
    the number of vertices in the path equal to the length of the target sequence.
    """
    b, n, m = emission_probs.shape
    _, seq_len = target_seq.shape
    dp = torch.zeros(b, seq_len, n, device=transition_matrix.device)
    dp.fill_(-float("inf"))
    initial_probs = torch.gather(
        emission_probs, dim=2, index=target_seq[:, 0].unsqueeze(1).unsqueeze(2)
    )
    dp[:, 0, 0] = initial_probs.squeeze(2).squeeze(1)
    backtrace = torch.zeros(
        b, seq_len - 1, n, dtype=torch.long, device=transition_matrix.device
    )

    emission_probs = emission_probs.transpose(-1, -2)
    for i in range(1, seq_len):
        currDp = dp[:, i - 1, :].unsqueeze(1).transpose(-1, -2)
        currDp = correct_dp_slice_for_spans(currDp, target_span_indices, i)
        dp[:, i - 1, :] = currDp
        currDp = currDp.unsqueeze(2)
        pathProbs = currDp + transition_matrix
        max_v, max_idx = torch.max(pathProbs, dim=1)
        res = max_v + vector_gather(emission_probs, target_seq[:, i])
        dp[:, i, :] = res
        backtrace[:, i - 1, :] = max_idx
    return dp, backtrace


def backtrace(
    backtrace: Tensor,
    target_lens: Tensor,
    vertex_lens: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    @param backtrace: (batch_size, seq_len - 1, n) tensor
    @param target_lens: (batch_size,) tensor
    @param vertex_lens: (batch_size,) tensor
    @param target_span_indices: (batch_size, seq_len) tensor
    @param target_span_indices_pad_value: int

    @return: (paths, is_padding) where `paths` is a (batch_size, seq_len) tensor
        and `is_padding` is a (batch_size, seq_len) tensor. `paths` contains the
        most likely path through the graph given the backtrace tensor. `is_padding`
        is a boolean tensor that indicates whether the path is padding or not.
        If the respective element in `is_padding` is True, then the path at that
        position is padding.

    Given the backtrace tensor, find the most likely path
    that starts from the first vertex and ends at the last vertex.

    For any padding vertices, we assume that the previous vertex
    is where the path comes from
    """
    b, t, v = backtrace.shape
    offset_target_lens = target_lens - 1
    start_vertex = vertex_lens - 1
    backtrace = backtrace.transpose(-1, -2)
    tracepath = torch.zeros(b, dtype=torch.long, device=backtrace.device) + v - 1
    traces = []
    for i in range(t - 1, -1, -1):
        tracepath = torch.where((i + 1) == offset_target_lens, start_vertex, tracepath)
        nextpath = vector_gather(backtrace, tracepath)[:, i]
        nextpath = torch.where(nextpath == 0, tracepath - 1, nextpath)
        tracepath = torch.where(i >= offset_target_lens, -1, tracepath)
        traces.append(tracepath)
        tracepath = nextpath
    traces.append(tracepath * 0)
    traces = torch.stack(traces[::-1], dim=-1)
    return traces, traces < 0
