import torch
from src.common.dp_correction import correct_dp_slice_for_spans
from src.common.logsumexp import logsumexp_infsafe as logsumexp
from src.common.vector_gather import vector_gather


def dag_loss_raw_lynchpin(targets, transition_matrix, emission_probs, assignments):
    """
    Calculates the directed acyclic graph (DAG) loss given the targets, transition matrix, and emission probabilities.
    It returns the dynamic programming table of which one of the entries is the DAG loss.

    Args:
        targets (torch.Tensor): The target sequence of shape (batch_size, m).
        transition_matrix (torch.Tensor): The transition matrix of shape (batch_size, l, l).
        emission_probs (torch.Tensor): The emission probabilities of shape (batch_size, l, vocab_size).
        assignments (torch.Tensor): The assignments of shape (batch_size, m).

    Returns:
        torch.Tensor: The DAG loss of shape (batch_size, m, l).

    The assignments tensor is a tensor of shape (batch_size, m) where each element is an integer from 0 to l-1.
    If the element is -1, then the corresponding element in the target sequence is free to be any of the l vertices.
    Otherwise, at that position in the target sequence, the vertex must be the one specified by the assignment.
    """
    batch_size, m = targets.shape
    _, l, vocab_size = emission_probs.shape
    dp = torch.ones((batch_size, m, l), device=transition_matrix.device)
    dp[dp == 1] = -float("inf")
    initial_probs = torch.gather(
        emission_probs, dim=2, index=targets[:, 0].unsqueeze(1).unsqueeze(2)
    )
    dp[:, 0, 0] = initial_probs.squeeze(2).squeeze(1)
    # assumes that transition_matrix and emission_probs are already in log space
    # also we need to tranpose emission_probs so it is vocab_size x l
    # so the vector gather works
    emission_probs = emission_probs.transpose(1, 2)
    for i in range(1, m):
        # before proceeding, we need to check the previous iteration's dp values
        # to see if that it agrees with the assignments, if not, we need to adjust
        # the dp values accordingly
        prev_dp = vector_gather(emission_probs, targets[:, i]) + (
            (
                logsumexp(
                    dp[:, i - 1, :].unsqueeze(1).transpose(1, 2) + transition_matrix,
                    dim=1,
                )
            ).squeeze(1)
        )
        prev_dp = prev_dp.unsqueeze(1).transpose(-1, -2)
        prev_dp = correct_dp_slice_for_spans(prev_dp, assignments, i + 1)
        dp[:, i, :] = prev_dp
    return dp


def process_dp(dp, target_lens, vertex_lens):
    """
    Processes the dynamic programming table (dp) to extract the correct loss values.
    The target lengths and vertex lengths are needed to determine which values to extract
    and which values are a result of padding and should be ignored.

    Args:
        dp (torch.Tensor): The dynamic programming table of shape (batch_size, m, l).
        target_lens (torch.Tensor): A tensor of shape (batch_size,) that describes the length of each target sequence.
        vertex_lens (torch.Tensor): A tensor of shape (batch_size,) that describes the number of non-padding vertices for each batch.

    Returns:
        torch.Tensor: The values corresponding to the last target and last vertex of shape (batch_size,).
    """
    dp_values = vector_gather(dp, target_lens - 1)
    values = torch.gather(dp_values, dim=1, index=(vertex_lens - 1).unsqueeze(-1))
    return values


def dag_loss(
    targets,
    transition_matrix,
    emission_probs,
    target_lens,
    vertex_lens,
    assignments=None,
):
    """
    Calculates the directed acyclic graph (DAG) loss given the targets, transition matrix, and emission probabilities.

    Args:
        targets (torch.Tensor): The target sequence of shape (batch_size, m).
        transition_matrix (torch.Tensor): The transition matrix of shape (batch_size, l, l).
        emission_probs (torch.Tensor): The emission probabilities of shape (batch_size, l, vocab_size).
        target_lens (torch.Tensor): A tensor of shape (batch_size,) that describes the length of each target sequence.
        vertex_lens (torch.Tensor): A tensor of shape (batch_size,) that describes the number of non-padding vertices for each batch.

    Returns:
        torch.Tensor: The DAG loss of shape (batch_size,).
    """
    dp = dag_loss_raw_lynchpin(
        targets,
        transition_matrix,
        emission_probs,
        (
            assignments
            if assignments is not None
            else -torch.ones_like(targets, device=targets.device, dtype=torch.long)
        ),
    )
    values = process_dp(dp, target_lens, vertex_lens)
    values = values / target_lens.unsqueeze(-1)
    return -torch.mean(values)
