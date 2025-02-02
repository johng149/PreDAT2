import torch
from src.common.dp_correction import correct_dp_slice_for_spans
from src.common.logsumexp import logsumexp_infsafe as logsumexp
from src.common.vector_gather import vector_gather

## DAG Loss
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

## Fuzzy Loss

from typing import Tuple, Union, List
from torch import Tensor
from collections import defaultdict

def find_ngrams(target_seqs, n, as_tensor=True) -> Union[Tuple[Tensor, Tensor], Tuple[List, List]]:
    """
    Given a 2D tensor of target sequences, and n-gram order, calculate
    which n-grams are present in the target sequences as well as the
    number of times they occur. We assume that the target sequences
    has no padding

    @param target_seqs: 2D tensor of target sequences
    @param n: n-gram order
    @param as_tensor: whether to return information as tensors or lists

    @return: n-grams present in the target sequences and their counts
    """
    batch_size, seq_len = target_seqs.shape
    assert seq_len >= n, "Sequence length must be greater than or equal to n-gram order"
    with torch.no_grad():
        ngrams = []
        counts = []

        for b in range(batch_size):
            ngram_dict = defaultdict(int)
            for i in range(seq_len - n + 1):
                ngram = tuple(target_seqs[b, i:i+n].tolist())
                ngram_dict[ngram] += 1

            ngrams.append(list(ngram_dict.keys()))
            counts.append(list(ngram_dict.values()))
        
        if as_tensor:
            ngrams = torch.tensor(ngrams, dtype=torch.long)
            counts = torch.tensor(counts, dtype=torch.long)
        
        return ngrams, counts
    
def passing_probability(transitions: Tensor, return_log: bool = False) -> Tensor:
    """
    Given a tensor of transition probabilities, calculate the probability
    of passing through each vertex in the graph, we'll assume that transitions
    is log-probabilities. If return_log is true, we return the passing
    probabilities in log-space otherwise we return them in linear space

    @param transitions: tensor of transition probabilities
    @return: tensor of passing probabilities
    """
    batch_size, num_vertices, _ = transitions.shape
    
    probs = torch.zeros(batch_size, num_vertices, device=transitions.device)
    probs[:, 0] = 1.0
    probs = torch.log(probs)

    for i in range(1, num_vertices):
        transition_column = transitions[:, :, i]
        current_sum = torch.logsumexp(probs + transition_column, dim=-1)
        probs[:, i] = current_sum
    
    if return_log:
        return probs
    else:
        return torch.exp(probs)
    
def log_bmm(log_A, log_B):
    """
    Performs a batch matrix multiplication in log space.

    Args:
        log_A: A tensor of shape (b, m, n) representing log(A).
        log_B: A tensor of shape (b, n, p) representing log(B).

    Returns:
        A tensor of shape (b, m, p) representing log(A @ B).
    """
    b, m, n = log_A.shape
    _, _, p = log_B.shape

    # 1. Expand dimensions to align for element-wise addition (broadcast)
    log_A_expanded = log_A.unsqueeze(3)  # Shape (b, m, n, 1)
    log_B_expanded = log_B.unsqueeze(1)  # Shape (b, 1, n, p)

    # 2. Perform addition in log-space for equivalent to product in linear space
    log_product = log_A_expanded + log_B_expanded  # Shape (b, m, n, p)

    # 3. LogSumExp over the `n` dimension (matrix multiplication reduction)
    log_C = torch.logsumexp(log_product, dim=2)  # Shape (b, m, p)

    return log_C

def fuzzy_ngram_loss(probs, transition, tgt_tokens, ngrams_order=2):
    # probs: batch_size x num_vertices x vocab_size
    # transition: batch_size x num_vertices x num_vertices
    # tgt_tokens: batch_size x tgt_len
    # we assume tgt_tokens have no padding (all the same length)
    ngrams, ngram_counts = find_ngrams(tgt_tokens, ngrams_order)

    passing_probs = passing_probability(transition, return_log=True)

    expected_tol_num_of_ngrams = passing_probs.unsqueeze(1)

    for i in range(ngrams_order-1):
        expected_tol_num_of_ngrams = log_bmm(expected_tol_num_of_ngrams, transition)


    expected_tol_num_of_ngrams = torch.logsumexp(expected_tol_num_of_ngrams, dim=-1)
    expected_tol_num_of_ngrams = torch.logsumexp(expected_tol_num_of_ngrams, dim=-1)


    ngram_target = ngrams[:,:,0].unsqueeze(-1) #bsz, number of ngram, 1

    #bsz, number of ngram, num vertices
    ngram_target_probs = torch.gather(input=probs.unsqueeze(1).expand(-1,ngram_target.size(-2),-1,-1),dim=-1,index=ngram_target.unsqueeze(2).expand(-1,-1,probs.size(-2),-1)).squeeze()

    expected_matched_num_of_ngrams = passing_probs.unsqueeze(1) + ngram_target_probs     

    for i in range(1,ngrams_order):
        ngram_target = ngrams[:,:,i].unsqueeze(-1) #bsz, number of ngram, 1

        #bsz, number of ngram, num vertices
        ngram_target_probs = torch.gather(input=probs.unsqueeze(1).expand(-1,ngram_target.size(-2),-1,-1),dim=-1,index=ngram_target.unsqueeze(2).expand(-1,-1,probs.size(-2),-1)).squeeze(dim=-1)

        expected_matched_num_of_ngrams = log_bmm(expected_matched_num_of_ngrams, transition)
        expected_matched_num_of_ngrams = expected_matched_num_of_ngrams + ngram_target_probs


    expected_matched_num_of_ngrams = torch.logsumexp(expected_matched_num_of_ngrams, dim=-1)

    ngram_counts = ngram_counts.log()
    cutted_expected_matched_num_of_ngrams = torch.min(expected_matched_num_of_ngrams, ngram_counts)#.sum(dim=-1)
    cutted_expected_matched_num_of_ngrams = torch.logsumexp(cutted_expected_matched_num_of_ngrams, dim=-1)

    cutted_precision = cutted_expected_matched_num_of_ngrams - expected_tol_num_of_ngrams

    loss = cutted_precision.exp().mean()

    return -loss