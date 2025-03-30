import torch
from torch import Tensor


def glancing_accuracy(
        paths: Tensor,
        is_padding: Tensor,
        emission_probs: Tensor,
        target_seq: Tensor
):
    """
    @param paths: (batch_size, seq_len) tensor
    @param is_padding: (batch_size, seq_len) tensor
    @param emission_probs: (batch_size, num_vertices, vocab_size) tensor
    @param target_seq: (batch_size, seq_len) tensor

    @return: accuracy: float

    `paths` is a (batch_size, seq_len) tensor that contains the most likely path
    through the graph for the respective target sequences. Each element in `paths`
    is greater than or equal to 0 and less than the number of vertices in the graph.

    `is_padding` is a (batch_size, seq_len) tensor that indicates whether the path
    is padding or not. If the respective element in `is_padding` is True, then
    the path at that position is padding.

    `emission_probs` is a (batch_size, num_vertices, vocab_size) tensor that
    contains the emission probabilities for each vertex in the graph in log-space.

    `target_seq` is a (batch_size, seq_len) tensor that contains the target sequences,
    each element in the target sequence is greater than or equal to 0 and
    less than the vocabulary size.

    The accuracy is calculated by first getting all the emission probabilities
    on the paths described by `paths` and then getting the most likely token
    for each of vertices in those paths. Then we compare the most likely token
    to the expected token in the target sequence. The accuracy is the number of
    correct tokens divided by the number of tokens in the target sequence.
    Note that padding tokens are ignored in the accuracy calculation, so
    any apparent match on a padding position will not be counted as correct,
    and the padding tokens will not count towards the total number of tokens
    in the target sequence.
    """
    with torch.no_grad():
        batch_size, seq_len = paths.shape
        _, num_vertices, vocab_size = emission_probs.shape

        vertices_on_paths = torch.gather(
            emission_probs, dim=1, index=paths.unsqueeze(-1).expand(-1, -1, vocab_size)
        ) # (batch_size, seq_len, vocab_size)

        most_likely = torch.argmax(vertices_on_paths, dim=-1) # (batch_size, seq_len)
        matches = most_likely == target_seq # (batch_size, seq_len)
        matches = matches & ~is_padding # (batch_size, seq_len)
        num_correct = matches.sum()
        num_total = (~is_padding).sum()
        accuracy = num_correct / num_total if num_total > 0 else 0
        return accuracy.item()