import torch
from typing import Union
from torch import device, Tensor


def acyclic_mask(vertices: int, device: Union[str, device]) -> Tensor:
    """
    Generates a mask that, when applied to the transition matrix, ensures that
    vertex i can only transition to vertices j where j > i.

    @param vertices: The number of vertices in the graph.
    @param device: The device to use for the mask.
    @return: The acyclic mask of shape (vertices, vertices), where value is non-zero
        if that part of the transition matrix should be masked.

    For example, with 3 vertices, the mask would be:
    [
        [1, 0, 0],
        [1, 1, 0],
        [1, 1, 1]
    ]
    """
    mask = torch.tril(torch.ones((vertices, vertices), device=device))
    return mask


def padding_transition_mask(
    batch_size: int, vertices: int, vertex_lens: Tensor, device: Union[str, device]
):
    """
    Generates a mask that, when applied to the transition matrix, prevents vertices
    from transitioning to padding vertices. It is assumed that the padding vertices
    are at the end of the sequence.

    @param batch_size: The number of sequences in the batch.
    @param vertices: The number of vertices in the graph after padding
    @param vertex_lens: A tensor of shape (batch_size,) that describes the number of
        non-padding vertices for each batch.
    @param device: The device to use for the mask.

    For example, with a batch size of 2, 4 vertices, and vertex_lens = [2, 3], the mask
    would be:

    tensor([[[0., 0., 1., 1.],
            [0., 0., 1., 1.],
            [0., 0., 1., 1.],
            [0., 0., 1., 1.]],

            [[0., 0., 0., 1.],
            [0., 0., 0., 1.],
            [0., 0., 0., 1.],
            [0., 0., 0., 1.]]])
    """
    vertex_lens_mask = torch.arange(vertices, device=device).repeat(
        len(vertex_lens), 1
    ) < vertex_lens.unsqueeze(-1)
    mask = torch.ones((batch_size, vertices, vertices), device=device)
    mask.transpose(1, 2)[vertex_lens_mask] = 0
    return mask


def masking_helper(batch_size, vertices, vertex_lens, device):
    """
    Creates masking to the transition matrix based on acyclic and padding masks.

    Args:
        transition_matrix (torch.Tensor): The transition matrix of shape (batch_size, vertices, vertices).
        vertex_lens (torch.Tensor): A tensor of shape (batch_size,) that describes the number of non-padding vertices for each batch.

    Returns:
        torch.Tensor: The masked transition matrix of shape (batch_size, vertices, vertices).
    """
    acyclic = acyclic_mask(vertices, device)
    padding = padding_transition_mask(batch_size, vertices, vertex_lens, device)
    return padding + acyclic


def masking(
    batch_size: int,
    vertices: int,
    vertex_lens: Tensor,
    device: Union[str, device],
):
    """
    Creates masking to the transition matrix based on acyclic and padding masks.
    Output is based on those for `masking_helper`, however, if there are any rows that
    contain only False (that is, all elements are masked), then every element
    in that row is set to True.

    Also returns a row mask that describes which rows are all masked

    Args:
        transition_matrix (torch.Tensor): The transition matrix of shape (batch_size, vertices, vertices).
        vertex_lens (torch.Tensor): A tensor of shape (batch_size,) that describes the number of non-padding vertices for each batch.

    Returns:
        torch.Tensor: The masked transition matrix of shape (batch_size, vertices, vertices).
        torch.Tensor: The row mask of shape (batch_size, vertices, 1).
    """
    m = masking_helper(batch_size, vertices, vertex_lens, device) == 0
    r = m.sum(dim=2, keepdim=True) == 0
    return m.masked_fill(r, True), r
