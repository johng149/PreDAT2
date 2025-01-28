import pytest
import torch

from src.training.loss_fn import dag_loss

def test_dag_loss_inf():
    
    num_vertices = 10
    vocab_size = 6

    # transition,  0  , 1  , 2  , 3  , 4  , 5  , 6  , 7  , 8  , 9
    transition0 = [0.0, 0.6, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    transition1 = [0.0, 0.0, 0.0, 0.3, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0]
    transition2 = [0.0, 0.0, 0.0, 0.7, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0]
    transition3 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.7, 0.3, 0.0, 0.0, 0.0]
    transition4 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0]
    transition5 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6, 0.4]
    transition6 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    transition7 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    transition8 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    transition9 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    transition_matrix = torch.tensor([transition0, transition1, transition2, transition3, transition4, transition5, transition6, transition7, transition8, transition9])
    transition_matrix = torch.log(transition_matrix)

    emission_matrix = torch.full((num_vertices, vocab_size), 1.0 / vocab_size)
    emission_matrix = torch.log(emission_matrix)

    transition_matrix = transition_matrix.unsqueeze(0)
    emission_matrix = emission_matrix.unsqueeze(0)

    targets = torch.tensor([
        [3, 1, 4, 5, 2]
    ])

    target_lens = torch.tensor([5])
    vertex_lens = torch.tensor([num_vertices])

    pins = torch.tensor([
        [-1, 3, -1, 5, -1]
    ])

    loss = dag_loss(
        targets=targets,
        transition_matrix=transition_matrix,
        emission_probs=emission_matrix,
        target_lens=target_lens,
        vertex_lens=vertex_lens,
        assignments=pins
    )

    assert torch.isinf(loss).all()

def test_dag_loss():
    num_vertices = 10
    vocab_size = 6

    # transition,  0  , 1  , 2  , 3  , 4  , 5  , 6  , 7  , 8  , 9
    transition0 = [0.0, 0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    transition1 = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    transition2 = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    transition3 = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    transition4 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.3, 0.3, 0.0, 0.0]
    transition5 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    transition6 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    transition7 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5]
    transition8 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    transition9 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    transition_matrix = torch.tensor([transition0, transition1, transition2, transition3, transition4, transition5, transition6, transition7, transition8, transition9])
    transition_matrix = torch.log(transition_matrix)

    emission_matrix = torch.full((num_vertices, vocab_size), 1.0 / vocab_size)
    emission_matrix = torch.log(emission_matrix)

    transition_matrix = transition_matrix.unsqueeze(0)
    emission_matrix = emission_matrix.unsqueeze(0)

    targets = torch.tensor([
        [3, 1, 4, 5, 2]
    ])

    target_lens = torch.tensor([5])
    vertex_lens = torch.tensor([num_vertices])

    pins = torch.tensor([
        [-1, 3, -1, 5, -1]
    ])

    emission_prob = 1.0 / vocab_size
    expected_prob = (1 * emission_prob * # zero vertex
                     0.2 * emission_prob * # 0 -> 3
                     1 * emission_prob * # 3 -> 4
                     0.4 * emission_prob * # 4 -> 5
                     1 * emission_prob # 5 -> 9
                    )
    expected_loss = -torch.log(torch.tensor([expected_prob])) / 5

    loss = dag_loss(
        targets=targets,
        transition_matrix=transition_matrix,
        emission_probs=emission_matrix,
        target_lens=target_lens,
        vertex_lens=vertex_lens,
        assignments=pins
    )

    assert torch.isclose(loss, expected_loss)