import torch

from src.training.glancing import findPathBatched, backtrace

def test_backtrace():
    num_vertices = 10
    vocab_size = 6

    # transition,  0  , 1  , 2  , 3  , 4  , 5  , 6  , 7  , 8  , 9
    transition0 = [0.0, 0.1, 0.5, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    transition1 = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    transition2 = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    transition3 = [0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]
    transition4 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    transition5 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.9, 0.0, 0.0]
    transition6 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5]
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
        [-1, 3, -1, 6, -1]
    ])

    dp, back = findPathBatched(
        transition_matrix=transition_matrix,
        emission_probs=emission_matrix,
        target_seq=targets,
        target_span_indices=pins,
    )

    path, _ = backtrace(back, target_lens, vertex_lens)
    expected_path = torch.tensor([
        [0, 3, 4, 6, 9]
    ])

    assert torch.equal(path, expected_path)