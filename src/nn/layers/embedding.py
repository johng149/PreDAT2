import torch
from torch.nn import Module, Embedding


class EmbeddingLayer(Module):
    # we use xavier initialization for the embedding layer
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        torch.nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, x):
        return self.embedding(x)
