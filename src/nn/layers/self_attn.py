import torch
from torch.nn import Module, Linear, Dropout, LayerNorm, ReLU
from torch.nn.functional import scaled_dot_product_attention as sdpa


class SelfAttn(Module):
    def __init__(self, emb_dim, num_heads, dropout=0.1):
        super().__init__()
        assert (
            emb_dim % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads"
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.dropout = dropout
        self.qkv = Linear(emb_dim, 3 * emb_dim)
        self.out = Linear(emb_dim, emb_dim)
        self.out_dropout = Dropout(dropout)

    def forward(self, x):
        # no need for mask, we assume that all sequences have the same length
        # this can be achieved through bucketing
        batch_size, seq_len, emb_dim = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.split(emb_dim, dim=-1)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        y = sdpa(q, k, v, dropout_p=self.dropout)
        y = y.transpose(1, 2).reshape(batch_size, seq_len, emb_dim)
        y = self.out(y)
        y = self.out_dropout(y)
        return y


class SelfAttnBlock(Module):
    def __init__(self, emb_dim, num_heads, mlp_dim=None, dropout=0.1):
        super().__init__()
        self.embed_dim = emb_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim if mlp_dim is not None else emb_dim * 3
        self.dropout = dropout
        self.attn = SelfAttn(emb_dim, num_heads, dropout)
        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)
        self.ff1 = Linear(emb_dim, mlp_dim)
        self.activation = ReLU()
        self.ff2 = Linear(mlp_dim, emb_dim)
        self.ff_dropout = Dropout(dropout)

    def forward(self, x, *args):
        x = x + self.attn(self.norm1(x))
        ff = self.ff1(self.norm2(x))
        ff = self.activation(ff)
        ff = self.ff2(ff)
        ff = self.ff_dropout(ff)
        x = x + ff
        return x
