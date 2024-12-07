import torch
from torch.nn import Module, Linear, Dropout, LayerNorm, ReLU
from torch.nn.functional import scaled_dot_product_attention as sdpa


class XAttn(Module):
    def __init__(self, emb_dim, num_heads, dropout=0.1):
        super().__init__()
        assert (
            emb_dim % num_heads == 0
        ), "Embedding dimension must be divisible by number of heads"
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.dropout = dropout
        self.q = Linear(emb_dim, emb_dim)
        self.kv = Linear(emb_dim, 2 * emb_dim)
        self.out_dropout = Dropout(dropout)

    def forward(self, q_x, kv_x):
        # we assume that all sequences have no padding
        batch_size, q_seq_len, emb_dim = q_x.shape
        _, kv_seq_len, _ = kv_x.shape

        q = self.q(q_x)
        kv = self.kv(kv_x)
        k, v = kv.split(emb_dim, dim=-1)

        q = q.view(batch_size, q_seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )
        v = v.view(batch_size, kv_seq_len, self.num_heads, self.head_dim).transpose(
            1, 2
        )

        y = sdpa(q, k, v, dropout_p=self.dropout)
        y = y.transpose(1, 2).reshape(batch_size, q_seq_len, emb_dim)
        y = self.out_dropout(y)
        return y


class XAttnBlock(Module):
    def __init__(self, emb_dim, num_heads, mlp_dim=None, dropout=0.1):
        super().__init__()
        self.embed_dim = emb_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim if mlp_dim is not None else emb_dim * 3
        self.dropout = dropout
        self.attn = XAttn(emb_dim, num_heads, dropout)
        self.norm1 = LayerNorm(emb_dim)
        self.norm2 = LayerNorm(emb_dim)
        self.ff_norm = LayerNorm(emb_dim)
        self.ff1 = Linear(emb_dim, mlp_dim)
        self.activation = ReLU()
        self.ff2 = Linear(mlp_dim, emb_dim)
        self.ff_dropout = Dropout(dropout)

    def forward(self, q_x, kv_x):
        x = q_x + self.attn(self.norm1(q_x), self.norm2(kv_x))
        ff = self.ff1(self.ff_norm(x))
        ff = self.activation(ff)
        ff = self.ff2(ff)
        ff = self.ff_dropout(ff)
        x = x + ff
        return x
