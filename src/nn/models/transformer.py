import torch
from torch.nn import Module, ModuleList
from src.nn.layers.dag_out import OutputDAG
from src.nn.layers.embedding import EmbeddingLayer
from src.nn.layers.self_attn import SelfAttnBlock
from src.nn.layers.x_attn import XAttnBlock


class Transformer(Module):
    def __init__(
        self,
        emb_dim: int,
        vocab_size: int,
        max_enc_len: int,
        max_dec_len: int,
        n_heads: int,
        n_enc: int,
        n_dec: int,
        mlp_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        self.n_heads = n_heads
        self.n_enc = n_enc
        self.n_dec = n_dec
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.kwargs = {
            "emb_dim": emb_dim,
            "vocab_size": vocab_size,
            "max_enc_len": max_enc_len,
            "max_dec_len": max_dec_len,
            "n_heads": n_heads,
            "n_enc": n_enc,
            "n_dec": n_dec,
            "mlp_dim": mlp_dim,
            "dropout": dropout,
        }
        self.vocab_embed = EmbeddingLayer(vocab_size, emb_dim)
        self.enc_pos_embed = EmbeddingLayer(max_enc_len, emb_dim)
        self.dec_pos_embed = EmbeddingLayer(max_dec_len, emb_dim)

        enc_layers = [
            SelfAttnBlock(emb_dim, n_heads, mlp_dim, dropout) for _ in range(n_enc)
        ]
        self.enc_layers = ModuleList(enc_layers)

        dec_layers = []
        for _ in range(n_dec - 1):
            dec_layers.append(XAttnBlock(emb_dim, n_heads, mlp_dim, dropout))
            dec_layers.append(SelfAttnBlock(emb_dim, n_heads, mlp_dim, dropout))
        self.dec_layers = ModuleList(dec_layers)

        self.output_dag = OutputDAG(emb_dim, vocab_size)

    def enc_forward(self, enc_x):
        batch_size, enc_len = enc_x.shape
        enc_pos = torch.arange(enc_len).unsqueeze(0)

        enc_x = self.vocab_embed(enc_x) + self.enc_pos_embed(enc_pos)
        for layer in self.enc_layers:
            enc_x = layer(enc_x)
        return enc_x

    def dec_forward(self, dec_x, enc_x):
        batch_size, dec_len = dec_x.shape
        dec_pos = torch.arange(dec_len).unsqueeze(0)

        dec_x = self.vocab_embed(dec_x) + self.dec_pos_embed(dec_pos)
        for i, layer in enumerate(self.dec_layers):
            if isinstance(layer, XAttnBlock):
                dec_x = layer(dec_x, enc_x)
            else:
                dec_x = layer(dec_x)
        return dec_x

    def forward(self, enc_x, dec_x):
        enc_x = self.enc_forward(enc_x)
        dec_x = self.dec_forward(dec_x, enc_x)
        return self.output_dag(dec_x)
