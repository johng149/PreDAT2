import torch
from torch import nn
from src.nn.utils.dag_masking import masking
from src.common.logsumexp import logsumexp_infsafe as logsumexp


class OutputDAG(nn.Module):
    def __init__(self, embed_dim, vocab_size, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.cs = embed_dim // num_heads  # chunk size
        self.attn = nn.Linear(self.embed_dim, self.embed_dim * 2, bias=False)
        self.gate = nn.Linear(self.embed_dim, self.num_heads, bias=False)
        self.lm_head = nn.Linear(self.embed_dim, self.vocab_size, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, vertex_lens):
        # since we are using bucketing, we assume that all sequences have the same length
        # that means that vertex_lens should just be a tensor of some constant value
        # repeated batch_size times, but we'll leave it in for now, just in case
        # we want to try non-bucketed sequences later
        batch_size, num_vertices, emb_dim = x.shape
        x = self.norm(x)
        m, r = masking(batch_size, num_vertices, vertex_lens, x.device)

        g = self.gate(x)
        g = torch.log_softmax(g, dim=-1)

        q, k = self.attn(x).split(self.embed_dim, dim=-1)
        q = q.reshape(batch_size, -1, self.num_heads, self.cs)
        k = k.reshape(batch_size, -1, self.num_heads, self.cs)
        # https://github.com/thu-coai/DA-Transformer/blob/245a90fe1397ba0dcaac04317bc327497c76cd9c/fs_plugins/models/glat_decomposed_with_link.py#L385
        attn_scores = torch.einsum("bicf,bjcf->bijc", q, k) / (self.cs**0.5)
        attn_scores = attn_scores.masked_fill(~m.unsqueeze(-1), float("-inf"))
        attn_scores = torch.log_softmax(attn_scores, dim=2)
        attn_scores = attn_scores.masked_fill(r.unsqueeze(-1), float("-inf"))

        # you might worry that performing a logsumexp might cause the
        # transition_matrix to contain values that map to probability
        # greater than 1 in linear space, however, experimentally this
        # additional g term seems to prevent that from happening
        # (no idea why though)
        transition_matrix = logsumexp(attn_scores + g.unsqueeze(2), dim=-1).squeeze(-1)

        vocab_log_probs = torch.log_softmax(self.lm_head(x), dim=-1)
        return transition_matrix, vocab_log_probs
