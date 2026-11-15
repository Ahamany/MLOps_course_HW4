"""Numerically stable logarithmic multi-head attention and a residual block."""

import torch
from torch import nn
from torch.nn import functional as F


class LogAttention(nn.Module):
    """Attention on log(softplus(Q/K)); boolean masks use True to block attention.

    Inputs have shape [batch, sequence, embedding]. Completely masked rows have
    zero attention weights. Returned weights are averaged over heads.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0 or embed_dim % num_heads:
            raise ValueError("embed_dim must be positive and divisible by positive num_heads")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = dropout
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        need_weights: bool = False,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        for tensor in (query, key, value):
            if tensor.ndim != 3 or tensor.shape[-1] != self.embed_dim or 0 in tensor.shape:
                raise ValueError("inputs must be nonempty [batch, sequence, embed_dim] tensors")
        batch, query_length, _ = query.shape
        key_length = key.shape[1]
        if key.shape[0] != batch or value.shape[:2] != (batch, key_length):
            raise ValueError("input batch sizes and key/value sequence lengths must match")

        def split(projection, tensor):
            return projection(tensor).reshape(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

        q = split(self.q_proj, query)
        k = split(self.k_proj, key)
        v = split(self.v_proj, value)
        # Linear projections may be negative: never apply log directly to them.
        tiny = torch.finfo(q.dtype).tiny
        q = F.softplus(q).clamp_min(tiny).log()
        k = F.softplus(k).clamp_min(tiny).log()
        scores = (q @ k.transpose(-2, -1)) * self.scale
        blocked = torch.zeros(batch, 1, query_length, key_length, dtype=torch.bool, device=q.device)
        if key_padding_mask is not None:
            if key_padding_mask.dtype != torch.bool or key_padding_mask.shape != (batch, key_length):
                raise ValueError("key_padding_mask must be boolean [batch, key_length]")
            blocked = blocked | key_padding_mask.to(q.device)[:, None, None, :]
        if attn_mask is not None:
            if attn_mask.dtype != torch.bool or attn_mask.shape != (query_length, key_length):
                raise ValueError("attn_mask must be boolean [query_length, key_length]")
            blocked = blocked | attn_mask.to(q.device)[None, None, :, :]
        if is_causal:
            blocked = blocked | torch.ones(query_length, key_length, dtype=torch.bool, device=q.device).triu(1)
        scores = scores.masked_fill(blocked, float("-inf"))
        # softmax([-inf, ...]) is NaN. Fill only fully blocked rows before softmax,
        # then explicitly zero every blocked weight, including those rows.
        scores = scores.masked_fill(blocked.all(dim=-1, keepdim=True), 0.0)
        weights = scores.softmax(dim=-1).masked_fill(blocked, 0.0)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        output = (weights @ v).transpose(1, 2).contiguous().reshape(batch, query_length, self.embed_dim)
        return self.out_proj(output), weights.mean(dim=1) if need_weights else None


class AttentionBlock(nn.Module):
    """Pre-normalized residual attention followed by a feed-forward network."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attention = LogAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        attended, _ = self.attention(normalized, normalized, normalized)
        x = x + self.dropout(attended)
        return x + self.dropout(self.feed_forward(self.norm2(x)))
