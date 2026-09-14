"""One bidirectional 4-query-head / 2-KV-head, 64-dimensional GQA block."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rope import apply_rope


class GlobalTemporalBlock64(nn.Module):
    """The sole late global-temporal block in LiteBN-G2."""

    d_model, query_heads, kv_heads, d_head = 64, 4, 2, 16

    def __init__(self) -> None:
        super().__init__()
        self.ln_attn = nn.LayerNorm(64)
        self.q = nn.Linear(64, 64)
        self.k = nn.Linear(64, 32)
        self.v = nn.Linear(64, 32)
        self.output = nn.Linear(64, 64)
        self.attention_dropout = nn.Dropout(.10)
        self.alpha_attn = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.ln_ffn = nn.LayerNorm(64)
        self.ffn = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Dropout(.10), nn.Linear(128, 64))
        self.alpha_ffn = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.last_attention_entropy = float("nan")
        self.last_normalized_attention_entropy = float("nan")
        self.last_entropy_quantiles = (float("nan"), float("nan"), float("nan"))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, width = tokens.shape
        if width != self.d_model:
            raise RuntimeError(f"GQA requires d_model=64, got {width}")
        h = self.ln_attn(tokens)
        query = self.q(h).view(batch, length, self.query_heads, self.d_head).transpose(1, 2)
        key = self.k(h).view(batch, length, self.kv_heads, self.d_head).transpose(1, 2)
        value = self.v(h).view(batch, length, self.kv_heads, self.d_head).transpose(1, 2)
        query, key = apply_rope(query), apply_rope(key)
        key, value = key.repeat_interleave(2, dim=1), value.repeat_interleave(2, dim=1)
        score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(float(self.d_head))
        probability = F.softmax(score, dim=-1)
        entropy = -(probability.clamp_min(1e-12) * probability.clamp_min(1e-12).log()).sum(dim=-1)
        self.last_attention_entropy = float(entropy.detach().mean().cpu())
        normalized = entropy / math.log(length) if length > 1 else torch.ones_like(entropy)
        self.last_normalized_attention_entropy = float(normalized.detach().mean().cpu())
        quantiles = torch.quantile(normalized.detach().float().flatten(), torch.tensor([.10, .50, .90], dtype=torch.float32, device=normalized.device))
        self.last_entropy_quantiles = tuple(float(value.cpu()) for value in quantiles)
        attended = torch.matmul(self.attention_dropout(probability), value)
        attended = self.output(attended.transpose(1, 2).reshape(batch, length, self.d_model))
        z1 = tokens + self.alpha_attn * attended
        return z1 + self.alpha_ffn * self.ffn(self.ln_ffn(z1))

    def parameter_breakdown(self) -> dict[str, int]:
        count = lambda module: int(sum(value.numel() for value in module.parameters()))
        return {
            "Q": count(self.q), "K": count(self.k), "V": count(self.v),
            "output_projection": count(self.output),
            "LayerNorm": count(self.ln_attn) + count(self.ln_ffn),
            "FFN": count(self.ffn), "ReZero_scalars": 2,
            "global_temporal_block_total": int(sum(value.numel() for value in self.parameters())),
        }
