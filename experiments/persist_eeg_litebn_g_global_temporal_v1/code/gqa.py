"""One bidirectional 4-query-head / 2-KV-head global temporal block."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rope import apply_rope


class GlobalTemporalBlock(nn.Module):
    d_model, query_heads, kv_heads, d_head = 48, 4, 2, 12

    def __init__(self):
        super().__init__()
        self.ln_attn = nn.LayerNorm(48)
        self.q = nn.Linear(48, 48)
        self.k = nn.Linear(48, 24)
        self.v = nn.Linear(48, 24)
        self.output = nn.Linear(48, 48)
        self.attention_dropout = nn.Dropout(.10)
        self.alpha_attn = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.ln_ffn = nn.LayerNorm(48)
        self.ffn = nn.Sequential(nn.Linear(48, 96), nn.GELU(), nn.Dropout(.10), nn.Linear(96, 48))
        self.alpha_ffn = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.last_attention_entropy = float("nan")
        self.last_normalized_attention_entropy = float("nan")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, width = tokens.shape
        if width != 48: raise RuntimeError(f"GQA requires d_model=48, got {width}")
        h = self.ln_attn(tokens)
        query = self.q(h).view(batch, length, 4, 12).transpose(1, 2)
        key = self.k(h).view(batch, length, 2, 12).transpose(1, 2)
        value = self.v(h).view(batch, length, 2, 12).transpose(1, 2)
        query, key = apply_rope(query), apply_rope(key)
        key, value = key.repeat_interleave(2, dim=1), value.repeat_interleave(2, dim=1)
        score = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(12.0)
        probability = F.softmax(score, dim=-1)
        safe = probability.clamp_min(1e-12)
        entropy = -(safe * safe.log()).sum(dim=-1)
        self.last_attention_entropy = float(entropy.detach().mean().cpu())
        self.last_normalized_attention_entropy = float((entropy / math.log(length)).detach().mean().cpu()) if length > 1 else 1.0
        attention = torch.matmul(self.attention_dropout(probability), value)
        attention = self.output(attention.transpose(1, 2).reshape(batch, length, 48))
        z1 = tokens + self.alpha_attn * attention
        return z1 + self.alpha_ffn * self.ffn(self.ln_ffn(z1))

    def parameter_breakdown(self) -> dict[str, int]:
        count = lambda module: int(sum(value.numel() for value in module.parameters()))
        return {"Q": count(self.q), "K": count(self.k), "V": count(self.v), "output_projection": count(self.output),
                "LayerNorm": count(self.ln_attn) + count(self.ln_ffn), "FFN": count(self.ffn),
                "ReZero_scalars": 2, "global_temporal_block_total": int(sum(value.numel() for value in self.parameters()))}
