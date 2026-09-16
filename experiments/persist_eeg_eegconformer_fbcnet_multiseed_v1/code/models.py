"""Fixed EEG-Conformer and FBCNet baseline architectures.

The implementations are clean-room transcriptions of the published architecture
contracts, not imports of the GPL EEG-Conformer repository. The source commits
and unavoidable input/head adaptations are documented in IMPLEMENTATION_AUDIT.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class _Attention(nn.Module):
    def __init__(self, width: int = 40, heads: int = 10, dropout: float = 0.5):
        super().__init__()
        if width % heads:
            raise ValueError("attention width must divide heads")
        self.width, self.heads = width, heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.drop = nn.Dropout(dropout)
        self.output = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        shape = lambda tensor: tensor.reshape(batch, tokens, self.heads, -1).transpose(1, 2)
        query, key, value = shape(self.query(x)), shape(self.key(x)), shape(self.value(x))
        # The released EEG-Conformer scales by sqrt(embedding width), not
        # sqrt(per-head width); preserve that exact architecture choice.
        scores = query @ key.transpose(-1, -2) / math.sqrt(self.width)
        return self.output((self.drop(scores.softmax(dim=-1)) @ value)
                           .transpose(1, 2).reshape(batch, tokens, self.width))


class _ConformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_norm = nn.LayerNorm(40)
        self.attn = _Attention()
        self.attn_drop = nn.Dropout(0.5)
        self.ffn_norm = nn.LayerNorm(40)
        self.ffn = nn.Sequential(nn.Linear(40, 160), nn.GELU(), nn.Dropout(0.5),
                                 nn.Linear(160, 40), nn.Dropout(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn_drop(self.attn(self.attn_norm(x)))
        return x + self.ffn(self.ffn_norm(x))


class EEGConformer_BASELINE(nn.Module):
    """Song et al. six-block, 40-wide EEG-Conformer with input-sized FC head."""

    def __init__(self, channels: int, samples: int, classes: int):
        super().__init__()
        if samples < 99:
            raise ValueError("EEG-Conformer patch tokenizer requires >=99 samples")
        self.channels, self.samples, self.classes = channels, samples, classes
        self.patch = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25)),
            nn.Conv2d(40, 40, (channels, 1)),
            nn.BatchNorm2d(40), nn.ELU(),
            nn.AvgPool2d((1, 75), stride=(1, 15)), nn.Dropout(0.5),
            nn.Conv2d(40, 40, (1, 1)),
        )
        tokens = ((samples - 25 + 1 - 75) // 15) + 1
        if tokens <= 0:
            raise ValueError("no temporal tokens")
        self.encoder = nn.Sequential(*(_ConformerBlock() for _ in range(6)))
        # The released head fixes the flatten width at 2440 for a 1000-sample
        # input. Only this width and output class count vary with the task.
        self.classifier = nn.Sequential(nn.Linear(tokens * 40, 256), nn.ELU(),
                                        nn.Dropout(0.5), nn.Linear(256, 32), nn.ELU(),
                                        nn.Dropout(0.3), nn.Linear(32, classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.channels, self.samples):
            raise ValueError(f"EEG-Conformer expected [B,{self.channels},{self.samples}]")
        tokens = self.patch(x.unsqueeze(1)).squeeze(2).transpose(1, 2)
        return self.classifier(self.encoder(tokens).flatten(1))


class _MaxNormConv(nn.Conv2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.weight.copy_(torch.renorm(self.weight, p=2, dim=0, maxnorm=2.0))
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


class _MaxNormLinear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.weight.copy_(torch.renorm(self.weight, p=2, dim=0, maxnorm=0.5))
        return F.linear(x, self.weight, self.bias)


class FBCNet_BASELINE(nn.Module):
    """Mane et al. nine fixed bands, grouped spatial conv, segmented log-var."""

    def __init__(self, channels: int, samples: int, classes: int):
        super().__init__()
        if samples < 8:
            raise ValueError("four temporal variance segments require at least 8 samples")
        self.channels, self.samples, self.classes = channels, samples, classes
        self.spatial = _MaxNormConv(9, 9 * 32, (channels, 1), groups=9)
        self.bn = nn.BatchNorm2d(9 * 32)
        self.head = _MaxNormLinear(9 * 32 * 4, classes)

    def forward(self, banks: torch.Tensor) -> torch.Tensor:
        if banks.ndim != 4 or banks.shape[1:] != (9, self.channels, self.samples):
            raise ValueError(f"FBCNet expected [B,9,{self.channels},{self.samples}]")
        z = self.bn(self.spatial(banks))
        z = z * torch.sigmoid(z)
        if self.samples % 4 == 0:
            segments = z.reshape(len(z), 9 * 32, 4, self.samples // 4)
            variance = segments.var(dim=-1, unbiased=True)
        else:
            # ERP has T=250. Four adjacent [63,63,62,62] segments retain every
            # observed sample; this is the smallest input-length adaptation.
            variance = torch.stack([part.var(dim=-1, unbiased=True)
                                    for part in torch.tensor_split(z, 4, dim=-1)], dim=-1)
        z = torch.log(variance.clamp(1e-6, 1e6)).flatten(1)
        return self.head(z)


def build_model(name: str, channels: int, samples: int, classes: int) -> nn.Module:
    if name == "EEGConformer":
        return EEGConformer_BASELINE(channels, samples, classes)
    if name == "FBCNet":
        return FBCNet_BASELINE(channels, samples, classes)
    raise ValueError(name)
