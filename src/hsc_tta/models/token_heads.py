from __future__ import annotations

import torch
from torch import nn


class TokenHeadBase(nn.Module):
    hidden_dim = 256

    def classify(self, hidden: torch.Tensor, return_hidden: bool) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        logits = self.classifier(hidden)
        return (logits, hidden) if return_hidden else logits


class OldMeanMLP(TokenHeadBase):
    def __init__(self, n_classes: int):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(200, 256), nn.GELU(), nn.Dropout(.2))
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, tokens: torch.Tensor, *, return_hidden: bool = False):
        return self.classify(self.encoder(tokens.mean(dim=(1, 2))), return_hidden)


class TemporalAttentionHead(TokenHeadBase):
    """Patch attention after channel-preserving aggregation; intended for formal single-channel sleep."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.norm = nn.LayerNorm(200)
        self.attention = nn.MultiheadAttention(200, 4, dropout=.1, batch_first=True)
        self.encoder = nn.Sequential(nn.Linear(200, 256), nn.GELU(), nn.Dropout(.2))
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, tokens: torch.Tensor, *, return_hidden: bool = False):
        sequence = self.norm(tokens.mean(dim=1))
        sequence, _ = self.attention(sequence, sequence, sequence, need_weights=False)
        return self.classify(self.encoder(sequence.mean(dim=1)), return_hidden)


class ChannelTemporalHead(TokenHeadBase):
    """Within-channel temporal attention followed by channel attention with channel identity."""

    def __init__(self, n_classes: int, max_channels: int = 64):
        super().__init__()
        self.channel_identity = nn.Parameter(torch.zeros(max_channels, 200))
        nn.init.normal_(self.channel_identity, std=.02)
        self.temporal = nn.MultiheadAttention(200, 4, dropout=.1, batch_first=True)
        self.channel = nn.MultiheadAttention(200, 4, dropout=.1, batch_first=True)
        self.norm = nn.LayerNorm(200)
        self.encoder = nn.Sequential(nn.Linear(200, 256), nn.GELU(), nn.Dropout(.2))
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, tokens: torch.Tensor, *, return_hidden: bool = False):
        batch, channels, patches, dim = tokens.shape
        temporal = self.norm(tokens).reshape(batch * channels, patches, dim)
        temporal, _ = self.temporal(temporal, temporal, temporal, need_weights=False)
        channel_tokens = temporal.mean(1).reshape(batch, channels, dim)
        channel_tokens = channel_tokens + self.channel_identity[:channels]
        channel_tokens, _ = self.channel(channel_tokens, channel_tokens, channel_tokens, need_weights=False)
        return self.classify(self.encoder(channel_tokens.mean(1)), return_hidden)


class OfficialAllPatchHead(TokenHeadBase):
    """Official CBraMod PhysioNet all-patch two-layer classifier, without claiming new official weights."""

    hidden_dim = 200

    def __init__(self, n_classes: int, channels: int = 64, patches: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(nn.Flatten(), nn.Linear(channels * patches * 200, 200),
                                     nn.ELU(), nn.Dropout(.2))
        self.classifier = nn.Linear(200, n_classes)

    def forward(self, tokens: torch.Tensor, *, return_hidden: bool = False):
        return self.classify(self.encoder(tokens), return_hidden)


def make_token_head(name: str, n_classes: int) -> TokenHeadBase:
    if name == "old_mean_mlp": return OldMeanMLP(n_classes)
    if name == "temporal_attention_head": return TemporalAttentionHead(n_classes)
    if name == "channel_temporal_head": return ChannelTemporalHead(n_classes)
    if name == "official_downstream_head": return OfficialAllPatchHead(n_classes)
    raise ValueError(f"unknown token head: {name}")
