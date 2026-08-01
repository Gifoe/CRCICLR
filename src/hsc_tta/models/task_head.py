from __future__ import annotations

import torch
from torch import nn


class TaskHead(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout))
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, embeddings: torch.Tensor, *, return_hidden: bool = False):
        hidden = self.hidden(embeddings.float())
        logits = self.classifier(hidden).float()
        return (logits, hidden) if return_hidden else logits
