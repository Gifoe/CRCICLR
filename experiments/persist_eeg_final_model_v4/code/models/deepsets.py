from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def configurations() -> list[dict[str, Any]]:
    return [
        {
            "hidden": hidden,
            "prior": prior,
            "learning_rate": 0.01,
            "weight_decay": 0.001,
            "epochs": 80,
        }
        for hidden in (8, 16, 32)
        for prior in (0.01, 0.1, 1.0)
    ]


class _DeepSetGate(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = int(hidden)
        self.phi = nn.Sequential(nn.Linear(14, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.score = nn.Sequential(nn.Linear(2 * hidden + 4, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(()))
        self.register_buffer("identity", torch.eye(6))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = x[:, :6]
        mask = x[:, 6:12]
        session = x[:, 12:14]
        count = mask.sum(1, keepdim=True).clamp_min(1.0)
        base = (logits * mask).sum(1, keepdim=True) / count
        centered = (logits - base) * mask
        std = torch.sqrt((centered.square().sum(1, keepdim=True) / count).clamp_min(1e-8))
        batch = len(x)
        identity = self.identity.unsqueeze(0).expand(batch, -1, -1)
        token = torch.cat(
            [
                logits.unsqueeze(-1),
                logits.abs().unsqueeze(-1),
                torch.sigmoid(logits).unsqueeze(-1),
                (logits - base).unsqueeze(-1),
                base.unsqueeze(1).expand(-1, 6, -1),
                std.unsqueeze(1).expand(-1, 6, -1),
                session.unsqueeze(1).expand(-1, 6, -1),
                identity,
            ],
            dim=-1,
        )
        hidden = self.phi(token)
        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / count
        pooled_token = pooled.unsqueeze(1).expand(-1, 6, -1)
        context = torch.cat(
            [
                (logits - base).unsqueeze(-1),
                logits.abs().unsqueeze(-1),
                session.unsqueeze(1).expand(-1, 6, -1),
            ],
            dim=-1,
        )
        score = self.score(torch.cat([hidden, pooled_token, context], dim=-1)).squeeze(-1)
        score = score.masked_fill(mask <= 0, -1e9)
        weight = torch.softmax(score, dim=1)
        pooled_logit = (weight * logits).sum(1)
        final_logit = torch.exp(self.log_scale.clamp(-2, 2)) * pooled_logit + self.bias
        uniform = mask / count
        return final_logit, weight, uniform


class DeepSetKeepGate:
    def __init__(self, configuration: dict[str, Any], seed: int):
        self.configuration = dict(configuration)
        self.seed = int(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: _DeepSetGate | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DeepSetKeepGate":
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        model = _DeepSetGate(int(self.configuration["hidden"])).to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.configuration["learning_rate"]),
            weight_decay=float(self.configuration["weight_decay"]),
        )
        values = torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)
        labels = torch.as_tensor(np.asarray(y, dtype=np.float32), device=self.device)
        prior = float(self.configuration["prior"])
        model.train()
        for _ in range(int(self.configuration["epochs"])):
            optimizer.zero_grad(set_to_none=True)
            logits, weight, uniform = model(values)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss = loss + prior * ((weight - uniform) ** 2).sum(1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        self.model = model.eval()
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("DeepSet gate is not fit")
        values = torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.device)
        output = []
        with torch.no_grad():
            for start in range(0, len(values), 4096):
                logits, _, _ = self.model(values[start : start + 4096])
                output.append(torch.sigmoid(logits).cpu().numpy())
        p1 = np.concatenate(output).astype(float)
        return np.column_stack([1 - p1, p1])


def build(configuration: dict[str, Any], seed: int) -> DeepSetKeepGate:
    return DeepSetKeepGate(configuration, seed)
