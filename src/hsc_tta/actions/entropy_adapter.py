from __future__ import annotations

import copy
import hashlib

import numpy as np
import torch
from torch import nn


class ResidualAdapter(nn.Module):
    def __init__(self, dimension: int = 200, bottleneck: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(dimension)
        self.down = nn.Linear(dimension, bottleneck)
        self.up = nn.Linear(bottleneck, dimension)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z + self.up(torch.nn.functional.gelu(self.down(self.norm(z))))


class EntropyAdapter:
    name = "entropy_adapter"

    def __init__(self, task_head: nn.Module, *, dimension: int = 200, bottleneck: int = 64,
                 steps: int = 5, learning_rate: float = 1e-4, beta: float = 0.1,
                 gamma: float = 1e-4, device: str = "cuda"):
        self.task_head = task_head
        self.device = torch.device(device)
        self.adapter = ResidualAdapter(dimension, bottleneck).to(self.device)
        self.initial_state = copy.deepcopy(self.adapter.state_dict())
        self.steps, self.learning_rate = int(steps), float(learning_rate)
        self.beta, self.gamma = float(beta), float(gamma)
        self.initial_hash = self.parameter_hash()
        self.diagnostics: dict[str, object] = {}

    def parameter_hash(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.adapter.state_dict().items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def reset(self) -> None:
        self.adapter.load_state_dict(self.initial_state)
        if self.parameter_hash() != self.initial_hash:
            raise RuntimeError("entropy adapter reset hash mismatch")

    @staticmethod
    def _entropy(probabilities: torch.Tensor) -> torch.Tensor:
        return -(probabilities * probabilities.clamp_min(1e-8).log()).sum(1)

    def adapt(self, context_embeddings: np.ndarray, context_logits: np.ndarray | None = None) -> "EntropyAdapter":
        self.reset()
        z = torch.as_tensor(context_embeddings, dtype=torch.float32, device=self.device)
        self.task_head.eval()
        for parameter in self.task_head.parameters():
            parameter.requires_grad_(False)
        with torch.no_grad():
            before = torch.softmax(self.task_head(z), dim=1)
        optimizer = torch.optim.Adam(self.adapter.parameters(), lr=self.learning_rate)
        gradient_nan, completed = False, 0
        for _ in range(self.steps):
            optimizer.zero_grad(set_to_none=True)
            probabilities = torch.softmax(self.task_head(self.adapter(z)), dim=1)
            sample_entropy = self._entropy(probabilities).mean()
            marginal_entropy = self._entropy(probabilities.mean(0, keepdim=True)).mean()
            penalty = sum((p - self.initial_state[n].to(self.device)).square().sum()
                          for n, p in self.adapter.named_parameters())
            loss = sample_entropy - self.beta * marginal_entropy + self.gamma * penalty
            if not torch.isfinite(loss):
                gradient_nan = True
                break
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.adapter.parameters()):
                gradient_nan = True
                break
            torch.nn.utils.clip_grad_norm_(self.adapter.parameters(), 1.0)
            optimizer.step()
            completed += 1
        with torch.no_grad():
            after = torch.softmax(self.task_head(self.adapter(z)), dim=1)
        before_entropy, after_entropy = float(self._entropy(before).mean()), float(self._entropy(after).mean())
        before_dist, after_dist = before.mean(0), after.mean(0)
        kl = float((after_dist * (after_dist.clamp_min(1e-8).log() - before_dist.clamp_min(1e-8).log())).sum())
        update_sq = sum((p.detach() - self.initial_state[n].to(self.device)).square().sum()
                        for n, p in self.adapter.named_parameters())
        update_norm = float(update_sq.sqrt())
        max_share = float(after.argmax(1).bincount(minlength=after.shape[1]).max() / len(after))
        collapse = bool(gradient_nan or max_share > 0.98 or after_entropy < 0.02 or not torch.isfinite(after).all())
        self.diagnostics = {
            "entropy_before": before_entropy, "entropy_after": after_entropy, "prediction_kl": kl,
            "adapter_update_norm": update_norm, "class_distribution_before": before_dist.cpu().tolist(),
            "class_distribution_after": after_dist.cpu().tolist(), "collapse_flag": collapse,
            "gradient_nan_flag": gradient_nan, "steps_completed": completed,
            "parameter_reset_hash": self.initial_hash,
        }
        return self

    @torch.inference_mode()
    def predict_proba(self, embeddings: np.ndarray, logits: np.ndarray | None = None) -> np.ndarray:
        z = torch.as_tensor(embeddings, dtype=torch.float32, device=self.device)
        return torch.softmax(self.task_head(self.adapter(z)), dim=1).float().cpu().numpy()
