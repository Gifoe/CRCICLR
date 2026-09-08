"""Strict loading and feature extraction for the two frozen CFRF carriers."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import torch
from torch import nn


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_sha256(module: nn.Module) -> str:
    payload = io.BytesIO()
    torch.save(module.state_dict(), payload)
    return hashlib.sha256(payload.getvalue()).hexdigest()


def freeze(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    if module.training or any(parameter.requires_grad for parameter in module.parameters()):
        raise RuntimeError("carrier freeze invariant failed")
    return module


def load_carrier(model_name: str, channels: int, checkpoint: Path, device: torch.device, eegnet: type[nn.Module], compact: type[nn.Module]) -> nn.Module:
    model = eegnet(channels) if model_name == "EEGNet" else compact(channels, "bn")
    model = model.to(device)
    incompat = model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False), strict=True)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(f"strict checkpoint mismatch: {checkpoint}")
    return freeze(model)


@torch.no_grad()
def extract(eegnet: nn.Module, litebn: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return logits and penultimate representations without changing carrier state."""
    eegnet.eval(); litebn.eval()
    z_a, h_a = eegnet(x)
    z_e, h_e = litebn(x)
    if h_a.ndim != 2 or h_e.ndim != 2 or h_a.shape[1] != 64 or h_e.shape[1] != 64:
        raise RuntimeError(f"unexpected penultimate dimensions: {tuple(h_a.shape)}, {tuple(h_e.shape)}")
    return z_a.float(), z_e.float(), h_a.float(), h_e.float()
