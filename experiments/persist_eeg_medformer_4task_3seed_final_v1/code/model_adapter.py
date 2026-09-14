"""CRCICLR boundary adapter for the vendored official Medformer classifier."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


VENDOR = Path(__file__).resolve().parent / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from models.Medformer import Model as OfficialMedformer  # noqa: E402


MODEL_NAME = "Medformer"
UPSTREAM_REPOSITORY = "DL4mHealth/Medformer"
UPSTREAM_COMMIT = "446275f27b713a9f09917a6ba0bc51a18e921597"
ARCHITECTURE = {
    "source_configuration": "official subject-independent TDBRAIN classification setting",
    "e_layers": 6,
    "d_model": 128,
    "d_ff": 256,
    "n_heads": 8,
    "patch_len_list": "8,8,8,16,16,16",
    "dropout": 0.1,
    "single_channel": False,
    "no_inter_attn": False,
    "augmentations": "none",
    "swa": False,
}


class MedformerAdapter(nn.Module):
    """Accept CRCICLR B,C,T and expose official classification logits B,K."""

    def __init__(self, channels: int, samples: int, classes: int):
        super().__init__()
        config = SimpleNamespace(
            task_name="classification",
            pred_len=0,
            output_attention=False,
            enc_in=int(channels),
            seq_len=int(samples),
            num_class=int(classes),
            single_channel=False,
            patch_len_list="8,8,8,16,16,16",
            augmentations="none",
            d_model=128,
            n_heads=8,
            dropout=0.1,
            no_inter_attn=False,
            e_layers=6,
            d_ff=256,
            activation="gelu",
        )
        self.model = OfficialMedformer(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected B,C,T; got {tuple(x.shape)}")
        return self.model(x.transpose(1, 2).contiguous(), None, None, None)


def build_model(*, channels: int, samples: int, classes: int) -> nn.Module:
    return MedformerAdapter(channels, samples, classes)


def model_metadata(model: nn.Module) -> dict[str, object]:
    return {
        "model": MODEL_NAME,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "architecture": ARCHITECTURE,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
    }
