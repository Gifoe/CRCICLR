"""CRCICLR boundary adapter for the vendored official SGN classifier."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


VENDOR = Path(__file__).resolve().parent / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from sgnmodels.SGN import Model as OfficialSGN  # noqa: E402


MODEL_NAME = "SGN"
UPSTREAM_REPOSITORY = "colison/SGN"
UPSTREAM_COMMIT = "c6d1b573dcb8c4255cde59b988f74334ea5da503"
ARCHITECTURE = {
    "source_configuration": "official PTB-XL standard classification configuration",
    "e_layers": 5,
    "depths": [2, 2, 2, 2, 1],
    "d_model": 64,
    "mlp_ratio": 2,
    "block_num": 1,
    "num_kernels": 7,
    "num_groups": 4,
    "period": 25,
    "dropout": 0.1,
}


class SGNAdapter(nn.Module):
    """Accept CRCICLR B,C,T and expose SGN classification logits B,K."""

    def __init__(self, channels: int, samples: int, classes: int):
        super().__init__()
        config = SimpleNamespace(
            task_name="classification",
            num_class=int(classes),
            e_layers=5,
            period=25,
            depths=[2, 2, 2, 2, 1],
            seq_len=int(samples),
            num_groups=4,
            enc_in=int(channels),
            d_model=64,
            embed="timeF",
            freq="h",
            dropout=0.1,
            block_num=1,
            kernel_size=3,
            mlp_ratio=2,
            num_kernels=7,
        )
        self.model = OfficialSGN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected B,C,T; got {tuple(x.shape)}")
        output = self.model(x.transpose(1, 2).contiguous(), None, None, None)
        # Official SGN also returns its similarity regularizer. The frozen
        # CRCICLR baseline loss is CE-only, so that optional term is not added.
        return output[0] if isinstance(output, tuple) else output


def build_model(*, channels: int, samples: int, classes: int) -> nn.Module:
    return SGNAdapter(channels, samples, classes)


def model_metadata(model: nn.Module) -> dict[str, object]:
    return {
        "model": MODEL_NAME,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "architecture": ARCHITECTURE,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "VGE_present": True,
        "MGWM_present": True,
        "PWSM_present": True,
    }
