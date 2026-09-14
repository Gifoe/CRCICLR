"""CRCICLR boundary adapter for the vendored official ModernTCN classifier."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


VENDOR = Path(__file__).resolve().parent / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.insert(0, str(VENDOR))

from models.ModernTCN import Model as OfficialModernTCN  # noqa: E402


MODEL_NAME = "ModernTCN"
UPSTREAM_REPOSITORY = "luodhhh/ModernTCN"
UPSTREAM_COMMIT = "56a9a2c018385cd5acef015378cae7f084d1b11c"
ARCHITECTURE = {
    "source_configuration": "official SelfRegulationSCP2 classification configuration",
    "ffn_ratio": 4,
    "patch_size": 32,
    "patch_stride": 16,
    "num_blocks": [1, 1],
    "large_size": [51, 49],
    "small_size": [5, 5],
    "dims": [64, 128],
    "head_dropout": 0.0,
    "dropout": 0.3,
    "class_dropout": 0.1,
    "use_multi_scale": False,
    "revin": True,
}


class ModernTCNAdapter(nn.Module):
    """Accept CRCICLR B,C,T and expose official classification logits B,K."""

    def __init__(self, channels: int, samples: int, classes: int):
        super().__init__()
        config = SimpleNamespace(
            task_name="classification",
            stem_ratio=6,
            downsample_ratio=2,
            ffn_ratio=4,
            num_blocks=[1, 1],
            large_size=[51, 49],
            small_size=[5, 5],
            dims=[64, 128],
            dw_dims=[64, 128],
            small_kernel_merged=False,
            dropout=0.3,
            head_dropout=0.0,
            use_multi_scale=False,
            revin=1,
            affine=0,
            subtract_last=0,
            freq="h",
            seq_len=int(samples),
            enc_in=int(channels),
            individual=0,
            pred_len=0,
            kernel_size=25,
            patch_size=32,
            patch_stride=16,
            class_dropout=0.1,
            num_class=int(classes),
            decomposition=0,
        )
        self.model = OfficialModernTCN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected B,C,T; got {tuple(x.shape)}")
        # The official wrapper consumes B,T,C and performs its own B,C,T swap.
        return self.model(x.transpose(1, 2).contiguous(), None, None, None)


def build_model(*, channels: int, samples: int, classes: int) -> nn.Module:
    return ModernTCNAdapter(channels, samples, classes)


def model_metadata(model: nn.Module) -> dict[str, object]:
    return {
        "model": MODEL_NAME,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "architecture": ARCHITECTURE,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
    }
