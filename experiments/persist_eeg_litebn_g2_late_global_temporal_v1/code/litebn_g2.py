"""LiteBN-G2: one late ReZero GQA+RoPE block after LiteBN backend block 2."""
from __future__ import annotations

import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from gqa64 import GlobalTemporalBlock64


def mapping_sha256(mapping: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(mapping.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8")); digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii")); digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class LiteBNG2(nn.Module):
    """Exact 64-feature LiteBN, with only a late 64-d global block added."""

    group_l_prefixes = ("depth2.", "point2.", "embedding.0.", "embedding.2.", "head.")
    frozen_parameter_prefixes = ("temporal.", "spatial.", "depth1.", "point1.")

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.global_block = GlobalTemporalBlock64()
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        for name, parameter in self.base.named_parameters():
            if name.startswith(self.group_l_prefixes):
                parameter.requires_grad_(True)
        for module in self.base.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                for parameter in module.parameters(recurse=False):
                    parameter.requires_grad_(False)
                module.eval()
        self._assert_groups()

    def _assert_groups(self) -> None:
        named = dict(self.base.named_parameters())
        allowed = {name for name in named if name.startswith(self.group_l_prefixes)}
        actual = {name for name, value in named.items() if value.requires_grad}
        if actual != allowed:
            raise RuntimeError(f"invalid Group-L set: {actual ^ allowed}")
        if any(parameter.requires_grad for module in self.base.modules()
               if isinstance(module, nn.modules.batchnorm._BatchNorm)
               for parameter in module.parameters(recurse=False)):
            raise RuntimeError("BatchNorm affine parameter was left trainable")

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self.base.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def group_g_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [(f"global_block.{name}", parameter) for name, parameter in self.global_block.named_parameters() if parameter.requires_grad]

    def group_l_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [(f"base.{name}", parameter) for name, parameter in self.base.named_parameters() if parameter.requires_grad]

    def frozen_parameter_sha256(self) -> str:
        values = {name: value for name, value in self.base.state_dict().items() if name.startswith(self.frozen_parameter_prefixes)}
        return mapping_sha256(values)

    def bn_sha256(self) -> str:
        values: dict[str, torch.Tensor] = {}
        for name, module in self.base.named_modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                values.update({f"{name}.{key}": value for key, value in module.state_dict().items()})
        return mapping_sha256(values)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        value = value.unsqueeze(1)
        branches = []
        for temporal, temporal_norm, spatial, spatial_norm in zip(self.base.temporal, self.base.temporal_norm, self.base.spatial, self.base.spatial_norm):
            branch = F.elu(temporal_norm(temporal(value)))
            branch = F.elu(spatial_norm(spatial(branch)))
            branch = F.avg_pool2d(branch, (1, 4))
            branches.append(F.dropout(branch, .20, self.base.training))
        value = torch.cat(branches, dim=1)
        value = F.dropout(F.avg_pool2d(F.elu(self.base.norm1(self.base.point1(self.base.depth1(value)))), (1, 2)), .15, self.base.training)
        value = F.dropout(F.avg_pool2d(F.elu(self.base.norm2(self.base.point2(self.base.depth2(value)))), (1, 2)), .15, self.base.training)
        value = self.global_block(value.squeeze(2).transpose(1, 2)).transpose(1, 2).unsqueeze(2)
        representation = self.base.drop(self.base.embedding(self.base.pool(value).flatten(1)))
        return self.base.head(representation), representation

    def model_spec(self) -> dict:
        return {
            "insertion_point": "after historical backend block 2 and before AdaptiveAvgPool(8)",
            "d_model": 64, "Hq": 4, "Hkv": 2, "d_head": 16,
            "RoPE_dimension": 16, "RoPE_base": 10000,
            "ffn_dimensions": [64, 128, 64], "attention_dropout": .10, "ffn_dropout": .10,
            "historical_dropout": {"stem": .20, "backend": .15, "readout": .25},
            "alpha_attn_initial": float(self.global_block.alpha_attn.detach().cpu()),
            "alpha_ffn_initial": float(self.global_block.alpha_ffn.detach().cpu()),
            "global_parameter_breakdown": self.global_block.parameter_breakdown(),
            "group_G_trainable_names": [name for name, _ in self.group_g_named_parameters()],
            "group_L_trainable_names": [name for name, _ in self.group_l_named_parameters()],
            "frozen_parameter_names": [f"base.{name}" for name, value in self.base.named_parameters() if not value.requires_grad],
        }
