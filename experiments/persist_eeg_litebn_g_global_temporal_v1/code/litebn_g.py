"""LiteBN-G: one ReZero GQA+RoPE block between frozen stem and LiteBN backend."""
from __future__ import annotations

import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from gqa import GlobalTemporalBlock


def mapping_sha256(mapping: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(mapping.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8")); digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii")); digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


class LiteBNG(nn.Module):
    """Exact 64-feature LiteBN with frozen early stem/BN and low-LR backend adaptation."""
    group_l_prefixes = ("depth1.", "point1.", "depth2.", "point2.", "embedding.0.", "embedding.2.", "head.")

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.global_block = GlobalTemporalBlock()
        for parameter in self.base.parameters(): parameter.requires_grad_(False)
        names = dict(self.base.named_parameters())
        for name, parameter in names.items():
            if name.startswith(self.group_l_prefixes): parameter.requires_grad_(True)
        # BatchNorm affine parameters are immutable even if a future base refactor changes a prefix.
        for module in self.base.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                for parameter in module.parameters(recurse=False): parameter.requires_grad_(False)
                module.eval()
        self._assert_groups()

    def _assert_groups(self) -> None:
        base_names = dict(self.base.named_parameters())
        allowed = {name for name in base_names if name.startswith(self.group_l_prefixes)}
        actual = {name for name, value in base_names.items() if value.requires_grad}
        if actual != allowed: raise RuntimeError(f"invalid Group-L set: {actual ^ allowed}")
        if any(parameter.requires_grad for module in self.base.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm) for parameter in module.parameters(recurse=False)):
            raise RuntimeError("BatchNorm affine parameter was left trainable")

    def train(self, mode: bool = True):
        super().train(mode)
        # Preserve regular dropout behavior throughout LiteBN while forcing every BN to historical eval statistics.
        for module in self.base.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm): module.eval()
        return self

    def group_g_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [(f"global_block.{name}", parameter) for name, parameter in self.global_block.named_parameters() if parameter.requires_grad]

    def group_l_named_parameters(self) -> list[tuple[str, nn.Parameter]]:
        return [(f"base.{name}", parameter) for name, parameter in self.base.named_parameters() if parameter.requires_grad]

    def frozen_stem_sha256(self) -> str:
        values = {name: value for name, value in self.base.state_dict().items()
                  if name.startswith(("temporal.", "temporal_norm.", "spatial.", "spatial_norm."))}
        return mapping_sha256(values)

    def bn_sha256(self) -> str:
        values = {}
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
            branches.append(F.dropout(branch, 0.20, self.base.training))
        stem = torch.cat(branches, dim=1)
        tokens = stem.squeeze(2).transpose(1, 2)
        stem = self.global_block(tokens).transpose(1, 2).unsqueeze(2)
        value = F.dropout(F.avg_pool2d(F.elu(self.base.norm1(self.base.point1(self.base.depth1(stem)))), (1, 2)), .15, self.base.training)
        value = F.dropout(F.avg_pool2d(F.elu(self.base.norm2(self.base.point2(self.base.depth2(value)))), (1, 2)), .15, self.base.training)
        representation = self.base.drop(self.base.embedding(self.base.pool(value).flatten(1)))
        return self.base.head(representation), representation

    def model_spec(self) -> dict:
        g = self.global_block.parameter_breakdown()
        return {"insertion_point": "after 48-channel three-branch concat and before historical backend block 1",
                "d_model": 48, "Hq": 4, "Hkv": 2, "d_head": 12, "RoPE_dimension": 12, "RoPE_base": 10000,
                "ffn_dimensions": [48, 96, 48], "attention_dropout": .10, "ffn_dropout": .10,
                "historical_dropout": {"stem": .20, "backend": .15, "readout": .25},
                "alpha_attn_initial": float(self.global_block.alpha_attn.detach().cpu()),
                "alpha_ffn_initial": float(self.global_block.alpha_ffn.detach().cpu()), "global_parameter_breakdown": g,
                "group_G_trainable_names": [name for name, _ in self.group_g_named_parameters()],
                "group_L_trainable_names": [name for name, _ in self.group_l_named_parameters()],
                "frozen_parameter_names": [f"base.{name}" for name, value in self.base.named_parameters() if not value.requires_grad]}
