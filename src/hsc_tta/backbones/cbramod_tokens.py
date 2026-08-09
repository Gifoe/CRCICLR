from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn

from hsc_tta.backbones.cbramod import module_sha256


class FrozenCBraModTokens(nn.Module):
    """Official CBraMod returning channel/patch tokens without pooling."""

    output_dim = 200

    def __init__(self, source_root: str | Path, checkpoint: str | Path):
        super().__init__()
        root = str(Path(source_root).resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        from models.cbramod import CBraMod  # type: ignore
        self.model = CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800,
                            seq_len=30, n_layer=12, nhead=8)
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.frozen_hash = module_sha256(self.model)

    def train(self, mode: bool = True) -> "FrozenCBraModTokens":
        super().train(False); self.model.eval(); return self

    def verify_frozen(self) -> None:
        if self.training or self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("CBraMod token backbone must remain frozen in eval mode")
        if module_sha256(self.model) != self.frozen_hash:
            raise RuntimeError("CBraMod token backbone parameter hash changed")

    @torch.inference_mode()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.verify_frozen()
        tokens = self.model(x.float()).float()
        if tokens.ndim != 4 or tokens.shape[-1] != self.output_dim:
            raise RuntimeError(f"unexpected token shape: {tuple(tokens.shape)}")
        if not torch.isfinite(tokens).all():
            raise FloatingPointError("CBraMod produced nonfinite tokens")
        return tokens
