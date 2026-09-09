"""Thin, source-faithful adapter for the pinned official TeCh implementation.

The benchmark cache is B x C x T while the official model consumes B x T x C.
This module deliberately performs no resampling, filtering, patching, or other
signal transformation.  It only supplies the official configuration and swaps
the two non-batch axes at the model boundary.
"""
from __future__ import annotations

import contextlib
import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import torch
from torch import nn


TECH_COMMIT = "9a378cc546a5d97c871eff282148175b3c7cd75b"


def official_root() -> Path:
    root = Path(os.environ["OFFICIAL_BACKBONE_ROOT"]).expanduser().resolve() / "TeCh"
    if not (root / "models" / "TeCh.py").is_file():
        raise FileNotFoundError(f"pinned official TeCh source is missing: {root}")
    return root


@contextlib.contextmanager
def _official_import_path(root: Path) -> Iterator[None]:
    """Import only the official TeCh package, avoiding a stale generic models module."""
    stale = {name: module for name, module in sys.modules.items()
             if name == "models" or name.startswith("models.") or name == "layers" or name.startswith("layers.")}
    for name in stale:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        sys.path.remove(str(root))
        for name in list(sys.modules):
            if name == "models" or name.startswith("models.") or name == "layers" or name.startswith("layers."):
                sys.modules.pop(name, None)
        sys.modules.update(stale)


def recipe_config(*, channels: int, samples: int, classes: int, recipe: str) -> SimpleNamespace:
    """Return precisely one of the two preregistered official recipe families."""
    if recipe == "TECH-T":
        values = dict(t_layer=6, v_layer=0, d_model=128, dropout=0.0, patch_len=6,
                      learning_rate=1e-4, batch_size=128, train_epochs=60,
                      augmentations="flip0,frequency0.2,jitter0,mask0,channel0,drop0.4")
    elif recipe == "TECH-TC":
        values = dict(t_layer=6, v_layer=6, d_model=256, dropout=0.0, patch_len=1,
                      learning_rate=1e-4, batch_size=128, train_epochs=40,
                      augmentations="flip0.2,frequency0.2,jitter0,mask0,channel0,drop0.4")
    else:
        raise ValueError(f"unknown preregistered TeCh recipe: {recipe}")
    return SimpleNamespace(enc_in=int(channels), seq_len=int(samples), num_class=int(classes), **values)


def official_model(config: SimpleNamespace) -> nn.Module:
    root = official_root()
    with _official_import_path(root):
        module = importlib.import_module("models.TeCh")
        return module.Model(config)


class TeChAdapter(nn.Module):
    """Exact official TeCh model with the sole B,C,T -> B,T,C boundary adapter."""
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        self.config = config
        self.model = official_model(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"TeCh requires B,C,T input; got {tuple(x.shape)}")
        if x.shape[1] != self.config.enc_in or x.shape[2] != self.config.seq_len:
            raise ValueError("TeCh input shape disagrees with frozen task metadata: "
                             f"{tuple(x.shape)} vs C={self.config.enc_in}, T={self.config.seq_len}")
        return self.model(x.transpose(1, 2).contiguous())
