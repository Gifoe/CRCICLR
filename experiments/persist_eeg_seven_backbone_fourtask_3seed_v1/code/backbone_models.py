"""Admitted single-model backbone constructors for the four-task benchmark.

This file has no cache loader and never accesses a label or a split.  Each
constructor is intentionally isolated so the official projects' generic
``models`` packages cannot contaminate one another when a cell is run in its
own Python process.
"""
from __future__ import annotations

import contextlib
import importlib
import json
import os
import pickle
import sys
import types
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch import nn

from tech_official import TeChAdapter, recipe_config


MODELS = ("EEGNet", "LiteBN", "TCFormer", "ST-EEGFormer-small", "LaBraM-base", "CBraMod", "TeCh")
MODEL_NATIVE_RESAMPLING = {
    "EEGNet": None,
    "LiteBN": None,
    "TCFormer": None,
    "TeCh": None,
    "ST-EEGFormer-small": {"source_hz": 250, "target_hz": 100, "method": "scipy.signal.resample_poly(up=2, down=5)"},
    "LaBraM-base": {"source_hz": 250, "target_hz": 200, "method": "scipy.signal.resample_poly(up=4, down=5)", "patch_samples": 200},
    "CBraMod": {"source_hz": 250, "target_hz": 200, "method": "scipy.signal.resample_poly(up=4, down=5)", "patch_samples": 200},
}


def official_root() -> Path:
    return Path(os.environ["OFFICIAL_BACKBONE_ROOT"]).expanduser().resolve()


@contextlib.contextmanager
def official_import_path(root: Path) -> Iterator[None]:
    """Use an official source tree without retaining a stale generic package."""
    prefixes = ("models", "utils", "layers", "modeling_finetune")
    stale = {name: value for name, value in sys.modules.items()
             if name in prefixes or name.startswith(tuple(value + "." for value in prefixes))}
    for name in stale:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        sys.path.remove(str(root))
        for name in list(sys.modules):
            if name in prefixes or name.startswith(tuple(value + "." for value in prefixes)):
                sys.modules.pop(name, None)
        sys.modules.update(stale)


def _trainable(model: nn.Module) -> int:
    return int(sum(value.numel() for value in model.parameters() if value.requires_grad))


def _copy_state(model: nn.Module, source: dict[str, torch.Tensor], *, allow: set[str] | None = None) -> dict[str, Any]:
    """Load only shape-identical tensors and report every exclusion.

    The sole non-identical official checkpoint tensor is handled explicitly by
    ST-EEGFormer below; silent partial checkpoint loading is forbidden.
    """
    target = model.state_dict()
    accepted = {key: value for key, value in source.items()
                if key in target and tuple(value.shape) == tuple(target[key].shape)}
    unexpected = sorted(key for key in source if key not in target)
    mismatched = {key: {"checkpoint": list(value.shape), "model": list(target[key].shape)}
                  for key, value in source.items() if key in target and tuple(value.shape) != tuple(target[key].shape)}
    missing = sorted(key for key in target if key not in accepted)
    permitted = allow or set()
    nonpermitted = sorted(set(missing) - permitted)
    model.load_state_dict(accepted, strict=False)
    return {
        "loaded_tensors": len(accepted), "model_tensors": len(target),
        "loaded_parameter_elements": int(sum(value.numel() for value in accepted.values())),
        "model_parameter_elements": int(sum(value.numel() for value in target.values())),
        "unexpected_checkpoint_keys": unexpected, "mismatched": mismatched,
        "missing_model_keys": missing, "nonpermitted_missing_model_keys": nonpermitted,
    }


class EEGNet(nn.Module):
    """Canonical benchmark EEGNet; classifier dimension is task metadata only."""
    def __init__(self, channels: int, samples: int, classes: int):
        super().__init__()
        self.temporal = nn.Conv2d(1, 8, (1, 64), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(8)
        self.spatial = nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1, self.drop1 = nn.AvgPool2d((1, 4)), nn.Dropout(0.25)
        self.depth = nn.Conv2d(16, 16, (1, 16), padding="same", groups=16, bias=False)
        self.point, self.bn3 = nn.Conv2d(16, 16, 1, bias=False), nn.BatchNorm2d(16)
        self.pool2, self.drop2 = nn.AvgPool2d((1, 8)), nn.Dropout(0.25)
        self.embedding = nn.Sequential(nn.Linear(16 * (samples // 4 // 8), 64), nn.ELU(), nn.LayerNorm(64))
        self.head = nn.Linear(64, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = x.unsqueeze(1)
        value = self.bn1(self.temporal(value))
        value = self.drop1(self.pool1(F.elu(self.bn2(self.spatial(value)))))
        value = self.drop2(self.pool2(F.elu(self.bn3(self.point(self.depth(value))))))
        return self.head(self.embedding(value.flatten(1)))


def _norm(kind: str, channels: int) -> nn.Module:
    return nn.BatchNorm2d(channels) if kind == "bn" else nn.GroupNorm(4 if channels in (8, 16) else 8, channels)


class LiteBN(nn.Module):
    """Locked CompactLite BN backbone with a task-size classifier head."""
    def __init__(self, channels: int, classes: int):
        super().__init__()
        self.temporal = nn.ModuleList([nn.Conv2d(1, 8, (1, kernel), padding="same", bias=False) for kernel in (15, 63, 127)])
        self.temporal_norm = nn.ModuleList([_norm("bn", 8) for _ in range(3)])
        self.spatial = nn.ModuleList([nn.Conv2d(8, 16, (channels, 1), groups=8, bias=False) for _ in range(3)])
        self.spatial_norm = nn.ModuleList([_norm("bn", 16) for _ in range(3)])
        self.depth1, self.point1, self.norm1 = nn.Conv2d(48, 48, (1, 9), padding="same", groups=48, bias=False), nn.Conv2d(48, 32, 1, bias=False), _norm("bn", 32)
        self.depth2, self.point2, self.norm2 = nn.Conv2d(32, 32, (1, 7), padding="same", groups=32, bias=False), nn.Conv2d(32, 32, 1, bias=False), _norm("bn", 32)
        self.pool = nn.AdaptiveAvgPool2d((1, 4))
        self.embedding = nn.Sequential(nn.Linear(128, 64), nn.ELU(), nn.LayerNorm(64))
        self.drop, self.head = nn.Dropout(0.25), nn.Linear(64, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = x.unsqueeze(1); branches = []
        for temporal, tnorm, spatial, snorm in zip(self.temporal, self.temporal_norm, self.spatial, self.spatial_norm):
            branch = F.elu(tnorm(temporal(value))); branch = F.elu(snorm(spatial(branch)))
            branches.append(F.dropout(F.avg_pool2d(branch, (1, 4)), 0.20, self.training))
        value = torch.cat(branches, 1)
        value = F.dropout(F.avg_pool2d(F.elu(self.norm1(self.point1(self.depth1(value)))), (1, 2)), 0.15, self.training)
        value = F.dropout(F.avg_pool2d(F.elu(self.norm2(self.point2(self.depth2(value)))), (1, 2)), 0.15, self.training)
        return self.head(self.drop(self.embedding(self.pool(value).flatten(1))))


class TCFormerAdapter(nn.Module):
    def __init__(self, channels: int, classes: int):
        super().__init__()
        root = official_root() / "TCFormer"
        with official_import_path(root):
            # The official ``models/__init__.py`` eagerly imports unrelated
            # Lightning training modules.  We construct its package shell and
            # import the source TCFormer module directly; its own relative
            # imports remain untouched and no source code is copied or edited.
            package = types.ModuleType("models")
            package.__path__ = [str(root / "models")]
            sys.modules["models"] = package
            # TCFormerModule is a plain torch.nn.Module, but its source file
            # also declares an unused Lightning training wrapper.  The server
            # environment intentionally contains no Lightning dependency.  A
            # minimal import-time base lets us instantiate the official core
            # module while excluding that unrelated training harness.
            training_shell = types.ModuleType("models.classification_module")
            training_shell.ClassificationModule = nn.Module
            sys.modules["models.classification_module"] = training_shell
            module = importlib.import_module("models.tcformer")
            self.model = module.TCFormerModule(n_channels=channels, n_classes=classes, F1=16,
                temp_kernel_lengths=(16, 32, 64), pool_length_1=8, pool_length_2=7, D=2,
                dropout_conv=.3, d_group=16, tcn_depth=2, kernel_length_tcn=4,
                dropout_tcn=.3, use_group_attn=True, q_heads=8, kv_heads=4,
                trans_depth=5, trans_dropout=.4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _channel_names(dataset: str) -> list[str]:
    repo = Path(os.environ["SEVEN_REPO"]).resolve()
    lock = repo / "experiments" / "persist_eeg_fm_rescue_stage0" / "protocol" / "FM_INPUT_PROTOCOL_LOCK.json"
    return list(json.loads(lock.read_text(encoding="utf-8"))[dataset]["channels"])


class STEEGFormerAdapter(nn.Module):
    """Official small model with a 250 Hz -> 100 Hz model-boundary adapter.

    The official checkpoint has a 145-row channel table while the current
    source declares a 256-row table.  The release mapping only uses rows
    0..141; copying its 145 published rows verbatim therefore preserves every
    mapped pretrained embedding.  F9/F10 are absent from that vocabulary and
    use the published neighbouring FT9/FT10 coordinates, recorded explicitly.
    """
    def __init__(self, dataset: str, channels: int, classes: int):
        super().__init__()
        root = official_root() / "STEEGFormer"
        source_root = root / "eeg_foundation_2025"
        sys.path.insert(0, str(source_root / "utils"))
        try:
            st = importlib.import_module("models_vit_eeg")
            base = st.vit_small_patch16(global_pool="token", num_classes=classes, head_drop_out=0.1)
        finally:
            sys.path.remove(str(source_root / "utils"))
            sys.modules.pop("models_vit_eeg", None)
        checkpoint = torch.load(root / "checkpoints" / "checkpoint-300.pth", map_location="cpu", weights_only=False)
        raw = checkpoint.get("model", checkpoint)
        source = {key: value for key, value in raw.items() if not key.startswith("dec_") and not key.startswith("decoder_")}
        target = base.state_dict()
        table_key = "enc_channel_emd.channel_transformation.weight"
        table = torch.zeros_like(target[table_key])
        table[:raw[table_key].shape[0]] = raw[table_key]
        source[table_key] = table
        # The checkpoint is a self-supervised pretraining checkpoint: the
        # task classifier (including its LayerNorm) is necessarily new, and
        # timm's disabled positional parameter was not saved by the release.
        self.load_report = _copy_state(base, source, allow={"cls_head.norm.weight", "cls_head.norm.bias",
                                                             "cls_head.final.weight", "cls_head.final.bias", "pos_embed"})
        if self.load_report["nonpermitted_missing_model_keys"]:
            raise RuntimeError(f"ST checkpoint partial-load failure: {self.load_report['nonpermitted_missing_model_keys']}")
        mapping_path = root / "pretrain" / "senloc_file" / "sen_chan_idx.pkl"
        mapping = pickle.load(mapping_path.open("rb"))["channels_mapping"]
        lower = {str(key).lower(): int(value) for key, value in mapping.items()}
        aliases = {"f9": "ft9", "f10": "ft10"}
        resolved, alias_rows = [], []
        for name in _channel_names(dataset):
            key = name.lower()
            replacement = aliases.get(key, key)
            if replacement not in lower:
                raise RuntimeError(f"official ST channel vocabulary has no legal mapping for {dataset}/{name}")
            resolved.append(lower[replacement])
            if replacement != key: alias_rows.append({"cache_channel": name, "official_alias": replacement.upper(), "official_index": lower[replacement]})
        if len(resolved) != channels or max(resolved) >= raw[table_key].shape[0]:
            raise RuntimeError("ST channel index/checkpoint compatibility failure")
        base.default_chan_idx = torch.tensor(resolved, dtype=torch.long)
        self.model, self.channel_mapping = base, {"mapping_file": str(mapping_path), "indices": resolved, "aliases": alias_rows}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] not in (100, 400): raise ValueError(f"ST expects resampled 100/400 samples, got {x.shape[-1]}")
        return self.model(x)


class LaBraMAdapter(nn.Module):
    def __init__(self, dataset: str, channels: int, classes: int):
        super().__init__()
        root = official_root() / "LaBraM"
        with official_import_path(root):
            module = importlib.import_module("modeling_finetune")
            base = module.labram_base_patch200_200(pretrained=False, num_classes=0, use_mean_pooling=True,
                use_rel_pos_bias=False, use_abs_pos_emb=True, init_values=0.1, qkv_bias=False)
        checkpoint = torch.load(root / "checkpoints" / "labram-base.pth", map_location="cpu", weights_only=False)
        raw = checkpoint.get("model", checkpoint)
        source = {key.removeprefix("student."): value for key, value in raw.items()
                  if not key.removeprefix("student.").startswith("head.")}
        # The release omits the downstream mean-pooling normalisation layer;
        # it is a finetuning component and is initialized by the official
        # constructor together with the new supervised head.
        self.load_report = _copy_state(base, source, allow={"fc_norm.weight", "fc_norm.bias"})
        if self.load_report["nonpermitted_missing_model_keys"]:
            raise RuntimeError(f"LaBraM checkpoint partial-load failure: {self.load_report['nonpermitted_missing_model_keys']}")
        repo = Path(os.environ["SEVEN_REPO"]).resolve()
        lock = json.loads((repo / "experiments" / "persist_eeg_fm_rescue_stage0" / "protocol" / "FM_INPUT_PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
        indices = list(lock["LaBraM"]["input_chans"][dataset])
        # Official ``forward_features`` reserves element zero in input_chans;
        # its frozen maps consequently contain C+1 indices for C cache rows.
        if len(indices) != channels + 1: raise RuntimeError("LaBraM channel map length mismatch")
        self.model, self.input_chans, self.head = base, indices, nn.Linear(200, classes)
        nn.init.trunc_normal_(self.head.weight, std=.02); nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 200: raise ValueError(f"LaBraM resampled length is not a 200-sample patch multiple: {x.shape}")
        value = x.reshape(x.shape[0], x.shape[1], x.shape[-1] // 200, 200)
        return self.head(self.model.forward_features(value, input_chans=self.input_chans))


class CBraModAdapter(nn.Module):
    def __init__(self, channels: int, classes: int):
        super().__init__()
        root = official_root() / "CBraMod"
        with official_import_path(root):
            module = importlib.import_module("models.cbramod")
            base = module.CBraMod(in_dim=200, out_dim=200, d_model=200, dim_feedforward=800, seq_len=30, n_layer=12, nhead=8)
        candidates = (root / "pretrained_weights" / "pretrained_weights.pth",
                      Path(r"D:\nips-temp\TotalP\P2\fm_rescue_runtime\CBraMod\pretrained_weights\pretrained_weights.pth"))
        checkpoint_path = next((path for path in candidates if path.is_file()), None)
        if checkpoint_path is None: raise FileNotFoundError("official CBraMod checkpoint is absent")
        source = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.load_report = _copy_state(base, source)
        if self.load_report["nonpermitted_missing_model_keys"] or self.load_report["mismatched"]:
            raise RuntimeError("CBraMod released checkpoint does not exactly match its official architecture")
        base.proj_out = nn.Identity()
        self.model, self.head, self.checkpoint_path = base, nn.Linear(200, classes), checkpoint_path
        nn.init.trunc_normal_(self.head.weight, std=.02); nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 200: raise ValueError(f"CBraMod resampled length is not a 200-sample patch multiple: {x.shape}")
        value = x.reshape(x.shape[0], x.shape[1], x.shape[-1] // 200, 200)
        return self.head(self.model(value).mean(dim=(1, 2)))


def build_model(name: str, *, dataset: str, channels: int, samples: int, classes: int, tech_recipe: str | None = None) -> nn.Module:
    if name == "EEGNet": return EEGNet(channels, samples, classes)
    if name == "LiteBN": return LiteBN(channels, classes)
    if name == "TCFormer": return TCFormerAdapter(channels, classes)
    if name == "ST-EEGFormer-small": return STEEGFormerAdapter(dataset, channels, classes)
    if name == "LaBraM-base": return LaBraMAdapter(dataset, channels, classes)
    if name == "CBraMod": return CBraModAdapter(channels, classes)
    if name == "TeCh":
        if tech_recipe is None: raise ValueError("TeCh requires a frozen recipe")
        return TeChAdapter(recipe_config(channels=channels, samples=samples, classes=classes, recipe=tech_recipe))
    raise KeyError(name)


def admission_metadata(model: nn.Module) -> dict[str, Any]:
    result: dict[str, Any] = {"trainable_parameters": _trainable(model)}
    for name, value in model.named_modules():
        if name.endswith("model") and hasattr(value, "load_report"):
            result["load_report"] = getattr(value, "load_report")
    if hasattr(model, "load_report"): result["load_report"] = getattr(model, "load_report")
    if hasattr(model, "channel_mapping"): result["channel_mapping"] = getattr(model, "channel_mapping")
    if hasattr(model, "checkpoint_path"): result["checkpoint_path"] = str(getattr(model, "checkpoint_path"))
    return result
