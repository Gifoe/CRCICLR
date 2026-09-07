"""Evaluate structurally distinct MI-specific population backbones."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from backbones import train_ea_eegnet as core
from common import CACHE, DIAGNOSTICS, LEADERBOARD, PROTOCOL, V7_SEED, ensure_directories, logit, sigmoid, stable_seed, write_csv, write_json
from protocol.datasets import load_fold


class FBCVarianceNet(nn.Module):
    """Learned filter-bank spatial conv with log-variance temporal statistics."""

    learning_rate = 1e-3
    batch_size = 512
    label_smoothing = 0.05

    def __init__(self, channels: int, channel_mean: np.ndarray | None = None, channel_std: np.ndarray | None = None):
        super().__init__()
        self.population_normalized = channel_mean is not None and channel_std is not None
        if self.population_normalized:
            self.batch_size = 256
            self.register_buffer("channel_mean", torch.as_tensor(channel_mean, dtype=torch.float32).view(1, channels, 1))
            self.register_buffer("channel_std", torch.as_tensor(channel_std, dtype=torch.float32).view(1, channels, 1))
            kernels = (33, 65, 129)
        else:
            self.register_buffer("channel_mean", torch.zeros(1, channels, 1))
            self.register_buffer("channel_std", torch.ones(1, channels, 1))
            kernels = (17, 33, 65)
        self.temporal_short = nn.Conv2d(1, 8, (1, kernels[0]), padding="same", bias=False)
        self.temporal_mid = nn.Conv2d(1, 8, (1, kernels[1]), padding="same", bias=False)
        self.temporal_long = nn.Conv2d(1, 8, (1, kernels[2]), padding="same", bias=False)
        self.temporal_norm = nn.BatchNorm2d(24)
        self.spatial = nn.Conv2d(24, 48, (channels, 1), groups=24, bias=False)
        self.spatial_norm = nn.BatchNorm2d(48)
        self.dropout = nn.Dropout(0.5)
        self.feature_norm = nn.LayerNorm(48 * 10)
        self.embedding = nn.Sequential(
            nn.Linear(48 * 10, 128),
            nn.ELU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.LayerNorm(64),
        )
        self.head = nn.Linear(64, 2)

    def front_end(self, x: torch.Tensor) -> torch.Tensor:
        if self.population_normalized:
            value = (x - self.channel_mean) / torch.clamp(self.channel_std, min=1e-8)
        else:
            value = x[..., ::2]
            value = value - value.mean(dim=-1, keepdim=True)
            energy = torch.sqrt(torch.clamp(torch.mean(torch.square(value), dim=(-2, -1), keepdim=True), min=1e-10))
            value = value / energy
        if self.training:
            keep = (torch.rand(value.shape[0], value.shape[1], 1, device=value.device) >= 0.03).to(value.dtype)
            value = value * keep
        return value

    def forward_features(self, x: torch.Tensor, subject_index: torch.Tensor) -> torch.Tensor:
        del subject_index
        value = self.front_end(x).unsqueeze(1)
        value = torch.cat((self.temporal_short(value), self.temporal_mid(value), self.temporal_long(value)), dim=1)
        value = self.temporal_norm(value)
        value = F.elu(self.spatial_norm(self.spatial(value))).squeeze(2)
        value = self.dropout(value)
        # Exactly ten non-overlapping windows at the 125 Hz effective rate.
        value = value.reshape(value.shape[0], value.shape[1], 10, value.shape[-1] // 10)
        variance = torch.var(value, dim=-1, unbiased=False)
        statistic = torch.log(torch.clamp(variance, min=1e-6)).flatten(1)
        return self.embedding(self.feature_norm(statistic))

    def forward(self, x: torch.Tensor, subject_index: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x, subject_index))


class CompactEEGConformer(nn.Module):
    learning_rate = 3e-4
    batch_size = 128
    label_smoothing = 0.05

    def __init__(self, channels: int, channel_mean: np.ndarray | None = None, channel_std: np.ndarray | None = None):
        super().__init__()
        self.population_normalized = channel_mean is not None and channel_std is not None
        if self.population_normalized:
            self.batch_size = 64
            self.register_buffer("channel_mean", torch.as_tensor(channel_mean, dtype=torch.float32).view(1, channels, 1))
            self.register_buffer("channel_std", torch.as_tensor(channel_std, dtype=torch.float32).view(1, channels, 1))
        else:
            self.register_buffer("channel_mean", torch.zeros(1, channels, 1))
            self.register_buffer("channel_std", torch.ones(1, channels, 1))
        self.temporal = nn.Conv2d(1, 40, (1, 25), bias=False)
        self.spatial = nn.Conv2d(40, 40, (channels, 1), bias=False)
        self.norm = nn.BatchNorm2d(40)
        self.pool = nn.AvgPool2d((1, 25), stride=(1, 10))
        self.dropout = nn.Dropout(0.4)
        layer = nn.TransformerEncoderLayer(
            d_model=40,
            nhead=4,
            dim_feedforward=160,
            dropout=0.3,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2, norm=nn.LayerNorm(40))
        self.position = nn.Parameter(torch.zeros(1, 100, 40))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.embedding = nn.Sequential(nn.Linear(40, 64), nn.ELU(), nn.LayerNorm(64))
        self.head = nn.Linear(64, 2)

    def front_end(self, x: torch.Tensor) -> torch.Tensor:
        if self.population_normalized:
            value = (x - self.channel_mean) / torch.clamp(self.channel_std, min=1e-8)
        else:
            value = x[..., ::2]
            value = value - value.mean(dim=-1, keepdim=True)
            energy = torch.sqrt(torch.clamp(torch.mean(torch.square(value), dim=(-2, -1), keepdim=True), min=1e-10))
            value = value / energy
        if self.training:
            keep = (torch.rand(value.shape[0], value.shape[1], 1, device=value.device) >= 0.03).to(value.dtype)
            value = value * keep
        return value

    def forward_features(self, x: torch.Tensor, subject_index: torch.Tensor) -> torch.Tensor:
        del subject_index
        value = self.front_end(x).unsqueeze(1)
        value = self.dropout(self.pool(F.elu(self.norm(self.spatial(self.temporal(value))))))
        tokens = value.squeeze(2).transpose(1, 2)
        tokens = tokens + self.position[:, :tokens.shape[1]]
        tokens = self.transformer(tokens)
        return self.embedding(tokens.mean(dim=1))

    def forward(self, x: torch.Tensor, subject_index: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x, subject_index))


def build(architecture: str, channels: int, channel_mean: np.ndarray | None = None, channel_std: np.ndarray | None = None) -> nn.Module:
    if architecture == "fbcvariance":
        return FBCVarianceNet(channels)
    if architecture == "fbcvariance_norm":
        return FBCVarianceNet(channels, channel_mean, channel_std)
    if architecture == "conformer":
        return CompactEEGConformer(channels)
    if architecture == "conformer_norm":
        return CompactEEGConformer(channels, channel_mean, channel_std)
    if architecture == "conformer_ccalign":
        model = CompactEEGConformer(channels, channel_mean, channel_std)
        model.session_alignment_weight = 0.2
        return model
    raise ValueError(architecture)


def _channel_statistics(raw: torch.Tensor, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total = torch.zeros(raw.shape[1], dtype=torch.float64, device=raw.device)
    square = torch.zeros_like(total)
    count = 0
    tensor_indices = torch.as_tensor(indices, dtype=torch.long, device=raw.device)
    for start in range(0, len(tensor_indices), 256):
        value = raw[tensor_indices[start:start + 256]].float()
        total += value.sum(dim=(0, 2), dtype=torch.float64)
        square += torch.square(value).sum(dim=(0, 2), dtype=torch.float64)
        count += value.shape[0] * value.shape[2]
    mean = total / count
    variance = torch.clamp(square / count - torch.square(mean), min=1e-16)
    return mean.float().cpu().numpy(), torch.sqrt(variance).float().cpu().numpy()


def run(benchmark: str, architecture: str, epochs: int, folds: tuple[int, ...]) -> None:
    ensure_directories()
    raw_path, metadata_path, _ = core._paths(benchmark)
    raw_disk = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    metadata = pd.read_parquet(metadata_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    raw = torch.from_numpy(np.asarray(raw_disk)).to(device) if device.type == "cuda" else raw_disk
    prefix = "OPENBMI" if benchmark == "openbmi" else "WBCIC"
    method_prefix = architecture.upper()
    anchor, anchor_method = core._strong_anchor(benchmark)
    anchor_index = anchor.set_index("trial_uid")
    predictions = [anchor]
    training_rows = []
    for fold in folds:
        data = load_fold(benchmark, fold)
        nonoutcome = set(data.nonoutcome_subjects)
        train_indices = np.flatnonzero(metadata.subject_id.astype(str).isin(nonoutcome).to_numpy())
        if architecture.endswith("_norm") or architecture == "conformer_ccalign":
            channel_mean, channel_std = _channel_statistics(raw, train_indices)
        else:
            channel_mean = channel_std = None
        model = build(architecture, int(raw.shape[1]), channel_mean, channel_std).to(device)
        history = core._train(model, raw, metadata, train_indices, epochs, device, stable_seed(V7_SEED, benchmark, architecture, fold))
        features, logits = core._extract(model, raw, metadata, device)
        cache_prefix = CACHE / f"{prefix}_{method_prefix}_FOLD_{fold}"
        np.save(cache_prefix.with_name(cache_prefix.name + "_FEATURES.npy"), features, allow_pickle=False)
        np.save(cache_prefix.with_name(cache_prefix.name + "_LOGITS.npy"), logits, allow_pickle=False)
        fold_metadata = metadata.copy()
        fold_metadata["outer_fold"] = fold
        fold_metadata.to_parquet(cache_prefix.with_name(cache_prefix.name + "_METADATA.parquet"), index=False)
        torch.save({
            "model": model.state_dict(),
            "benchmark": benchmark,
            "architecture": architecture,
            "epochs": epochs,
            "nonoutcome_subjects": sorted(nonoutcome),
            "target_future_labels_used_for_fit": False,
            "OUTER_TEST_USED": False,
        }, cache_prefix.with_suffix(".pt"))
        for subject in data.outcome_subjects:
            subject_mask = metadata.subject_id.astype(str).eq(str(subject)).to_numpy()
            history_mask = subject_mask & metadata.session_id.astype(int).isin(data.history_sessions).to_numpy()
            future_mask = subject_mask & metadata.session_id.astype(int).eq(data.future_session).to_numpy()
            history_y = metadata.loc[history_mask, "label"].to_numpy(int)
            frozen_logit = logits[future_mask].astype(float)
            head_logit = core._subject_head(features[history_mask], history_y, features[future_mask])
            calibrated_logit = core._calibration(logits[history_mask].astype(float), history_y, frozen_logit)
            values = {
                f"{method_prefix}_FROZEN": frozen_logit,
                f"{method_prefix}_FIXED_HEAD": 0.5 * (frozen_logit + head_logit),
                f"{method_prefix}_FIXED_CALIBRATION": 0.5 * (frozen_logit + calibrated_logit),
            }
            anchor_probability = anchor_index.loc[metadata.loc[future_mask, "trial_uid"].astype(str), "probability"].to_numpy(float)
            anchor_logit = logit(anchor_probability)
            for method_id, value in list(values.items()):
                values[f"ANCHOR_BLEND__{method_id}"] = 0.5 * (anchor_logit + value)
            for method_id, value in values.items():
                predictions.append(core._frame(data, subject, metadata, future_mask, sigmoid(value), method_id, "FIXED_HEAD" in method_id or "CALIBRATION" in method_id))
        training_rows.append({
            "benchmark": data.benchmark,
            "outer_fold": fold,
            "architecture": architecture,
            "epochs": epochs,
            "last_train_loss": history[-1]["train_loss"],
            "history": json.dumps(history),
            "OUTER_TEST_USED": False,
        })
        print(f"[{benchmark} {architecture}] fold={fold} loss={history[-1]['train_loss']:.4f}", flush=True)
    predictions = pd.concat(predictions, ignore_index=True)
    if folds != (0, 1, 2, 3, 4):
        allowed = set().union(*(set(load_fold(benchmark, fold).outcome_subjects) for fold in folds))
        predictions = predictions.loc[predictions.subject_id.astype(str).isin(allowed)].copy()
    leaderboard, subjects = core._leaderboard(predictions, anchor_method)
    tag = f"{prefix}_{method_prefix}"
    write_csv(LEADERBOARD / f"{tag}.csv", leaderboard)
    write_csv(DIAGNOSTICS / f"{tag}_PREDICTIONS.csv", predictions)
    write_csv(DIAGNOSTICS / f"{tag}_SUBJECT_RESULTS.csv", subjects)
    write_csv(DIAGNOSTICS / f"{tag}_TRAINING.csv", pd.DataFrame(training_rows))
    write_json(PROTOCOL / f"{tag}_AUDIT.json", {
        "architecture": architecture,
        "population_fit": "all sessions from non-outcome subjects",
        "outcome_adaptation": "fixed history-only head/calibration",
        "outcome_future_labels_used_for_fit_or_selection": False,
        "WBCIC_outer_split_opened": False,
        "OUTER_TEST_USED": False,
    })
    print(leaderboard.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("openbmi", "wbcic"), required=True)
    parser.add_argument("--architecture", choices=("fbcvariance", "fbcvariance_norm", "conformer", "conformer_norm", "conformer_ccalign"), required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--fold", type=int, choices=range(5))
    args = parser.parse_args()
    run(args.benchmark, args.architecture, args.epochs, (args.fold,) if args.fold is not None else (0, 1, 2, 3, 4))


if __name__ == "__main__":
    main()
