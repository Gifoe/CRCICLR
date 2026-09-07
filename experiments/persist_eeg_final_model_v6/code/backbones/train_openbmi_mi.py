"""Train MI-specific population backbones and legal target-history adaptation."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from common import CACHE, DIAGNOSTICS, LEADERBOARD, PROTOCOL, RESEARCH_LOG, V6_SEED, sha256_file, stable_seed, stage0_root, write_csv, write_json
from evaluation.metrics import summarize
from protocol.datasets import OPENBMI_BEST_EPOCHS, load_openbmi_fold


class ManifestDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, subjects: Sequence[str], sessions: Sequence[int]):
        selected = manifest.subject_id.astype(str).isin(list(map(str, subjects))) & manifest.session_id.astype(int).isin(list(map(int, sessions)))
        self.rows = manifest.loc[selected].reset_index(drop=True)
        self.root = stage0_root()
        self.arrays: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[int(index)]
        path = str(row.signal_cache_path)
        if path not in self.arrays:
            self.arrays[path] = np.load(self.root / path, mmap_mode="r", allow_pickle=False)
        x = np.array(self.arrays[path][int(row.cache_index)], dtype=np.float32, copy=True)
        y = 0 if str(row.event_label) == "left_hand" else 1
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long), str(row.subject_id), str(row.trial_id)


class StandardEEGNet(nn.Module):
    def __init__(self, f1: int = 8, f2: int = 16, dropout: float = 0.25):
        super().__init__()
        self.temporal = nn.Conv2d(1, f1, (1, 64), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(f1)
        self.spatial = nn.Conv2d(f1, 2 * f1, (62, 1), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(2 * f1)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        self.depth = nn.Conv2d(2 * f1, 2 * f1, (1, 16), padding="same", groups=2 * f1, bias=False)
        self.point = nn.Conv2d(2 * f1, f2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        self.embedding = nn.Sequential(nn.Linear(f2 * 31, 64), nn.ELU(), nn.LayerNorm(64))
        self.head = nn.Linear(64, 2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.bn1(self.temporal(x))
        x = self.drop1(self.pool1(F.elu(self.bn2(self.spatial(x)))))
        x = self.drop2(self.pool2(F.elu(self.bn3(self.point(self.depth(x))))))
        return self.embedding(x.flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class ShallowConvNet(nn.Module):
    def __init__(self, filters: int = 40, dropout: float = 0.5):
        super().__init__()
        self.temporal = nn.Conv2d(1, filters, (1, 25), bias=False)
        self.spatial = nn.Conv2d(filters, filters, (62, 1), groups=1, bias=False)
        self.bn = nn.BatchNorm2d(filters)
        self.pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Sequential(nn.Linear(filters * 61, 64), nn.ELU(), nn.LayerNorm(64))
        self.head = nn.Linear(64, 2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.spatial(self.temporal(x.unsqueeze(1))))
        x = torch.square(x)
        x = torch.log(torch.clamp(self.pool(x), min=1e-6))
        return self.embedding(self.dropout(x).flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


ARCHITECTURES = (
    {"architecture": "eegnet", "f1": 8, "f2": 16, "dropout": 0.25, "lr": 3e-4, "weight_decay": 5e-4},
    {"architecture": "eegnet", "f1": 16, "f2": 32, "dropout": 0.5, "lr": 3e-4, "weight_decay": 5e-4},
    {"architecture": "shallow", "filters": 40, "dropout": 0.5, "lr": 1e-3, "weight_decay": 5e-4},
)


def build(configuration: dict[str, Any]) -> nn.Module:
    if configuration["architecture"] == "eegnet":
        return StandardEEGNet(int(configuration["f1"]), int(configuration["f2"]), float(configuration["dropout"]))
    return ShallowConvNet(int(configuration["filters"]), float(configuration["dropout"]))


def _normalizer(fold: int) -> tuple[np.ndarray, np.ndarray]:
    checkpoint = stage0_root() / "outputs" / "persist_eeg_p2p3" / "backbone" / "checkpoints" / "eegnet" / f"fold-{fold}" / "seed-0" / "trajectory" / f"epoch-{OPENBMI_BEST_EPOCHS[fold]:03d}.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return np.asarray(payload["channel_mean"], dtype=np.float32), np.asarray(payload["channel_std"], dtype=np.float32)


def _normalize(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean[None, :, None]) / torch.clamp(std[None, :, None], min=1e-8)


def _subject_ba(labels: np.ndarray, probability: np.ndarray, subjects: np.ndarray) -> float:
    return float(np.mean([balanced_accuracy_score(labels[subjects == subject], probability[subjects == subject] >= 0.5) for subject in np.unique(subjects)]))


def _evaluate(model, loader, device, mean, std):
    model.eval()
    probabilities, labels, subjects, trial_ids = [], [], [], []
    with torch.inference_mode():
        for x, y, subject, trial_id in loader:
            x = _normalize(x.to(device, non_blocking=True), mean, std)
            probability = torch.softmax(model(x), dim=1)[:, 1]
            probabilities.append(probability.cpu().numpy())
            labels.append(y.numpy())
            subjects.extend(map(str, subject))
            trial_ids.extend(map(str, trial_id))
    return np.concatenate(probabilities), np.concatenate(labels), np.asarray(subjects), np.asarray(trial_ids)


def _train(
    configuration: dict[str, Any],
    train_loader: DataLoader,
    validation_loader: DataLoader | None,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    seed: int,
    fixed_epochs: int | None = None,
) -> tuple[nn.Module, int, list[dict[str, float]]]:
    torch.manual_seed(seed)
    model = build(configuration).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(configuration["lr"]), weight_decay=float(configuration["weight_decay"]))
    best_state = None
    best_key = None
    best_epoch = -1
    history = []
    maximum = fixed_epochs if fixed_epochs is not None else 60
    patience = 12
    stale = 0
    for epoch in range(maximum):
        model.train()
        total_loss = 0.0
        seen = 0
        for x, y, _, _ in train_loader:
            x = _normalize(x.to(device, non_blocking=True), mean, std)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)
            seen += len(y)
        row = {"epoch": epoch + 1, "train_loss": total_loss / max(seen, 1)}
        if validation_loader is not None:
            probability, labels, subjects, _ = _evaluate(model, validation_loader, device, mean, std)
            row["validation_mean_subject_BA"] = _subject_ba(labels, probability, subjects)
            key = (row["validation_mean_subject_BA"], -row["train_loss"], -(epoch + 1))
            if best_key is None or key > best_key:
                best_key = key
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1
                stale = 0
            else:
                stale += 1
        history.append(row)
        if validation_loader is not None and stale >= patience and epoch + 1 >= 20:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        best_epoch = maximum
    return model, best_epoch, history


def _adapt(model: nn.Module, x: np.ndarray, y: np.ndarray, configuration: dict[str, Any], device: torch.device, mean: torch.Tensor, std: torch.Tensor, seed: int) -> nn.Module:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    adapted = copy.deepcopy(model).to(device)
    strategy = configuration["strategy"]
    if strategy == "frozen":
        return adapted
    for parameter in adapted.parameters():
        parameter.requires_grad_(False)
    for name, parameter in adapted.named_parameters():
        if strategy == "full" or (strategy == "head" and name.startswith("head")) or (strategy == "tail" and (name.startswith("embedding") or name.startswith("head"))):
            parameter.requires_grad_(True)
    parameters = [value for value in adapted.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(configuration["lr"]), weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(int(configuration["epochs"])):
        adapted.train()
        permutation = torch.randperm(len(y), generator=generator)
        for start in range(0, len(y), 32):
            index = permutation[start : start + 32]
            xb = _normalize(torch.as_tensor(x[index], dtype=torch.float32, device=device), mean, std)
            yb = torch.as_tensor(y[index], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(adapted(xb), yb)
            loss.backward()
            optimizer.step()
    return adapted


def _raw_subject(manifest: pd.DataFrame, subject: str):
    rows = manifest.loc[manifest.subject_id.astype(str).eq(subject)].sort_values(["session_id", "cache_index"])
    root = stage0_root()
    values, labels, sessions, uids = [], [], [], []
    for relative, group in rows.groupby("signal_cache_path", sort=False):
        source = np.load(root / str(relative), mmap_mode="r", allow_pickle=False)
        values.extend(np.asarray(source[group.cache_index.to_numpy(int)], dtype=np.float32))
        labels.extend(group.event_label.astype(str).map({"left_hand": 0, "right_hand": 1}).astype(int).tolist())
        sessions.extend(group.session_id.astype(int).tolist())
        uids.extend(("OpenBMI_nm000273_MI:" + group.trial_id.astype(str)).tolist())
    values = np.stack(values)
    labels = np.asarray(labels)
    sessions = np.asarray(sessions)
    uids = np.asarray(uids)
    return values[sessions == 1], labels[sessions == 1], values[sessions == 2], labels[sessions == 2], uids[sessions == 2]


def run() -> None:
    root = stage0_root()
    manifest = pd.read_parquet(root / "outputs" / "persist_eeg_stage0" / "manifests" / "openbmi_trials.parquet")
    manifest = manifest.loc[manifest.paradigm.eq("mi")].copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = []
    training_rows = []
    for fold in range(5):
        data = load_openbmi_fold(fold)
        mean_np, std_np = _normalizer(fold)
        mean = torch.as_tensor(mean_np, device=device)
        std = torch.as_tensor(std_np, device=device)
        train_dataset = ManifestDataset(manifest, data.model_fit_subjects, (1, 2))
        discovery_dataset = ManifestDataset(manifest, data.discovery_subjects, (2,))
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
        discovery_loader = DataLoader(discovery_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
        candidates = []
        for order, configuration in enumerate(ARCHITECTURES):
            model, best_epoch, history = _train(configuration, train_loader, discovery_loader, device, mean, std, stable_seed(V6_SEED, "MI-backbone", fold, order))
            probability, labels, subjects, _ = _evaluate(model, discovery_loader, device, mean, std)
            ba = _subject_ba(labels, probability, subjects)
            candidates.append({"order": order, "configuration": configuration, "best_epoch": best_epoch, "discovery_BA": ba, "model": model})
            training_rows.append({"outer_fold": fold, "candidate_order": order, "configuration": json.dumps(configuration, sort_keys=True), "best_epoch": best_epoch, "discovery_BA": ba, "history": json.dumps(history), "OUTER_TEST_USED": False})
            print(f"[MI backbone] fold={fold} candidate={order} BA={ba:.4f} epoch={best_epoch}", flush=True)
        selected = max(candidates, key=lambda row: (row["discovery_BA"], -row["order"]))
        raw_discovery = {subject: _raw_subject(manifest, subject) for subject in data.discovery_subjects}
        adaptation_configs = (
            {"strategy": "frozen", "lr": 0.0, "epochs": 0},
            {"strategy": "head", "lr": 1e-3, "epochs": 10},
            {"strategy": "head", "lr": 3e-4, "epochs": 30},
            {"strategy": "tail", "lr": 1e-4, "epochs": 10},
            {"strategy": "tail", "lr": 3e-4, "epochs": 10},
            {"strategy": "full", "lr": 1e-5, "epochs": 10},
            {"strategy": "full", "lr": 3e-5, "epochs": 10},
            {"strategy": "full", "lr": 3e-5, "epochs": 30},
        )
        adaptation_rows = []
        for order, configuration in enumerate(adaptation_configs):
            ba_values = []
            for subject, (hx, hy, fx, fy, _) in raw_discovery.items():
                adapted = _adapt(selected["model"], hx, hy, configuration, device, mean, std, stable_seed(V6_SEED, "MI-adapt", fold, order, subject))
                probability = _evaluate(adapted, DataLoader(ManifestDataset(manifest, [subject], (2,)), batch_size=128, shuffle=False), device, mean, std)[0]
                ba_values.append(balanced_accuracy_score(fy, probability >= 0.5))
            adaptation_rows.append({"order": order, "configuration": configuration, "discovery_BA": float(np.mean(ba_values))})
        selected_adaptation = max(adaptation_rows, key=lambda row: (row["discovery_BA"], -row["order"]))
        # Standard outer-CV refit: architecture/epoch/adaptation are fixed before
        # adding discovery subjects to the source-training set.
        refit_dataset = ManifestDataset(manifest, tuple(data.model_fit_subjects) + tuple(data.discovery_subjects), (1, 2))
        refit_loader = DataLoader(refit_dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
        refit_model, _, _ = _train(selected["configuration"], refit_loader, None, device, mean, std, stable_seed(V6_SEED, "MI-refit", fold), fixed_epochs=int(selected["best_epoch"]))
        checkpoint = CACHE / f"OPENBMI_MI_SPECIFIC_BACKBONE_FOLD_{fold}.pt"
        torch.save({"model": refit_model.state_dict(), "configuration": selected["configuration"], "epochs": selected["best_epoch"], "adaptation": selected_adaptation["configuration"], "OUTER_TEST_USED": False}, checkpoint)
        for subject in data.outcome_subjects:
            hx, hy, fx, fy, uid = _raw_subject(manifest, subject)
            adapted = _adapt(refit_model, hx, hy, selected_adaptation["configuration"], device, mean, std, stable_seed(V6_SEED, "MI-outcome-adapt", fold, subject))
            adapted.eval()
            probabilities = []
            with torch.inference_mode():
                for start in range(0, len(fx), 128):
                    xb = _normalize(torch.as_tensor(fx[start : start + 128], dtype=torch.float32, device=device), mean, std)
                    probabilities.append(torch.softmax(adapted(xb), dim=1)[:, 1].cpu().numpy())
            probability = np.concatenate(probabilities)
            predictions.append(pd.DataFrame({"benchmark": data.benchmark, "method_id": "MI_SPECIFIC_BACKBONE_ADAPTED", "trial_uid": uid, "subject_id": subject, "outer_fold": fold, "label": fy, "probability": probability, "prediction": (probability >= 0.5).astype(int), "target_history_labels_used": selected_adaptation["configuration"]["strategy"] != "frozen", "target_future_labels_used_for_fit": False, "exploratory": True, "OUTER_TEST_USED": False}))
        training_rows.append({"outer_fold": fold, "candidate_order": "selected", "configuration": json.dumps(selected["configuration"], sort_keys=True), "best_epoch": selected["best_epoch"], "discovery_BA": selected["discovery_BA"], "adaptation": json.dumps(selected_adaptation["configuration"], sort_keys=True), "adaptation_discovery_BA": selected_adaptation["discovery_BA"], "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint), "OUTER_TEST_USED": False})
        print(f"[MI backbone] fold={fold} selected={selected['configuration']['architecture']} adapt={selected_adaptation['configuration']} BA={selected_adaptation['discovery_BA']:.4f}", flush=True)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    baseline = pd.read_csv(DIAGNOSTICS / "OPENBMI_BASELINE_PREDICTIONS.csv")
    reference = baseline.loc[baseline.method_id.eq("B_HISTORY_FUSION_LDA")].copy()
    row, subjects, folds = summarize(prediction_frame, reference=reference)
    table = pd.DataFrame([row])
    write_csv(LEADERBOARD / "OPENBMI_MI_SPECIFIC_BACKBONE.csv", table)
    write_csv(DIAGNOSTICS / "OPENBMI_MI_SPECIFIC_BACKBONE_TRAINING.csv", pd.DataFrame(training_rows))
    write_csv(DIAGNOSTICS / "OPENBMI_MI_SPECIFIC_BACKBONE_PREDICTIONS.csv", prediction_frame)
    write_csv(DIAGNOSTICS / "OPENBMI_MI_SPECIFIC_BACKBONE_SUBJECT_RESULTS.csv", subjects)
    write_csv(DIAGNOSTICS / "OPENBMI_MI_SPECIFIC_BACKBONE_FOLD_RESULTS.csv", folds)
    write_json(PROTOCOL / "OPENBMI_MI_SPECIFIC_BACKBONE_AUDIT.json", {"population_model_fit": "model-fit subjects S1/S2", "architecture_epoch_selection": "discovery subjects S2", "outer_refit": "model-fit plus discovery after selection", "target_adaptation": "outcome subject S1 only", "target_future_labels_used_for_fit": False, "exploratory": True, "OUTER_TEST_USED": False})
    (RESEARCH_LOG / "ITERATION_003_OPENBMI.md").write_text("# Iteration 003 — MI-specific backbone\n\nThe multitask global-pooling representation was replaced by MI-specific standard EEGNet and shallow-conv candidates, followed by legal S1 target adaptation.\n\n```text\n" + table.to_string(index=False) + "\n```\n", encoding="utf-8")
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
