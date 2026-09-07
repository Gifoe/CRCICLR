"""Train the same enlarged MI-specific EEGNet family on WBCIC."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, log_loss
from torch.utils.data import DataLoader

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from adapters.run_encoder_finetuning import _adapt, _load_wbcic_raw
from common import ABLATIONS, CACHE, DIAGNOSTICS, LEADERBOARD, PROTOCOL, RESEARCH_LOG, V6_SEED, logit, sigmoid, stable_seed, v5_output_root, wbcic_source_root, write_csv, write_json
from evaluation.metrics import summarize
from protocol.datasets import load_wbcic_fold


class LargeEEGNet(nn.Module):
    def __init__(self, dropout: float = 0.5):
        super().__init__()
        self.temporal = nn.Conv2d(1, 16, (1, 64), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.spatial = nn.Conv2d(16, 32, (58, 1), groups=16, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        self.separable_depth = nn.Conv2d(32, 32, (1, 16), padding="same", groups=32, bias=False)
        self.separable_point = nn.Conv2d(32, 32, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)
        self.embedding = nn.Linear(32 * 31, 64)
        self.embedding_norm = nn.LayerNorm(64)
        self.head = nn.Linear(64, 2)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        value = x.unsqueeze(1)
        value = self.bn1(self.temporal(value))
        value = self.drop1(self.pool1(F.elu(self.bn2(self.spatial(value)))))
        value = self.separable_depth(value)
        value = self.separable_point(value)
        value = self.drop2(self.pool2(F.elu(self.bn3(value))))
        return self.embedding_norm(F.elu(self.embedding(value.flatten(1))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


TARGET_CONFIGS = (
    {"strategy": "frozen", "lr": 0.0, "epochs": 0, "augment": False},
    {"strategy": "head", "lr": 3e-4, "epochs": 24, "augment": False},
    {"strategy": "tail", "lr": 1e-4, "epochs": 8, "augment": False},
    {"strategy": "full", "lr": 1e-4, "epochs": 8, "augment": False},
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _evaluate_loader(model, loader, device):
    model.eval()
    probabilities, labels, subject_indices = [], [], []
    with torch.inference_mode():
        for x, y, subject_index, _ in loader:
            probability = torch.softmax(model(x.to(device, non_blocking=True)), dim=1)[:, 1]
            probabilities.append(probability.cpu().numpy())
            labels.append(y.numpy())
            subject_indices.append(subject_index.numpy())
    return np.concatenate(probabilities), np.concatenate(labels), np.concatenate(subject_indices)


def _subject_ba(labels, probability, subject_indices):
    return float(np.mean([balanced_accuracy_score(labels[subject_indices == subject], probability[subject_indices == subject] >= 0.5) for subject in np.unique(subject_indices)]))


def _train(core, subjects, validation_subjects, device, seed, fixed_epochs=None):
    torch.manual_seed(seed)
    model = LargeEEGNet().to(device)
    train = core.EpochDataset(subjects, (0, 1, 2))
    train_loader = DataLoader(train, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)
    validation_loader = None
    if validation_subjects:
        validation = core.EpochDataset(validation_subjects, (2,))
        validation_loader = DataLoader(validation, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    maximum = int(fixed_epochs) if fixed_epochs is not None else 60
    best_state = None
    best_key = None
    best_epoch = maximum
    stale = 0
    history = []
    for epoch in range(maximum):
        model.train()
        total = 0.0
        count = 0
        for x, y, _, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(y)
            count += len(y)
        row = {"epoch": epoch + 1, "train_loss": total / max(count, 1)}
        if validation_loader is not None:
            probability, labels, subject_index = _evaluate_loader(model, validation_loader, device)
            row["validation_mean_subject_BA"] = _subject_ba(labels, probability, subject_index)
            key = (row["validation_mean_subject_BA"], -row["train_loss"], -(epoch + 1))
            if best_key is None or key > best_key:
                best_key = key
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                best_epoch = epoch + 1
                stale = 0
            else:
                stale += 1
        history.append(row)
        print(f"[WBCIC large EEGNet] epoch={epoch + 1}/{maximum} train_loss={row['train_loss']:.4f} validation_BA={row.get('validation_mean_subject_BA')}", flush=True)
        if validation_loader is not None and epoch + 1 >= 20 and stale >= 10:
            break
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    return model, best_epoch, history


def _predict(model, x, device):
    model.eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(x), 128):
            parts.append(torch.softmax(model(torch.as_tensor(x[start : start + 128], dtype=torch.float32, device=device)), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(parts)


def _v5():
    frame = pd.read_csv(v5_output_root() / "diagnostics" / "WBCIC_MULTI_SEED_OOF_PREDICTIONS.csv")
    frame = frame.loc[frame.seed.astype(int).eq(V6_SEED)].copy().rename(columns={"dataset": "benchmark"})
    frame["benchmark"] = "WBCIC_S1S2_to_S3_authorized_development"
    frame["method_id"] = "V5_CS_LGS_ANCHOR"
    frame["target_future_labels_used_for_fit"] = False
    return frame


def run() -> None:
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core_path = wbcic_source_root() / "experiments" / "persist_eeg_wbcic_actionability_v2" / "code" / "core.py"
    predictions, training_rows = [], []
    for fold in range(5):
        data = load_wbcic_fold(fold)
        core = _load_module(core_path, f"v6_large_wbcic_core_{fold}")
        model, best_epoch, history = _train(core, data.model_fit_subjects, data.discovery_subjects, device, stable_seed(V6_SEED, "WBCIC-large", fold))
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        discovery_raw = {subject: _load_wbcic_raw(subject) for subject in data.discovery_subjects}
        adaptation_rows = []
        for order, configuration in enumerate(TARGET_CONFIGS):
            records = []
            for subject, raw in discovery_raw.items():
                probability, _ = _adapt("wbcic", model, state, raw, configuration, device, stable_seed(V6_SEED, "WBCIC-large-adapt", fold, order, subject))
                records.append((raw.future_y, probability))
            ba = float(np.mean([balanced_accuracy_score(y, p >= 0.5) for y, p in records]))
            nll = float(np.mean([log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]) for y, p in records]))
            adaptation_rows.append({"order": order, "configuration": configuration, "discovery_BA": ba, "discovery_NLL": nll})
        selected_adaptation = max(adaptation_rows, key=lambda row: (row["discovery_BA"], -row["discovery_NLL"], -row["order"]))
        nonoutcome = tuple(data.model_fit_subjects) + tuple(data.discovery_subjects)
        refit, _, refit_history = _train(core, nonoutcome, (), device, stable_seed(V6_SEED, "WBCIC-large-refit", fold), fixed_epochs=best_epoch)
        refit_state = {name: value.detach().cpu().clone() for name, value in refit.state_dict().items()}
        checkpoint = CACHE / f"WBCIC_LARGE_EEGNET_FOLD_{fold}.pt"
        torch.save({"model": refit_state, "epochs": best_epoch, "adaptation": selected_adaptation["configuration"], "OUTER_TEST_USED": False}, checkpoint)
        for subject in data.outcome_subjects:
            raw = _load_wbcic_raw(subject)
            frozen_probability = _predict(refit, raw.future_x, device)
            adapted_probability, _ = _adapt("wbcic", refit, refit_state, raw, selected_adaptation["configuration"], device, stable_seed(V6_SEED, "WBCIC-large-outcome", fold, subject))
            for method_id, probability, history_used in (
                ("WBCIC_LARGE_EEGNET_FROZEN", frozen_probability, False),
                ("WBCIC_LARGE_EEGNET_TARGET_ADAPTED", adapted_probability, selected_adaptation["configuration"]["strategy"] != "frozen"),
            ):
                predictions.append(pd.DataFrame({"benchmark": data.benchmark, "method_id": method_id, "trial_uid": raw.future_uid, "subject_id": subject, "outer_fold": fold, "label": raw.future_y, "probability": probability, "prediction": (probability >= 0.5).astype(int), "target_history_labels_used": history_used, "target_future_labels_used_for_fit": False, "exploratory": True, "OUTER_TEST_USED": False}))
        training_rows.append({"outer_fold": fold, "best_epoch": best_epoch, "history": json.dumps(history), "refit_history": json.dumps(refit_history), "adaptation_candidates": json.dumps(adaptation_rows), "selected_adaptation": json.dumps(selected_adaptation["configuration"], sort_keys=True), "checkpoint": str(checkpoint), "OUTER_TEST_USED": False})
        print(f"[WBCIC large EEGNet] fold={fold} best_epoch={best_epoch} adaptation={selected_adaptation['configuration']}", flush=True)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    anchor = _v5()
    aligned_anchor = anchor.set_index("trial_uid")
    for method in ("WBCIC_LARGE_EEGNET_FROZEN", "WBCIC_LARGE_EEGNET_TARGET_ADAPTED"):
        part = prediction_frame.loc[prediction_frame.method_id.eq(method)].copy()
        anchor_probability = aligned_anchor.loc[part.trial_uid, "probability"].to_numpy(float)
        probability = sigmoid(0.5 * (logit(anchor_probability) + logit(part.probability.to_numpy(float))))
        part["method_id"] = "V5_FIXED_BLEND__" + method
        part["probability"] = probability
        part["prediction"] = (probability >= 0.5).astype(int)
        part["target_history_labels_used"] = True
        prediction_frame = pd.concat([prediction_frame, part], ignore_index=True)
    rows, subject_parts, fold_parts = [], [], []
    anchor_row, anchor_subjects, anchor_folds = summarize(anchor)
    rows.append(anchor_row); subject_parts.append(anchor_subjects); fold_parts.append(anchor_folds)
    for method in prediction_frame.method_id.unique():
        part = prediction_frame.loc[prediction_frame.method_id.eq(method)].copy()
        row, subjects, folds = summarize(part, reference=anchor)
        rows.append(row); subject_parts.append(subjects); fold_parts.append(folds)
    table = pd.DataFrame(rows).sort_values("mean_subject_BA", ascending=False)
    write_csv(LEADERBOARD / "WBCIC_LARGE_EEGNET.csv", table)
    write_csv(DIAGNOSTICS / "WBCIC_LARGE_EEGNET_TRAINING.csv", pd.DataFrame(training_rows))
    write_csv(DIAGNOSTICS / "WBCIC_LARGE_EEGNET_PREDICTIONS.csv", prediction_frame)
    write_csv(DIAGNOSTICS / "WBCIC_LARGE_EEGNET_SUBJECT_RESULTS.csv", pd.concat(subject_parts, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_LARGE_EEGNET_FOLD_RESULTS.csv", pd.concat(fold_parts, ignore_index=True))
    write_csv(ABLATIONS / "WBCIC_BACKBONE_CAPACITY_ABLATION.csv", table)
    write_json(PROTOCOL / "WBCIC_LARGE_EEGNET_AUDIT.json", {"population_fit": "model-fit subjects S1/S2/S3", "selection": "discovery subjects S3", "refit": "all non-outcome subjects S1/S2/S3 after epoch/adaptation selection", "outcome_adaptation": "S1/S2 labels only", "outcome_S3_labels_used_for_fit": False, "OUTER_TEST_USED": False})
    (RESEARCH_LOG / "ITERATION_006_WBCIC.md").write_text("# Iteration 006 — enlarged MI-specific EEGNet\n\nA 2x-width 64-D EEGNet is trained on legal model-fit history-to-future episodes and optionally adapted from target history.\n\n```text\n" + table.to_string(index=False) + "\n```\n", encoding="utf-8")
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
