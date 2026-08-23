"""Strict unseen-subject vanilla EEGNet re-establishment for OpenBMI MI.

This runner deliberately uses only the authorized V8_SEARCH subjects.  It
does not use target S1 trials, target labels, adaptation, population heads, or
any of the cached representation experiments.  The raw epoch cache is read
only after the 14-subject V8 internal holdout has been removed from the row
index.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss


ROOT = Path(os.environ.get("FOLD_CORRECT_EXPERIMENT", ".")).resolve()
V8_ROOT = Path(os.environ.get(
    "PERSIST_V8_RUNTIME",
    r"D:\nips-temp\TotalP\P1\CRCICLR_V8_HEADROOM_FIRST",
))
V7_ROOT = Path(os.environ.get(
    "PERSIST_V7_RUNTIME",
    r"D:\nips-temp\TotalP\P1\CRCICLR_V7_FUTURE_UTILITY_META",
))
STAGE0_ROOT = Path(os.environ.get(
    "PERSIST_STAGE0_REPO",
    r"D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full",
))
VENDOR = Path(os.environ.get(
    "PERSIST_PYARROW_VENDOR",
    r"D:\nips-temp\TotalP\P1\CRCICLR_V3_WORK\vendor",
))
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))
import pyarrow.parquet as pq


SEEDS = (0, 1, 2)
VARIANTS = {
    "S1S2_SOURCE_TO_S2": (1, 2),
    "S1_ONLY_SOURCE_TO_S2": (1,),
}
MAX_EPOCHS = 60
PATIENCE = 12
MIN_EPOCHS = 20
BATCH_SIZE = 512
BASE_SEED = 20260823
HISTORICAL_BA = 0.75297


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StandardEEGNet(nn.Module):
    """The exact project MI EEGNet used by the historical backbone code.

    Input is (batch, 62 channels, 1000 samples).  This is the basic EEGNet
    implementation, with no EA, subject index, P/U/D head, or adaptation.
    """

    def __init__(self, channels: int = 62, samples: int = 1000, dropout: float = 0.25):
        super().__init__()
        if channels != 62 or samples != 1000:
            raise ValueError(f"locked OpenBMI shape expected (62,1000), got ({channels},{samples})")
        f1, d, f2 = 8, 2, 16
        self.temporal = nn.Conv2d(1, f1, (1, 64), padding="same", bias=False)
        self.bn1 = nn.BatchNorm2d(f1)
        self.spatial = nn.Conv2d(f1, f1 * d, (channels, 1), groups=f1, bias=False)
        self.bn2 = nn.BatchNorm2d(f1 * d)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)
        self.depth = nn.Conv2d(f1 * d, f1 * d, (1, 16), padding="same", groups=f1 * d, bias=False)
        self.point = nn.Conv2d(f1 * d, f2, 1, bias=False)
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


def load_protocol() -> tuple[dict, dict, pd.DataFrame, torch.Tensor]:
    split_path = V8_ROOT / "experiments" / "persist_eeg_final_model_v8" / "outputs" / "protocol" / "V8_SEARCH_SPLIT.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    assert split["OUTER_TEST_USED"] is False
    search = set(map(str, split["openbmi"]["V8_SEARCH"]))
    holdout = set(map(str, split["openbmi"]["V8_INTERNAL_HOLDOUT"]))
    assert len(search) == 40 and len(holdout) == 14 and not search & holdout
    freeze_path = STAGE0_ROOT / "delivery" / "persist_eeg_stage0" / "SPLIT_FREEZE.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8-sig"))
    folds = freeze["openbmi"]["folds"]
    test_union = set()
    test_intersections = set()
    for k, row in enumerate(folds):
        test = set(map(str, row["outer_test_subjects"])) & search
        assert test
        assert not test_union & test
        test_union |= test
    assert test_union == search
    manifest_path = V7_ROOT / "experiments" / "persist_eeg_final_model_v7" / "outputs" / "cache" / "OPENBMI_RAW_METADATA.parquet"
    metadata = pq.read_table(manifest_path).to_pandas()
    metadata["subject_id"] = metadata.subject_id.astype(str)
    metadata["session_id"] = metadata.session_id.astype(int)
    metadata["label"] = metadata.label.astype(int)
    keep = metadata.subject_id.isin(search).to_numpy()
    metadata = metadata.loc[keep].reset_index(drop=True)
    assert len(metadata) == 8_000
    assert metadata.groupby(["subject_id", "session_id"]).size().eq(100).all()
    raw_path = V7_ROOT / "experiments" / "persist_eeg_final_model_v7" / "outputs" / "cache" / "OPENBMI_RAW_EPOCHS_FLOAT16.npy"
    raw_disk = np.load(raw_path, mmap_mode="r", allow_pickle=False)
    assert raw_disk.shape == (10_800, 62, 1_000)
    # The advanced index is formed before materialising the tensor, so no
    # internal-holdout signal is copied into the GPU tensor.
    raw_search_np = np.asarray(raw_disk[keep], dtype=np.float32)
    assert raw_search_np.shape == (8_000, 62, 1_000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.from_numpy(raw_search_np).to(device, non_blocking=True)
    del raw_search_np, raw_disk
    return split, {"folds": folds, "search": sorted(search, key=lambda x: int(x)), "holdout": sorted(holdout, key=lambda x: int(x))}, metadata, raw


def subject_ba(y: np.ndarray, p: np.ndarray, subjects: np.ndarray) -> float:
    values = []
    for subject in sorted(np.unique(subjects).tolist(), key=lambda x: int(x)):
        mask = subjects == subject
        values.append(balanced_accuracy_score(y[mask], p[mask] >= 0.5))
    return float(np.mean(values)) if values else float("nan")


def evaluate(model: nn.Module, raw: torch.Tensor, indices: np.ndarray, labels: np.ndarray, subjects: np.ndarray, mean: torch.Tensor, std: torch.Tensor, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    if len(indices) == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=int)
    model.eval()
    probs, logits = [], []
    with torch.inference_mode():
        for start in range(0, len(indices), BATCH_SIZE):
            idx = torch.as_tensor(indices[start:start + BATCH_SIZE], dtype=torch.long, device=device)
            xb = (raw[idx] - mean[None, :, None]) / torch.clamp(std[None, :, None], min=1e-6)
            out = model(xb)
            logits.append(out.detach().float().cpu().numpy())
    output = np.concatenate(logits, axis=0)
    probability = torch.softmax(torch.from_numpy(output), dim=1)[:, 1].numpy()
    return probability.astype(float), output.astype(float)


def train_one(model: nn.Module, raw: torch.Tensor, train_idx: np.ndarray, val_idx: np.ndarray, labels: np.ndarray, subjects: np.ndarray, mean: torch.Tensor, std: torch.Tensor, seed: int, device: torch.device) -> tuple[nn.Module, int, list[dict]]:
    set_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train_tensor = torch.as_tensor(train_idx, dtype=torch.long, device=device)
    best_state = None
    best_key = None
    best_epoch = MAX_EPOCHS
    stale = 0
    history = []
    for epoch in range(MAX_EPOCHS):
        model.train()
        permutation = train_tensor[torch.randperm(len(train_tensor), generator=generator, device="cpu").to(device)]
        total_loss, seen = 0.0, 0
        for start in range(0, len(permutation), BATCH_SIZE):
            idx = permutation[start:start + BATCH_SIZE]
            cpu_idx = idx.detach().cpu().numpy()
            xb = (raw[idx] - mean[None, :, None]) / torch.clamp(std[None, :, None], min=1e-6)
            yb = torch.as_tensor(labels[cpu_idx], dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(idx)
            seen += len(idx)
        val_prob, val_logits = evaluate(model, raw, val_idx, labels[val_idx], subjects[val_idx], mean, std, device)
        val_ba = subject_ba(labels[val_idx], val_prob, subjects[val_idx])
        val_nll = float(log_loss(labels[val_idx], np.clip(val_prob, 1e-7, 1 - 1e-7), labels=[0, 1]))
        row = {"epoch": epoch + 1, "train_loss": total_loss / max(seen, 1), "validation_mean_subject_BA": val_ba, "validation_NLL": val_nll}
        history.append(row)
        key = (val_ba, -val_nll, -(epoch + 1))
        if best_key is None or key > best_key:
            best_key = key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        if epoch + 1 >= MIN_EPOCHS and stale >= PATIENCE:
            break
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[vanilla] seed={seed} epoch={epoch + 1} train_loss={row['train_loss']:.4f} val_BA={val_ba:.4f}", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, history


def bootstrap_ci(values: Iterable[float], seed: int, n_boot: int = 5000) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    rng = np.random.default_rng(seed)
    sample = rng.choice(array, size=(n_boot, len(array)), replace=True).mean(axis=1)
    return float(np.percentile(sample, 2.5)), float(np.percentile(sample, 97.5))


def run_variant(variant: str, sessions: tuple[int, ...], metadata: pd.DataFrame, raw: torch.Tensor, folds: list[dict], search: list[str], device: torch.device, out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subject_rows, fold_rows, seed_rows = [], [], []
    labels_all = metadata.label.to_numpy(int)
    subjects_all = metadata.subject_id.to_numpy(str)
    sessions_all = metadata.session_id.to_numpy(int)
    for seed in SEEDS:
        for fold_id, fold in enumerate(folds):
            train_subjects = set(map(str, fold["train_subjects"])) & set(search)
            validation_subjects = set(map(str, fold["validation_subjects"])) & set(search)
            outcome_subjects = set(map(str, fold["outer_test_subjects"])) & set(search)
            train_mask = np.isin(subjects_all, list(train_subjects)) & np.isin(sessions_all, list(sessions))
            val_mask = np.isin(subjects_all, list(validation_subjects)) & np.isin(sessions_all, list(sessions))
            test_mask = np.isin(subjects_all, list(outcome_subjects)) & (sessions_all == 2)
            assert not np.any(train_mask & val_mask) and not np.any(train_mask & test_mask) and not np.any(val_mask & test_mask)
            train_idx, val_idx, test_idx = np.flatnonzero(train_mask), np.flatnonzero(val_mask), np.flatnonzero(test_mask)
            assert len(train_idx) > 0 and len(val_idx) > 0 and len(test_idx) == len(outcome_subjects) * 100
            # Fit all population statistics on the train subjects only.
            mean_np = np.asarray(raw[torch.as_tensor(train_idx, device=device)].mean(dim=(0, 2)).detach().cpu(), dtype=np.float32)
            std_np = np.asarray(raw[torch.as_tensor(train_idx, device=device)].std(dim=(0, 2), unbiased=False).detach().cpu(), dtype=np.float32)
            std_np = np.maximum(std_np, 1e-6)
            mean, std = torch.as_tensor(mean_np, device=device), torch.as_tensor(std_np, device=device)
            model = StandardEEGNet().to(device)
            run_seed = BASE_SEED + 10000 * SEEDS.index(seed) + 100 * (0 if variant == "S1S2_SOURCE_TO_S2" else 1) + fold_id
            model, best_epoch, history = train_one(model, raw, train_idx, val_idx, labels_all, subjects_all, mean, std, run_seed, device)
            probability, logits = evaluate(model, raw, test_idx, labels_all[test_idx], subjects_all[test_idx], mean, std, device)
            y = labels_all[test_idx]
            sids = subjects_all[test_idx]
            for subject in sorted(np.unique(sids).tolist(), key=lambda x: int(x)):
                mask = sids == subject
                p = probability[mask]
                yy = y[mask]
                pred = (p >= 0.5).astype(int)
                subject_rows.append({
                    "variant": variant, "seed": seed, "fold": fold_id, "subject_id": subject,
                    "BA": float(balanced_accuracy_score(yy, pred)), "macro_F1": float(f1_score(yy, pred, average="macro")),
                    "accuracy": float(accuracy_score(yy, pred)), "NLL": float(log_loss(yy, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1])),
                    "source_sessions": "+".join(map(str, sessions)), "target_S1_used": False, "target_adaptation": False,
                    "strict_unseen_subject": True, "best_epoch": best_epoch, "OUTER_TEST_USED": False,
                })
            fold_part = [r for r in subject_rows if r["variant"] == variant and r["seed"] == seed and r["fold"] == fold_id]
            fold_rows.append({
                "variant": variant, "seed": seed, "fold": fold_id, "subjects": len(fold_part),
                "mean_BA": float(np.mean([r["BA"] for r in fold_part])), "macro_F1": float(np.mean([r["macro_F1"] for r in fold_part])),
                "accuracy": float(np.mean([r["accuracy"] for r in fold_part])), "NLL": float(np.mean([r["NLL"] for r in fold_part])),
                "best_epoch": best_epoch, "train_subjects": len(train_subjects), "validation_subjects": len(validation_subjects),
                "test_subjects": len(outcome_subjects), "OUTER_TEST_USED": False,
            })
            ckpt = out / "checkpoints" / variant / f"seed-{seed}" / f"fold-{fold_id}.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "seed": seed, "fold": fold_id, "variant": variant, "source_sessions": sessions, "best_epoch": best_epoch, "train_subjects": sorted(train_subjects), "validation_subjects": sorted(validation_subjects), "target_subjects": sorted(outcome_subjects), "target_S1_used": False, "target_adaptation": False, "OUTER_TEST_USED": False}, ckpt)
            print(f"[vanilla] variant={variant} seed={seed} fold={fold_id} test_BA={fold_rows[-1]['mean_BA']:.4f} best_epoch={best_epoch}", flush=True)
    subject_frame = pd.DataFrame(subject_rows)
    fold_frame = pd.DataFrame(fold_rows)
    for seed in SEEDS:
        part = subject_frame.loc[(subject_frame.variant == variant) & (subject_frame.seed == seed)]
        ci = bootstrap_ci(part.BA, BASE_SEED + seed)
        seed_rows.append({
            "variant": variant, "seed": seed, "subjects": len(part), "mean_BA": float(part.BA.mean()),
            "median_BA": float(part.BA.median()), "SD_BA": float(part.BA.std(ddof=1)), "macro_F1": float(part.macro_F1.mean()),
            "accuracy": float(part.accuracy.mean()), "NLL": float(part.NLL.mean()), "bootstrap_CI95_L": ci[0], "bootstrap_CI95_U": ci[1],
            "worst_subject_BA": float(part.BA.min()), "best_subject_BA": float(part.BA.max()), "target_S1_used": False,
            "target_adaptation": False, "strict_unseen_subject": True, "OUTER_TEST_USED": False,
        })
    return subject_frame, fold_frame, pd.DataFrame(seed_rows)


def main() -> None:
    out = ROOT
    for name in ("results", "protocol", "checkpoints"):
        (out / name).mkdir(parents=True, exist_ok=True)
    split, roles, metadata, raw = load_protocol()
    device = raw.device
    folds = roles["folds"]
    subjects, folds_all, seeds_all = [], [], []
    for variant, sessions in VARIANTS.items():
        sub, fold, seed = run_variant(variant, sessions, metadata, raw, folds, roles["search"], device, out)
        subjects.append(sub); folds_all.append(fold); seeds_all.append(seed)
    subject_frame = pd.concat(subjects, ignore_index=True)
    fold_frame = pd.concat(folds_all, ignore_index=True)
    seed_frame = pd.concat(seeds_all, ignore_index=True)
    summary_rows = []
    for variant in VARIANTS:
        part = seed_frame.loc[seed_frame.variant == variant]
        subject_mean = subject_frame.loc[subject_frame.variant == variant].groupby("subject_id").BA.mean()
        ci = bootstrap_ci(subject_mean, BASE_SEED + 991)
        summary_rows.append({
            "Method": "Fresh vanilla EEGNet source S1+S2 -> target S2" if variant == "S1S2_SOURCE_TO_S2" else "Fresh vanilla EEGNet source S1 only -> target S2",
            "variant": variant, "Subjects": 40, "Source sessions": "+".join(map(str, VARIANTS[variant])), "Target S1 used?": False,
            "Target adaptation?": False, "Strict unseen subject?": True, "Seed": "0,1,2", "Mean BA": float(part.mean_BA.mean()),
            "SD across seeds": float(part.mean_BA.std(ddof=1)), "Macro-F1": float(part.macro_F1.mean()), "Accuracy": float(part.accuracy.mean()),
            "NLL": float(part.NLL.mean()), "95% CI": f"[{ci[0]:.6f}, {ci[1]:.6f}]", "CI_L": ci[0], "CI_U": ci[1], "OUTER_TEST_USED": False,
        })
    historical = {"Method": "Historical inferred old EEGNet", "variant": "HISTORICAL_INFERRED", "Subjects": 54, "Source sessions": "unknown", "Target S1 used?": "unknown", "Target adaptation?": "unknown", "Strict unseen subject?": "unknown", "Seed": "historical", "Mean BA": HISTORICAL_BA, "SD across seeds": None, "Macro-F1": None, "Accuracy": None, "NLL": None, "95% CI": "not available", "CI_L": None, "CI_U": None, "OUTER_TEST_USED": "unknown"}
    summary_frame = pd.DataFrame([historical] + summary_rows)
    subject_frame.to_csv(out / "results" / "VANILLA_EEGNET_SUBJECT_RESULTS.csv", index=False)
    fold_frame.to_csv(out / "results" / "VANILLA_EEGNET_FOLD_RESULTS.csv", index=False)
    seed_frame.to_csv(out / "results" / "VANILLA_EEGNET_SEED_RESULTS.csv", index=False)
    summary_frame.to_csv(out / "results" / "VANILLA_EEGNET_SUMMARY.csv", index=False)
    architecture = {
        "implementation": "StandardEEGNet from PERSIST-EEG V6 train_openbmi_mi.py, reproduced without EA or target adaptation",
        "input_channels": 62, "input_samples": 1000, "sampling_frequency_hz": 250, "temporal_kernel": [1, 64],
        "F1": 8, "D": 2, "F2": 16, "dropout": 0.25, "pooling": [[1, 4], [1, 8]], "activation": "ELU",
        "classifier": "Linear(64,2)", "embedding": "Linear(496,64)+ELU+LayerNorm(64)", "parameter_count": sum(p.numel() for p in StandardEEGNet().parameters()),
        "biases": False, "target_specific_layers": False,
    }
    write_json(out / "protocol" / "VANILLA_EEGNET_ARCHITECTURE.json", architecture)
    write_json(out / "protocol" / "VANILLA_EEGNET_PROTOCOL.json", {
        "dataset": "OpenBMI / Lee2019 MI", "authorized_subjects": 40, "sealed_internal_holdout_subjects": 14,
        "split_source": str(STAGE0_ROOT / "delivery" / "persist_eeg_stage0" / "SPLIT_FREEZE.json"), "folds": 5,
        "test_union_equals_search": True, "source_sessions_primary": [1, 2], "target_session": 2,
        "training_subject_role": "original train_subjects intersect V8_SEARCH", "validation_subject_role": "original validation_subjects intersect V8_SEARCH",
        "target_subject_role": "original outer_test_subjects intersect V8_SEARCH", "target_S1_used": False, "target_S2_labels_used_for_fit_or_selection": False,
        "target_adaptation": False, "population_head": False, "PUD": False, "DGUG": False, "anchor_blend": False, "PCA_head": False,
        "logistic_posthoc_head": False, "EA": False, "meta_controller": False, "future_session_adaptation": False,
        "seeds": list(SEEDS), "max_epochs": MAX_EPOCHS, "early_stopping": {"patience": PATIENCE, "minimum_epoch": MIN_EPOCHS},
        "primary_metric": "mean subject balanced accuracy", "internal_holdout_used": False, "outer_test_used": False, "wbcic_used": False,
    })
    write_json(out / "protocol" / "VANILLA_EEGNET_PREPROCESSING.json", {
        "raw_cache": str(V7_ROOT / "experiments" / "persist_eeg_final_model_v7" / "outputs" / "cache" / "OPENBMI_RAW_EPOCHS_FLOAT16.npy"),
        "cache_shape": [10800, 62, 1000], "cache_dtype": "float16", "materialized_rows": "8000 V8_SEARCH only",
        "channels": 62, "sampling_frequency_hz": 250, "trial_crop": "locked 1000 samples from stage0 OpenBMI MI cache",
        "baseline_correction": "already applied in stage0 cache; no new target-derived correction", "bandpass": "stage0 cache provenance; no new filter in baseline",
        "notch": "stage0 cache provenance", "normalization": "per-fold channel mean/std fit on train subjects and source sessions only",
        "artifact_rejection": "stage0 cache provenance", "trial_count": "100 per subject/session", "classes": ["left_hand", "right_hand"],
        "class_balance": "50 trials per class per subject/session", "validation_statistics_used_for_train": False, "test_statistics_used_for_train": False,
        "internal_holdout_used": False, "outer_test_used": False, "wbcic_used": False,
    })
    (out / "BASELINE_DEFINITION.md").write_text("""# True vanilla EEGNet definition\n\nThe primary baseline is the reproduced `StandardEEGNet` (F1=8, depth multiplier D=2, F2=16, dropout=0.25) trained on original fold train subjects and both source sessions (S1+S2), with train-only channel normalization. Validation subjects are used only for predeclared epoch selection. The target subject is an entirely unseen outer-test subject and only its Session-2 trials are scored. No target S1 trials or labels, adaptation, EA, PCA/logistic head, anchor blend, P/U/D, DGUG, Conformer, or meta-controller is used.\n\nThe S1-only sensitivity changes only the source session set; it is not selected as the main result.\n""", encoding="utf-8")
    (out / "VANILLA_EEGNET_BASELINE_REPORT.md").write_text(summary_frame.to_string(index=False) + "\n\nAll 40 development subjects are tested exactly once in the frozen original folds. Holdout, historical outer-test, and WBCIC are sealed.\n", encoding="utf-8")
    (out / "HISTORICAL_BASELINE_RECONCILIATION.md").write_text(f"Historical inferred EEGNet reference: {HISTORICAL_BA:.6f} ({HISTORICAL_BA*100:.3f}%). Fresh values are in `results/VANILLA_EEGNET_SUMMARY.csv`; they were not tuned to reproduce the historical aggregate.\n", encoding="utf-8")
    print(summary_frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
