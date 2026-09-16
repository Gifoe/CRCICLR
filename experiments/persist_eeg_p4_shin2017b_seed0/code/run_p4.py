"""Seed-0-only repair of the independent Shin2017B P4 replication.

This runner imports the final server-side SIRE implementation; it contains no
local LiteBN/SIRE reimplementation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import shutil
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import KFold

ROOT = Path(os.environ.get("P4_ROOT", "/root/p4_shin2017b_seed0"))
CACHE = Path(os.environ.get("SHIN2017B_CACHE", "/root/shin2017b_eeg_cache"))
SESSIONS = ("1arithmetic", "3arithmetic", "5arithmetic")
SEED, EPOCHS, MIN_EPOCH, BATCH, LR, WD, CLIP = 0, 60, 10, 128, 3e-4, 5e-4, 5.0
MODELS = ("EEGNet", "SIRE-EEG")

# The final-heldout checkpoint manifest points to this exact source. Hash drift
# is a hard stop, never an excuse to silently substitute another LiteBN variant.
SIRE_REPO = Path(os.environ.get("SIRE_FINAL_REPO", "/root/rivermind-data/CRCICLR_FINAL_CONFIRM_WORK"))
SIRE_SOURCE = SIRE_REPO / "experiments/persist_eeg_carrier_dualdataset_screen_v1/code/run_carrier_screen.py"
SIRE_SOURCE_SHA256 = "920af131aabc272317da128f42be9961d5592619ce85ca99192029d1181f126f"
SIRE_TRAINING_CONFIG = SIRE_REPO / "experiments/persist_eeg_carrier_5fold_multiseed_stability_v1/protocol/TRAINING_PROTOCOL.json"
SIRE_HELDOUT_CHECKPOINT = Path("/root/rivermind-data/carrier_5fold_multiseed_stability_runtime/openbmi_fold0_seed0_litebn/selected_best.pt")
SIRE_HELDOUT_CHECKPOINT_SHA256 = "6ac976196e0658f66888c0fe23c909c6745c567e57c05cb7e5b2f2113adc5c8c"
SIRE_FINAL_HELDOUT_MANIFEST = SIRE_REPO / "experiments/persist_eeg_final_heldout_confirmation_v1/protocol/SOURCE_CHECKPOINTS.json"
SIRE_MODULE: Any | None = None


class EEGNet(nn.Module):
    """Frozen existing P4 EEGNet; this repair never retrains it."""

    def __init__(self, channels: int, samples: int, classes: int = 2):
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


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([clean(row) for row in rows])
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def stable(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") % (2**32 - 1)


def seed_all(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def bootstrap(values: list[float] | np.ndarray, *parts: Any) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty set")
    rng = np.random.default_rng(stable(*parts))
    draws = values[rng.integers(0, len(values), size=(20_000, len(values)))].mean(1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def natural(values: list[str] | np.ndarray) -> list[str]:
    return sorted(map(str, set(values)), key=lambda value: int(value.replace("sub-", "")))


def load_authoritative_sire() -> Any:
    """Direct import of current server source, rather than a copied class."""
    global SIRE_MODULE
    if SIRE_MODULE is not None:
        return SIRE_MODULE
    if not SIRE_SOURCE.is_file():
        raise FileNotFoundError(f"authoritative SIRE source missing: {SIRE_SOURCE}")
    observed = sha(SIRE_SOURCE)
    if observed != SIRE_SOURCE_SHA256:
        raise RuntimeError(f"authoritative SIRE source hash drift: expected {SIRE_SOURCE_SHA256}, got {observed}")
    old_repo = os.environ.get("R2EEG_REPO")
    os.environ["R2EEG_REPO"] = str(SIRE_REPO)
    source_parent = str(SIRE_SOURCE.parent)
    sys.path.insert(0, source_parent)
    try:
        spec = importlib.util.spec_from_file_location("p4_authoritative_sire", SIRE_SOURCE)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import authoritative SIRE source: {SIRE_SOURCE}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        if sys.path and sys.path[0] == source_parent:
            sys.path.pop(0)
        if old_repo is None:
            os.environ.pop("R2EEG_REPO", None)
        else:
            os.environ["R2EEG_REPO"] = old_repo
    if not hasattr(module, "CompactLite"):
        raise RuntimeError("authoritative source has no CompactLite final SIRE class")
    SIRE_MODULE = module
    return module


def _assert_shape(state: dict[str, torch.Tensor], key: str, expected: tuple[int, ...]) -> None:
    observed = tuple(state[key].shape)
    if observed != expected:
        raise RuntimeError(f"SIRE architecture mismatch for {key}: expected {expected}, observed {observed}")


def audit_sire_instance(model: nn.Module, channels: int = 30, samples: int = 2000) -> dict[str, Any]:
    """Architecture, forward and parameter audit. Any mismatch halts execution."""
    if channels != 30 or samples != 2000:
        raise RuntimeError(f"P4 SIRE audit expects C=30,T=2000, got C={channels},T={samples}")
    if tuple(model.pool.output_size) != (1, 8):
        raise RuntimeError(f"SIRE audit failed: expected adaptive pool (1,8), got {model.pool.output_size}")
    if not isinstance(model.embedding[2], nn.LayerNorm) or tuple(model.embedding[2].normalized_shape) != (64,):
        raise RuntimeError("SIRE audit failed: final embedding must contain LayerNorm(64)")
    if not isinstance(model.drop, nn.Dropout) or model.drop.p != 0.25:
        raise RuntimeError("SIRE audit failed: expected final Dropout(0.25)")
    state = model.state_dict()
    expected = {
        "temporal.0.weight": (8, 1, 1, 15), "temporal.1.weight": (8, 1, 1, 63),
        "temporal.2.weight": (8, 1, 1, 127), "spatial.0.weight": (16, 1, channels, 1),
        "spatial.1.weight": (16, 1, channels, 1), "spatial.2.weight": (16, 1, channels, 1),
        "depth1.weight": (48, 1, 1, 15), "point1.weight": (64, 48, 1, 1),
        "depth2.weight": (64, 1, 1, 31), "point2.weight": (64, 64, 1, 1),
        "embedding.0.weight": (64, 512), "embedding.0.bias": (64,),
        "embedding.2.weight": (64,), "embedding.2.bias": (64,),
        "head.weight": (2, 64), "head.bias": (2,),
    }
    for key, shape in expected.items():
        _assert_shape(state, key, shape)
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    expected_parameters = 44_872 + 48 * channels + 65 * 2
    if parameters != expected_parameters or parameters != 46_442:
        raise RuntimeError(f"SIRE parameter audit failed: expected 46,442, got {parameters}")
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, channels, samples))
    if not isinstance(output, tuple) or len(output) != 2 or tuple(output[0].shape) != (1, 2) or tuple(output[1].shape) != (1, 64):
        raise RuntimeError("SIRE audit failed: authoritative model must return logits (1,2) and 64-D embedding")
    return {
        "source_path": str(SIRE_SOURCE), "source_sha256": SIRE_SOURCE_SHA256, "class": "CompactLite",
        "channels": channels, "samples": samples, "classes": 2, "parameter_formula": "44872 + 48*C + 65*K",
        "trainable_parameters": parameters, "embedding_location": "CompactLite.embedding output / CompactLite.head pre-hook",
        "critical_state_shapes": {key: list(shape) for key, shape in expected.items()},
    }


def audit_authoritative_chain() -> dict[str, Any]:
    module = load_authoritative_sire()
    if not SIRE_TRAINING_CONFIG.is_file():
        raise FileNotFoundError(f"authoritative SIRE training config missing: {SIRE_TRAINING_CONFIG}")
    config = json.loads(SIRE_TRAINING_CONFIG.read_text(encoding="utf-8"))
    recipe = {
        "optimizer": "AdamW", "lr": 3e-4, "weight_decay": 5e-4, "batch_size": 64,
        "episode_batch_size": 128, "gradient_clipping": 5.0, "epochs": 60, "scheduler": "none",
        "loss": "ordinary cross entropy", "checkpoint_selection": "inner future-session mean-subject BA, eligible epochs 10..60, earliest tie",
    }
    for key, expected in recipe.items():
        if config.get(key) != expected:
            raise RuntimeError(f"authoritative SIRE config drift for {key}: expected {expected!r}, got {config.get(key)!r}")
    if not SIRE_HELDOUT_CHECKPOINT.is_file():
        raise FileNotFoundError(f"authoritative held-out SIRE checkpoint missing: {SIRE_HELDOUT_CHECKPOINT}")
    checkpoint_hash = sha(SIRE_HELDOUT_CHECKPOINT)
    if checkpoint_hash != SIRE_HELDOUT_CHECKPOINT_SHA256:
        raise RuntimeError(f"authoritative held-out checkpoint hash drift: expected {SIRE_HELDOUT_CHECKPOINT_SHA256}, got {checkpoint_hash}")
    heldout = module.CompactLite(62, "bn")
    saved = torch.load(SIRE_HELDOUT_CHECKPOINT, map_location="cpu", weights_only=False)
    missing, unexpected = heldout.load_state_dict(saved, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"final-heldout checkpoint/model incompatibility: missing={missing}, unexpected={unexpected}")
    heldout_parameters = sum(parameter.numel() for parameter in heldout.parameters() if parameter.requires_grad)
    if heldout_parameters != 47_978:
        raise RuntimeError(f"final-heldout parameter audit failed: expected 47,978, got {heldout_parameters}")
    result = audit_sire_instance(module.CompactLite(30, "bn"))
    result.update({
        "training_config_path": str(SIRE_TRAINING_CONFIG), "training_config_sha256": sha(SIRE_TRAINING_CONFIG),
        "training_recipe": recipe, "final_heldout_manifest_path": str(SIRE_FINAL_HELDOUT_MANIFEST),
        "final_heldout_checkpoint_path": str(SIRE_HELDOUT_CHECKPOINT), "final_heldout_checkpoint_sha256": checkpoint_hash,
        "final_heldout_C62_K2_trainable_parameters": heldout_parameters,
    })
    return result


def model(name: str, channels: int, samples: int) -> nn.Module:
    if name == "EEGNet":
        return EEGNet(channels, samples)
    if name != "SIRE-EEG":
        raise ValueError(name)
    net = load_authoritative_sire().CompactLite(channels, "bn")
    audit_sire_instance(net, channels, samples)
    return net


def output_logits(output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def reference() -> Any:
    path = ROOT / "code" / "peeh_reference.py"
    spec = importlib.util.spec_from_file_location("p4_peeh_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen PEEH reference: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def paths(subject: int, session: str) -> Path:
    return CACHE / f"sub-{subject:02d}" / f"ses-{session}_eeg_task-mental-arithmetic.npz"


def load_cell(subject: int, session: str) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    with np.load(paths(subject, session)) as archive:
        return archive["X"].astype(np.float32), (archive["y"] == 3).astype(np.int64), tuple(archive["channel_names"].tolist())


def assemble(subjects: np.ndarray, sessions: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    values, labels, subject_ids, session_ids, channels = [], [], [], [], None
    for subject in subjects:
        for session in sessions:
            value, label, names = load_cell(int(subject), session)
            channels = channels or names
            if names != channels or value.shape != (20, 30, 2000) or sorted(np.unique(label).tolist()) != [0, 1]:
                raise RuntimeError(f"invalid cache cell {subject}/{session}")
            values.append(value); labels.append(label)
            subject_ids.extend([f"sub-{int(subject):02d}"] * len(label)); session_ids.extend([session] * len(label))
    return np.concatenate(values), np.concatenate(labels), np.asarray(subject_ids), np.asarray(session_ids), channels


def normalize(train: np.ndarray, *others: np.ndarray) -> tuple[list[np.ndarray], dict[str, str]]:
    mean = train.mean(axis=(0, 2), keepdims=True)
    std = train.std(axis=(0, 2), keepdims=True)
    std[std < 1e-6] = 1
    return [(item - mean) / std for item in (train, *others)], {"mean_std_sha256": hashlib.sha256(mean.tobytes() + std.tobytes()).hexdigest()}


def logits(net: nn.Module, values: np.ndarray, device: torch.device) -> np.ndarray:
    rows = []
    net.eval()
    with torch.inference_mode():
        for start in range(0, len(values), BATCH):
            batch = torch.from_numpy(np.ascontiguousarray(values[start:start + BATCH])).to(device)
            rows.append(output_logits(net(batch)).float().cpu().numpy())
    return np.concatenate(rows)


def reps(net: nn.Module, values: np.ndarray, device: torch.device) -> np.ndarray:
    """Frozen diagnostic embedding: exact 64-D classifier input."""
    rows, captured = [], []
    hook = net.head.register_forward_pre_hook(lambda _module, args: captured.append(args[0].detach()))
    try:
        net.eval()
        with torch.inference_mode():
            for start in range(0, len(values), BATCH):
                captured.clear()
                net(torch.from_numpy(np.ascontiguousarray(values[start:start + BATCH])).to(device))
                if len(captured) != 1:
                    raise RuntimeError("classifier-input embedding hook did not fire exactly once")
                rows.append(captured[0].reshape(len(captured[0]), -1).float().cpu().numpy())
    finally:
        hook.remove()
    result = np.concatenate(rows).astype(np.float32)
    if result.shape[1] != 64:
        raise RuntimeError(f"expected 64-D final SIRE representation, got {result.shape}")
    return result


def eval_subjects(net: nn.Module, values: np.ndarray, labels: np.ndarray, subjects: np.ndarray, session: str, device: torch.device, base: dict[str, Any]) -> list[dict[str, Any]]:
    prediction = logits(net, values, device).argmax(1)
    rows = []
    for subject in natural(subjects):
        mask = subjects == subject
        rows.append({**base, "subject": subject, "session": session, "BA": float(balanced_accuracy_score(labels[mask], prediction[mask])), "Macro_F1": float(f1_score(labels[mask], prediction[mask], average="macro", zero_division=0)), "trials": int(mask.sum())})
    return rows


def ridge_fit(values: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean, std = values.mean(0), values.std(0)
    std[std < 1e-6] = 1
    design, target = np.c_[(values - mean) / std, np.ones(len(values))], np.eye(2)[labels]
    penalty = np.eye(design.shape[1]); penalty[-1, -1] = 0
    return np.linalg.solve(design.T @ design + 0.01 * penalty, design.T @ target), mean, std


def ridge_pred(values: np.ndarray, fit: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    weights, mean, std = fit
    return (np.c_[(values - mean) / std, np.ones(len(values))] @ weights).argmax(1)


def splits() -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    subjects = np.arange(1, 30)
    outer = KFold(n_splits=5, shuffle=True, random_state=SEED)
    result = []
    for fold, (train_index, test_index) in enumerate(outer.split(subjects)):
        available = subjects[train_index]
        validation = np.sort(np.random.default_rng(stable("P4-inner-validation", fold, SEED)).choice(available, 5, replace=False))
        training = np.array([subject for subject in available if subject not in set(validation)])
        result.append((fold, np.sort(training), validation, np.sort(subjects[test_index])))
    return result


def manifest() -> None:
    rows = []
    for fold, training, validation, testing in splits():
        rows.extend({"fold": fold, "subject": int(subject), "role": "train"} for subject in training)
        rows.extend({"fold": fold, "subject": int(subject), "role": "inner_val"} for subject in validation)
        rows.extend({"fold": fold, "subject": int(subject), "role": "outer_test"} for subject in testing)
    write_csv(ROOT / "subject_split_manifest.csv", rows)


def dataset_audit() -> None:
    record = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
    lines = ["# Shin2017B P4 dataset audit", "", "- Dataset: Shin2017B / nm000268, EEG only.", "- Subjects: 29 (sub-01 through sub-29).", "- Sessions: 1arithmetic, 3arithmetic, 5arithmetic (chronological S1/S2/S3).", "- Task labels: raw 3=subtraction/mental arithmetic, 4=rest; cached binary 1/0.", "- Sampling: 200 Hz; no resampling; each whole task trial is 10 s = 2,000 samples.", "- Tensor: every session `(20, 30, 2000)`, with 10 trials per class.", "- Channels (preserved order): `" + ", ".join(record["records"][0]["channel_names"]) + "`.", "- Missing/invalid recordings: none after cache validation."]
    (ROOT / "dataset_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def protocol() -> None:
    text = """# P4 seed-0 protocol

Shin2017B / nm000268 uses only the 30 EEG channels and the three mental-arithmetic sessions (S1=1arithmetic, S2=3arithmetic, S3=5arithmetic). Whole 10-second task trials are used once; no MI, NIRS, EOG, pseudo-trials, calibration, adaptation, or outer-subject information is used.

Outer CV is deterministic 5-fold subject-disjoint (seed 0); each fold reserves five remaining subjects for S3 inner validation. Training uses TRAIN S1+S2. Channel-wise mean/std is fit only on TRAIN S1+S2. EEGNet is preserved as previously run. Corrected SIRE-EEG imports the authoritative current server class and uses the frozen final recipe: 60 epochs, checkpoint eligibility epoch 10--60, earliest tie, inner future-session mean-subject BA, AdamW lr=3e-4, weight decay=5e-4, batch 128, clip=5.0. Only C=30, T=2000 and the binary output are used; kernels, widths, pooling, normalization and dropout are unchanged.

Primary full-model endpoint is outer S3. BA and macro-F1 are subject-equal; WS-BA=min(S1,S2,S3). PEEH/PSWA reuse the frozen PERSIST implementation: 200 persistence permutations, 100 equal-rank controls, ridge alpha .01, and 20,000 biological-subject bootstrap draws. Coordinates and Protected selection use TRAIN S1/S2 only; PEEH probe fits TRAIN S1 only and evaluates outer S3. PSWA probes use TRAIN S1/S2 and evaluate outer S1/S2/S3. Empty Protected assignments contribute exactly 0 PEEH effect and remain undefined/omitted for PSWA.
"""
    (ROOT / "P4_PROTOCOL.md").write_text(text, encoding="utf-8")


def coerce_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    integer_fields = {"fold", "selected_epoch", "trials", "protected_rank", "active_rank"}
    float_fields = {"BA", "Macro_F1", "intact_BA", "protected_erased_BA", "random_erased_BA", "PEEH_pp", "protected_WSBA", "random_WSBA", "PSWA_pp", "protected_BA_S1", "protected_BA_S2", "protected_BA_S3", "random_BA_S1", "random_BA_S2", "random_BA_S3"}
    for row in rows:
        for key, value in list(row.items()):
            if value == "": row[key] = None
            elif key in integer_fields: row[key] = int(value)
            elif key in float_fields: row[key] = float(value)
    return rows


def archive_invalid_sire() -> Path:
    """Copy mixed reports, then recoverably move only invalid SIRE checkpoints."""
    destination = ROOT / "provenance" / "invalid_old_sire_seed0_pre_final_repair"
    if destination.exists():
        raise RuntimeError(f"invalid-SIRE provenance already exists: {destination}; refusing to overwrite it")
    destination.mkdir(parents=True)
    copied = {}
    for name in ("P4_SEED0_SUMMARY.md", "seed0_fullmodel_subject_metrics.csv", "seed0_fullmodel_summary.csv", "seed0_peeh_subject_results.csv", "seed0_pswa_subject_results.csv", "seed0_diagnostic_summary.csv", "diagnostic_assignments.json", "run.log"):
        source = ROOT / name
        if source.is_file():
            target = destination / name
            shutil.copy2(source, target)
            copied[name] = sha(target)
    old_checkpoints = ROOT / "checkpoints" / "SIRE-EEG"
    moved = None
    if old_checkpoints.exists():
        moved = destination / "checkpoints_SIRE-EEG_old_implementation"
        shutil.move(str(old_checkpoints), str(moved))
    atomic_json(destination / "INVALID_SIRE_PROVENANCE.json", {"status": "INVALID_REIMPLEMENTED_SIRE_ARCHIVED", "archived_at_utc": datetime.now(timezone.utc).isoformat(), "prior_branch_commit": "665e5757119bd9d222275a107d2f6e4034290e5a", "prior_runner_class": "run_p4.py::LiteBN", "prior_runner_parameter_formula": "13512 + 48*C + 65*K", "prior_runner_C30_K2_parameters": 15_082, "invalid_features": "48->32 refinement, k=9/7, AdaptiveAvgPool2d((1,4)), Linear(128,64)", "moved_old_sire_checkpoints": None if moved is None else str(moved), "copied_artifact_sha256": copied})
    return destination


def preserved_eegnet_outputs() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    full_path, peeh_path, pswa_path, assignment_path = ROOT / "seed0_fullmodel_subject_metrics.csv", ROOT / "seed0_peeh_subject_results.csv", ROOT / "seed0_pswa_subject_results.csv", ROOT / "diagnostic_assignments.json"
    for path in (full_path, peeh_path, pswa_path, assignment_path):
        if not path.is_file(): raise FileNotFoundError(f"existing EEGNet artifact required for no-retrain repair is missing: {path}")
    full = [row for row in coerce_rows(read_csv(full_path)) if row["model"] == "EEGNet"]
    peeh = [row for row in coerce_rows(read_csv(peeh_path)) if row["model"] == "EEGNet"]
    pswa = [row for row in coerce_rows(read_csv(pswa_path)) if row["model"] == "EEGNet"]
    assignments = [row for row in json.loads(assignment_path.read_text(encoding="utf-8")) if row["model"] == "EEGNet"]
    if len(full) != 87 or len(peeh) != 29 or len(pswa) != 29 or len(assignments) != 5:
        raise RuntimeError(f"existing EEGNet provenance incomplete; refusing retraining: full={len(full)}, peeh={len(peeh)}, pswa={len(pswa)}, assignments={len(assignments)}")
    for row in peeh:
        if row["status"] == "EMPTY_PROTECTED": row["PEEH_pp"] = 0.0
    return full, peeh, pswa, assignments


def train_sire_cell(fold: int, training: np.ndarray, validation: np.ndarray, testing: np.ndarray, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    cell = ROOT / "checkpoints" / "SIRE-EEG" / f"fold{fold}"
    checkpoint, record_path = cell / "selected.pt", cell / "record.json"
    cell.mkdir(parents=True, exist_ok=True)
    training_x, training_y, _, _, _ = assemble(training, SESSIONS[:2])
    validation_x, validation_y, validation_subjects, _, _ = assemble(validation, (SESSIONS[2],))
    (training_x, validation_x), normalizer = normalize(training_x, validation_x)
    seed_all(SEED)
    net = model("SIRE-EEG", 30, 2000).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    x_train, y_train = torch.from_numpy(training_x).to(device), torch.from_numpy(training_y).to(device)
    best, best_epoch, best_state, history = -1.0, 0, None, []
    for epoch in range(1, EPOCHS + 1):
        net.train(); losses = []
        rng = np.random.default_rng(stable("P4-batch", "SIRE-EEG", fold, SEED, epoch))
        for indices in (part for part in np.array_split(rng.permutation(len(training_y)), max(1, int(np.ceil(len(training_y) / BATCH))) ) if len(part)):
            index = torch.as_tensor(indices, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(output_logits(net(x_train.index_select(0, index))), y_train.index_select(0, index))
            if not torch.isfinite(loss): raise RuntimeError(f"non-finite corrected SIRE loss in fold {fold}, epoch {epoch}")
            loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), CLIP); optimizer.step(); losses.append(float(loss.detach().cpu()))
        prediction = logits(net, validation_x, device).argmax(1)
        score = float(np.mean([balanced_accuracy_score(validation_y[validation_subjects == subject], prediction[validation_subjects == subject]) for subject in natural(validation_subjects)]))
        selected = epoch >= MIN_EPOCH and score > best + 1e-12
        if selected:
            best, best_epoch, best_state = score, epoch, deepcopy({key: value.detach().cpu() for key, value in net.state_dict().items()})
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "inner_val_subject_equal_BA": score, "selected": selected})
    if best_state is None: raise RuntimeError(f"no corrected SIRE checkpoint selected in fold {fold}")
    net.load_state_dict(best_state)
    torch.save({"state_dict": net.state_dict(), "selected_epoch": best_epoch}, checkpoint)
    record = {"model": "SIRE-EEG", "fold": fold, "seed": 0, "selected_epoch": best_epoch, "best_inner_validation_BA": best, "recipe": {"epochs": EPOCHS, "min_epoch": MIN_EPOCH, "batch_size": BATCH, "lr": LR, "weight_decay": WD, "gradient_clip": CLIP, "optimizer": "AdamW", "checkpoint_selection": "inner S3 mean-subject BA, eligible epochs 10..60, earliest tie"}, "normalizer": normalizer, "checkpoint_sha256": sha(checkpoint), "history": history, "train_subjects": list(map(int, training)), "inner_val_subjects": list(map(int, validation)), "outer_subjects": list(map(int, testing)), "authoritative_source_sha256": SIRE_SOURCE_SHA256, "trainable_parameters": 46_442}
    atomic_json(record_path, record)
    return net.eval(), record


def diagnostics(name: str, fold: int, training: np.ndarray, testing: np.ndarray, net: nn.Module, device: torch.device, ref: Any, checkpoint: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Frozen PEEH/PSWA definitions; only empty-assignment aggregation is amended."""
    train_raw, labels, train_subjects, sessions, _ = assemble(training, SESSIONS[:2])
    outer_x, outer_y, outer_subjects = [], [], []
    for session in SESSIONS:
        value, label, subjects, _, _ = assemble(testing, (session,))
        outer_x.append(value); outer_y.append(label); outer_subjects.append(subjects)
    (train_x, *outer_x), _ = normalize(train_raw, *outer_x)
    h_train, h_outer = reps(net, train_x, device), [reps(net, value, device) for value in outer_x]
    session_number = np.asarray([1 if value == SESSIONS[0] else 2 for value in sessions])
    spectrum = ref.spectrum(h_train, labels, train_subjects, session_number, "Shin2017B", name, fold); spectrum["classes"] = 2
    protected, assignment = ref.select_protected(h_train, labels, train_subjects, session_number, spectrum, name, "Shin2017B", fold)
    fit_index = np.flatnonzero(sessions == SESSIONS[0])
    engine = ref.FixedStandardizerKernelRidge(h_train[fit_index], labels[fit_index], h_outer[2], 2, spectrum, qfit=ref.canonical(h_train[fit_index], spectrum), qeval=ref.canonical(h_outer[2], spectrum), raw_base=ref._erasure_base(spectrum))
    intact, _ = engine.predict(())
    protected_prediction, _ = engine.predict(protected)
    all_dimensions = np.arange(spectrum["rank"])
    random_predictions = []
    for draw in range(100):
        dimensions = [] if not protected else np.random.default_rng(ref.stable_seed("final-random", name, "Shin2017B", fold, draw)).choice(all_dimensions, len(protected), replace=False).tolist()
        prediction, _ = engine.predict(dimensions); random_predictions.append(prediction)
    peeh_rows = []
    for subject in natural(outer_subjects[2]):
        mask = outer_subjects[2] == subject
        intact_ba = balanced_accuracy_score(outer_y[2][mask], intact[mask])
        protected_ba = balanced_accuracy_score(outer_y[2][mask], protected_prediction[mask])
        random_ba = float(np.mean([balanced_accuracy_score(outer_y[2][mask], prediction[mask]) for prediction in random_predictions]))
        peeh_rows.append({"model": name, "fold": fold, "subject": subject, "status": "VALID" if protected else "EMPTY_PROTECTED", "protected_rank": len(protected), "intact_BA": intact_ba, "protected_erased_BA": protected_ba, "random_erased_BA": random_ba, "PEEH_pp": 0.0 if not protected else 100 * (random_ba - protected_ba), "checkpoint_sha256": checkpoint["checkpoint_sha256"]})
    pswa_rows = []
    if protected:
        q_train, q_outer = ref.canonical(h_train, spectrum), [ref.canonical(value, spectrum) for value in h_outer]
        protected_prediction = [ridge_pred(value[:, protected], ridge_fit(q_train[:, protected], labels)) for value in q_outer]
        random_sets = [np.random.default_rng(ref.stable_seed("final-random", name, "Shin2017B", fold, draw)).choice(all_dimensions, len(protected), replace=False) for draw in range(100)]
        random_prediction = [[ridge_pred(value[:, dimensions], ridge_fit(q_train[:, dimensions], labels)) for value in q_outer] for dimensions in random_sets]
        for subject in natural(outer_subjects[0]):
            protected_bas, random_wsbas = [], []
            row = {"model": name, "fold": fold, "subject": subject, "status": "VALID", "protected_rank": len(protected), "checkpoint_sha256": checkpoint["checkpoint_sha256"]}
            for index, label in enumerate(("S1", "S2", "S3")):
                mask = outer_subjects[index] == subject
                protected_ba = balanced_accuracy_score(outer_y[index][mask], protected_prediction[index][mask])
                random_ba = float(np.mean([balanced_accuracy_score(outer_y[index][mask], item[index][mask]) for item in random_prediction]))
                row[f"protected_BA_{label}"] = protected_ba; row[f"random_BA_{label}"] = random_ba; protected_bas.append(protected_ba)
            for item in random_prediction:
                random_wsbas.append(min(balanced_accuracy_score(outer_y[index][outer_subjects[index] == subject], item[index][outer_subjects[index] == subject]) for index in range(3)))
            row["protected_WSBA"] = min(protected_bas); row["random_WSBA"] = float(np.mean(random_wsbas)); row["PSWA_pp"] = 100 * (row["protected_WSBA"] - row["random_WSBA"]); pswa_rows.append(row)
    else:
        pswa_rows = [{"model": name, "fold": fold, "subject": subject, "status": "EMPTY_PROTECTED", "protected_rank": 0, "PSWA_pp": None, "checkpoint_sha256": checkpoint["checkpoint_sha256"]} for subject in natural(outer_subjects[0])]
    return peeh_rows, pswa_rows, {"model": name, "fold": fold, "protected_rank": len(protected), "active_rank": spectrum["rank"], "assignment": assignment, "checkpoint_sha256": checkpoint["checkpoint_sha256"]}


def full_metrics(rows: list[dict[str, Any]], name: str) -> tuple[dict[str, Any], list[float], list[float]]:
    selected = [row for row in rows if row["model"] == name]
    by_subject = {subject: {row["session"]: row for row in selected if row["subject"] == subject} for subject in natural([row["subject"] for row in selected])}
    if len(by_subject) != 29 or any(set(cells) != {"S1", "S2", "S3"} for cells in by_subject.values()): raise RuntimeError(f"full-model metrics incomplete for {name}")
    future = [cells["S3"]["BA"] for cells in by_subject.values()]
    macro_f1 = [cells["S3"]["Macro_F1"] for cells in by_subject.values()]
    wsba = [min(cells[session]["BA"] for session in ("S1", "S2", "S3")) for cells in by_subject.values()]
    return {"model": name, "future BA": float(np.mean(future)), "future Macro-F1": float(np.mean(macro_f1)), "WS-BA": float(np.mean(wsba)), "subjects": len(by_subject)}, future, wsba


def diagnostic_summary(peeh_rows: list[dict[str, Any]], pswa_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for name in MODELS:
        peeh = [row for row in peeh_rows if row["model"] == name]
        pswa = [row for row in pswa_rows if row["model"] == name and row["status"] == "VALID"]
        if len(peeh) != 29: raise RuntimeError(f"PEEH must retain all 29 subjects for {name}; got {len(peeh)}")
        effects = [0.0 if row["status"] == "EMPTY_PROTECTED" else float(row["PEEH_pp"]) for row in peeh]
        peeh_mean, peeh_low, peeh_high = bootstrap(effects, "P4", "peeh", name)
        peeh_folds = len({int(row["fold"]) for row in peeh if row["status"] == "VALID"})
        pswa_folds = len({int(row["fold"]) for row in pswa})
        if pswa: pswa_mean, pswa_low, pswa_high = bootstrap([float(row["PSWA_pp"]) for row in pswa], "P4", "pswa", name)
        else: pswa_mean = pswa_low = pswa_high = None
        summary.append({"model": name, "PEEH_pp": peeh_mean, "PEEH_CI95_low": peeh_low, "PEEH_CI95_high": peeh_high, "PEEH_subjects_including_empty_as_zero": len(peeh), "PEEH_coverage": f"{peeh_folds}/5", "PSWA_pp": pswa_mean, "PSWA_CI95_low": pswa_low, "PSWA_CI95_high": pswa_high, "PSWA_subjects_nonempty_only": len(pswa), "PSWA_coverage": f"{pswa_folds}/5"})
    return summary


def write_sire_audit(audit: dict[str, Any], records: list[dict[str, Any]], archive: Path) -> None:
    checkpoints = [f"| {record['fold']} | {record['selected_epoch']} | {record['best_inner_validation_BA']:.6f} | `{record['checkpoint_sha256']}` |" for record in sorted(records, key=lambda item: item["fold"])]
    lines = ["# SIRE implementation audit", "", "## Authoritative current-server chain", "", f"- Final-heldout manifest: `{audit['final_heldout_manifest_path']}`.", f"- Frozen final-heldout checkpoint tested: `{audit['final_heldout_checkpoint_path']}` (SHA-256 `{audit['final_heldout_checkpoint_sha256']}`); strict-loads into the current C=62, K=2 class with {audit['final_heldout_C62_K2_trainable_parameters']:,} trainable parameters.", f"- Training script/model source: `{audit['source_path']}`; SHA-256 `{audit['source_sha256']}`; imported directly at runtime as `CompactLite`.", f"- Frozen training config: `{audit['training_config_path']}`; SHA-256 `{audit['training_config_sha256']}`. Recipe: AdamW 3e-4, weight decay 5e-4, 60 epochs, clip 5.0, no scheduler, and inner future-session mean-subject BA selection in epochs 10--60 with earliest tie.", "- Diagnostic embedding: the 64-D input to `CompactLite.head`, equivalently the output of `CompactLite.embedding` before final dropout/head in evaluation mode.", "", "## Exact architecture and P4 adaptation", "", "- Temporal branches: Conv2d 1->8 with kernels 15/63/127, BatchNorm2d(8), ELU, branch spatial Conv2d 8->16 with groups=8, BatchNorm2d(16), pool (1,4), dropout 0.20.", "- Concatenated width 48; depthwise temporal k=15 48->48 plus pointwise 48->64/BatchNorm2d(64), then depthwise k=31 64->64 plus pointwise 64->64/BatchNorm2d(64); each refinement has pool (1,2) and dropout 0.15.", "- AdaptiveAvgPool2d((1,8)); Linear(512,64), ELU, LayerNorm(64), dropout 0.25, Linear(64,K).", "- Parameter formula: `44,872 + 48*C + 65*K`. For Shin2017B C=30, K=2: **46,442 trainable parameters** (instantiated and asserted before every corrected fold).", "- Only C=30, T=2000 (parameter-invariant due adaptive pooling), and K=2 are dataset-bound. Kernels, widths, pooling, normalization, dropout, optimizer and selection rule are otherwise unchanged.", "", "## Rejected prior P4 implementation", "", "- Prior P4 `run_p4.py::LiteBN` was not authoritative: it used 48->32 refinement, k=9/7, 4-bin pooling and Linear(128,64). Its formula was `13,512 + 48*C + 65*K`, or 15,082 parameters at C=30, K=2.", "- The historical seven-backbone LiteBN has the same obsolete structure (`experiments/persist_eeg_seven_backbone_fourtask_3seed_v1/code/backbone_models.py::LiteBN`) and was explicitly excluded. It is historical main-table provenance, not the source used here.", f"- Invalid old-SIRE checkpoints and mixed outputs were preserved at `{archive}`; EEGNet artifacts were preserved and never retrained.", "", "## Corrected SIRE checkpoint selection", "", "| Fold | Selected epoch | Inner S3 subject-equal BA | Checkpoint SHA-256 |", "|---:|---:|---:|---|", *checkpoints, "", "All five corrected checkpoints passed source/config/state_dict/architecture/parameter audits before training. Any mismatch raises an error; no copied LiteBN fallback exists."]
    (ROOT / "SIRE_IMPLEMENTATION_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_repair() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    audit = audit_authoritative_chain()
    existing_full, existing_peeh, existing_pswa, existing_assignments = preserved_eegnet_outputs()
    archive = archive_invalid_sire()
    protocol(); dataset_audit(); manifest()
    ref, device = reference(), torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corrected_full, corrected_peeh, corrected_pswa, corrected_assignments, records = [], [], [], [], []
    for fold, training, validation, testing in splits():
        net, record = train_sire_cell(fold, training, validation, testing, device); records.append(record)
        train_raw, _, _, _, _ = assemble(training, SESSIONS[:2])
        for session, short in zip(SESSIONS, ("S1", "S2", "S3")):
            value, label, subjects, _, _ = assemble(testing, (session,))
            value = normalize(train_raw, value)[0][1]
            corrected_full.extend(eval_subjects(net, value, label, subjects, short, device, {"model": "SIRE-EEG", "fold": fold, "checkpoint_sha256": record["checkpoint_sha256"], "selected_epoch": record["selected_epoch"]}))
        peeh, pswa, assignment = diagnostics("SIRE-EEG", fold, training, testing, net, device, ref, record)
        corrected_peeh.extend(peeh); corrected_pswa.extend(pswa); corrected_assignments.append(assignment)
        del net
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print(f"P4_CORRECTED_SIRE_CELL_COMPLETE fold={fold}", flush=True)
    if len(corrected_full) != 87 or len(corrected_peeh) != 29 or len(corrected_pswa) != 29 or len(corrected_assignments) != 5:
        raise RuntimeError("corrected SIRE output cardinality failure")
    full, peeh, pswa, assignments = existing_full + corrected_full, existing_peeh + corrected_peeh, existing_pswa + corrected_pswa, existing_assignments + corrected_assignments
    write_csv(ROOT / "seed0_fullmodel_subject_metrics.csv", full); write_csv(ROOT / "seed0_peeh_subject_results.csv", peeh); write_csv(ROOT / "seed0_pswa_subject_results.csv", pswa); atomic_json(ROOT / "diagnostic_assignments.json", assignments)
    eegnet_summary, eegnet_future, eegnet_wsba = full_metrics(full, "EEGNet")
    sire_summary, sire_future, sire_wsba = full_metrics(full, "SIRE-EEG")
    future_delta, wsba_delta = np.asarray(sire_future) - np.asarray(eegnet_future), np.asarray(sire_wsba) - np.asarray(eegnet_wsba)
    _, future_low, future_high = bootstrap(future_delta, "P4", "future")
    _, wsba_low, wsba_high = bootstrap(wsba_delta, "P4", "wsba")
    summary = [eegnet_summary, sire_summary, {"model": "SIRE-EEG minus EEGNet", "metric": "future BA contrast", "mean": float(future_delta.mean()), "CI95_low": future_low, "CI95_high": future_high, "bootstrap_draws": 20_000}, {"model": "SIRE-EEG minus EEGNet", "metric": "WS-BA contrast", "mean": float(wsba_delta.mean()), "CI95_low": wsba_low, "CI95_high": wsba_high, "bootstrap_draws": 20_000}]
    write_csv(ROOT / "seed0_fullmodel_summary.csv", summary)
    diagnostics_summary = diagnostic_summary(peeh, pswa)
    write_csv(ROOT / "seed0_diagnostic_summary.csv", diagnostics_summary)
    write_sire_audit(audit, records, archive)
    by_model = {row["model"]: row for row in diagnostics_summary}
    lines = ["# P4 seed-0 summary", "", f"- EEGNet (preserved, not retrained): future BA {eegnet_summary['future BA']:.3f}, future Macro-F1 {eegnet_summary['future Macro-F1']:.3f}, WS-BA {eegnet_summary['WS-BA']:.3f}.", f"- Corrected SIRE-EEG: future BA {sire_summary['future BA']:.3f}, future Macro-F1 {sire_summary['future Macro-F1']:.3f}, WS-BA {sire_summary['WS-BA']:.3f}.", f"- Paired SIRE-EEG minus EEGNet future BA: {future_delta.mean():.3f} [{future_low:.3f}, {future_high:.3f}].", f"- Paired SIRE-EEG minus EEGNet WS-BA: {wsba_delta.mean():.3f} [{wsba_low:.3f}, {wsba_high:.3f}].", "", "## Frozen diagnostics", "", f"- EEGNet: PEEH {by_model['EEGNet']['PEEH_pp']:.3f} pp, nonempty-fold coverage {by_model['EEGNet']['PEEH_coverage']}; PSWA {by_model['EEGNet']['PSWA_pp'] if by_model['EEGNet']['PSWA_pp'] is not None else 'undefined'} pp, nonempty-fold coverage {by_model['EEGNet']['PSWA_coverage']}.", f"- Corrected SIRE-EEG: PEEH {by_model['SIRE-EEG']['PEEH_pp']:.3f} pp, nonempty-fold coverage {by_model['SIRE-EEG']['PEEH_coverage']}; PSWA {by_model['SIRE-EEG']['PSWA_pp'] if by_model['SIRE-EEG']['PSWA_pp'] is not None else 'undefined'} pp, nonempty-fold coverage {by_model['SIRE-EEG']['PSWA_coverage']}.", "- Empty Protected contributes exactly zero to PEEH and remains undefined/omitted for PSWA. Coverage is nonempty fold/checkpoint coverage, not outer-subject count.", "", "The corrected SIRE source/configuration/state/architecture audit passed before training; all five corrected folds completed. Diagnostic direction is descriptive and does not change selection, thresholds or coverage."]
    (ROOT / "P4_SEED0_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("P4_CORRECTED_SIRE_SEED0_COMPLETE", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-sire", action="store_true", help="archive invalid old SIRE and rerun only corrected SIRE seed 0")
    args = parser.parse_args()
    if not args.repair_sire:
        raise RuntimeError("this repaired runner requires --repair-sire and never retrains EEGNet")
    run_repair()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
