"""Independent BNCI2015-001 seed-0 matched episodic replication.

This runner intentionally imports the frozen final EEGNet, CompactLite, and
OpenBMI episode-manifest builder.  It neither reimplements those models nor
uses IID mini-batches.  Raw data and checkpoints live outside the Git tree.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import mne
import moabb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score


ROOT = Path(os.environ.get("BNCI_P4_ROOT", "/root/p4_bnci2015_001_seed0_matched_episodic_v1"))
CACHE = Path(os.environ.get("BNCI2015_001_CACHE", "/root/bnci2015_001_eeg_cache"))
FINAL_REPO = Path(os.environ.get("SIRE_FINAL_REPO", "/root/rivermind-data/CRCICLR_FINAL_CONFIRM_WORK"))

SEED, TRAINING_SEED = 0, 100_000
EPOCHS, MIN_EPOCH, EPISODE_TRIALS = 60, 10, 128
LR, WEIGHT_DECAY, GRADIENT_CLIP = 3e-4, 5e-4, 5.0
SUBJECTS = tuple(f"sub-{value:02d}" for value in range(1, 13))
CHANNELS = ("FC3", "FCz", "FC4", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "CP3", "CPz", "CP4")
SESSION_IDS = {"0A": 1, "1B": 2}
EVENT_LABELS = {"right_hand": 1, "feet": 2}

SIRE_SOURCE = FINAL_REPO / "experiments/persist_eeg_carrier_dualdataset_screen_v1/code/run_carrier_screen.py"
EEGNET_SOURCE = FINAL_REPO / "experiments/persist_eeg_carrier_dualdataset_screen_v1/code/eegnet_locked.py"
EPISODE_SOURCE = FINAL_REPO / "experiments/persist_eeg_r2eeg_stage1_v1/code/run_stage1.py"
CORE_SOURCE = FINAL_REPO / "experiments/persist_eeg_r2eeg_stage1_v1/code/stage1_core.py"
TRAINING_CONFIG = FINAL_REPO / "experiments/persist_eeg_carrier_5fold_multiseed_stability_v1/protocol/TRAINING_PROTOCOL.json"
TRAINING_SOURCE = FINAL_REPO / "experiments/persist_eeg_carrier_5fold_multiseed_stability_v1/code/train_grid.py"
EXPECTED_SHA256 = {
    SIRE_SOURCE: "920af131aabc272317da128f42be9961d5592619ce85ca99192029d1181f126f",
    EEGNET_SOURCE: "f7c513c3f3cd1f326a74b4e419bd693e15ee8980a7c378f1c0bee8215b8b89dd",
    EPISODE_SOURCE: "40cd24d90a1919db14d9d7b5a0b976bc5e4943e82e8a9fec10bbb82d70d78b75",
    CORE_SOURCE: "5277ec7055974c953acf32df3fe58a05761e0439b66c43eca3049563940c21ce",
    TRAINING_CONFIG: "e03ba36815df3d8f808e558211a2cba739b7fc53c0179fc9cc8c6cab31adf936",
    TRAINING_SOURCE: "23f67f2c6ee75ce0babaab56a9e37b3210fa7b589d0347a417d1437fb1290c61",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([clean(row) for row in rows])
    os.replace(temporary, path)


def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def stable_seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") % (2**32 - 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def state_hash(state: dict[str, torch.Tensor]) -> str:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def direct_import(name: str, path: Path, prepend: list[Path]) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"authoritative source missing: {path}")
    inserted = [str(value) for value in prepend]
    sys.path[:0] = inserted
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for value in inserted:
            if value in sys.path:
                sys.path.remove(value)


def load_authoritative() -> tuple[Any, Any, Any, dict[str, Any]]:
    for path, expected in EXPECTED_SHA256.items():
        observed = sha256(path) if path.is_file() else None
        if observed != expected:
            raise RuntimeError(f"authoritative source/config hash drift: {path}; expected {expected}, got {observed}")
    config = json.loads(TRAINING_CONFIG.read_text(encoding="utf-8"))
    required = {
        "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY,
        "episode_batch_size": EPISODE_TRIALS, "gradient_clipping": GRADIENT_CLIP,
        "epochs": EPOCHS, "scheduler": "none", "loss": "ordinary cross entropy",
        "checkpoint_selection": "inner future-session mean-subject BA, eligible epochs 10..60, earliest tie",
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise RuntimeError(f"final training config mismatch: {key}={config.get(key)!r}, expected {expected!r}")
    previous = os.environ.get("R2EEG_REPO")
    os.environ["R2EEG_REPO"] = str(FINAL_REPO)
    try:
        stage1 = direct_import("run_stage1", EPISODE_SOURCE, [EPISODE_SOURCE.parent])
        eegnet = direct_import("bnci_final_eegnet", EEGNET_SOURCE, [EEGNET_SOURCE.parent])
        sire = direct_import("bnci_final_compact", SIRE_SOURCE, [SIRE_SOURCE.parent])
    finally:
        if previous is None:
            os.environ.pop("R2EEG_REPO", None)
        else:
            os.environ["R2EEG_REPO"] = previous
    if not hasattr(stage1, "make_manifest") or not hasattr(stage1, "core"):
        raise RuntimeError("final episode source lacks make_manifest/core")
    if not hasattr(eegnet, "EEGNet") or not hasattr(sire, "CompactLite"):
        raise RuntimeError("final model source lacks EEGNet/CompactLite")
    return stage1, eegnet, sire, required


def audit_models(eegnet_module: Any, sire_module: Any) -> dict[str, Any]:
    eegnet = eegnet_module.EEGNet(13, 1000)
    sire = sire_module.CompactLite(13, "bn")
    expected = {
        "EEGNet": {
            "temporal.weight": (8, 1, 1, 64), "spatial.weight": (16, 1, 13, 1),
            "depth.weight": (16, 1, 1, 16), "point.weight": (16, 16, 1, 1),
            "embedding.0.weight": (64, 496), "embedding.2.weight": (64,), "head.weight": (2, 64),
        },
        "SIRE-EEG": {
            "temporal.0.weight": (8, 1, 1, 15), "temporal.1.weight": (8, 1, 1, 63),
            "temporal.2.weight": (8, 1, 1, 127), "spatial.0.weight": (16, 1, 13, 1),
            "spatial.1.weight": (16, 1, 13, 1), "spatial.2.weight": (16, 1, 13, 1),
            "depth1.weight": (48, 1, 1, 15), "point1.weight": (64, 48, 1, 1),
            "depth2.weight": (64, 1, 1, 31), "point2.weight": (64, 64, 1, 1),
            "embedding.0.weight": (64, 512), "embedding.2.weight": (64,), "head.weight": (2, 64),
        },
    }
    objects = {"EEGNet": eegnet, "SIRE-EEG": sire}
    counts: dict[str, int] = {}
    for name, model in objects.items():
        state = model.state_dict()
        for key, shape in expected[name].items():
            if key not in state or tuple(state[key].shape) != shape:
                got = tuple(state[key].shape) if key in state else None
                raise RuntimeError(f"{name} state audit mismatch: {key} {got} != {shape}")
        counts[name] = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        with torch.inference_mode():
            logits, embedding = model(torch.zeros(1, 13, 1000))
        if tuple(logits.shape) != (1, 2) or tuple(embedding.shape) != (1, 64):
            raise RuntimeError(f"{name} forward audit failed: logits={tuple(logits.shape)}, embedding={tuple(embedding.shape)}")
    if counts["SIRE-EEG"] != 45_626:
        raise RuntimeError(f"SIRE formula audit failed: {counts['SIRE-EEG']} != 45626")
    if counts["EEGNet"] != 33_378:
        raise RuntimeError(f"EEGNet authoritative C13/T1000 audit failed: {counts['EEGNet']} != 33378")
    return {
        "EEGNet": {"class": "eegnet_locked.EEGNet", "source": str(EEGNET_SOURCE), "sha256": EXPECTED_SHA256[EEGNET_SOURCE], "parameters_C13_T1000_K2": counts["EEGNet"], "state_shapes": expected["EEGNet"]},
        "SIRE-EEG": {"class": "run_carrier_screen.CompactLite(kind='bn')", "source": str(SIRE_SOURCE), "sha256": EXPECTED_SHA256[SIRE_SOURCE], "parameters_C13_T1000_K2": counts["SIRE-EEG"], "formula": "44872 + 48*C + 65*K", "state_shapes": expected["SIRE-EEG"]},
    }


def source_availability() -> dict[str, list[str]]:
    # Exact condition in MOABB 1.7.2 bnci_2015._load_data_001_2015.
    return {f"sub-{subject:02d}": (["0A", "1B", "2C"] if subject in (8, 9, 10, 11) else ["0A", "1B"]) for subject in range(1, 13)}


def official_cache() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Download only official A/B MAT files and use MOABB's exact MAT-to-Raw converter."""
    from moabb.datasets.bnci.base import BNCI_URL, _convert_mi, data_path

    availability = source_availability()
    if any(not {"0A", "1B"}.issubset(sessions) for sessions in availability.values()):
        raise RuntimeError("STOP: full 12-subject common A/B cohort is unavailable")
    downloads = CACHE / "official_bnci_downloads"
    prepared = CACHE / "prepared"
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for subject in range(1, 13):
        subject_id = f"sub-{subject:02d}"
        for session in ("0A", "1B"):
            cell = prepared / subject_id / f"ses-{session}.npz"
            try:
                url = f"{BNCI_URL}001-2015/S{subject:02d}{session[-1]}.mat"
                filename = Path(data_path(url, str(downloads), force_update=False, update_path=False, verbose=False)[0])
                if cell.is_file():
                    with np.load(cell, allow_pickle=False) as archive:
                        x, y = archive["X"], archive["y"]
                        names = tuple(archive["channel_names"].tolist())
                    if x.shape != (200, 13, 1000) or y.shape != (200,) or np.bincount(y, minlength=2).tolist() != [100, 100] or names != CHANNELS:
                        raise RuntimeError(f"prepared cache validation failed: {cell}")
                    records.append({"subject": subject_id, "session": session, "url": url, "file": str(filename), "file_bytes": filename.stat().st_size, "cache_file": str(cell), "cache_bytes": cell.stat().st_size, "trials": int(len(y)), "right_hand": int((y == 0).sum()), "feet": int((y == 1).sum())})
                    continue
                runs, event_id = _convert_mi(str(filename), list(CHANNELS), ["eeg"] * len(CHANNELS), dataset_code="BNCI2015-001", subject_id=subject)
                if event_id != EVENT_LABELS:
                    raise RuntimeError(f"event mapping {event_id} != {EVENT_LABELS}")
                trials, labels = [], []
                for raw in runs:
                    if tuple(raw.ch_names[:13]) != CHANNELS or raw.info["sfreq"] != 512.0:
                        raise RuntimeError(f"channel/sfreq drift: {raw.ch_names[:13]}, {raw.info['sfreq']}")
                    events = mne.find_events(raw, shortest_event=1, verbose=False)
                    raw, events = raw.resample(250.0, npad="auto", events=events, verbose=False)
                    values = raw.get_data(picks=list(CHANNELS)).astype(np.float32, copy=False)
                    for onset, _, code in events:
                        if int(code) not in (1, 2):
                            continue
                        start, stop = int(onset) + 312, int(onset) + 312 + 1000
                        if start < 0 or stop > values.shape[1]:
                            raise RuntimeError(f"sustained window outside recording: start={start}, stop={stop}, n={values.shape[1]}")
                        trials.append(values[:, start:stop])
                        labels.append(0 if int(code) == 1 else 1)
                x = np.ascontiguousarray(np.stack(trials), dtype=np.float32)
                y = np.asarray(labels, dtype=np.int64)
                if x.ndim != 3 or x.shape[1:] != (13, 1000) or len(y) != len(x) or np.bincount(y, minlength=2).tolist() != [100, 100]:
                    raise RuntimeError(f"invalid trial balance/shape {x.shape}, {np.bincount(y, minlength=2).tolist()}")
                cell.parent.mkdir(parents=True, exist_ok=True)
                if not cell.exists():
                    np.savez_compressed(cell, X=x, y=y, channel_names=np.asarray(CHANNELS), source_url=url)
                with np.load(cell, allow_pickle=False) as archive:
                    if archive["X"].shape != (200, 13, 1000) or not np.array_equal(archive["y"], y):
                        raise RuntimeError(f"prepared cache validation failed: {cell}")
                records.append({"subject": subject_id, "session": session, "url": url, "file": str(filename), "file_bytes": filename.stat().st_size, "cache_file": str(cell), "cache_bytes": cell.stat().st_size, "trials": int(len(y)), "right_hand": int((y == 0).sum()), "feet": int((y == 1).sum())})
            except Exception as error:
                errors.append(f"{subject_id}/{session}: {type(error).__name__}: {error}")
                raise RuntimeError("official BNCI A/B cache failed: " + errors[-1]) from error
    source_files = sorted(downloads.rglob("*.mat"))
    if len(source_files) != 24:
        raise RuntimeError(f"unnecessary/missing official downloads: expected 24 A/B files, found {len(source_files)}")
    if any(path.name.endswith("C.mat") for path in source_files):
        raise RuntimeError("session C was downloaded despite the exclusion rule")
    return {"availability": availability, "errors": errors, "raw_download_bytes": sum(path.stat().st_size for path in source_files), "raw_files": len(source_files), "cache_bytes": sum(Path(row["cache_file"]).stat().st_size for row in records), "moabb_version": moabb.__version__}, records


def load_bundle(stage1: Any) -> tuple[Any, np.ndarray]:
    rows, cells = [], []
    for subject in SUBJECTS:
        for session, session_id in SESSION_IDS.items():
            path = CACHE / "prepared" / subject / f"ses-{session}.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as archive:
                x, y = archive["X"].astype(np.float32, copy=False), archive["y"].astype(np.int64, copy=False)
                names = tuple(archive["channel_names"].tolist())
            if x.shape != (200, 13, 1000) or y.shape != (200,) or np.bincount(y, minlength=2).tolist() != [100, 100] or names != CHANNELS:
                raise RuntimeError(f"cache cell invariant failed: {path}")
            cells.append(np.ascontiguousarray(x))
            rows.extend(stage1.core.Row(subject, session_id, str(path), trial, int(y[trial])) for trial in range(200))
    raw = np.concatenate(cells, axis=0)
    if raw.shape != (4800, 13, 1000):
        raise RuntimeError(f"assembled cache shape mismatch: {raw.shape}")
    # name='OpenBMI' invokes the authoritative one-source-session/future-session 4+4 sampler branch.
    return stage1.core.DatasetBundle("OpenBMI", list(SUBJECTS), rows, None, 13), raw


def make_splits() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = np.random.default_rng(SEED).permutation(np.asarray(SUBJECTS))
    outer_parts = np.array_split(shuffled, 5)
    folds, rows = [], []
    for fold_id, part in enumerate(outer_parts):
        outer = sorted(part.tolist())
        remaining = sorted(set(SUBJECTS) - set(outer))
        validation = [str(np.random.default_rng(1000 + fold_id).choice(np.asarray(remaining)))]
        train = sorted(set(remaining) - set(validation))
        cells = [set(train), set(validation), set(outer)]
        if len(train) < 8 or len(validation) != 1 or any(left & right for n, left in enumerate(cells) for right in cells[n + 1:]) or len(set.union(*cells)) != 12:
            raise RuntimeError(f"invalid deterministic split fold={fold_id}")
        folds.append({"fold_id": fold_id, "fold_seed": SEED, "inner_train_subjects": train, "inner_val_subjects": validation, "outer_test_subjects": outer})
        for role, group in (("inner_train", train), ("inner_val", validation), ("outer_test", outer)):
            rows.extend({"seed": SEED, "fold": fold_id, "role": role, "subject": subject} for subject in group)
    outer_all = [subject for fold in folds for subject in fold["outer_test_subjects"]]
    if sorted(outer_all) != list(SUBJECTS) or len(set(outer_all)) != 12:
        raise RuntimeError("outer fold coverage invariant failed")
    return folds, rows


def normalize(raw: np.ndarray, bundle: Any, subjects: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    indices = bundle.indices(subjects, (1,))
    if len(indices) != len(subjects) * 200:
        raise RuntimeError("normalizer must use exactly inner-train S1 trials")
    source = raw[indices].astype(np.float64, copy=False)
    mean, std = source.mean(axis=(0, 2)).astype(np.float32), source.std(axis=(0, 2)).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std, {"subjects": sorted(subjects), "session": "S1=0A", "session_id": 1, "trials": int(len(indices)), "samples_per_channel": int(len(indices) * 1000), "mean_std_sha256": hashlib.sha256(mean.tobytes() + std.tobytes()).hexdigest()}


class GPUCache:
    def __init__(self, raw: np.ndarray, bundle: Any, mean: np.ndarray, std: np.ndarray, device: torch.device):
        x = (raw - mean[None, :, None]) / np.maximum(std[None, :, None], 1e-6)
        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device, non_blocking=True)
        self.y = torch.as_tensor(bundle.labels(np.arange(len(bundle.search_rows), dtype=np.int64)), dtype=torch.long, device=device)
        self.device = device

    def batch(self, indices: list[int] | np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        item = torch.as_tensor(np.asarray(indices, dtype=np.int64), device=self.device)
        return self.x.index_select(0, item), self.y.index_select(0, item)


def build_manifest(stage1: Any, bundle: Any, fold: dict[str, Any]) -> tuple[list[list[dict[str, Any]]], dict[str, Any], Path]:
    stage1.RUNTIME = ROOT / "runtime" / "authoritative_episode_sampler"
    final_fold = {"fold_id": fold["fold_id"], "fold_seed": SEED, "inner_train_subjects": fold["inner_train_subjects"], "inner_val_subjects": fold["inner_val_subjects"], "outer_dev_subjects": fold["outer_test_subjects"]}
    manifest, provenance = stage1.make_manifest(bundle, final_fold)
    path = stage1.RUNTIME / "episode_manifests" / f"openbmi_fold{fold['fold_id']}.json"
    if len(manifest) != 60 or provenance["steps_per_epoch"] != 20 or not path.is_file():
        raise RuntimeError(f"authoritative sampler schedule mismatch: {provenance}")
    audit = {"fold": fold["fold_id"], "epochs": len(manifest), "manifest_sha256": sha256(path), "source": str(EPISODE_SOURCE), "source_sha256": EXPECTED_SHA256[EPISODE_SOURCE], **provenance, "episodes": 0, "support_trials": 0, "query_trials": 0, "subject_disjoint_episodes": 0}
    allowed, prohibited = set(fold["inner_train_subjects"]), set(fold["inner_val_subjects"]) | set(fold["outer_test_subjects"])
    for epoch in manifest:
        for episode in epoch:
            support, query = set(episode["support_subjects"]), set(episode["query_subjects"])
            si, qi = episode["support_indices"], episode["query_indices"]
            if len(support) != 4 or len(query) != 4 or support & query or not (support | query) <= allowed or (support | query) & prohibited or len(si) != 64 or len(qi) != 64:
                raise RuntimeError("authoritative episode invariant failed")
            if any(bundle.search_rows[index].session != 1 or bundle.search_rows[index].subject not in support for index in si):
                raise RuntimeError("support must be S1 and support subject")
            if any(bundle.search_rows[index].session != 2 or bundle.search_rows[index].subject not in query for index in qi):
                raise RuntimeError("query must be S2 and query subject")
            audit["episodes"] += 1; audit["support_trials"] += len(si); audit["query_trials"] += len(qi); audit["subject_disjoint_episodes"] += 1
    return manifest, audit, path


def constructor(name: str, eegnet_module: Any, sire_module: Any) -> nn.Module:
    if name == "EEGNet":
        return eegnet_module.EEGNet(13, 1000)
    if name == "SIRE-EEG":
        return sire_module.CompactLite(13, "bn")
    raise ValueError(name)


def evaluate_session(model: nn.Module, bundle: Any, cache: GPUCache, subjects: list[str], session: int) -> dict[str, dict[str, float]]:
    model.eval(); result: dict[str, dict[str, float]] = {}
    with torch.inference_mode():
        for subject in sorted(subjects):
            indices, y = bundle.indices([subject], (session,)), bundle.labels(bundle.indices([subject], (session,)))
            chunks = []
            for start in range(0, len(indices), 128):
                x, _ = cache.batch(indices[start:start + 128]); chunks.append(model(x)[0].float().cpu().numpy())
            prediction = np.concatenate(chunks).argmax(1)
            result[subject] = {"BA": float(balanced_accuracy_score(y, prediction)), "Macro_F1": float(f1_score(y, prediction, average="macro", zero_division=0)), "trials": int(len(y))}
    return result


def train_one(name: str, fold: dict[str, Any], manifest: list[list[dict[str, Any]]], manifest_path: Path, bundle: Any, cache: GPUCache, eegnet_module: Any, sire_module: Any, models: dict[str, Any], normalizer: dict[str, Any]) -> dict[str, Any]:
    set_seed(SEED)
    model = constructor(name, eegnet_module, sire_module).to(cache.device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if parameter_count != models[name]["parameters_C13_T1000_K2"]:
        raise RuntimeError(f"{name} per-fold authoritative parameter audit failed")
    set_seed(TRAINING_SEED)
    directory = ROOT / "runtime" / "checkpoints" / name / f"fold{fold['fold_id']}_seed0"; directory.mkdir(parents=True, exist_ok=True)
    latest, selected_path = directory / "checkpoint_latest.pt", directory / "selected_best.pt"
    init_sha, manifest_sha = state_hash(copy.deepcopy(model.state_dict())), sha256(manifest_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    amp = cache.device.type == "cuda"; scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, history, best, best_epoch, best_state = 1, [], -float("inf"), None, None
    if latest.exists():
        saved = torch.load(latest, map_location=cache.device, weights_only=False)
        if saved["init_sha256"] != init_sha or saved["manifest_sha256"] != manifest_sha or saved["normalizer_sha256"] != normalizer["mean_std_sha256"]:
            raise RuntimeError(f"unsafe resume checkpoint: {latest}")
        model.load_state_dict(saved["current_state"]); optimizer.load_state_dict(saved["optimizer"]); scaler.load_state_dict(saved["scaler"]); restore_rng(saved["rng"])
        start, history, best, best_epoch, best_state = int(saved["epoch"]) + 1, saved["history"], saved["best_val_BA"], saved["best_epoch"], saved["best_state"]
    started = time.perf_counter()
    for epoch in range(start, EPOCHS + 1):
        model.train(); losses = []
        for episode in manifest[epoch - 1]:
            x, y = cache.batch(episode["support_indices"] + episode["query_indices"])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=cache.device.type, dtype=torch.float16, enabled=amp):
                logits, _ = model(x); loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite ordinary CE: {name}/fold{fold['fold_id']}")
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP); scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach().cpu()))
        validation = evaluate_session(model, bundle, cache, fold["inner_val_subjects"], 2)
        val_ba = float(np.mean([item["BA"] for item in validation.values()]))
        selected = epoch >= MIN_EPOCH and val_ba > best + 1e-12
        if selected:
            best, best_epoch, best_state = val_ba, epoch, copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch, "CE": float(np.mean(losses)), "inner_val_S2_subject_BA": val_ba, "selected": bool(selected)})
        torch.save({"epoch": epoch, "history": history, "best_val_BA": best, "best_epoch": best_epoch, "best_state": best_state, "current_state": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(), "manifest_sha256": manifest_sha, "init_sha256": init_sha, "normalizer_sha256": normalizer["mean_std_sha256"]}, latest)
        if epoch == 1 or epoch % 5 == 0 or selected:
            print(f"[{name} fold={fold['fold_id']} seed=0] epoch={epoch:02d} CE={history[-1]['CE']:.4f} valS2BA={val_ba:.4f}", flush=True)
    if best_state is None or best_epoch is None:
        raise RuntimeError(f"no eligible checkpoint: {name}/fold{fold['fold_id']}")
    model.load_state_dict(best_state); torch.save(model.state_dict(), selected_path)
    record = {"model": name, "fold": fold["fold_id"], "seed": SEED, "selected_epoch": int(best_epoch), "best_inner_val_S2_subject_BA": float(best), "checkpoint_path": str(selected_path), "checkpoint_sha256": sha256(selected_path), "latest_checkpoint_path": str(latest), "init_sha256": init_sha, "manifest_sha256": manifest_sha, "normalizer_sha256": normalizer["mean_std_sha256"], "parameter_count": parameter_count, "amp": amp, "elapsed_seconds": time.perf_counter() - started, "history": history, "inner_train_subjects": fold["inner_train_subjects"], "inner_val_subjects": fold["inner_val_subjects"], "outer_test_subjects": fold["outer_test_subjects"]}
    del model
    if cache.device.type == "cuda": torch.cuda.empty_cache()
    return record


def bootstrap(values: np.ndarray, label: str) -> tuple[float, float, float]:
    rng = np.random.default_rng(stable_seed("BNCI2015-001", SEED, label))
    draws = values[rng.integers(0, len(values), size=(20_000, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def summarize_subjects(rows: list[dict[str, Any]], model: str) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    chosen = [row for row in rows if row["model"] == model]; subjects = sorted({row["subject"] for row in chosen}); result = {}
    for subject in subjects:
        sessions = {row["session"]: row for row in chosen if row["subject"] == subject}
        if set(sessions) != {"S1", "S2"}: raise RuntimeError(f"missing outer evaluation {model}/{subject}")
        result[subject] = {"future S2 BA": float(sessions["S2"]["BA"]), "future S2 Macro-F1": float(sessions["S2"]["Macro_F1"]), "WS-BA": min(float(sessions["S1"]["BA"]), float(sessions["S2"]["BA"]))}
    if subjects != list(SUBJECTS): raise RuntimeError(f"outer subject coverage failed for {model}: {subjects}")
    return {metric: float(np.mean([item[metric] for item in result.values()])) for metric in ("future S2 BA", "future S2 Macro-F1", "WS-BA")}, result


def write_reports(dataset: dict[str, Any], cells: list[dict[str, Any]], models: dict[str, Any], recipe: dict[str, Any], splits: list[dict[str, Any]], audits: list[dict[str, Any]], normalizers: dict[int, dict[str, Any]], summary: dict[str, dict[str, float]] | None = None, contrast: dict[str, tuple[float, float, float]] | None = None, records: list[dict[str, Any]] | None = None, outer_fold: dict[tuple[str, int], float] | None = None) -> None:
    source_note = "MOABB official BNCI downloader and its _convert_mi MAT-to-Raw converter; only explicit A/B URLs were requested."
    protocol = ["# BNCI2015-001 matched episodic protocol", "", "Independent seed-0 external replication. It is separate from all Shin2017B directories and contains no PEEH, PSWA, ScaleCollapse, PRD, BN-state, suppression, adaptation, tuning, extra baseline, or seed-1/2 run.", "", "- Dataset/task: BNCI2015-001 / NEMAR nm000140; right-hand MI (class 0) versus feet MI (class 1).", "- Cohort/session contract: 12 subjects; S1=0A and S2=1B only. 2C is never downloaded, cached, normalized, sampled, selected, or evaluated.", "- Signal contract: fixed native 512 Hz official data; deterministic MNE polyphase resampling to 250 Hz; from each trial marker retain the fixed sustained imagery interval [cue+1.25 s, cue+5.25 s), yielding 13x1000. No frequency search, BNCI-specific band, CSP, or post-hoc crop appears.", "- Splits: seed-0 five-fold outer subject CV; one inner-validation subject and at least eight inner-train subjects per fold. Outer and validation subjects are excluded from episodes, normalization and checkpoint selection.", "- Episode: direct call to final `run_stage1.make_manifest` through its OpenBMI branch. Four S1 support subjects and four disjoint S2 query subjects, each 8 trials/class, produce 64+64=128 ordinary-CE trials.", "- Normalization: channel mean/std fit only on inner-train S1 trials.", "- Frozen recipe: AdamW lr=3e-4, weight decay=5e-4, 60 epochs, clip=5.0, AMP on CUDA, eligible epochs 10..60, S2 inner-validation subject-equal BA, earliest strict tie.", "- Primary endpoint: subject-equal outer S2 BA. Secondary: outer S2 Macro-F1 and WS-BA=min(S1 BA,S2 BA).", "", "## Provenance", f"- {source_note}", f"- MOABB version: `{dataset['moabb_version']}`.", f"- Final SIRE: `{SIRE_SOURCE}` SHA-256 `{EXPECTED_SHA256[SIRE_SOURCE]}`.", f"- Final EEGNet: `{EEGNET_SOURCE}` SHA-256 `{EXPECTED_SHA256[EEGNET_SOURCE]}`.", f"- Final episode source: `{EPISODE_SOURCE}` SHA-256 `{EXPECTED_SHA256[EPISODE_SOURCE]}`."]
    write_text(ROOT / "BNCI2015_001_PROTOCOL.md", protocol)
    audit_lines = ["# BNCI2015-001 data audit", "", f"- Dataset: BNCI2015-001 / NEMAR nm000140; source: {source_note}", f"- MOABB version: `{dataset['moabb_version']}`; official source URL prefix: `https://lampx.tugraz.at/~bci/database/001-2015/`.", "- Expected/observed cohort: 12 subjects, all with common `0A` and `1B`. The public loader lists `2C` only for sub-08, sub-09, sub-10 and sub-11; it was deliberately excluded before downloading.", f"- Native source: 13 EEG channels at 512 Hz; ordered `{list(CHANNELS)}`.", "- Source event map: `right_hand=1`, `feet=2`; cached y map: right_hand=0, feet=1.", "- Source preprocessing recorded by MOABB metadata: 0.5--100 Hz bandpass, 50 Hz notch, CAR. This replication adds no dataset-specific frequency filter or CSP.", "- Deterministic resampling: MNE Raw.resample(250 Hz, npad='auto'); fixed interval [event+1.25s,event+5.25s) yields exactly T=1000.", f"- Official raw A/B downloads: {dataset['raw_files']} MAT files, {dataset['raw_download_bytes']:,} bytes. Session-C files: zero.", f"- Prepared cache: {dataset['cache_bytes']:,} compressed bytes, 24 cells; each X=[200,13,1000] float32 and y=[200] int64.", "", "| Subject | Available public sessions | Used session | Trials | right_hand | feet |", "|---|---|---|---:|---:|---:|"]
    for subject in SUBJECTS:
        available = ", ".join(dataset["availability"][subject])
        for cell in [item for item in cells if item["subject"] == subject]: audit_lines.append(f"| {subject} | {available} | {cell['session']} | {cell['trials']} | {cell['right_hand']} | {cell['feet']} |")
    audit_lines.extend(["", f"Missing/corrupt A/B recordings: {'none' if not dataset['errors'] else '; '.join(dataset['errors'])}."])
    write_text(ROOT / "DATASET_AUDIT.md", audit_lines)
    model_lines = ["# Authoritative model audit", "", "Both classes are imported directly from the final benchmark sources. This experiment does not maintain a copied/reconstructed model.", "", "| Model | Final class/source | SHA-256 | C/T/K parameters | Forward output |", "|---|---|---|---:|---|"]
    for name in ("EEGNet", "SIRE-EEG"):
        item = models[name]; model_lines.append(f"| {name} | `{item['class']}`<br>`{item['source']}` | `{item['sha256']}` | {item['parameters_C13_T1000_K2']:,} | logits [1,2], embedding [1,64] |")
    model_lines.extend(["", "## Frozen SIRE structure", "", "Temporal branches k=15/63/127 (1->8); branch spatial grouped convolutions 8->16 groups=8; concatenated width 48; depthwise k=15 then pointwise 48->64; depthwise k=31 then pointwise 64->64; adaptive pool 8; Linear 512->64; ELU; LayerNorm(64); dropout; Linear 64->2.", "", "SIRE count assertion: 44,872 + 48*13 + 65*2 = 45,626. Any state shape, count, or forward-shape mismatch raises before training. EEGNet's independently audited final count is 33,378.", "", "## Critical state shapes", ""])
    for name in ("EEGNet", "SIRE-EEG"):
        model_lines.append(f"### {name}"); model_lines.extend([f"- `{key}`: {tuple(shape)}" for key, shape in models[name]["state_shapes"].items()]); model_lines.append("")
    write_text(ROOT / "AUTHORITATIVE_MODEL_AUDIT.md", model_lines)
    episode_lines = ["# Episode-manifest audit", "", "The direct final OpenBMI sampler was called after adapting only dataset metadata: `0A -> session 1` and `1B -> session 2`. Its `name='OpenBMI'` branch creates the specified S1 support/S2 future query construction.", "", "| Fold | Inner train | Inner val | Outer | Epochs | Steps/epoch | Episodes | Support trials | Query trials | Disjoint episodes | Manifest SHA-256 |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for fold, item in zip(splits, audits): episode_lines.append(f"| {fold['fold_id']} | {len(fold['inner_train_subjects'])} | {','.join(fold['inner_val_subjects'])} | {','.join(fold['outer_test_subjects'])} | {item['epochs']} | {item['steps_per_epoch']} | {item['episodes']} | {item['support_trials']} | {item['query_trials']} | {item['subject_disjoint_episodes']} | `{item['manifest_sha256']}` |")
    episode_lines.extend(["", "Every audit checks 4 support and 4 non-overlapping query subjects, 8 trials/class/subject, 64+64 trials, S1-only support, S2-only query, and exclusion of inner-val/outer subjects."])
    write_text(ROOT / "EPISODE_MANIFEST_AUDIT.md", episode_lines)
    if summary is None or contrast is None or records is None or outer_fold is None: return
    result_lines = ["# BNCI2015-001 seed-0 result summary", "", "Primary estimates are biological-subject-equal means across the 12 outer-test subjects. The paired bootstrap has 20,000 biological-subject resamples.", "", "| Model | Future S2 BA | Future S2 Macro-F1 | WS-BA |", "|---|---:|---:|---:|"]
    for name in ("EEGNet", "SIRE-EEG"): result_lines.append(f"| {name} | {summary[name]['future S2 BA']:.4f} | {summary[name]['future S2 Macro-F1']:.4f} | {summary[name]['WS-BA']:.4f} |")
    result_lines.extend(["", "## Paired SIRE-EEG minus EEGNet", "", "| Metric | Mean difference | 95% bootstrap CI | Unit | N |", "|---|---:|---:|---|---:|"])
    for metric, (mean, low, high) in contrast.items(): result_lines.append(f"| {metric} | {mean:+.4f} | [{low:+.4f}, {high:+.4f}] | biological subject | 12 |")
    result_lines.extend(["", "## Selected checkpoints and outer S2 BA", "", "| Fold | Model | Selected epoch | Inner-val S2 BA | Outer S2 BA | Checkpoint SHA-256 |", "|---:|---|---:|---:|---:|---|"])
    for record in sorted(records, key=lambda item: (item["fold"], item["model"])): result_lines.append(f"| {record['fold']} | {record['model']} | {record['selected_epoch']} | {record['best_inner_val_S2_subject_BA']:.4f} | {outer_fold[(record['model'], record['fold'])]:.4f} | `{record['checkpoint_sha256']}` |")
    result_lines.extend(["", "Hard stop honored: this report contains seed 0 only; no diagnostics, alternative preprocessing, other baselines, or additional seeds were run."])
    write_text(ROOT / "SEED0_RESULT_SUMMARY.md", result_lines)


def run() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    stage1, eegnet_module, sire_module, recipe = load_authoritative()
    models = audit_models(eegnet_module, sire_module)
    dataset, cells = official_cache()
    bundle, raw = load_bundle(stage1)
    folds, split_rows = make_splits()
    write_csv(ROOT / "subject_split_manifest.csv", split_rows, ["seed", "fold", "role", "subject"])
    normalizers, manifests, manifest_paths, audits = {}, {}, {}, []
    for fold in folds:
        mean, std, normalizer = normalize(raw, bundle, fold["inner_train_subjects"])
        manifest, audit, path = build_manifest(stage1, bundle, fold)
        normalizers[fold["fold_id"]] = (mean, std, normalizer); manifests[fold["fold_id"]] = manifest; manifest_paths[fold["fold_id"]] = path; audits.append(audit)
    write_reports(dataset, cells, models, recipe, folds, audits, {key: value[2] for key, value in normalizers.items()})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = []
    for fold in folds:
        mean, std, norm = normalizers[fold["fold_id"]]; cache = GPUCache(raw, bundle, mean, std, device)
        for name in ("EEGNet", "SIRE-EEG"):
            records.append(train_one(name, fold, manifests[fold["fold_id"]], manifest_paths[fold["fold_id"]], bundle, cache, eegnet_module, sire_module, models, norm))
        del cache
        if device.type == "cuda": torch.cuda.empty_cache()
    write_json(ROOT / "seed0_training_records.json", {"seed": SEED, "device": str(device), "final_repository": str(FINAL_REPO), "final_repository_commit": "cf1db5a6d8f337626544b44f13057e86abe54dc8", "training_recipe": recipe, "normalizers": {str(key): value[2] for key, value in normalizers.items()}, "records": records})
    index = {(item["model"], item["fold"]): item for item in records}; subject_rows: list[dict[str, Any]] = []; outer_fold: dict[tuple[str, int], float] = {}
    for fold in folds:
        mean, std, norm = normalizers[fold["fold_id"]]; cache = GPUCache(raw, bundle, mean, std, device)
        for name in ("EEGNet", "SIRE-EEG"):
            record = index[(name, fold["fold_id"])]; model = constructor(name, eegnet_module, sire_module).to(device); model.load_state_dict(torch.load(record["checkpoint_path"], map_location=device, weights_only=False))
            for session, session_id in (("S1", 1), ("S2", 2)):
                evaluation = evaluate_session(model, bundle, cache, fold["outer_test_subjects"], session_id)
                if session == "S2": outer_fold[(name, fold["fold_id"])] = float(np.mean([metric["BA"] for metric in evaluation.values()]))
                for subject, metric in evaluation.items(): subject_rows.append({"model": name, "fold": fold["fold_id"], "seed": SEED, "checkpoint_sha256": record["checkpoint_sha256"], "selected_epoch": record["selected_epoch"], "normalizer_sha256": norm["mean_std_sha256"], "subject": subject, "session": session, **metric})
            del model
        del cache
        if device.type == "cuda": torch.cuda.empty_cache()
    write_csv(ROOT / "seed0_subject_metrics.csv", subject_rows, ["model", "fold", "seed", "checkpoint_sha256", "selected_epoch", "normalizer_sha256", "subject", "session", "BA", "Macro_F1", "trials"])
    summary, per_subject = {}, {}
    for name in ("EEGNet", "SIRE-EEG"): summary[name], per_subject[name] = summarize_subjects(subject_rows, name)
    contrast = {metric: bootstrap(np.asarray([per_subject["SIRE-EEG"][subject][metric] - per_subject["EEGNet"][subject][metric] for subject in SUBJECTS]), metric) for metric in ("future S2 BA", "WS-BA")}
    summary_rows = [{"row_type": "model", "model": name, "fold": "", **summary[name], "metric": "", "mean": "", "CI95_low": "", "CI95_high": "", "subjects": 12, "bootstrap_resamples": ""} for name in ("EEGNet", "SIRE-EEG")]
    for name in ("EEGNet", "SIRE-EEG"):
        for fold in folds:
            record = index[(name, fold["fold_id"])]; summary_rows.append({"row_type": "fold", "model": name, "fold": fold["fold_id"], "future S2 BA": outer_fold[(name, fold["fold_id"])], "future S2 Macro-F1": "", "WS-BA": "", "metric": "inner_val_S2_BA", "mean": record["best_inner_val_S2_subject_BA"], "CI95_low": "", "CI95_high": "", "subjects": len(fold["outer_test_subjects"]), "bootstrap_resamples": ""})
    for metric, (mean, low, high) in contrast.items(): summary_rows.append({"row_type": "paired_contrast", "model": "SIRE-EEG minus EEGNet", "fold": "", "future S2 BA": "", "future S2 Macro-F1": "", "WS-BA": "", "metric": metric, "mean": mean, "CI95_low": low, "CI95_high": high, "subjects": 12, "bootstrap_resamples": 20_000})
    write_csv(ROOT / "seed0_summary.csv", summary_rows, ["row_type", "model", "fold", "future S2 BA", "future S2 Macro-F1", "WS-BA", "metric", "mean", "CI95_low", "CI95_high", "subjects", "bootstrap_resamples"])
    write_reports(dataset, cells, models, recipe, folds, audits, {key: value[2] for key, value in normalizers.items()}, summary, contrast, records, outer_fold)
    print("BNCI2015_001_MATCHED_EPISODIC_SEED0_COMPLETE", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run", action="store_true", help="run the only permitted seed-0 protocol")
    arguments = parser.parse_args()
    if not arguments.run: raise RuntimeError("pass --run; no other experiment modes exist")
    run(); return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"BNCI2015_001_PROTOCOL_INVALID: {type(error).__name__}: {error}", flush=True)
        raise
