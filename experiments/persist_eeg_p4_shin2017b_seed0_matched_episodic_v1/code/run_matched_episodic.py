"""Seed-0 matched episodic Shin2017B comparison.

This is intentionally separate from the existing IID P4 replication.  It
imports the final benchmark model classes and calls the final episode-manifest
builder directly through a WBCIC-shaped metadata adapter: session IDs 0, 1,
and 2 stand for Shin S1, S2, and S3 respectively.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score


ROOT = Path(os.environ.get("MATCHED_EPISODIC_ROOT", "/root/p4_shin2017b_seed0_matched_episodic_v1"))
CACHE = Path(os.environ.get("SHIN2017B_CACHE", "/root/shin2017b_eeg_cache"))
IID_ROOT = Path(os.environ.get("P4_IID_ROOT", "/root/p4_shin2017b_seed0"))
FINAL_REPO = Path(os.environ.get("SIRE_FINAL_REPO", "/root/rivermind-data/CRCICLR_FINAL_CONFIRM_WORK"))

SEED, TRAINING_SEED = 0, 100_000
EPOCHS, MIN_EPOCH, EPISODE_TRIALS = 60, 10, 128
LR, WEIGHT_DECAY, GRADIENT_CLIP = 3e-4, 5e-4, 5.0
SESSIONS = ("1arithmetic", "3arithmetic", "5arithmetic")
SESSION_TO_ID = {name: index for index, name in enumerate(SESSIONS)}
EXPECTED_SPLIT_SHA256 = "f1088d106dfbbfded0f4f69fae3ca90ae04976cf8e59b1204f1f7afa3dd4daea"

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


def stable_seed(*parts: Any) -> int:
    text = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "little") % (2**32 - 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state() -> dict[str, Any]:
    result: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state_all()
    return result


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def state_hash(state: dict[str, torch.Tensor]) -> str:
    import io
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def direct_import(name: str, path: Path, prepend: list[Path] = []) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"authoritative source missing: {path}")
    inserted = [str(item) for item in prepend]
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
        for item in inserted:
            if item in sys.path:
                sys.path.remove(item)


def load_authoritative() -> tuple[Any, Any, Any, dict[str, Any]]:
    """Hard-pin all final source files, then import the classes and sampler."""
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
            raise RuntimeError(f"final training config mismatch for {key}: {config.get(key)!r} != {expected!r}")

    old_repo = os.environ.get("R2EEG_REPO")
    os.environ["R2EEG_REPO"] = str(FINAL_REPO)
    try:
        # The final grid imports this exact module as run_stage1 and calls
        # make_manifest; keeping the canonical module name makes the same
        # source visible to the CompactLite module as well.
        stage1 = direct_import("run_stage1", EPISODE_SOURCE, [EPISODE_SOURCE.parent])
        eegnet_module = direct_import("matched_final_eegnet", EEGNET_SOURCE, [EEGNET_SOURCE.parent])
        sire_module = direct_import("matched_final_compact", SIRE_SOURCE, [SIRE_SOURCE.parent])
    finally:
        if old_repo is None:
            os.environ.pop("R2EEG_REPO", None)
        else:
            os.environ["R2EEG_REPO"] = old_repo
    if not hasattr(stage1, "make_manifest") or not hasattr(stage1, "core"):
        raise RuntimeError("final episode source lacks make_manifest/core")
    if not hasattr(eegnet_module, "EEGNet") or not hasattr(sire_module, "CompactLite"):
        raise RuntimeError("final model source lacks EEGNet or CompactLite")
    return stage1, eegnet_module, sire_module, {"config": config, "required": required}


def audit_models(eegnet_module: Any, sire_module: Any) -> dict[str, Any]:
    eegnet = eegnet_module.EEGNet(30, 2000)
    sire = sire_module.CompactLite(30, "bn")
    eeg_state, sire_state = eegnet.state_dict(), sire.state_dict()
    eeg_expected = {
        "temporal.weight": (8, 1, 1, 64), "spatial.weight": (16, 1, 30, 1),
        "depth.weight": (16, 1, 1, 16), "point.weight": (16, 16, 1, 1),
        "embedding.0.weight": (64, 992), "embedding.0.bias": (64,),
        "embedding.2.weight": (64,), "embedding.2.bias": (64,),
        "head.weight": (2, 64), "head.bias": (2,),
    }
    sire_expected = {
        "temporal.0.weight": (8, 1, 1, 15), "temporal.1.weight": (8, 1, 1, 63),
        "temporal.2.weight": (8, 1, 1, 127), "spatial.0.weight": (16, 1, 30, 1),
        "spatial.1.weight": (16, 1, 30, 1), "spatial.2.weight": (16, 1, 30, 1),
        "depth1.weight": (48, 1, 1, 15), "point1.weight": (64, 48, 1, 1),
        "depth2.weight": (64, 1, 1, 31), "point2.weight": (64, 64, 1, 1),
        "embedding.0.weight": (64, 512), "embedding.0.bias": (64,),
        "embedding.2.weight": (64,), "embedding.2.bias": (64,),
        "head.weight": (2, 64), "head.bias": (2,),
    }
    for label, state, expected in (("EEGNet", eeg_state, eeg_expected), ("SIRE-EEG", sire_state, sire_expected)):
        for key, shape in expected.items():
            if key not in state or tuple(state[key].shape) != shape:
                raise RuntimeError(f"{label} architecture audit mismatch: {key} expected {shape}, got {tuple(state[key].shape) if key in state else None}")
    eeg_count = sum(parameter.numel() for parameter in eegnet.parameters() if parameter.requires_grad)
    sire_count = sum(parameter.numel() for parameter in sire.parameters() if parameter.requires_grad)
    if eeg_count != 65_394:
        raise RuntimeError(f"canonical EEGNet C=30,T=2000 count mismatch: {eeg_count}")
    if sire_count != 46_442:
        raise RuntimeError(f"final SIRE C=30,K=2 count mismatch: {sire_count}")
    with torch.inference_mode():
        for label, model in (("EEGNet", eegnet), ("SIRE-EEG", sire)):
            logits, embedding = model(torch.zeros(1, 30, 2000))
            if tuple(logits.shape) != (1, 2) or tuple(embedding.shape) != (1, 64):
                raise RuntimeError(f"{label} forward audit failed: {tuple(logits.shape)}, {tuple(embedding.shape)}")
    return {
        "EEGNet": {"class": "eegnet_locked.EEGNet", "source": str(EEGNET_SOURCE), "source_sha256": EXPECTED_SHA256[EEGNET_SOURCE], "trainable_parameters_C30_T2000_K2": eeg_count, "critical_state_shapes": eeg_expected},
        "SIRE-EEG": {"class": "run_carrier_screen.CompactLite(kind='bn')", "source": str(SIRE_SOURCE), "source_sha256": EXPECTED_SHA256[SIRE_SOURCE], "trainable_parameters_C30_T2000_K2": sire_count, "parameter_formula": "44872 + 48*C + 65*K", "critical_state_shapes": sire_expected},
    }


def load_splits() -> list[dict[str, Any]]:
    source = IID_ROOT / "subject_split_manifest.csv"
    if not source.is_file():
        raise FileNotFoundError(f"existing P4 split manifest missing: {source}")
    if sha256(source) != EXPECTED_SPLIT_SHA256:
        raise RuntimeError("existing 5-fold split manifest drifted; refusing to generate a new split")
    rows = list(csv.DictReader(source.open(newline="", encoding="utf-8")))
    copied = ROOT / "subject_split_manifest.csv"
    copied.parent.mkdir(parents=True, exist_ok=True)
    if not copied.exists():
        shutil.copy2(source, copied)
    if sha256(copied) != EXPECTED_SPLIT_SHA256:
        raise RuntimeError("matched experiment split copy does not match the frozen IID manifest")
    folds = []
    for fold in range(5):
        roles: dict[str, list[str]] = {"train": [], "inner_val": [], "outer_test": []}
        for row in rows:
            if int(row["fold"]) == fold:
                roles[row["role"]].append(f"sub-{int(row['subject']):02d}")
        if sorted(map(int, (name[-2:] for name in roles["train"]))) != sorted(map(int, (name[-2:] for name in roles["train"]))):
            raise RuntimeError("unreachable split parsing failure")
        if len(roles["train"]) not in (18, 19) or len(roles["inner_val"]) != 5 or len(roles["outer_test"]) not in (5, 6):
            raise RuntimeError(f"unexpected split sizes fold {fold}: { {key: len(value) for key, value in roles.items()} }")
        sets = [set(roles[key]) for key in ("train", "inner_val", "outer_test")]
        if any(left & right for index, left in enumerate(sets) for right in sets[index + 1:]) or len(set.union(*sets)) != 29:
            raise RuntimeError(f"subject-disjoint split invariant failed for fold {fold}")
        folds.append({"fold_id": fold, "fold_seed": 0, "inner_train_subjects": roles["train"], "inner_val_subjects": roles["inner_val"], "outer_dev_subjects": roles["outer_test"]})
    outer = [subject for fold in folds for subject in fold["outer_dev_subjects"]]
    if sorted(outer) != [f"sub-{index:02d}" for index in range(1, 30)] or len(set(outer)) != 29:
        raise RuntimeError("outer subjects must cover the 29 subjects exactly once")
    return folds


def load_bundle(stage1: Any) -> tuple[Any, np.ndarray, tuple[str, ...]]:
    """Create a final-sampler-compatible metadata bundle plus a raw array."""
    rows, values, channels = [], [], None
    for subject in range(1, 30):
        subject_id = f"sub-{subject:02d}"
        for session_name, session_id in SESSION_TO_ID.items():
            path = CACHE / subject_id / f"ses-{session_name}_eeg_task-mental-arithmetic.npz"
            if not path.is_file():
                raise FileNotFoundError(f"EEG cache cell missing: {path}")
            with np.load(path, allow_pickle=False) as archive:
                x = archive["X"].astype(np.float32, copy=False)
                raw_y = archive["y"]
                names = tuple(archive["channel_names"].tolist())
            y = (raw_y == 3).astype(np.int64)
            if x.shape != (20, 30, 2000) or y.shape != (20,) or set(map(int, np.unique(y))) != {0, 1}:
                raise RuntimeError(f"invalid Shin cell {subject_id}/{session_name}: {x.shape}, {y.shape}, {np.unique(y)}")
            if channels is None:
                channels = names
            if names != channels:
                raise RuntimeError(f"channel order drift in {path}")
            values.append(np.ascontiguousarray(x))
            rows.extend(stage1.core.Row(subject_id, session_id, str(path), trial, int(y[trial])) for trial in range(20))
    raw = np.concatenate(values, axis=0)
    if raw.shape != (29 * 3 * 20, 30, 2000) or channels is None:
        raise RuntimeError(f"invalid assembled Shin tensor: {raw.shape}")
    # The final make_manifest only uses DatasetBundle.search_rows, .indices,
    # and .labels.  `name='WBCIC'` deliberately selects its two-source-session
    # 4-per-class support / one-future-session 8-per-class query branch.
    bundle = stage1.core.DatasetBundle("WBCIC", [f"sub-{index:02d}" for index in range(1, 30)], rows, None, 30)
    return bundle, raw, channels


def normalize(raw: np.ndarray, bundle: Any, subjects: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source_indices = bundle.indices(subjects, (0, 1))
    if len(source_indices) != len(subjects) * 2 * 20:
        raise RuntimeError("normalizer source count must be inner-train S1+S2 only")
    source = raw[source_indices].astype(np.float64, copy=False)
    mean = source.mean(axis=(0, 2)).astype(np.float32)
    std = source.std(axis=(0, 2)).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std, {
        "subjects": sorted(subjects), "sessions": ["S1", "S2"], "session_ids": [0, 1],
        "trials": int(len(source_indices)), "samples_per_channel": int(len(source_indices) * raw.shape[-1]),
        "mean_std_sha256": hashlib.sha256(mean.tobytes() + std.tobytes()).hexdigest(),
    }


class GPUCache:
    def __init__(self, raw: np.ndarray, bundle: Any, mean: np.ndarray, std: np.ndarray, device: torch.device):
        x = (raw - mean[None, :, None]) / np.maximum(std[None, :, None], 1e-6)
        self.x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device, non_blocking=True)
        self.y = torch.as_tensor(bundle.labels(np.arange(len(bundle.search_rows), dtype=np.int64)), dtype=torch.long, device=device)
        self.device = device

    def batch(self, indices: list[int] | np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        item = torch.as_tensor(np.asarray(indices, dtype=np.int64), device=self.device)
        return self.x.index_select(0, item), self.y.index_select(0, item)


def build_manifest(stage1: Any, bundle: Any, fold: dict[str, Any]) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    stage1.RUNTIME = ROOT / "runtime" / "final_sampler"
    manifest, provenance = stage1.make_manifest(bundle, fold)
    if len(manifest) != EPOCHS or provenance["steps_per_epoch"] != 20:
        raise RuntimeError(f"final episode sampler returned wrong schedule: {len(manifest)} epochs, {provenance}")
    audit = {"fold": fold["fold_id"], "manifest_source": str(EPISODE_SOURCE), "manifest_source_sha256": EXPECTED_SHA256[EPISODE_SOURCE], **provenance,
             "support_session_mapping": {"0": "S1=1arithmetic", "1": "S2=3arithmetic"}, "query_session_mapping": {"2": "S3=5arithmetic"},
             "episodes": 0, "subject_disjoint_episodes": 0, "support_trials": 0, "query_trials": 0}
    legal = set(fold["inner_train_subjects"])
    prohibited = set(fold["inner_val_subjects"]) | set(fold["outer_dev_subjects"])
    for epoch in manifest:
        for episode in epoch:
            support, query = set(episode["support_subjects"]), set(episode["query_subjects"])
            si, qi = episode["support_indices"], episode["query_indices"]
            if len(support) != 4 or len(query) != 4 or support & query or not (support | query) <= legal or (support | query) & prohibited:
                raise RuntimeError("final sampler violated subject-disjoint inner-train episode construction")
            if len(si) != 64 or len(qi) != 64:
                raise RuntimeError("final sampler did not produce a 128-trial episode")
            for index in si:
                row = bundle.search_rows[index]
                if row.subject not in support or row.session not in (0, 1):
                    raise RuntimeError("support episode contains a non-S1/S2 or wrong-subject trial")
            for index in qi:
                row = bundle.search_rows[index]
                if row.subject not in query or row.session != 2:
                    raise RuntimeError("query episode contains a non-S3 or wrong-subject trial")
            audit["episodes"] += 1
            audit["subject_disjoint_episodes"] += 1
            audit["support_trials"] += len(si)
            audit["query_trials"] += len(qi)
    return manifest, audit


def constructor(name: str, eegnet_module: Any, sire_module: Any) -> nn.Module:
    if name == "EEGNet":
        return eegnet_module.EEGNet(30, 2000)
    if name == "SIRE-EEG":
        return sire_module.CompactLite(30, "bn")
    raise ValueError(name)


def evaluate_session(model: nn.Module, bundle: Any, cache: GPUCache, subjects: list[str], session: int) -> dict[str, dict[str, float]]:
    model.eval()
    result: dict[str, dict[str, float]] = {}
    with torch.inference_mode():
        for subject in sorted(subjects):
            indices = bundle.indices([subject], (session,))
            y = bundle.labels(indices)
            logits = []
            for start in range(0, len(indices), 128):
                x, _ = cache.batch(indices[start:start + 128])
                logits.append(model(x)[0].float().cpu().numpy())
            prediction = np.concatenate(logits).argmax(1)
            result[subject] = {"BA": float(balanced_accuracy_score(y, prediction)), "Macro_F1": float(f1_score(y, prediction, average="macro", zero_division=0)), "trials": int(len(y))}
    return result


def train_one(name: str, fold: dict[str, Any], manifest: list[list[dict[str, Any]]], bundle: Any, cache: GPUCache, eegnet_module: Any, sire_module: Any, model_audit: dict[str, Any]) -> dict[str, Any]:
    set_seed(SEED)
    model = constructor(name, eegnet_module, sire_module).to(cache.device)
    observed_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if observed_parameters != model_audit[name]["trainable_parameters_C30_T2000_K2"]:
        raise RuntimeError(f"{name} training instantiation failed authoritative parameter audit")
    set_seed(TRAINING_SEED)
    directory = ROOT / "runtime" / "checkpoints" / name / f"fold{fold['fold_id']}_seed0"
    directory.mkdir(parents=True, exist_ok=True)
    latest, selected_path = directory / "checkpoint_latest.pt", directory / "selected_best.pt"
    init_sha = state_hash(copy.deepcopy(model.state_dict()))
    manifest_sha = sha256(Path(str(ROOT / "runtime" / "final_sampler" / "episode_manifests" / f"wbcic_fold{fold['fold_id']}.json")))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    amp = cache.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, history, best, best_epoch, best_state = 1, [], -float("inf"), None, None
    if latest.exists():
        saved = torch.load(latest, map_location=cache.device, weights_only=False)
        if saved["init_sha256"] != init_sha or saved["manifest_sha256"] != manifest_sha:
            raise RuntimeError(f"resume invariant mismatch: {latest}")
        model.load_state_dict(saved["current_state"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        start, history = int(saved["epoch"]) + 1, saved["history"]
        best, best_epoch, best_state = saved["best_val_BA"], saved["best_epoch"], saved["best_state"]
        restore_rng(saved["rng"])
    started = time.perf_counter()
    for epoch in range(start, EPOCHS + 1):
        model.train()
        losses = []
        for episode in manifest[epoch - 1]:
            x, y = cache.batch(episode["support_indices"] + episode["query_indices"])
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=cache.device.type, dtype=torch.float16, enabled=amp):
                logits, _ = model(x)
                loss = F.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite ordinary CE for {name}, fold {fold['fold_id']}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        val = evaluate_session(model, bundle, cache, fold["inner_val_subjects"], 2)
        val_ba = float(np.mean([row["BA"] for row in val.values()]))
        selected = epoch >= MIN_EPOCH and val_ba > best + 1e-12
        if selected:
            best, best_epoch, best_state = val_ba, epoch, copy.deepcopy(model.state_dict())
        history.append({"epoch": epoch, "CE": float(np.mean(losses)), "inner_val_subject_BA": val_ba, "selected": bool(selected)})
        torch.save({"epoch": epoch, "history": history, "best_val_BA": best, "best_epoch": best_epoch, "best_state": best_state, "current_state": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(), "manifest_sha256": manifest_sha, "init_sha256": init_sha}, latest)
        if epoch == 1 or epoch % 5 == 0 or selected:
            print(f"[{name} fold={fold['fold_id']} seed=0] epoch={epoch:02d} CE={history[-1]['CE']:.4f} valBA={val_ba:.4f}", flush=True)
    if best_state is None or best_epoch is None:
        raise RuntimeError(f"no eligible selected checkpoint for {name} fold {fold['fold_id']}")
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), selected_path)
    record = {"model": name, "fold": fold["fold_id"], "seed": SEED, "selected_epoch": int(best_epoch), "best_inner_val_subject_BA": float(best), "history": history, "checkpoint_path": str(selected_path), "checkpoint_sha256": sha256(selected_path), "latest_checkpoint_path": str(latest), "init_sha256": init_sha, "manifest_sha256": manifest_sha, "parameter_count": observed_parameters, "amp": amp, "elapsed_seconds": time.perf_counter() - started, "inner_train_subjects": fold["inner_train_subjects"], "inner_val_subjects": fold["inner_val_subjects"], "outer_subjects": fold["outer_dev_subjects"]}
    del model
    if cache.device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def bootstrap(values: np.ndarray, label: str) -> tuple[float, float, float]:
    rng = np.random.default_rng(stable_seed("matched-episodic", SEED, label))
    draws = values[rng.integers(0, len(values), size=(20_000, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def summary_from_rows(rows: list[dict[str, Any]], model: str) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    selected = [row for row in rows if row["model"] == model]
    subjects = sorted({row["subject"] for row in selected})
    by_subject: dict[str, dict[str, float]] = {}
    for subject in subjects:
        sessions = {row["session"]: row for row in selected if row["subject"] == subject}
        if set(sessions) != {"S1", "S2", "S3"}:
            raise RuntimeError(f"incomplete outer evaluation for {model}/{subject}")
        by_subject[subject] = {"future BA": float(sessions["S3"]["BA"]), "future Macro-F1": float(sessions["S3"]["Macro_F1"]), "WS-BA": min(float(sessions[name]["BA"]) for name in ("S1", "S2", "S3"))}
    if len(by_subject) != 29:
        raise RuntimeError(f"{model} requires exactly 29 outer-subject results")
    return {metric: float(np.mean([values[metric] for values in by_subject.values()])) for metric in ("future BA", "future Macro-F1", "WS-BA")}, by_subject


def iid_summary() -> tuple[dict[str, dict[str, float]], str]:
    path = IID_ROOT / "seed0_fullmodel_subject_metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(f"IID control subject metrics missing: {path}")
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    converted = [{"model": row["model"], "subject": row["subject"], "session": row["session"], "BA": float(row["BA"]), "Macro_F1": float(row["Macro_F1"])} for row in rows]
    output = {name: summary_from_rows(converted, name)[0] for name in ("EEGNet", "SIRE-EEG")}
    return output, sha256(path)


def write_protocol(model_audit: dict[str, Any]) -> None:
    text = [
        "# Matched episodic Shin2017B protocol", "",
        "This is a clean, separate seed-0 comparison. The existing IID P4 directory is read only and is retained as a control.", "",
        "- Dataset: Shin2017B / nm000268, EEG only; 29 subjects, 30 channels in cached order, 2,000 samples/trial.",
        "- Session mapping: S1=`1arithmetic`, S2=`3arithmetic`, S3=`5arithmetic`.",
        "- The existing five subject-disjoint folds are copied byte-for-byte from the IID P4 manifest (SHA-256 `f1088d106dfbbfded0f4f69fae3ca90ae04976cf8e59b1204f1f7afa3dd4daea`).",
        "- A source episode uses four inner-train support subjects and S1/S2, sampling four trials per class per session per subject (64 trials). A query uses four different inner-train subjects and S3, sampling eight trials per class per subject (64 trials). Thus every episode is 128 trials and support/query subjects are disjoint.",
        "- The final `run_stage1.make_manifest` is called directly through its WBCIC metadata branch: numerical sessions 0/1/2 are exactly S1/S2/S3. This preserves its 20 steps/epoch, 60 epochs, fixed RNG formula, and 4/4 subject construction rather than using shuffled IID mini-batches.",
        "- Mean/std is fit only on inner-train S1+S2. Inner-validation and outer subjects never occur in an episode or checkpoint selection. Query S3 is only from inner-train subjects.",
        "- Both directly imported final models use AdamW (lr 3e-4, weight decay 5e-4), ordinary cross entropy over the complete support+query episode, gradient clip 5.0, CUDA AMP when available, 60 epochs, and S3 inner-validation subject-equal BA selection over epochs 10--60 with earliest strict tie.",
        "- All outer S1/S2/S3 evaluations happen only after all 10 fold/model trainings finish; the frozen selected checkpoint is used.",
        "",
        "## Imported authoritative models", "",
        f"- EEGNet: `{model_audit['EEGNet']['source']}`; SHA-256 `{model_audit['EEGNet']['source_sha256']}`; C=30,T=2000,K=2 parameters {model_audit['EEGNet']['trainable_parameters_C30_T2000_K2']:,}.",
        f"- SIRE-EEG/CompactLite: `{model_audit['SIRE-EEG']['source']}`; SHA-256 `{model_audit['SIRE-EEG']['source_sha256']}`; C=30,T=2000,K=2 parameters {model_audit['SIRE-EEG']['trainable_parameters_C30_T2000_K2']:,}.",
        f"- Final episode source: `{EPISODE_SOURCE}`; SHA-256 `{EXPECTED_SHA256[EPISODE_SOURCE]}`. Final grid source: `{TRAINING_SOURCE}`; SHA-256 `{EXPECTED_SHA256[TRAINING_SOURCE]}`.",
    ]
    (ROOT / "MATCHED_EPISODIC_PROTOCOL.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def write_comparison(iid: dict[str, dict[str, float]], matched: dict[str, dict[str, float]], contrast: dict[str, tuple[float, float, float]], iid_hash: str) -> None:
    rows = ["# IID versus matched episodic comparison", "", f"The IID control is read from `{IID_ROOT / 'seed0_fullmodel_subject_metrics.csv'}` (SHA-256 `{iid_hash}`); it has not been overwritten.", "", "| Model | IID future BA | Episodic future BA | Change | IID Macro-F1 | Episodic Macro-F1 | Change | IID WS-BA | Episodic WS-BA | Change |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for model in ("EEGNet", "SIRE-EEG"):
        old, new = iid[model], matched[model]
        rows.append(f"| {model} | {old['future BA']:.3f} | {new['future BA']:.3f} | {new['future BA'] - old['future BA']:+.3f} | {old['future Macro-F1']:.3f} | {new['future Macro-F1']:.3f} | {new['future Macro-F1'] - old['future Macro-F1']:+.3f} | {old['WS-BA']:.3f} | {new['WS-BA']:.3f} | {new['WS-BA'] - old['WS-BA']:+.3f} |")
    rows.extend(["", "## Episodic paired SIRE-EEG minus EEGNet", "", "| Metric | Mean difference | 95% bootstrap CI | Biological subjects | Bootstrap resamples |", "|---|---:|---:|---:|---:|"])
    for metric in ("future BA", "WS-BA"):
        mean, low, high = contrast[metric]
        rows.append(f"| {metric} | {mean:+.3f} | [{low:+.3f}, {high:+.3f}] | 29 | 20,000 |")
    rows.extend(["", "No diagnostics, ScaleCollapse, additional baseline, or seed-1/2 run is included in this experiment."])
    (ROOT / "IID_VS_EPISODIC_COMPARISON.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def run() -> None:
    if ROOT.resolve() == IID_ROOT.resolve():
        raise RuntimeError("matched episodic root must differ from the IID P4 root")
    ROOT.mkdir(parents=True, exist_ok=True)
    stage1, eegnet_module, sire_module, training_audit = load_authoritative()
    model_audit = audit_models(eegnet_module, sire_module)
    folds = load_splits()
    bundle, raw, channel_names = load_bundle(stage1)
    write_protocol(model_audit)
    write_json(ROOT / "authoritative_implementation_audit.json", {"final_repository": str(FINAL_REPO), "final_repository_commit": "cf1db5a6d8f337626544b44f13057e86abe54dc8", "sources": {str(path): digest for path, digest in EXPECTED_SHA256.items()}, "training_recipe": training_audit["required"], "models": model_audit, "Shin_adaptations_only": {"channels": 30, "samples": 2000, "classes": 2, "episode_session_mapping": {"0": "S1", "1": "S2", "2": "S3"}}})
    write_json(ROOT / "dataset_audit.json", {"dataset": "Shin2017B/nm000268", "modalities": ["EEG"], "subjects": 29, "sessions": list(SESSIONS), "shape_per_subject_session": [20, 30, 2000], "channel_names": list(channel_names), "cache_manifest_sha256": sha256(CACHE / "manifest.json")})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalizers, manifests, manifest_audits = {}, {}, []
    for fold in folds:
        mean, std, norm = normalize(raw, bundle, fold["inner_train_subjects"])
        manifest, manifest_audit = build_manifest(stage1, bundle, fold)
        normalizers[fold["fold_id"]] = (mean, std, norm)
        manifests[fold["fold_id"]] = manifest
        manifest_audits.append(manifest_audit)
    write_json(ROOT / "episode_manifest_audit.json", {"seed": 0, "source": str(EPISODE_SOURCE), "source_sha256": EXPECTED_SHA256[EPISODE_SOURCE], "folds": manifest_audits})
    write_json(ROOT / "normalizer_audit.json", {str(fold["fold_id"]): normalizers[fold["fold_id"]][2] for fold in folds})

    training_records = []
    # Train every fold/model before an outer label is evaluated.
    for fold in folds:
        mean, std, _ = normalizers[fold["fold_id"]]
        cache = GPUCache(raw, bundle, mean, std, device)
        for name in ("EEGNet", "SIRE-EEG"):
            training_records.append(train_one(name, fold, manifests[fold["fold_id"]], bundle, cache, eegnet_module, sire_module, model_audit))
        del cache
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_json(ROOT / "seed0_matched_episode_training_records.json", {"seed": 0, "device": str(device), "records": training_records})

    subject_rows: list[dict[str, Any]] = []
    record_index = {(record["model"], record["fold"]): record for record in training_records}
    for fold in folds:
        mean, std, norm = normalizers[fold["fold_id"]]
        cache = GPUCache(raw, bundle, mean, std, device)
        for name in ("EEGNet", "SIRE-EEG"):
            record = record_index[(name, fold["fold_id"])]
            model = constructor(name, eegnet_module, sire_module).to(device)
            model.load_state_dict(torch.load(record["checkpoint_path"], map_location=device, weights_only=False))
            for session_name, session_id in (("S1", 0), ("S2", 1), ("S3", 2)):
                evaluation = evaluate_session(model, bundle, cache, fold["outer_dev_subjects"], session_id)
                for subject, metric in evaluation.items():
                    subject_rows.append({"model": name, "fold": fold["fold_id"], "seed": 0, "checkpoint_sha256": record["checkpoint_sha256"], "selected_epoch": record["selected_epoch"], "normalizer_sha256": norm["mean_std_sha256"], "subject": subject, "session": session_name, **metric})
            del model
        del cache
        if device.type == "cuda":
            torch.cuda.empty_cache()
    fields = ["model", "fold", "seed", "checkpoint_sha256", "selected_epoch", "normalizer_sha256", "subject", "session", "BA", "Macro_F1", "trials"]
    write_csv(ROOT / "seed0_matched_episode_subject_metrics.csv", subject_rows, fields)
    matched, per_subject = {}, {}
    for name in ("EEGNet", "SIRE-EEG"):
        matched[name], per_subject[name] = summary_from_rows(subject_rows, name)
    contrast = {metric: bootstrap(np.asarray([per_subject["SIRE-EEG"][subject][metric] - per_subject["EEGNet"][subject][metric] for subject in sorted(per_subject["EEGNet"])]), metric) for metric in ("future BA", "WS-BA")}
    summary_rows = []
    for name in ("EEGNet", "SIRE-EEG"):
        summary_rows.append({"model": name, **matched[name], "subjects": 29, "metric": "", "mean": "", "CI95_low": "", "CI95_high": "", "bootstrap_draws": ""})
    for metric in ("future BA", "WS-BA"):
        mean, low, high = contrast[metric]
        summary_rows.append({"model": "SIRE-EEG minus EEGNet", "future BA": "", "future Macro-F1": "", "WS-BA": "", "subjects": 29, "metric": f"{metric} contrast", "mean": mean, "CI95_low": low, "CI95_high": high, "bootstrap_draws": 20_000})
    summary_fields = ["model", "future BA", "future Macro-F1", "WS-BA", "subjects", "metric", "mean", "CI95_low", "CI95_high", "bootstrap_draws"]
    write_csv(ROOT / "seed0_matched_episode_summary.csv", summary_rows, summary_fields)
    iid, iid_hash = iid_summary()
    write_comparison(iid, matched, contrast, iid_hash)
    print("MATCHED_EPISODIC_SHIN2017B_SEED0_COMPLETE", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="run the isolated seed-0 experiment")
    arguments = parser.parse_args()
    if not arguments.run:
        raise RuntimeError("pass --run; this script has no other experiment modes")
    run()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"MATCHED_EPISODIC_PROTOCOL_INVALID: {type(error).__name__}: {error}", flush=True)
        raise
