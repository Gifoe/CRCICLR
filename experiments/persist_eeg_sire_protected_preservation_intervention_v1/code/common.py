"""Shared frozen-source helpers for Experiment 2.

The only neural model used here is the exact historical CompactLite instance
that the repository designates as final SIRE-EEG.  This module intentionally
imports the matched PRD implementation and the Appendix-L PEEH selector rather
than providing alternative versions of their semantics.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments/persist_eeg_sire_protected_preservation_intervention_v1"
PROTOCOL = EXP / "protocol"
RUNTIME = EXP / "runtime"
OUT = EXP / "outputs"
PRD_MULTI = ROOT / "experiments/persist_eeg_litebn_prd_multiseed_v1"
PRD_SEED0 = ROOT / "experiments/persist_eeg_litebn_prd_seed0_v1"
PEEH = ROOT / "experiments/persist_eeg_litebn_tfformer_peeh_v1/code/run_peeh.py"

TASKS = ("OpenBMI_MI", "WBCIC_MI")
FOLDS = tuple(range(5))
SEEDS = (0, 1, 2)
PRD_LAMBDA = 0.25
PRESERVE_LAMBDA = 0.25
STAGE2_EPOCHS = 20
STEPS_PER_EPOCH = 20
LR = 3e-4
WEIGHT_DECAY = 5e-4
CLIP = 5.0
RANDOM_POOL_SIZE = 100
RANDOM_CONTROL_IDS = (0, 1, 2)
BOOTSTRAPS = 20_000
EXPECTED_PARAMS = {"OpenBMI_MI": 47978, "WBCIC_MI": 47786}


def import_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_sources():
    """Load exact PRD/evaluation code and repaired Appendix-L selector."""
    prd = import_file("preserve_prd_source", PRD_SEED0 / "code/run_litebn_prd_seed0.py")
    prd.base_runner = prd.module("preserve_prd_base_runner", prd.BASE_RUN)
    prd.base_global = prd.base_runner.base
    prd.csgd_global = prd.module("preserve_prd_csgd", prd.CSGD_RUN)
    runtime = prd.csgd_global.load_runtime()
    peeh = import_file("preserve_appendix_l_peeh", PEEH)
    return prd, runtime, peeh


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") % (2**63 - 1)


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, torch.Tensor):
        return clean(value.detach().cpu().tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(clean(value), sort_keys=True, separators=(",", ":")).encode()


def sha_obj(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    (rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)).to_csv(temporary, index=False)
    os.replace(temporary, path)


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        array = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.numpy().tobytes())
    return digest.hexdigest()


def bn_hash(state: dict[str, torch.Tensor]) -> str:
    return state_hash({key: value for key, value in state.items() if "running_" in key or "num_batches_tracked" in key})


def parameter_count(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def model_definition_hash(model: torch.nn.Module) -> str:
    source = inspect.getsourcefile(model.__class__)
    if source is None:
        raise RuntimeError("model source is not inspectable")
    return sha256_file(Path(source))


def reference_records() -> pd.DataFrame:
    matching = pd.read_csv(PRD_MULTI / "protocol/MATCHING_AUDIT.csv")
    reference = matching[matching.condition == "CE"].copy()
    if len(reference) != 30:
        raise RuntimeError(f"expected 30 CE reference rows, got {len(reference)}")
    if set(reference.task) != set(TASKS) or set(reference.fold) != set(FOLDS) or set(reference.seed) != set(SEEDS):
        raise RuntimeError("CE reference coverage mismatch")
    if reference.duplicated(["task", "fold", "seed"]).any():
        raise RuntimeError("duplicate CE reference cell")
    return reference.sort_values(["task", "fold", "seed"]).reset_index(drop=True)


def load_reference_model(prd, task: str, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise RuntimeError(f"invalid PRD checkpoint schema: {checkpoint}")
    model = prd.base_global.build_model("LiteBN_BASELINE", task)
    model.load_state_dict(payload["state_dict"], strict=True)
    if parameter_count(model) != EXPECTED_PARAMS[task]:
        raise RuntimeError(f"not the official ~48k SIRE model: {task}/{parameter_count(model)}")
    return model.to(device)


def normalizer(prd, task: str, fold: int):
    path = Path(prd.csgd_global.normalizer_path(task, fold))
    if not path.is_file():
        raise FileNotFoundError(path)
    mean, std, metadata = prd.base_global.load_tensor_pair(path)
    return path, mean, std, metadata


def persistence_sessions(task: str) -> tuple[int, int]:
    return (1, 2) if task == "OpenBMI_MI" else (0, 1)


def source_sessions(prd, task: str) -> tuple[int, ...]:
    return tuple(map(int, prd.base_global.TASKS[task]["source_sessions"]))


def subset_bundle(base, bundle, sessions: Iterable[int]):
    wanted = set(map(int, sessions))
    rows = [row for row in bundle.rows if int(row.session) in wanted]
    return base.SignalBundle(bundle.task, bundle.subjects, rows)


def metadata(bundle) -> pd.DataFrame:
    return pd.DataFrame({
        "subject_id": [str(row.subject) for row in bundle.rows],
        "session_id": [int(row.session) for row in bundle.rows],
        "label": [int(row.label) for row in bundle.rows],
    })


def infer_embeddings(model, raw, mean: np.ndarray, std: np.ndarray, device: torch.device) -> np.ndarray:
    values = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(raw.x), 128):
            indices = np.arange(start, min(start + 128, len(raw.x)), dtype=np.int64)
            x, _ = raw.batch(indices, mean, std)
            output = model(x)
            if not isinstance(output, tuple) or len(output) != 2 or output[1].shape[1] != 64:
                raise RuntimeError("SIRE model does not return the required 64-d pre-classifier representation")
            values.append(output[1].float().cpu().numpy())
    return np.concatenate(values, axis=0).astype(np.float32)


def evaluate_sessions(prd, model, bundle, raw, mean, std, subjects, sessions):
    """Identical metrics and inference order to the matched PRD evaluator."""
    model.eval()
    rows = []
    with torch.inference_mode():
        for subject in map(str, subjects):
            for session in sessions:
                indices = bundle.indices([subject], [int(session)])
                if not len(indices):
                    raise RuntimeError(f"empty evaluation cell {bundle.task}/{subject}/S{session}")
                logits = []
                for start in range(0, len(indices), 128):
                    x, _ = raw.batch(indices[start:start + 128], mean, std)
                    logits.append(model(x)[0].float().cpu().numpy())
                metrics = prd.base_global.classification_metrics(bundle.labels(indices), np.concatenate(logits, axis=0))
                rows.append({"subject_id": subject, "session": f"S{session}", "BA": float(metrics["BA"]),
                             "macro_F1": float(metrics["macro_F1"]), "trials": int(len(indices))})
    return rows


def evaluation_bundle(prd, runtime, task: str):
    subjects, sessions = prd.csgd_global.subjects_and_sessions(task)
    bundle = prd.csgd_global.build_wbcic_outer_bundle(runtime)[0] if task == "WBCIC_MI" else prd.base_global.build_bundle(task, subjects)
    return bundle, tuple(map(str, subjects)), tuple(map(int, sessions))


def stage2_bundle(prd, task: str, fold: dict):
    subjects = list(map(str, fold["inner_train_subjects"] + fold["inner_val_subjects"]))
    return prd.base_global.build_bundle(task, subjects)


def target_file(task: str, fold: int, seed: int) -> Path:
    return RUNTIME / "targets" / task / f"fold{fold}_seed{seed}.npz"


def coordinate_cache_file(task: str, fold: int, seed: int) -> Path:
    return RUNTIME / "reference_coordinate_cache" / task / f"fold{fold}_seed{seed}.npz"


def stage_manifest_file(task: str, fold: int, seed: int) -> Path:
    return RUNTIME / "manifests" / task / f"fold{fold}_seed{seed}.json"


def checkpoint_file(task: str, fold: int, seed: int, arm: str) -> Path:
    return RUNTIME / "checkpoints" / task / f"fold{fold}_seed{seed}" / arm / "final.pt"


def target_arrays(task: str, fold: int, seed: int) -> dict[str, Any]:
    path = target_file(task, fold, seed)
    with np.load(path, allow_pickle=False) as values:
        result = {key: np.asarray(values[key]) for key in values.files if key != "metadata"}
        result["metadata"] = json.loads(str(values["metadata"].item()))
    return result


def coordinate_cache(task: str, fold: int, seed: int) -> dict[str, Any]:
    path = coordinate_cache_file(task, fold, seed)
    with np.load(path, allow_pickle=False) as values:
        # The cache's subject/session labels are provenance-only and never
        # participate in optimization.  Restrict the safe numeric load to the
        # two reference tensors actually used by Stage 2, so an old object-
        # dtype label sidecar cannot prevent loading frozen coordinates.
        result = {key: np.asarray(values[key]) for key in ("q0", "h0") if key in values.files}
        result["metadata"] = json.loads(str(values["metadata"].item()))
    return result


def bootstrap(values: np.ndarray, *key: Any) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(stable_seed("preservation-bootstrap", *key))
    means = np.empty(BOOTSTRAPS, dtype=np.float64)
    for start in range(0, BOOTSTRAPS, 2_000):
        stop = min(start + 2_000, BOOTSTRAPS)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))
