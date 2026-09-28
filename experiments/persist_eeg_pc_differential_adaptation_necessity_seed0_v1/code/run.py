"""Frozen-backbone, source-only P/C differential adaptation audit.

Commands are deliberately staged: `prepare` reads TRAIN only, `select` uses
TRAIN pseudo-target labels only, and `outer` reads OUTER only after selection
has been sealed. No code path opens the formal final-heldout EEG cohort.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler

EXP = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("PCDA_REPO", str(EXP.parents[1]))).resolve()
RUNTIME = Path(os.environ.get("PCDA_RUNTIME", str(REPO.parent / "pc_differential_adaptation_runtime"))).resolve()
SDG_CODE = Path(os.environ.get("PCDA_SDG_CODE", str(REPO / "experiments/persist_eeg_shared_decision_geometry_characterization_seed0_v1/code/run.py")))
EEGNET_PEEH_CODE = Path(os.environ.get("PCDA_EEGNET_PEEH_CODE", str(REPO / "experiments/persist_eeg_crossbackbone_peeh_v1/code/run_crossbackbone_peeh.py")))
CONFORMER_PEEH_CODE = Path(os.environ.get("PCDA_CONFORMER_PEEH_CODE", str(REPO / "experiments/persist_eeg_eegconformer_fbcnet_peeh_pswa_v1/code/run_eegconformer_fbcnet_peeh.py")))
PEEH_EEGNET_RUNTIME = Path(os.environ.get("PCDA_PEEH_EEGNET_RUNTIME", r"D:\nips-temp\TotalP\P1\crossbackbone_peeh_runtime"))
PEEH_CONFORMER_RUNTIME = Path(os.environ.get("PCDA_PEEH_CONFORMER_RUNTIME", r"D:\nips-temp\TotalP\P1\eegconformer_fbcnet_peeh_runtime"))
PROTOCOL = json.loads((EXP / "protocol/PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
ALPHAS = np.asarray(PROTOCOL["alpha_grid"], dtype=np.float64)
LAMBDAS = tuple(map(float, PROTOCOL["shrinkage_lambda_grid"]))
FOLDS = range(5)
CELLS = tuple(tuple(x) for x in PROTOCOL["cells"])
EPS = float(PROTOCOL["epsilon"])


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


SDG = load_module("pcda_pinned_sdg", SDG_CODE)


def helper(model: str):
    return load_module(f"pcda_peeh_{model.lower()}", EEGNET_PEEH_CODE if model == "EEGNet" else CONFORMER_PEEH_CODE)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seed(*parts: object) -> int:
    return SDG.stable_seed("pcda-v1", *parts)


def clean(value):
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)): return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)): return bool(value)
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    return value


def write_json_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    part = path.with_name(path.name + f".{os.getpid()}.part")
    with part.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(clean(value), stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    part.rename(path)


def save_npz_new(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    part = path.with_name(path.name + f".{os.getpid()}.part")
    with part.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    part.rename(path)


def cell_id(model: str, task: str, fold: int) -> str:
    if (model, task) not in CELLS or fold not in FOLDS: raise ValueError((model, task, fold))
    return f"{model.lower()}_{task.lower()}_fold{fold}_seed0"


def expected_peeh(model: str, task: str, fold: int) -> Path:
    root = PEEH_EEGNET_RUNTIME if model == "EEGNet" else PEEH_CONFORMER_RUNTIME
    return root / "cells" / model.lower() / task.lower() / f"fold{fold}_seed0.json"


def ortho_from_spec(spec: dict, dims: list[int]) -> np.ndarray:
    d = int(spec["representation_dim"])
    if not dims: return np.empty((d, 0), dtype=np.float64)
    # Exact canonical-coordinate erasure vectors, then an orthonormal raw-h
    # span. The intervention is an orthogonal projector, not canonical erasure.
    rows = (spec["directions"][:, dims].T * spec["scale"][None, :]) @ spec["basis"].T
    u, s, _ = np.linalg.svd(rows.T.astype(np.float64), full_matrices=False)
    if len(s) != len(dims) or s[-1] < s[0] * 1e-8:
        raise RuntimeError("selected canonical raw-space span is singular")
    return np.asarray(u[:, :len(dims)], dtype=np.float64)


def capped_geometry_arrays(h: np.ndarray, y: np.ndarray, sub: np.ndarray, sess: np.ndarray,
                           model: str, task: str, fold: int, mod):
    pieces = []
    for session, purpose in ((1, "train-source"), (2, "train-future")):
        where = np.flatnonzero(sess == session)
        picked = mod.capped_indices(sub[where], y[where], session, task, fold, purpose)
        pieces.append(where[picked])
    idx = np.concatenate(pieces)
    return h[idx], y[idx], sub[idx], sess[idx]


def fit_geometry(h: np.ndarray, y: np.ndarray, sub: np.ndarray, sess: np.ndarray,
                 model: str, task: str, fold: int, mod):
    hc, yc, sc, sec = capped_geometry_arrays(h, y, sub, sess, model, task, fold, mod)
    spec = mod.spectrum(hc, yc, sc, sec, task, model, fold)
    spec["classes"] = int(y.max() + 1)
    protected, assignment = mod.select_protected(hc, yc, sc, sec, spec, model, task, fold)
    persistent = sorted({d for block, row in zip(spec["blocks"], spec["support"])
                         if row["persistence_supported"] for d in block})
    if not set(protected).issubset(persistent): raise RuntimeError("Protected is not a persistence subset")
    p = ortho_from_spec(spec, protected)
    pa = ortho_from_spec(spec, persistent)
    return p, pa, {"rank_active": int(spec["rank"]), "rank_protected": len(protected),
                   "rank_persist_all": len(persistent), "protected_dims": protected,
                   "persistent_dims": persistent, "capped_rows": len(hc),
                   "basis_sha256": SDG.digest_array(spec["mean"], spec["basis"], spec["scale"], spec["directions"]),
                   "assignment": assignment, "selector_code_sha256": sha(EEGNET_PEEH_CODE if model == "EEGNet" else CONFORMER_PEEH_CODE)}


def train_arrays(model_name: str, task: str, fold: int):
    ids, sessions, split_sha, cache_name = SDG.split_role(task, fold)
    inner = SDG.module(f"pcda_inner_{fold}", SDG.SEVEN_CODE / "tech_recipe_selection.py")
    source, sy, ss, mapping = inner._openbmi_rows(SDG.CACHE, list(ids["TRAIN_GEOMETRY"]), (sessions[0],), cache_name)
    mean, std, norm_sha = SDG.historical_normalizer(source)
    model, head, proof = SDG.checkpoint(model_name, task, fold, split_sha, norm_sha, (source.shape[1], source.shape[2]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    before = SDG.state_sha(model)
    x = ((source - mean[None, :, None]) / np.maximum(std[None, :, None], 1e-6)).astype(np.float32)
    del source
    first = {"h": SDG.extract(model, head, x, device), "y": sy.astype(np.int64),
             "subject": ss.astype(str), "session": np.full(len(sy), sessions[0], np.int64)}
    del x
    future, fy, fs, mapping2 = inner._openbmi_rows(SDG.CACHE, list(ids["TRAIN_GEOMETRY"]), (sessions[1],), cache_name, mapping)
    if mapping2 != mapping: raise RuntimeError("class mapping changed between sessions")
    x = ((future - mean[None, :, None]) / np.maximum(std[None, :, None], 1e-6)).astype(np.float32)
    del future
    second = {"h": SDG.extract(model, head, x, device), "y": fy.astype(np.int64),
              "subject": fs.astype(str), "session": np.full(len(fy), sessions[1], np.int64)}
    del x
    if SDG.state_sha(model) != before: raise RuntimeError("frozen neural state mutated during TRAIN extraction")
    if set(first["subject"]) != set(ids["TRAIN_GEOMETRY"]) or set(second["subject"]) != set(ids["TRAIN_GEOMETRY"]):
        raise RuntimeError("TRAIN subject inventory mismatch")
    if first["h"].shape[1] != int(head.in_features) or second["h"].shape[1] != int(head.in_features):
        raise RuntimeError("penultimate dimension/head mismatch")
    arrays = {key: np.concatenate((first[key], second[key])) for key in first}
    arrays["head_weight"] = head.weight.detach().cpu().numpy().copy()
    arrays["head_bias"] = head.bias.detach().cpu().numpy().copy()
    proof.update({"split_sha256": split_sha, "normalizer_sha256": norm_sha,
                  "normalizer_source_session_rows": len(first["h"]),
                  "class_mapping": mapping, "role_subjects": {k: list(v) for k, v in ids.items()},
                  "model_state_sha256_before": before, "model_state_sha256_after": SDG.state_sha(model),
                  "head_weight_sha256": SDG.digest_array(arrays["head_weight"], arrays["head_bias"]),
                  "train_rows": len(arrays["h"]), "formal_final_heldout_eeg_reads": 0,
                  "sdg_source_sha256": sha(SDG_CODE)})
    return arrays, proof


def prepare(model: str, task: str, fold: int) -> None:
    name = cell_id(model, task, fold)
    target = RUNTIME / "train" / f"{name}.npz"
    audit = RUNTIME / "train" / f"{name}.json"
    if target.exists() or audit.exists(): raise FileExistsError("TRAIN stage already exists; no overwrite")
    arrays, proof = train_arrays(model, task, fold)
    h, y, sub, sess = (arrays[k] for k in ("h", "y", "subject", "session"))
    mod = helper(model)
    print("GEOMETRY_FULL_START", name, flush=True)
    full_p, full_all, full_audit = fit_geometry(h, y, sub, sess, model, task, fold, mod)
    stored_path = expected_peeh(model, task, fold)
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    if (stored.get("Model"), stored.get("Task"), int(stored.get("fold", -1))) != (model, task, fold):
        raise RuntimeError("historical PEEH identity mismatch")
    if stored["checkpoint_sha256"].lower() != proof["checkpoint_sha256"].lower():
        raise RuntimeError("historical PEEH checkpoint mismatch")
    if int(stored["rank"]) != full_audit["rank_active"] or sorted(stored["protected_blocks"]) != full_audit["protected_dims"]:
        raise RuntimeError("exact historical Protected selector did not reproduce")
    subjects = sorted(set(sub), key=int)
    rng = np.random.default_rng(seed("crossfit-half", model, task, fold))
    perm = rng.permutation(subjects)
    half = (sorted(perm[:len(perm) // 2], key=int), sorted(perm[len(perm) // 2:], key=int))
    inner = []
    matrices = {}
    for k in range(2):
        source = half[1 - k]
        mask = np.isin(sub, source)
        print("GEOMETRY_CROSSFIT_START", name, k, len(source), flush=True)
        p, pa, row = fit_geometry(h[mask], y[mask], sub[mask], sess[mask], model, task, fold, mod)
        matrices[f"p_half{k}"] = p
        matrices[f"persist_half{k}"] = pa
        inner.append({"pseudo_target_half": k, "fit_subjects": source, **row})
    matrices["p_final"] = full_p
    matrices["persist_final"] = full_all
    arrays.update(matrices)
    save_npz_new(target, **arrays)
    write_json_new(audit, {"schema": "PCDA_TRAIN_GEOMETRY_V1", "model": model, "task": task,
                           "fold": fold, "seed": 0, "status": "PROTECTED_NOT_ESTIMABLE" if full_p.shape[1] == 0 else "COMPLETE",
                           "proof": proof, "historical_peeh_path": str(stored_path), "historical_peeh_sha256": sha(stored_path),
                           "full": full_audit, "crossfit_half_subjects": half, "crossfit": inner,
                           "train_cache_sha256": sha(target), "final_heldout_eeg_reads": 0})
    print("PREPARE_COMPLETE", name, full_p.shape[1], sha(target), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "select", "outer"))
    parser.add_argument("--model", choices=("EEGNet", "EEGConformer"), required=True)
    parser.add_argument("--task", choices=("OpenBMI_MI", "OpenBMI_ERP"), required=True)
    parser.add_argument("--fold", type=int, choices=FOLDS, required=True)
    args = parser.parse_args()
    if args.mode == "prepare": prepare(args.model, args.task, args.fold)
    else: raise NotImplementedError(f"{args.mode} stage is not yet implemented")


if __name__ == "__main__": main()
