"""Frozen EEGNet data and inner-train-only P/PathFit geometry.

Importing this module does not open EEG arrays. ``load_train`` is the only
entry point allowed before ``fit_geometry`` has been committed to disk.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("PERSIST_SOURCE_REPO", str(EXP.parents[1]))).resolve()
RUNTIME = Path(os.environ.get("P_SEMANTICS_RUNTIME", str(ROOT.parent / "p_semantics_canonical_geometry_runtime"))).resolve()
EEN_PATH = ROOT / "experiments/persist_eeg_crossbackbone_peeh_v1/code/run_crossbackbone_peeh.py"
PW_PATH = ROOT / "experiments/persist_eeg_protected_pathway_mechanism_seed0_v1/code/run_protected_pathway.py"
STAGES = ("temporal_bn", "spatial_elu_pool1", "depth_point_elu_pool2", "embedding_64d")
ROLES = ("TRAIN_GEOMETRY", "CHECKPOINT_VALIDATION", "OUTER_DEVELOPMENT")


def imported(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def upstream():
    # Existing loaders, exact model constructor, PERSIST convention and PathFit.
    een = imported("psem_eegnet_peeh", EEN_PATH)
    pw = imported("psem_pathfit", PW_PATH)
    return een, pw, pw.UP


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(value)
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") % (2**32 - 1)


def role_context(fold: int, up):
    modern, _ = up.BENCH._sources()
    split, _, split_sha = modern.load_split()
    role = next(row for row in split["OpenBMI"] if int(row["fold_id"]) == fold)
    ids = {
        "TRAIN_GEOMETRY": tuple(map(str, role["inner_train_subjects"])),
        "CHECKPOINT_VALIDATION": tuple(map(str, role["inner_val_subjects"])),
        "OUTER_DEVELOPMENT": tuple(map(str, role["outer_dev_subjects"])),
    }
    if any(set(ids[a]) & set(ids[b]) for i, a in enumerate(ROLES) for b in ROLES[i + 1:]):
        raise RuntimeError("fold subject roles overlap")
    sessions = (int(modern.SOURCE_SESSIONS["OpenBMI"][0]), int(modern.EVAL_SESSION))
    if sessions[0] == sessions[1]:
        raise RuntimeError("session identity collision")
    return ids, sessions, split_sha


def _normalizer(source: np.ndarray):
    observations = source.shape[0] * source.shape[2]
    mean = (source.sum(axis=(0, 2), dtype=np.float64) / observations).astype(np.float32)
    square = np.square(source, dtype=np.float64).sum(axis=(0, 2), dtype=np.float64)
    std = np.sqrt(np.maximum(square / observations - mean.astype(np.float64) ** 2, 1e-12)).astype(np.float32)
    digest = hashlib.sha256(mean.tobytes() + std.tobytes()).hexdigest()
    return mean, std, digest


@dataclass
class Population:
    x: np.ndarray
    y: np.ndarray
    subject: np.ndarray
    session: np.ndarray
    role: str


def load_train(fold: int, een, up):
    """Read inner-train subjects only; never pass validation/outer IDs here."""
    ids, sessions, split_sha = role_context(fold, up)
    source, sy, ss, mapping = up.INNER._openbmi_rows(een.OPENBMI_CACHE, list(ids["TRAIN_GEOMETRY"]), (sessions[0],), "mi")
    future, fy, fs, _ = up.INNER._openbmi_rows(een.OPENBMI_CACHE, list(ids["TRAIN_GEOMETRY"]), (sessions[1],), "mi", mapping)
    mean, std, norm_sha = _normalizer(source)
    apply = lambda value: ((value - mean[None, :, None]) / np.maximum(std[None, :, None], 1e-6)).astype(np.float32)
    source_rows, future_rows = len(source), len(future)
    a = een.capped_indices(ss, sy, sessions[0], "OpenBMI_MI", fold, "train-source")
    b = een.capped_indices(fs, fy, sessions[1], "OpenBMI_MI", fold, "train-future")
    x = np.concatenate((apply(source[a]), apply(future[b])))
    y = np.concatenate((sy[a], fy[b])).astype(np.int64)
    owner = np.concatenate((ss[a], fs[b])).astype(str)
    sess = np.concatenate((np.full(len(a), sessions[0]), np.full(len(b), sessions[1]))).astype(np.int64)
    if set(owner) != set(ids["TRAIN_GEOMETRY"]) or set(np.unique(y)) != {0, 1}:
        raise RuntimeError("unexpected TRAIN subject or label inventory")
    return Population(x, y, owner, sess, "TRAIN_GEOMETRY"), {
        "ids": ids, "sessions": sessions, "split_sha256": split_sha,
        "normalizer_sha256": norm_sha, "normalizer_mean": mean, "normalizer_std": std,
        "label_mapping": mapping, "raw_source_rows": source_rows, "raw_future_rows": future_rows,
        "capped_source_rows": len(a), "capped_future_rows": len(b),
    }


def load_held(role: str, context: dict, een, up):
    if role not in ("CHECKPOINT_VALIDATION", "OUTER_DEVELOPMENT"):
        raise ValueError(role)
    subjects = list(context["ids"][role])
    if not subjects or set(subjects) & set(context["ids"]["TRAIN_GEOMETRY"]):
        raise RuntimeError("held-role identity invalid")
    parts = []
    for session in context["sessions"]:
        raw, y, owner, _ = up.INNER._openbmi_rows(een.OPENBMI_CACHE, subjects, (session,), "mi", context["label_mapping"])
        x = ((raw - context["normalizer_mean"][None, :, None]) /
             np.maximum(context["normalizer_std"][None, :, None], 1e-6)).astype(np.float32)
        parts.append((x, y.astype(np.int64), owner.astype(str), np.full(len(y), session, dtype=np.int64)))
    return Population(*(np.concatenate([part[i] for part in parts]) for i in range(4)), role)


def checkpoint(fold: int, train: Population, context: dict, een):
    record, path = een.cell("EEGNet", "OpenBMI_MI", fold)
    if record["split_sha256"] != context["split_sha256"]:
        raise RuntimeError("canonical checkpoint/split SHA mismatch")
    if record["normalizer"]["mean_std_sha256"] != context["normalizer_sha256"]:
        raise RuntimeError("canonical checkpoint/TRAIN normalizer SHA mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    row = {"Model": "EEGNet", "Task": "OpenBMI_MI", "fold": fold, "seed": 0,
           "channels": int(train.x.shape[1]), "samples": int(train.x.shape[2]),
           "classes": 2, "checkpoint_path": str(path),
           "recipe_name": record.get("recipe", {}).get("name")}
    model, head = een.build_model(row, device)
    if head is not model.head:
        raise RuntimeError("not the canonical EEGNet native head")
    return record, path, model, head, device


def model_sha(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        a = value.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def extract(pop: Population, stages, *, batch: int = 32):
    features = {key: [] for key in STAGES}
    logits = []
    stages.m.eval()
    with torch.inference_mode():
        for start in range(0, len(pop.x), batch):
            x = stages.tensor(pop.x[start:start + batch])
            values, embedding, native = stages.all(x)
            direct = stages.m(x)
            if not torch.allclose(native, direct, rtol=1e-5, atol=1e-6):
                raise RuntimeError("manual stage/native forward mismatch")
            for key in STAGES:
                features[key].append(values[key].reshape(len(native), -1).float().cpu().numpy())
            logits.append(native.float().cpu().numpy())
    return {key: np.concatenate(rows).astype(np.float32) for key, rows in features.items()}, np.concatenate(logits).astype(np.float32)


def capped(train: Population, een, fold: int):
    # load_train already uses the canonical, deterministic PEEH cap32 draws.
    return np.arange(len(train.y), dtype=np.int64)


def _centroids(features: np.ndarray, pop: Population):
    keys = sorted(set(zip(pop.subject.astype(str), pop.session.astype(int), pop.y.astype(int))))
    if len(keys) != len(set(pop.subject)) * 4:
        raise RuntimeError("subject/session/class centroid grid incomplete")
    x = np.stack([features[(pop.subject == s) & (pop.session == t) & (pop.y == y)].mean(0)
                  for s, t, y in keys]).astype(np.float32)
    return x, keys


def orthonormal(columns: np.ndarray, rank: int) -> np.ndarray:
    if rank <= 0:
        raise RuntimeError("empty Protected rank; required contrasts undefined")
    q, r = np.linalg.qr(columns.astype(np.float64), mode="reduced")
    if q.shape[1] < rank or np.min(np.abs(np.diag(r[:rank, :rank]))) < 1e-10:
        raise RuntimeError("PathFit/final-P numerical rank deficiency")
    q = q[:, :rank].astype(np.float32)
    if np.max(np.abs(q.T @ q - np.eye(rank))) >= 1e-5:
        raise RuntimeError("projector orthogonality failure")
    return q


def fit_geometry(fold: int, train: Population, features: dict, context: dict, een, pw):
    """All arguments are TRAIN_GEOMETRY. No held EEG may be passed here."""
    if train.role != "TRAIN_GEOMETRY" or set(train.subject) != set(context["ids"]["TRAIN_GEOMETRY"]):
        raise RuntimeError("geometry fit population violation")
    ix = capped(train, een, fold)
    h = features["embedding_64d"]
    spec = een.spectrum(h[ix], train.y[ix], train.subject[ix], train.session[ix], "OpenBMI_MI", "EEGNet", fold)
    spec["classes"] = 2
    selected, assignment = een.select_protected(h[ix], train.y[ix], train.subject[ix], train.session[ix], spec,
                                                 "EEGNet", "OpenBMI_MI", fold)
    if not selected:
        raise RuntimeError("fresh TRAIN-only P selection is empty; cannot silently use historical P")
    raw = ((spec["directions"][:, selected].T * spec["scale"][None, :]) @ spec["basis"].T).T
    q_final = orthonormal(raw, len(selected))
    target = (h - h[ix].mean(0)) @ q_final
    target_cent, keys = _centroids(target, train)
    geometry = {}
    diagnostics = {}
    for stage in STAGES:
        a = features[stage]
        mu = a.mean(0, dtype=np.float64).astype(np.float32)
        if stage == "embedding_64d":
            q = q_final
            diagnostics[stage] = {"method": "fresh TRAIN-only final P", "train_centroid_R2": 1.0}
        else:
            cent, stage_keys = _centroids(a, train)
            if keys != stage_keys:
                raise RuntimeError("PathFit key alignment mismatch")
            fit = pw.PathFit(cent)
            qz, singular = fit.q(target_cent)
            q = orthonormal(qz / fit.std[:, None], min(len(selected), qz.shape[1]))
            # Train-centroid projection is descriptive; held-subject claims use
            # downstream frozen decoders, never this in-sample fit score.
            projected = (cent - mu) @ q
            diagnostics[stage] = {"method": "TRAIN-only PathFit to final-P coordinates",
                                  "singular_values": singular.tolist(),
                                  "train_centroid_projected_variance": float(np.var(projected)),
                                  "fit_centroid_count": len(keys), "ridge_alpha": float(pw.RIDGE_ALPHA)}
        if q.shape[1] > len(selected):
            raise RuntimeError("intermediate P rank exceeds final P")
        geometry[stage] = {"mu": mu, "q": q, "rank": q.shape[1], "sha256": array_sha(mu, q)}
    provenance = {"fold": fold, "population": "TRAIN_GEOMETRY only", "fit_subjects": sorted(set(train.subject)),
                  "fit_subject_ids_sha256": hashlib.sha256(json.dumps(sorted(set(train.subject))).encode()).hexdigest(),
                  "fit_rows": len(ix), "all_train_rows": len(train.y), "checkpoint_split_sha256": context["split_sha256"],
                  "checkpoint_normalizer_sha256": context["normalizer_sha256"],
                  "final_P_rank": len(selected), "final_P_selected_coordinates": selected,
                  "final_P_selection_assignment": assignment, "final_whitening_mean_sha256": array_sha(spec["mean"]),
                  "final_whitening_basis_sha256": array_sha(spec["basis"], spec["scale"], spec["directions"]),
                  "stage_geometry_sha256": {key: val["sha256"] for key, val in geometry.items()},
                  "stage_diagnostics": diagnostics, "outer_development_used_for_fit": False,
                  "checkpoint_validation_used_for_fit": False, "final_heldout_accessed": False}
    return geometry, provenance


def random_subspace(dimension: int, rank: int, fold: int, stage: str, draw: int) -> np.ndarray:
    rng = np.random.default_rng(seed("psem-random-subspace", fold, stage, draw))
    return orthonormal(rng.standard_normal((dimension, rank)).astype(np.float32), rank)


def representation(x: np.ndarray, mu: np.ndarray, q: np.ndarray, family: str) -> np.ndarray:
    centered = x - mu
    if family == "FULL":
        return centered
    projection = (centered @ q) @ q.T
    if family in ("P", "RANDOM"):
        return projection.astype(np.float32)
    if family == "C":
        return (centered - projection).astype(np.float32)
    raise KeyError(family)
