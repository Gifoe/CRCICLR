"""Frozen-backbone, TRAIN-only shared decision geometry characterization.

The only EEG reader in this program is ``_openbmi_rows`` with subject IDs
drawn from the frozen train/validation/outer-development split.  In
particular, no final-heldout manifest or EEG path is opened.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

EXP = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("SDG_REPO", str(EXP.parents[1]))).resolve()
SEVEN_REPO = Path(os.environ.get("SEVEN_REPO", str(REPO))).resolve()
CONFORMER_REPO = Path(os.environ.get("CONFORMER_REPO", str(REPO))).resolve()
RUNTIME = Path(os.environ.get("SDG_RUNTIME", str(REPO.parent / "shared_decision_geometry_runtime"))).resolve()
SEVEN_RUNTIME = Path(os.environ.get("SEVEN_RUNTIME", r"D:\nips-temp\TotalP\P1\seven_backbone_fourtask_3seed_runtime"))
CONFORMER_RUNTIME = Path(os.environ.get("NEW_BASELINE_RUNTIME", r"D:\nips-temp\TotalP\P1\eegconformer_fbcnet_runtime"))
CACHE = Path(os.environ.get("FULL_OPENBMI_CACHE", r"D:\nips-temp\TotalP\P1\persist_eeg_stage0_repo_full\outputs\persist_eeg_stage0\cache\openbmi"))
SEVEN_CODE = SEVEN_REPO / "experiments/persist_eeg_seven_backbone_fourtask_3seed_v1/code"
CONFORMER_CODE = CONFORMER_REPO / "experiments/persist_eeg_eegconformer_fbcnet_multiseed_v1/code"
TASK_CODE = SEVEN_REPO / "experiments/persist_eeg_openbmi_task_generality_v1/code/task_datasets.py"
MODERN_CODE = SEVEN_REPO / "experiments/persist_eeg_outcome_blind_modern_backbone_seed0_v1/code/modern_common.py"
if str(SEVEN_CODE) not in sys.path:
    sys.path.insert(0, str(SEVEN_CODE))
os.environ.setdefault("SEVEN_REPO", str(SEVEN_REPO))
os.environ.setdefault("SEVEN_RUNTIME", str(SEVEN_RUNTIME))
os.environ.setdefault("MODERN_REPO", str(SEVEN_REPO))
os.environ.setdefault("TASK_GENERALITY_REPO", str(SEVEN_REPO))
CELLS = (("EEGNet", "OpenBMI_MI"), ("EEGNet", "OpenBMI_ERP"), ("EEGConformer", "OpenBMI_MI"))
ROLES = ("TRAIN_GEOMETRY", "CHECKPOINT_VALIDATION", "OUTER_DEVELOPMENT")
COORDS = ("COMMON_STANDARDIZED", "COMMON_WHITENED")
CAP = None  # use every frozen-role trial; no diagnostic trial cap
LABEL_NULLS = 200
ISOTROPIC_NULLS = 1000
RANDOM_BASES = 100
SEED = 0


def module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest_array(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(value)
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little") % (2**32 - 1)


def save_json_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    part = path.with_name(path.name + f".{os.getpid()}.part")
    def convert(x):
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, (np.bool_,)): return bool(x)
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, Path): return str(x)
        raise TypeError(f"JSON type not supported: {type(x).__name__}")
    with part.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False, default=convert)
        stream.write("\n")
    part.rename(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    if not columns:
        raise RuntimeError(f"no rows for {path.name}")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def state_sha(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        a = value.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def split_role(task: str, fold: int) -> tuple[dict, tuple[int, int], str, str]:
    if task == "OpenBMI_MI":
        modern = module(f"sdg_modern_{fold}", MODERN_CODE)
        folds, _, split_sha = modern.load_split()
        row = next(r for r in folds["OpenBMI"] if int(r["fold_id"]) == fold)
        sessions, cache_name = (int(modern.SOURCE_SESSIONS["OpenBMI"][0]), int(modern.EVAL_SESSION)), "mi"
    elif task == "OpenBMI_ERP":
        tasks = module(f"sdg_tasks_{fold}", TASK_CODE)
        # The historical split source suffices.  Do not call split_reference,
        # which unnecessarily opens the final-heldout subject manifest.
        split_path = tasks.FIVEFOLD
        split = json.loads(split_path.read_text(encoding="utf-8"))
        if split.get("protocol") != "CARRIER_5FOLD_MULTISEED_STABILITY_V1":
            raise RuntimeError("ERP split protocol drift")
        row = next(r for r in split["folds"]["OpenBMI"] if int(r["fold_id"]) == fold)
        split_sha = sha(split_path)
        spec = tasks.TASKS["ERP"]
        sessions, cache_name = (int(spec["source_session"]), int(spec["future_session"])), spec["cache_name"]
    else:
        raise ValueError(task)
    ids = {"TRAIN_GEOMETRY": tuple(map(str, row["inner_train_subjects"])),
           "CHECKPOINT_VALIDATION": tuple(map(str, row["inner_val_subjects"])),
           "OUTER_DEVELOPMENT": tuple(map(str, row["outer_dev_subjects"]))}
    if any(set(ids[a]) & set(ids[b]) for i, a in enumerate(ROLES) for b in ROLES[i + 1:]):
        raise RuntimeError("subject-role overlap")
    if sessions[0] == sessions[1]:
        raise RuntimeError("physical sessions coincide")
    return ids, sessions, split_sha, cache_name


def all_indices(owner: np.ndarray, y: np.ndarray, task: str, fold: int, role: str, session: int) -> np.ndarray:
    labels = owner.astype(str)
    for subject in sorted(set(labels), key=int):
        if set(np.unique(y[labels == subject])) != {0, 1}:
            raise RuntimeError(f"empty subject/session/class: {task}/{fold}/{role}/{subject}/{session}")
    return np.arange(len(y), dtype=np.int64)


def historical_normalizer(source: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    observations = source.shape[0] * source.shape[2]
    mean = (source.sum(axis=(0, 2), dtype=np.float64) / observations).astype(np.float32)
    square = np.square(source, dtype=np.float64).sum(axis=(0, 2), dtype=np.float64)
    std = np.sqrt(np.maximum(square / observations - mean.astype(np.float64) ** 2, 1e-12)).astype(np.float32)
    return mean, std, hashlib.sha256(mean.tobytes() + std.tobytes()).hexdigest()


def checkpoint(model_name: str, task: str, fold: int, split_sha: str, norm_sha: str, shape: tuple[int, int]):
    if model_name == "EEGNet":
        folder = SEVEN_RUNTIME / "search_cells" / task.lower() / "eegnet" / f"fold{fold}_seed0"
        models = module(f"sdg_eegnet_{fold}", SEVEN_CODE / "backbone_models.py")
        construct = lambda: models.build_model("EEGNet", dataset="OpenBMI", channels=shape[0], samples=shape[1], classes=2)
        source = SEVEN_CODE / "backbone_models.py"
    else:
        folder = CONFORMER_RUNTIME / "cells" / "eegconformer" / task.lower() / f"fold{fold}_seed0"
        models = module(f"sdg_conformer_{fold}", CONFORMER_CODE / "models.py")
        construct = lambda: models.build_model("EEGConformer", channels=shape[0], samples=shape[1], classes=2)
        source = CONFORMER_CODE / "models.py"
    record_path, path = folder / "record.json", folder / "selected.pt"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (record.get("model"), record.get("task"), int(record.get("fold", -1)), int(record.get("seed", -1))) != (model_name, task, fold, 0):
        raise RuntimeError(f"checkpoint identity mismatch: {record_path}")
    if record["checkpoint_sha256"] != sha(path):
        raise RuntimeError(f"checkpoint SHA mismatch: {path}")
    if record["split_sha256"] != split_sha or record["normalizer"]["mean_std_sha256"] != norm_sha:
        raise RuntimeError("historical split or neural normalizer mismatch")
    recorded_shape = (int(record.get("channels") or shape[0]), int(record.get("samples") or shape[1]), int(record["classes"]))
    if recorded_shape != (*shape, 2):
        raise RuntimeError("checkpoint shape mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = construct()
    model.load_state_dict(payload.get("state_dict", payload), strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    head = model.head if model_name == "EEGNet" else model.classifier[-1]
    if not isinstance(head, torch.nn.Linear):
        raise RuntimeError("final classifier is not affine")
    return model, head, {"checkpoint_path": str(path), "checkpoint_sha256": sha(path), "record_sha256": sha(record_path),
                         "model_source_path": str(source), "model_source_sha256": sha(source), "selected_epoch": int(record["selected_epoch"]),
                         "embedding_dim": int(head.in_features), "penultimate_hook": "head input" if model_name == "EEGNet" else "classifier[-1] input"}


def extract(model: torch.nn.Module, head: torch.nn.Module, x: np.ndarray, device: torch.device) -> np.ndarray:
    captured = []
    handle = head.register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach()))
    features = []
    try:
        model.eval()
        with torch.inference_mode():
            for first in range(0, len(x), 64):
                batch = torch.from_numpy(np.ascontiguousarray(x[first:first + 64])).to(device)
                captured.clear()
                logits = model(batch)
                if len(captured) != 1 or captured[0].ndim != 2:
                    raise RuntimeError("penultimate hook fired incorrectly")
                exact = head(captured[0])
                if not torch.allclose(exact, logits, rtol=1e-5, atol=1e-6):
                    raise RuntimeError("native logits/penultimate affine mismatch")
                features.append(captured[0].float().cpu().numpy())
    finally:
        handle.remove()
    return np.concatenate(features).astype(np.float32)


def population_features(inner, model, head, ids, task, fold, role, sessions, cache_name, mapping, mean, std, device):
    all_h, all_y, all_subject, all_session = [], [], [], []
    for session in sessions:
        raw, y, owner, new_mapping = inner._openbmi_rows(CACHE, list(ids[role]), (session,), cache_name, mapping)
        if new_mapping != mapping:
            raise RuntimeError("class mapping changed")
        ix = all_indices(owner, y, task, fold, role, session)
        x = ((raw - mean[None, :, None]) / np.maximum(std[None, :, None], 1e-6)).astype(np.float32)
        del raw
        all_h.append(extract(model, head, x, device)); del x
        all_y.append(y[ix].astype(np.int64)); all_subject.append(owner[ix].astype(str)); all_session.append(np.full(len(ix), session, dtype=np.int64))
    return {"h": np.concatenate(all_h), "y": np.concatenate(all_y), "subject": np.concatenate(all_subject),
            "session": np.concatenate(all_session), "role": role}


def load_fold(model_name: str, task: str, fold: int):
    ids, sessions, split_sha, cache_name = split_role(task, fold)
    inner = module(f"sdg_inner_{fold}", SEVEN_CODE / "tech_recipe_selection.py")
    source, sy, ss, mapping = inner._openbmi_rows(CACHE, list(ids["TRAIN_GEOMETRY"]), (sessions[0],), cache_name)
    mean, std, norm_sha = historical_normalizer(source)
    model, head, proof = checkpoint(model_name, task, fold, split_sha, norm_sha, (source.shape[1], source.shape[2]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    before = state_sha(model)
    ix = all_indices(ss, sy, task, fold, "TRAIN_GEOMETRY", sessions[0])
    x = ((source - mean[None, :, None]) / np.maximum(std[None, :, None], 1e-6)).astype(np.float32)
    del source
    h0 = extract(model, head, x, device); del x
    train = {"h": h0, "y": sy[ix].astype(np.int64), "subject": ss[ix].astype(str),
             "session": np.full(len(ix), sessions[0], dtype=np.int64), "role": "TRAIN_GEOMETRY"}
    source2, y2, s2, mapping2 = inner._openbmi_rows(CACHE, list(ids["TRAIN_GEOMETRY"]), (sessions[1],), cache_name, mapping)
    if mapping2 != mapping:
        raise RuntimeError("TRAIN future-session mapping drift")
    jx = all_indices(s2, y2, task, fold, "TRAIN_GEOMETRY", sessions[1])
    x2 = ((source2 - mean[None, :, None]) / np.maximum(std[None, :, None], 1e-6)).astype(np.float32)
    del source2
    train2 = {"h": extract(model, head, x2, device), "y": y2[jx].astype(np.int64), "subject": s2[jx].astype(str),
              "session": np.full(len(jx), sessions[1], dtype=np.int64), "role": "TRAIN_GEOMETRY"}
    del x2
    for key in ("h", "y", "subject", "session"):
        train[key] = np.concatenate((train[key], train2[key]))
    held = {role: population_features(inner, model, head, ids, task, fold, role, sessions, cache_name, mapping, mean, std, device)
            for role in ROLES[1:]}
    after = state_sha(model)
    if before != after:
        raise RuntimeError("frozen model parameters or BN buffers mutated")
    proof.update({"split_sha256": split_sha, "normalizer_sha256": norm_sha, "model_state_sha256_before_after": before,
                  "class_mapping": mapping, "source_session": sessions[0], "future_session": sessions[1],
                  "role_subjects": {k: list(v) for k, v in ids.items()}, "cap_per_subject_session_class": CAP,
                  "train_geometry_includes_train_subject_future_session": True,
                  "historical_neural_normalizer_fit": "all uncapped TRAIN_GEOMETRY source-session trials",
                  "final_heldout_eeg_reads": 0})
    for role, pop in (("TRAIN_GEOMETRY", train), *held.items()):
        if set(pop["subject"]) != set(ids[role]) or set(np.unique(pop["y"])) != {0, 1}:
            raise RuntimeError(f"population role inventory mismatch: {role}")
    return train, held, proof


def classifier(task: str):
    return LogisticRegression(penalty="l2", C=1.0, solver="liblinear", class_weight="balanced" if task == "OpenBMI_ERP" else None,
                              random_state=SEED, max_iter=500, tol=1e-6)


def local_vectors(z: np.ndarray, pop: dict, task: str, fold: int, coord: str, audit: list[dict], *, permute: bool = False, draw: int = -1):
    ws, keys, norms = [], [], []
    for subject in sorted(set(pop["subject"]), key=int):
        for session in sorted(set(pop["session"])):
            ix = np.flatnonzero((pop["subject"] == subject) & (pop["session"] == session))
            y = pop["y"][ix]
            if set(y) != {0, 1}:
                raise RuntimeError("local class inventory missing")
            fit_y = np.random.default_rng(stable_seed("label-null", task, fold, coord, draw, subject, session)).permutation(y) if permute else y
            fit = classifier(task).fit(z[ix], fit_y)
            beta = fit.coef_[0].astype(np.float64)
            norm = float(np.linalg.norm(beta))
            if norm <= 1e-10:
                raise RuntimeError("zero local discriminant normal")
            w = beta / norm
            ws.append(w); keys.append((subject, int(session))); norms.append(norm)
            if not permute:
                audit.append({"backbone": pop["backbone"], "task": task, "fold": fold, "coordinate": coord,
                              "role": pop["role"], "subject": subject, "session": int(session), "rows": len(ix),
                              "class_0_rows": int((y == 0).sum()), "class_1_rows": int((y == 1).sum()),
                              "local_train_BA": float(balanced_accuracy_score(y, fit.predict(z[ix]))),
                              "raw_beta_norm": norm, "unit_vector_sha256": digest_array(w),
                              "orientation": "increasing class-1 logit", "label_assisted_evaluation_only": pop["role"] == "OUTER_DEVELOPMENT"})
    return np.stack(ws), keys, np.asarray(norms)


def spectrum(w: np.ndarray) -> dict:
    singular = np.linalg.svd(w, compute_uv=False, full_matrices=False)
    energy = np.square(singular)
    cumulative = np.cumsum(energy) / np.sum(energy)
    rank = int(np.sum(singular > max(1e-10, singular[0] * 1e-8)))
    k = lambda threshold: int(np.searchsorted(cumulative, threshold) + 1)
    return {"singular": singular, "cumulative": cumulative, "rank": rank,
            "k50": k(.5), "k75": k(.75), "k90": k(.9), "k95": k(.95),
            "effective_rank": float(energy.sum() ** 2 / np.square(energy).sum()),
            "normalized_effective_rank": float(energy.sum() ** 2 / np.square(energy).sum() / min(w.shape))}


def basis(w: np.ndarray, k: int) -> np.ndarray:
    _, singular, vt = np.linalg.svd(w, full_matrices=False)
    if k < 1 or k > len(singular) or singular[k - 1] < 1e-9:
        raise RuntimeError("basis rank invalid")
    return vt[:k].T


def geometry(w: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    projected = (w @ b) @ b.T
    coverage = np.sum(projected ** 2, axis=1) / np.maximum(np.sum(w ** 2, axis=1), 1e-15)
    cosine = np.sum(w * projected, axis=1) / np.maximum(np.linalg.norm(w, axis=1) * np.linalg.norm(projected, axis=1), 1e-15)
    return coverage, cosine, np.sqrt(np.maximum(0, 1 - coverage))


def random_basis(dim: int, k: int, *seed_parts: object):
    a = np.random.default_rng(stable_seed("random-basis", *seed_parts)).normal(size=(dim, k))
    q, _ = np.linalg.qr(a, mode="reduced")
    return q[:, :k]


def metrics(y: np.ndarray, probability: np.ndarray, subjects: np.ndarray) -> dict:
    pred = (probability >= .5).astype(int)
    rows = []
    for subject in sorted(set(subjects), key=int):
        ix = subjects == subject
        rows.append((balanced_accuracy_score(y[ix], pred[ix]), f1_score(y[ix], pred[ix], average="macro", zero_division=0),
                     log_loss(y[ix], np.clip(probability[ix], 1e-8, 1 - 1e-8), labels=[0, 1])))
    return {"subject_equal_BA": float(np.mean([r[0] for r in rows])),
            "subject_equal_macro_F1": float(np.mean([r[1] for r in rows])),
            "subject_equal_NLL": float(np.mean([r[2] for r in rows])),
            "subject_count": len(rows), "trial_count": len(y)}


def transform_view(z: np.ndarray, b: np.ndarray, name: str) -> np.ndarray:
    if name == "SHARED": return z @ b
    if name == "PRIVATE": return z - (z @ b) @ b.T
    if name == "FULL_LINEAR_REFERENCE": return z
    raise ValueError(name)


def decode(train_z, train_y, held_z, b, task, view):
    fit = classifier(task).fit(transform_view(train_z, b, view), train_y)
    return fit.predict_proba(transform_view(held_z, b, view))[:, 1], fit


def grouped_oof(h: np.ndarray, y: np.ndarray, owner: np.ndarray, session: np.ndarray, task: str, fold: int, rank_name: str):
    n = min(5, len(set(owner)))
    if n < 2: raise RuntimeError("too few TRAIN biological subjects for OOF")
    margins = np.empty((len(y), 2), dtype=np.float64)
    for part, (fit_ix, held_ix) in enumerate(GroupKFold(n_splits=n).split(h, y, owner)):
        if set(owner[fit_ix]) & set(owner[held_ix]): raise RuntimeError("OOF biological subject leakage")
        scale = StandardScaler().fit(h[fit_ix])
        z_fit, z_held = scale.transform(h[fit_ix]), scale.transform(h[held_ix])
        subpop = {"y": y[fit_ix], "subject": owner[fit_ix], "session": session[fit_ix], "role": "TRAIN_GEOMETRY", "backbone": "OOF"}
        w, _, _ = local_vectors(z_fit, subpop, task, fold, f"OOF_PART_{part}", [])
        k90 = spectrum(w)["k90"]
        k = k90 if rank_name == "k90" else min(4, k90)
        b = basis(w, k)
        for j, view in enumerate(("SHARED", "PRIVATE")):
            fit = classifier(task).fit(transform_view(z_fit, b, view), y[fit_ix])
            margins[held_ix, j] = fit.decision_function(transform_view(z_held, b, view))
    if not np.isfinite(margins).all(): raise RuntimeError("OOF margins nonfinite")
    return margins


def principal_stability(a: np.ndarray, b: np.ndarray) -> dict:
    corr = np.clip(np.linalg.svd(a.T @ b, compute_uv=False), 0, 1)
    angles = np.degrees(np.arccos(corr))
    projector = float(np.linalg.norm(a @ a.T - b @ b.T, "fro") / np.sqrt(2 * a.shape[1]))
    return {"mean_principal_angle_deg": float(np.mean(angles)), "max_principal_angle_deg": float(np.max(angles)),
            "mean_squared_canonical_correlation": float(np.mean(corr ** 2)), "normalized_projector_distance": projector}


def tagged(backbone: str, task: str, fold: int, coord: str, **rest) -> dict:
    return {"backbone": backbone, "task": task, "fold": fold, "coordinate": coord, **rest}


def run_coordinate(backbone: str, task: str, fold: int, train: dict, held: dict, coord: str, z_train: np.ndarray, z_held: dict):
    out = {name: [] for name in OUTPUT_STEMS}
    train["backbone"] = backbone
    w, keys, strengths = local_vectors(z_train, train, task, fold, coord, out["LOCAL_DECISION_VECTORS_AUDIT"])
    spec = spectrum(w); wc = w - w.mean(0)
    centered = spectrum(wc)
    for centered_name, value in (("UNCENTERED_PRIMARY", spec), ("CENTERED_SENSITIVITY", centered)):
        for index in range(min(16, len(value["singular"]))):
            out["DECISION_SPECTRUM"].append(tagged(backbone, task, fold, coord, centering=centered_name, k=index + 1,
                singular_value=float(value["singular"][index]), cumulative_energy=float(value["cumulative"][index])))
        out["DECISION_EFFECTIVE_RANK"].append(tagged(backbone, task, fold, coord, centering=centered_name,
            decision_vectors=len(w), embedding_dim=w.shape[1], k50=value["k50"], k75=value["k75"], k90=value["k90"], k95=value["k95"],
            effective_rank=value["effective_rank"], normalized_effective_rank=value["normalized_effective_rank"],
            E1=float(value["cumulative"][0]), E2=float(value["cumulative"][min(1, len(value["cumulative"]) - 1)]),
            E4=float(value["cumulative"][min(3, len(value["cumulative"]) - 1)])))
    for draw in range(LABEL_NULLS):
        null, _, _ = local_vectors(z_train, train, task, fold, coord, [], permute=True, draw=draw)
        s = spectrum(null)
        out["DECISION_SPECTRUM_NULLS"].append(tagged(backbone, task, fold, coord, null="WITHIN_CELL_LABEL_PERMUTATION",
            draw=draw, k90=s["k90"], effective_rank=s["effective_rank"], E4=float(s["cumulative"][min(3, len(s["cumulative"]) - 1)])))
    for draw in range(ISOTROPIC_NULLS):
        rng = np.random.default_rng(stable_seed("isotropic", backbone, task, fold, coord, draw))
        null = rng.normal(size=w.shape); null /= np.linalg.norm(null, axis=1, keepdims=True)
        s = spectrum(null)
        out["DECISION_SPECTRUM_NULLS"].append(tagged(backbone, task, fold, coord, null="ISOTROPIC_UNIT_DIRECTIONS",
            draw=draw, k90=s["k90"], effective_rank=s["effective_rank"], E4=float(s["cumulative"][min(3, len(s["cumulative"]) - 1)])))
    for draw in range(LABEL_NULLS):
        rng = np.random.default_rng(stable_seed("strength-matched", backbone, task, fold, coord, draw))
        null = rng.normal(size=w.shape); null /= np.linalg.norm(null, axis=1, keepdims=True)
        weighted = null * strengths[:, None]
        s = spectrum(weighted)
        out["DECISION_SPECTRUM_NULLS"].append(tagged(backbone, task, fold, coord, null="STRENGTH_MATCHED_RANDOM_ORIENTATION_DESCRIPTIVE",
            draw=draw, k90=s["k90"], effective_rank=s["effective_rank"], E4=float(s["cumulative"][min(3, len(s["cumulative"]) - 1)])))
    rank_choices = {"k90": spec["k90"], "compact": min(4, spec["k90"])}
    outer = held["OUTER_DEVELOPMENT"]
    outer["backbone"] = backbone
    w_outer, outer_keys, _ = local_vectors(z_held["OUTER_DEVELOPMENT"], outer, task, fold, coord,
                                           out["LOCAL_DECISION_VECTORS_AUDIT"])
    pca = np.linalg.svd(z_train - z_train.mean(0), full_matrices=False)[2].T
    for rank_name, k in rank_choices.items():
        b = basis(w, k)
        b_sha = digest_array(b)
        # Independently estimated physical-session bases; same locked rank.
        s1, s2 = sorted(set(train["session"]))
        w1 = w[[i for i, key in enumerate(keys) if key[1] == s1]]
        w2 = w[[i for i, key in enumerate(keys) if key[1] == s2]]
        session_k = min(k, np.linalg.matrix_rank(w1), np.linalg.matrix_rank(w2))
        if session_k < 1: raise RuntimeError("session-specific rank deficient")
        b1, b2 = basis(w1, session_k), basis(w2, session_k)
        out["CROSS_SESSION_BASIS_STABILITY"].append(tagged(backbone, task, fold, coord, rank_choice=rank_name, k=session_k,
            full_train_rank_choice_k=k, session_rank_capped=session_k < k,
            **principal_stability(b1, b2)))
        for subject in sorted(set(train["subject"]), key=int):
            excluded = [i for i, key in enumerate(keys) if key[0] != subject]
            w_ex = w[excluded]; ex_spec = spectrum(w_ex)
            k_ex = ex_spec["k90"] if rank_name == "k90" else min(4, ex_spec["k90"])
            b_ex = basis(w_ex, k_ex)
            for i, key in enumerate(keys):
                if key[0] != subject: continue
                coverage, cosine, residual = geometry(w[i:i + 1], b_ex)
                out["LOSO_DECISION_GEOMETRY"].append(tagged(backbone, task, fold, coord, rank_choice=rank_name,
                    subject=subject, session=key[1], k=k_ex, coverage=float(coverage[0]), reconstruction_cosine=float(cosine[0]),
                    private_fraction=float(1 - coverage[0]), residual_norm_fraction=float(residual[0]),
                    excluded_subject_from_basis=True, excluded_subject_from_rank_selection=True))
            # Cross-session source basis leaves out the target biological subject.
            for source, target in ((s1, s2), (s2, s1)):
                source_ix = [i for i, key in enumerate(keys) if key[0] != subject and key[1] == source]
                target_ix = next(i for i, key in enumerate(keys) if key == (subject, target))
                source_w = w[source_ix]; source_k90 = spectrum(source_w)["k90"]
                source_k = source_k90 if rank_name == "k90" else min(4, source_k90)
                source_b = basis(source_w, source_k)
                coverage, cosine, residual = geometry(w[target_ix:target_ix + 1], source_b)
                out["CROSS_SESSION_DECISION_COVERAGE"].append(tagged(backbone, task, fold, coord, rank_choice=rank_name,
                    subject=subject, source_session=source, target_session=target, k=source_k, coverage=float(coverage[0]),
                    reconstruction_cosine=float(cosine[0]), residual_norm_fraction=float(residual[0])))
        cover, cos, res = geometry(w_outer, b)
        for i, key in enumerate(outer_keys):
            out["OUTER_DECISION_GEOMETRY_COVERAGE"].append(tagged(backbone, task, fold, coord, rank_choice=rank_name,
                k=k, subject=key[0], session=key[1], coverage=float(cover[i]), reconstruction_cosine=float(cos[i]),
                private_fraction=float(1 - cover[i]), basis_sha256=b_sha, label_assisted_evaluation_only=True))
        for control, q in (("TRAIN_EMBEDDING_PCA_R", pca[:, :k]),):
            c, cosine, _ = geometry(w_outer, q)
            for i, key in enumerate(outer_keys):
                out["DECISION_BASIS_COVERAGE_CONTROLS"].append(tagged(backbone, task, fold, coord, rank_choice=rank_name,
                    control=control, draw=-1, k=k, subject=key[0], session=key[1], coverage=float(c[i]), reconstruction_cosine=float(cosine[i])))
        for draw in range(RANDOM_BASES):
            q = random_basis(w.shape[1], k, backbone, task, fold, coord, rank_name, draw)
            c, cosine, _ = geometry(w_outer, q)
            for i, key in enumerate(outer_keys):
                out["DECISION_BASIS_COVERAGE_CONTROLS"].append(tagged(backbone, task, fold, coord, rank_choice=rank_name,
                    control="RANK_MATCHED_RANDOM", draw=draw, k=k, subject=key[0], session=key[1],
                    coverage=float(c[i]), reconstruction_cosine=float(cosine[i])))
        decoder_p = {}
        for role, pop in (("TRAIN_GEOMETRY", train), *held.items()):
            z = z_train if role == "TRAIN_GEOMETRY" else z_held[role]
            for view in ("SHARED", "PRIVATE", "FULL_LINEAR_REFERENCE"):
                prob, _ = decode(z_train, train["y"], z, b, task, view)
                decoder_p[(role, view)] = prob
                dest = "SHARED_GEOMETRY_DECODER" if view != "PRIVATE" else "PRIVATE_GEOMETRY_DECODER"
                out[dest].append(tagged(backbone, task, fold, coord, rank_choice=rank_name, k=k, role=role,
                                        decoder=view, **metrics(pop["y"], prob, pop["subject"])))
        y, subjects = outer["y"], outer["subject"]
        ps, pp = decoder_p[("OUTER_DEVELOPMENT", "SHARED")], decoder_p[("OUTER_DEVELOPMENT", "PRIVATE")]
        pf = decoder_p[("OUTER_DEVELOPMENT", "FULL_LINEAR_REFERENCE")]
        cs, cp, cf = (ps >= .5) == y, (pp >= .5) == y, (pf >= .5) == y
        for subject in sorted(set(subjects), key=int):
            ix = subjects == subject
            a, d = cs[ix], cp[ix]
            out["SHARED_PRIVATE_ERROR_COMPLEMENTARITY"].append(tagged(backbone, task, fold, coord, rank_choice=rank_name,
                subject=subject, k=k, both_correct=int((a & d).sum()), shared_only_correct=int((a & ~d).sum()),
                private_only_correct=int((~a & d).sum()), both_wrong=int((~a & ~d).sum()),
                private_rescue=float(np.mean(d[~a])) if (~a).any() else None,
                shared_rescue=float(np.mean(a[~d])) if (~d).any() else None,
                full_correct=int(cf[ix].sum()), rows=int(ix.sum())))
        oracle = cs | cp
        oracle_pred = np.where(cs, (ps >= .5).astype(int), (pp >= .5).astype(int))
        oracle_metrics = metrics(y, oracle_pred.astype(float), subjects)
        out["SHARED_PRIVATE_ORACLE_UNION"].append(tagged(backbone, task, fold, coord, rank_choice=rank_name,
            k=k, oracle_correct_fraction=float(oracle.mean()), diagnostic="NONDEPLOYABLE_ORACLE_UNION",
            subject_equal_BA=oracle_metrics["subject_equal_BA"], subject_equal_macro_F1=oracle_metrics["subject_equal_macro_F1"]))
        oof = grouped_oof(train["h"], train["y"], train["subject"], train["session"], task, fold, rank_name)
        stack = classifier(task).fit(oof, train["y"])
        # Final base decoders and basis use all TRAIN; OUTER only enters predict.
        fit_s = classifier(task).fit(transform_view(z_train, b, "SHARED"), train["y"])
        fit_p = classifier(task).fit(transform_view(z_train, b, "PRIVATE"), train["y"])
        margins = np.column_stack((fit_s.decision_function(transform_view(z_held["OUTER_DEVELOPMENT"], b, "SHARED")),
                                   fit_p.decision_function(transform_view(z_held["OUTER_DEVELOPMENT"], b, "PRIVATE"))))
        stacked = stack.predict_proba(margins)[:, 1]
        out["SHARED_PRIVATE_STACKED_FUSION"].append(tagged(backbone, task, fold, coord, rank_choice=rank_name, k=k,
            stacker_fit="TRAIN subject-grouped OOF only; OOF scaler and basis refit excluding validation subjects",
            **{"stacked_" + name: value for name, value in metrics(y, stacked, subjects).items()},
            **{"shared_" + name: value for name, value in metrics(y, ps, subjects).items()},
            **{"private_" + name: value for name, value in metrics(y, pp, subjects).items()},
            **{"full_" + name: value for name, value in metrics(y, pf, subjects).items()}))
        for scope, ww, kk in (("TRAIN", w, keys), ("OUTER_EVAL", w_outer, outer_keys)):
            shared = (ww @ b) @ b.T; private = ww - shared
            for i, (subject, session) in enumerate(kk):
                other = [j for j, key in enumerate(kk) if key[0] == subject and key[1] != session]
                cross = [j for j, key in enumerate(kk) if key[0] != subject]
                def mean_cos(a, ix):
                    if not ix or np.linalg.norm(a[i]) < 1e-10: return None
                    return float(np.mean([a[i] @ a[j] / max(1e-12, np.linalg.norm(a[i]) * np.linalg.norm(a[j])) for j in ix]))
                out["PRIVATE_DIRECTION_STRUCTURE"].append(tagged(backbone, task, fold, coord, rank_choice=rank_name,
                    role=scope, subject=subject, session=session, private_energy_fraction=float(np.sum(private[i] ** 2) / np.sum(ww[i] ** 2)),
                    shared_cross_session_cosine=mean_cos(shared, other), private_cross_session_cosine=mean_cos(private, other),
                    shared_cross_subject_cosine=mean_cos(shared, cross), private_cross_subject_cosine=mean_cos(private, cross),
                    label_assisted_evaluation_only=scope == "OUTER_EVAL"))
    return out, {"k90": spec["k90"], "compact": rank_choices["compact"], "effective_rank": spec["effective_rank"],
                 "normalized_effective_rank": spec["normalized_effective_rank"], "E1": float(spec["cumulative"][0]),
                 "E2": float(spec["cumulative"][1]), "E4": float(spec["cumulative"][3]), "embedding_dim": w.shape[1],
                 "train_decision_vectors": len(w)}


OUTPUT_STEMS = ("LOCAL_DECISION_VECTORS_AUDIT", "DECISION_SPECTRUM", "DECISION_EFFECTIVE_RANK", "DECISION_SPECTRUM_NULLS",
                "CROSS_SESSION_BASIS_STABILITY", "CROSS_SESSION_DECISION_COVERAGE", "LOSO_DECISION_GEOMETRY",
                "OUTER_DECISION_GEOMETRY_COVERAGE", "DECISION_BASIS_COVERAGE_CONTROLS", "SHARED_GEOMETRY_DECODER",
                "PRIVATE_GEOMETRY_DECODER", "SHARED_PRIVATE_ERROR_COMPLEMENTARITY", "SHARED_PRIVATE_ORACLE_UNION",
                "SHARED_PRIVATE_STACKED_FUSION", "PRIVATE_DIRECTION_STRUCTURE")


def run_fold(backbone: str, task: str, fold: int):
    if (backbone, task) not in CELLS or fold not in range(5):
        raise ValueError("outside frozen 15 fold-cells")
    target = RUNTIME / "folds" / f"{backbone.lower()}_{task.lower()}_fold{fold}.json"
    if target.exists(): raise FileExistsError(target)
    torch.set_num_threads(4)
    train, held, proof = load_fold(backbone, task, fold)
    scale = StandardScaler().fit(train["h"])
    z_train = scale.transform(train["h"]).astype(np.float64)
    z_held = {role: scale.transform(pop["h"]).astype(np.float64) for role, pop in held.items()}
    covariance = np.cov(train["h"].astype(np.float64), rowvar=False)
    eigen, vector = np.linalg.eigh(covariance)
    floor = max(1e-4 * float(eigen[-1]), 1e-8)
    white = (vector * (1 / np.sqrt(np.maximum(eigen, floor)))[None, :]) @ vector.T
    white_train = (train["h"].astype(np.float64) - train["h"].mean(axis=0, dtype=np.float64)) @ white
    white_held = {role: (pop["h"].astype(np.float64) - train["h"].mean(axis=0, dtype=np.float64)) @ white for role, pop in held.items()}
    all_rows = {name: [] for name in OUTPUT_STEMS}
    summaries = {}
    for coord, x, h in ((COORDS[0], z_train, z_held), (COORDS[1], white_train, white_held)):
        rows, summary = run_coordinate(backbone, task, fold, train, held, coord, x, h)
        for name in OUTPUT_STEMS: all_rows[name].extend(rows[name])
        summaries[coord] = summary
    proof.update({"common_standardizer_sha256": digest_array(scale.mean_, scale.scale_),
                  "common_whitening_sha256": digest_array(train["h"].mean(axis=0, dtype=np.float64), white),
                  "whitening_eigenvalue_floor": floor, "standardizer_fit_role": "TRAIN_GEOMETRY only",
                  "whitener_fit_role": "TRAIN_GEOMETRY only", "outer_used_for_scaling_basis_rank_decoder_or_stacker": False,
                  "held_rows": {role: len(pop["y"]) for role, pop in held.items()}, "train_rows": len(train["y"]),
                  "population_embedding_sha256": {"TRAIN_GEOMETRY": digest_array(train["h"]),
                       **{role: digest_array(pop["h"]) for role, pop in held.items()}}})
    save_json_new(target, {"backbone": backbone, "task": task, "fold": fold, "proof": proof,
                           "summaries": summaries, "rows": all_rows})
    print(f"COMPLETE {backbone} {task} fold{fold} {sha(target)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("fold", "aggregate"))
    parser.add_argument("--backbone", choices=("EEGNet", "EEGConformer"))
    parser.add_argument("--task", choices=("OpenBMI_MI", "OpenBMI_ERP"))
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    if args.mode == "fold":
        if args.backbone is None or args.task is None or args.fold is None: parser.error("fold needs backbone/task/fold")
        run_fold(args.backbone, args.task, args.fold)
    else:
        aggregate()


def aggregate():
    outdir = EXP / "outputs"
    if outdir.exists():
        raise FileExistsError("aggregate is immutable; outputs already exist")
    fold_outputs = []
    for backbone, task in CELLS:
        for fold in range(5):
            path = RUNTIME / "folds" / f"{backbone.lower()}_{task.lower()}_fold{fold}.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            if (value["backbone"], value["task"], value["fold"]) != (backbone, task, fold):
                raise RuntimeError("fold-output identity mismatch")
            if value["proof"]["final_heldout_eeg_reads"] != 0 or value["proof"]["outer_used_for_scaling_basis_rank_decoder_or_stacker"]:
                raise RuntimeError("population-isolation audit failed")
            fold_outputs.append((path, value))
    all_rows = {name: [] for name in OUTPUT_STEMS}
    for _, fold_value in fold_outputs:
        for name in OUTPUT_STEMS:
            all_rows[name].extend(fold_value["rows"][name])
    table = {name: all_rows[name] for name in OUTPUT_STEMS}
    fold_summaries = []
    sensitivity = []
    for _, value in fold_outputs:
        model, task, fold = value["backbone"], value["task"], value["fold"]
        per = {}
        for coord in COORDS:
            prefix = lambda r: r["backbone"] == model and r["task"] == task and r["fold"] == fold and r["coordinate"] == coord
            spec = next(r for r in table["DECISION_EFFECTIVE_RANK"] if prefix(r) and r["centering"] == "UNCENTERED_PRIMARY")
            null = [r for r in table["DECISION_SPECTRUM_NULLS"] if prefix(r) and r["null"] == "WITHIN_CELL_LABEL_PERMUTATION"]
            if len(null) != LABEL_NULLS: raise RuntimeError("label-null count mismatch")
            thresholds = {"label_null_k90_p05": float(np.percentile([r["k90"] for r in null], 5)),
                          "label_null_effective_rank_p05": float(np.percentile([r["effective_rank"] for r in null], 5)),
                          "label_null_E4_p95": float(np.percentile([r["E4"] for r in null], 95))}
            low = (spec["k90"] < thresholds["label_null_k90_p05"] and
                   spec["effective_rank"] < thresholds["label_null_effective_rank_p05"] and
                   spec["E4"] > thresholds["label_null_E4_p95"])
            per[coord] = {"k90": spec["k90"], "effective_rank": spec["effective_rank"], "E4": spec["E4"], **thresholds}
            for rank_name in ("compact", "k90"):
                rank_filter = lambda r: prefix(r) and r.get("rank_choice") == rank_name
                outer = [r for r in table["OUTER_DECISION_GEOMETRY_COVERAGE"] if rank_filter(r)]
                controls = [r for r in table["DECISION_BASIS_COVERAGE_CONTROLS"] if rank_filter(r) and r["control"] == "RANK_MATCHED_RANDOM"]
                if len(controls) != RANDOM_BASES * len(outer): raise RuntimeError("random-basis control count mismatch")
                means = {draw: (float(np.mean([r["coverage"] for r in controls if r["draw"] == draw])),
                                float(np.mean([r["reconstruction_cosine"] for r in controls if r["draw"] == draw])))
                         for draw in range(RANDOM_BASES)}
                real_cov = float(np.mean([r["coverage"] for r in outer]))
                real_cos = float(np.mean([r["reconstruction_cosine"] for r in outer]))
                random_cov95 = float(np.percentile([v[0] for v in means.values()], 95))
                random_cos95 = float(np.percentile([v[1] for v in means.values()], 95))
                decoder = {r["decoder"]: r for r in table["SHARED_GEOMETRY_DECODER"] + table["PRIVATE_GEOMETRY_DECODER"]
                           if rank_filter(r) and r["role"] == "OUTER_DEVELOPMENT"}
                shared, private, full = (decoder[name] for name in ("SHARED", "PRIVATE", "FULL_LINEAR_REFERENCE"))
                error = [r for r in table["SHARED_PRIVATE_ERROR_COMPLEMENTARITY"] if rank_filter(r)]
                rescue_values = [r["private_rescue"] for r in error if r["private_rescue"] is not None]
                rescue = float(np.mean(rescue_values)) if rescue_values else 0.0
                fusion = next(r for r in table["SHARED_PRIVATE_STACKED_FUSION"] if rank_filter(r))
                session = next(r for r in table["CROSS_SESSION_BASIS_STABILITY"] if rank_filter(r))
                loso = [r for r in table["LOSO_DECISION_GEOMETRY"] if rank_filter(r)]
                private_struct = [r for r in table["PRIVATE_DIRECTION_STRUCTURE"] if rank_filter(r) and r["role"] == "TRAIN"]
                row = tagged(model, task, fold, coord, rank_choice=rank_name, k=outer[0]["k"],
                    embedding_dim=spec["embedding_dim"], train_decision_vectors=spec["decision_vectors"],
                    k50=spec["k50"], k90=spec["k90"], effective_rank=spec["effective_rank"],
                    normalized_effective_rank=spec["normalized_effective_rank"], E1=spec["E1"], E2=spec["E2"], E4=spec["E4"],
                    **thresholds, low_rank_fold_criterion=low, cross_session_canonical_R2=session["mean_squared_canonical_correlation"],
                    cross_session_projector_distance=session["normalized_projector_distance"],
                    loso_mean_coverage=float(np.mean([r["coverage"] for r in loso])),
                    outer_mean_coverage=real_cov, outer_mean_reconstruction_cosine=real_cos,
                    random_basis_coverage_p95=random_cov95, random_basis_cosine_p95=random_cos95,
                    outer_coverage_beats_random=real_cov > random_cov95, outer_cosine_beats_random=real_cos > random_cos95,
                    shared_only_BA=shared["subject_equal_BA"], shared_only_F1=shared["subject_equal_macro_F1"],
                    shared_only_NLL=shared["subject_equal_NLL"], private_only_BA=private["subject_equal_BA"],
                    private_only_F1=private["subject_equal_macro_F1"], private_only_NLL=private["subject_equal_NLL"],
                    full_linear_BA=full["subject_equal_BA"], full_linear_F1=full["subject_equal_macro_F1"],
                    full_linear_NLL=full["subject_equal_NLL"], private_rescue_fraction=rescue,
                    private_rescue_defined_subjects=len(rescue_values),
                    stacked_BA=fusion["stacked_subject_equal_BA"], stacked_F1=fusion["stacked_subject_equal_macro_F1"],
                    stacked_NLL=fusion["stacked_subject_equal_NLL"],
                    stacked_gain_BA=fusion["stacked_subject_equal_BA"] - shared["subject_equal_BA"],
                    stacked_gain_NLL=shared["subject_equal_NLL"] - fusion["stacked_subject_equal_NLL"],
                    private_direction_cross_session_cosine=float(np.mean([r["private_cross_session_cosine"] for r in private_struct if r["private_cross_session_cosine"] is not None])),
                    shared_direction_cross_session_cosine=float(np.mean([r["shared_cross_session_cosine"] for r in private_struct if r["shared_cross_session_cosine"] is not None])))
                fold_summaries.append(row)
        std_row = next(r for r in fold_summaries if r["backbone"] == model and r["task"] == task and r["fold"] == fold and r["coordinate"] == COORDS[0] and r["rank_choice"] == "k90")
        white_row = next(r for r in fold_summaries if r["backbone"] == model and r["task"] == task and r["fold"] == fold and r["coordinate"] == COORDS[1] and r["rank_choice"] == "k90")
        sensitivity.append(tagged(model, task, fold, "COORDINATE_COMPARISON",
            standardized_k90=per[COORDS[0]]["k90"], whitened_k90=per[COORDS[1]]["k90"],
            standardized_E4=per[COORDS[0]]["E4"], whitened_E4=per[COORDS[1]]["E4"],
            standardized_effective_rank=per[COORDS[0]]["effective_rank"], whitened_effective_rank=per[COORDS[1]]["effective_rank"],
            standardized_low_rank_fold_criterion=std_row["low_rank_fold_criterion"],
            whitened_low_rank_fold_criterion=white_row["low_rank_fold_criterion"],
            standardized_outer_coverage=std_row["outer_mean_coverage"], whitened_outer_coverage=white_row["outer_mean_coverage"],
            standardized_shared_BA=std_row["shared_only_BA"], whitened_shared_BA=white_row["shared_only_BA"],
            standardized_private_rescue=std_row["private_rescue_fraction"], whitened_private_rescue=white_row["private_rescue_fraction"]))
    table["SHARED_PRIVATE_SUMMARY"] = fold_summaries
    table["NORMALIZATION_SENSITIVITY"] = sensitivity
    primary = [r for r in fold_summaries if r["coordinate"] == COORDS[0] and r["rank_choice"] == "k90"]
    cell_results = {}
    for model, task in CELLS:
        rows = sorted((r for r in primary if r["backbone"] == model and r["task"] == task), key=lambda x: x["fold"])
        low_k_count = sum(r["k90"] < r["label_null_k90_p05"] for r in rows)
        low_rank_count = sum(r["effective_rank"] < r["label_null_effective_rank_p05"] for r in rows)
        high_e4_count = sum(r["E4"] > r["label_null_E4_p95"] for r in rows)
        low_joint_count = sum(r["low_rank_fold_criterion"] for r in rows)
        coverage_count = sum(r["outer_coverage_beats_random"] for r in rows)
        cosine_count = sum(r["outer_cosine_beats_random"] for r in rows)
        # Locked operational meaning of 'clearly above chance and meaningfully retained'.
        predict_count = sum(r["shared_only_BA"] >= .55 and r["shared_only_BA"] >= r["full_linear_BA"] - .05 for r in rows)
        fusion_ba_count = sum(r["stacked_gain_BA"] >= .005 for r in rows)
        fusion_nll_count = sum(r["stacked_gain_NLL"] > 0 for r in rows)
        rescue = float(np.mean([r["private_rescue_fraction"] for r in rows]))
        cell_results[f"{model}/{task}"] = {
            "low_rank_supported": low_k_count >= 4 and low_rank_count >= 4 and high_e4_count >= 4,
            "low_k90_fold_count": low_k_count, "low_effective_rank_fold_count": low_rank_count,
            "high_E4_fold_count": high_e4_count, "low_rank_joint_fold_count": low_joint_count,
            "unseen_subject_supported": coverage_count >= 4 and cosine_count >= 4 and predict_count >= 4,
            "random_coverage_fold_count": coverage_count, "random_cosine_fold_count": cosine_count,
            "prediction_retention_fold_count": predict_count,
            "private_complement_supported": rescue >= .10 and (fusion_ba_count >= 3 or fusion_nll_count >= 4),
            "private_rescue_mean": rescue, "fusion_BA_gain_fold_count": fusion_ba_count,
            "fusion_NLL_gain_fold_count": fusion_nll_count,
            "normalized_effective_rank_mean": float(np.mean([r["normalized_effective_rank"] for r in rows])),
            "k90_mean": float(np.mean([r["k90"] for r in rows])),
            "E1_mean": float(np.mean([r["E1"] for r in rows])),
            "E2_mean": float(np.mean([r["E2"] for r in rows])),
            "E4_mean": float(np.mean([r["E4"] for r in rows])),
            "outer_coverage_mean": float(np.mean([r["outer_mean_coverage"] for r in rows])),
            "shared_to_full_BA_ratio": float(np.mean([r["shared_only_BA"] / max(r["full_linear_BA"], 1e-9) for r in rows])),
            "stacked_gain_mean": float(np.mean([r["stacked_gain_BA"] for r in rows]))}
    a, b, c = (cell_results[f"{model}/{task}"] for model, task in CELLS)
    a_good = a["low_rank_supported"] and a["unseen_subject_supported"]
    b_good = b["low_rank_supported"] and b["unseen_subject_supported"]
    c_good = c["low_rank_supported"] and c["unseen_subject_supported"]
    if a_good and b_good and c_good:
        if sum(x["private_complement_supported"] for x in (a, b, c)) >= 2:
            interpretation, action = "SHARED_GEOMETRY_WITH_COMPLEMENTARY_PRIVATE_UTILITY", "PROCEED_SHARED_PRIVATE_DECISION_GEOMETRY_MODEL"
        else:
            interpretation, action = "SHARED_GEOMETRY_WITHOUT_PRIVATE_NEED", "PROCEED_SHARED_DECISION_GEOMETRY_MODEL"
    elif a_good != b_good:
        interpretation, action = "TASK_SPECIFIC_DECISION_GEOMETRY", "CHARACTERIZE_TASK_DEPENDENCE_BEFORE_MODEL"
    elif a_good and not c_good:
        interpretation, action = "BACKBONE_SPECIFIC_DECISION_GEOMETRY", "CHARACTERIZE_BACKBONE_DEPENDENCE_BEFORE_MODEL"
    elif any(x["low_rank_supported"] for x in (a, b, c)) and not all(x["unseen_subject_supported"] for x in (a, b, c)):
        interpretation, action = "LOW_RANK_WITHOUT_UNSEEN_SUBJECT_TRANSFER", "DO_NOT_BUILD_SHARED_GEOMETRY_MODEL"
    else:
        interpretation, action = "SHARED_DECISION_GEOMETRY_NOT_SUPPORTED", "STOP_THIS_MODEL_DIRECTION"
    cross_task = []
    cross_backbone = []
    for comparison, target in (((CELLS[0], CELLS[1]), cross_task), ((CELLS[0], CELLS[2]), cross_backbone)):
        for model, task in comparison:
            for r in primary:
                if (r["backbone"], r["task"]) == (model, task):
                    target.append({key: r[key] for key in ("backbone", "task", "fold", "embedding_dim", "normalized_effective_rank", "k90", "E1", "E2", "E4", "outer_mean_coverage", "private_rescue_fraction", "stacked_gain_BA")} |
                                  {"k90_over_embedding_dim": r["k90"] / r["embedding_dim"],
                                   "shared_to_full_BA_ratio": r["shared_only_BA"] / max(r["full_linear_BA"], 1e-9)})
    table["CROSS_TASK_GEOMETRY_SUMMARY"] = cross_task
    table["CROSS_BACKBONE_GEOMETRY_SUMMARY"] = cross_backbone
    for name, rows in table.items(): write_csv(outdir / f"{name}.csv", rows)
    proof = [{"backbone": value["backbone"], "task": value["task"], "fold": value["fold"],
              "fold_runtime_sha256": sha(path), **value["proof"]} for path, value in fold_outputs]
    save_json_new(EXP / "protocol/REPRESENTATION_PROVENANCE.json", {"schema": "SDG_REPRESENTATION_PROVENANCE_V1", "folds": proof,
        "head_input_proof": "Every batch verifies final affine(head-input) equals native logits within atol=1e-6, rtol=1e-5",
        "neural_state_proof": "parameter and buffer hash unchanged before/after extraction"})
    save_json_new(EXP / "protocol/SOURCE_PROVENANCE.json", {"schema": "SDG_SOURCE_PROVENANCE_V1",
        "source_commits": ["8a8b708b30e19a20a83902d272fbe53fe8279f12", "d5184c5423806693a3b218f2d0329cd6b29d5d9a", "378f71f46fcfc7f8644e8885f15a95af169038d0"],
        "historical_model_source_repo_heads": {"EEGNet_benchmark": "96a0b1435da07ee5b8a424a5b465f1f9be6bc349",
                                               "EEGConformer_baseline": "e19a1aa547b77eb7eca405c1441796b544364f9f"},
        "fold_runner_sha256": sha(Path(os.environ.get("SDG_FOLD_RUNNER_PATH", str(Path(__file__).resolve())))),
        "aggregate_runner_sha256": sha(Path(__file__).resolve()),
        "fold_runtime_sha256": {f"{v['backbone']}/{v['task']}/fold{v['fold']}": sha(p) for p, v in fold_outputs},
        "source_file_sha256": {str(p): sha(p) for p in {SEVEN_CODE / "backbone_models.py", SEVEN_CODE / "benchmark_data.py",
            SEVEN_CODE / "tech_recipe_selection.py", CONFORMER_CODE / "models.py", CONFORMER_CODE / "train_queue.py", TASK_CODE, MODERN_CODE}}})
    save_json_new(outdir / "FINAL_HELDOUT_EXCLUSION_AUDIT.json", {"formal_final_heldout_eeg_reads": 0,
        "allowed_eeg_reader": "tech_recipe_selection._openbmi_rows on exact frozen TRAIN_GEOMETRY/CHECKPOINT_VALIDATION/OUTER_DEVELOPMENT subject IDs only",
        "all_fold_proofs_zero": all(v["proof"]["final_heldout_eeg_reads"] == 0 for _, v in fold_outputs),
        "outer_used_for_fit_or_rank": False, "checkpoint_validation_historically_used_for_checkpoint_selection": True,
        "outer_label_use": "metrics and LABEL_ASSISTED_EVALUATION_ONLY local decision vectors"})
    summary = {"schema": "SDG_DECISION_SUMMARY_V1", "cells": cell_results, "overall_interpretation": interpretation,
        "next_action": action, "formal_final_heldout_eeg_reads": 0,
        "criterion_interpretation": "Shared prediction clearly above chance means subject-equal BA >=0.55 in >=4 folds; meaningfully retained means BA >= full-linear BA -0.05 in >=4 folds; predeclared before OUTER read.",
        "coordinate_primary": COORDS[0], "coordinate_sensitivity": COORDS[1]}
    save_json_new(outdir / "DECISION_SUMMARY.json", summary)
    report = ["# Shared decision geometry characterization (seed 0)", "", "Frozen-backbone diagnostic only. No neural network was trained, no model architecture was implemented, and formal final-heldout EEG reads were 0.",
              "", "All basis/rank/scaler/whitening/decoder/fusion fits used TRAIN_GEOMETRY only. OUTER_DEVELOPMENT labels entered evaluation metrics and explicitly labelled evaluation-only local decision vectors. CHECKPOINT_VALIDATION was previously used for neural checkpoint selection.",
              "", f"Primary overall interpretation: `{interpretation}`. Exact next action: `{action}`.", "",
              "Common-coordinate primary analysis uses a single TRAIN-only StandardScaler per fold on every TRAIN_GEOMETRY trial embedding. A separate TRAIN-only regularized whitening sensitivity uses floor max(1e-4*largest covariance eigenvalue, 1e-8). No diagnostic trial cap is applied; the historical neural input normalizer is fitted on all TRAIN source-session trials.", "",
              "The decomposition is imposed after training: learned local decision normals are described by a shared basis plus orthogonal residual. It is not an explicit factorization of the neural network.", "",
              "An earlier cap-32 diagnostic preview was excluded because it did not use every TRAIN trial embedding; see `protocol/CAP32_PREVIEW_EXCLUSION.json`. Operational startup/serialization failures are disclosed in `protocol/FAILURE_DISCLOSURES.json`; failed artifacts were preserved outside Git, and no scientific fold result was silently overwritten.", ""]
    prompts = ["Q1 — Are decision directions low-dimensional?", "Q2 — Is the basis stable across sessions?", "Q3 — Does a subject-excluded basis explain held-subject rules?", "Q4 — Does shared geometry preserve unseen-subject prediction?", "Q5 — What remains outside the shared basis?", "Q6 — Is private information complementary?", "Q7 — MI to ERP generality", "Q8 — EEGNet to EEGConformer generality", "Q9 — Most defensible mechanism statement", "Q10 — Should a new model be built?"]
    for model, task in CELLS:
        rows = [r for r in primary if (r["backbone"], r["task"]) == (model, task)]
        whitened = [r for r in fold_summaries if (r["backbone"], r["task"], r["coordinate"], r["rank_choice"]) == (model, task, COORDS[1], "k90")]
        cell = cell_results[f"{model}/{task}"]
        avg = lambda key: float(np.mean([r[key] for r in rows]))
        white_counts = (sum(r["k90"] < r["label_null_k90_p05"] for r in whitened),
                        sum(r["effective_rank"] < r["label_null_effective_rank_p05"] for r in whitened),
                        sum(r["E4"] > r["label_null_E4_p95"] for r in whitened))
        white_low = all(count >= 4 for count in white_counts)
        sensitivity_note = ("The low-rank support label CHANGES under TRAIN-only whitening; the structural claim is coordinate-dependent."
                            if white_low != cell["low_rank_supported"] else
                            "The low-rank support label is unchanged under TRAIN-only whitening; numeric differences remain in NORMALIZATION_SENSITIVITY.csv.")
        report += [f"## {model} / {task}", "",
                   f"Q1: k90 mean {avg('k90'):.2f}, effective rank {avg('effective_rank'):.2f}, E(1/2/4) {avg('E1'):.3f}/{avg('E2'):.3f}/{avg('E4'):.3f}; label-null E4 95th percentile {avg('label_null_E4_p95'):.3f}. All singular curves and null draws are in DECISION_SPECTRUM*.csv. Low-rank fold gates k90/r_eff/E4 {cell['low_k90_fold_count']}/{cell['low_effective_rank_fold_count']}/{cell['high_E4_fold_count']} of 5; label {cell['low_rank_supported']}.", "",
                   f"Q2: mean squared canonical correlation {avg('cross_session_canonical_R2'):.3f}, normalized projector distance {avg('cross_session_projector_distance'):.3f}; principal angles in CROSS_SESSION_BASIS_STABILITY.csv.", "",
                   f"Q3: strict LOSO coverage {avg('loso_mean_coverage'):.3f}; OUTER evaluation-only decision coverage {avg('outer_mean_coverage'):.3f}, reconstruction cosine {avg('outer_mean_reconstruction_cosine'):.3f}; rank-matched random coverage p95 {avg('random_basis_coverage_p95'):.3f}.", "",
                   f"Q4: frozen OUTER subject-equal shared-only BA/F1/NLL {avg('shared_only_BA'):.3f}/{avg('shared_only_F1'):.3f}/{avg('shared_only_NLL'):.3f}; full-linear BA {avg('full_linear_BA'):.3f}.", "",
                   f"Q5: OUTER private decision-energy fraction {1-avg('outer_mean_coverage'):.3f}; private-only BA {avg('private_only_BA'):.3f}. Private structure is not called noise.", "",
                   f"Q6: subject-equal private rescue {avg('private_rescue_fraction'):.3f}; stacked BA {avg('stacked_BA'):.3f}, gain over shared {avg('stacked_gain_BA'):.3f}; private-complement label {cell['private_complement_supported']}. Oracle union is nondeployable.", "",
                   f"Coordinate sensitivity: whitened low-rank criteria k90/r_eff/E4 pass {white_counts[0]}/{white_counts[1]}/{white_counts[2]} of 5 folds; whitened OUTER coverage {np.mean([r['outer_mean_coverage'] for r in whitened]):.3f} and shared-only BA {np.mean([r['shared_only_BA'] for r in whitened]):.3f}. {sensitivity_note}", ""]
    report += ["## Cross-cell answers", "", f"Q7: EEGNet MI low/unseen labels {a['low_rank_supported']}/{a['unseen_subject_supported']}; EEGNet ERP {b['low_rank_supported']}/{b['unseen_subject_supported']}. See CROSS_TASK_GEOMETRY_SUMMARY.csv; raw bases are not compared across task semantics.", "",
               f"Q8: EEGNet MI low/unseen labels {a['low_rank_supported']}/{a['unseen_subject_supported']}; EEGConformer MI {c['low_rank_supported']}/{c['unseen_subject_supported']}. See CROSS_BACKBONE_GEOMETRY_SUMMARY.csv; bases from different representation coordinates are not directly compared.", "",
               "Q9: under the predeclared common-standardized coordinates, the three cells' low-rank/unseen-subject labels are "
               + "; ".join(f"{model}/{task}: {cell_results[f'{model}/{task}']['low_rank_supported']}/{cell_results[f'{model}/{task}']['unseen_subject_supported']}" for model, task in CELLS)
               + ". The low-rank gate compares TRAIN decision spectra with TRAIN label-permutation nulls; the unseen-subject gate uses frozen OUTER evaluation. Whitening sensitivity is reported separately in each cell. These are post-hoc geometric descriptions, not causal or architectural claims.", "",
               "Private-complement labels are "
               + "; ".join(f"{model}/{task}: {cell_results[f'{model}/{task}']['private_complement_supported']} (mean stacked BA gain {cell_results[f'{model}/{task}']['stacked_gain_mean']:.3f})" for model, task in CELLS)
               + ". Rescue and NLL can satisfy the locked complement gate even when BA gain is nonpositive; do not equate the gate with a practical accuracy improvement.", "",
               f"Q10: `{action}`. Only a PROCEED label licenses a later model-design experiment; no new model was trained here.", "",
               "Whitening and centered-W results are separate sensitivity analyses. If they conflict with common-standardized uncentered results, the primary claim is coordinate-dependent and should be narrowed.", ""]
    if action.startswith("PROCEED_"):
        report += ["Future model hypothesis only: a trainable model could constrain `w[s,t] ≈ B_shared a[s,t] + r[s,t]` to test whether shared population discriminants plus context-specific private correction help. This branch does not implement that model.", ""]
    report += ["## Primary fold table", "", "| Cell | Dim | Train w | k50 | k90 | r_eff | E1 | E2 | E4 | Null E4 p95 | Low rank | Session R² | OUTER coverage | OUTER cosine | Random coverage p95 | Shared BA | Private BA | Full BA | Private rescue | Stacked BA | Gain |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in primary:
        report.append(f"| {r['backbone']}/{r['task']}/f{r['fold']} | {r['embedding_dim']} | {r['train_decision_vectors']} | {r['k50']} | {r['k90']} | {r['effective_rank']:.2f} | {r['E1']:.3f} | {r['E2']:.3f} | {r['E4']:.3f} | {r['label_null_E4_p95']:.3f} | {r['low_rank_fold_criterion']} | {r['cross_session_canonical_R2']:.3f} | {r['outer_mean_coverage']:.3f} | {r['outer_mean_reconstruction_cosine']:.3f} | {r['random_basis_coverage_p95']:.3f} | {r['shared_only_BA']:.3f} | {r['private_only_BA']:.3f} | {r['full_linear_BA']:.3f} | {r['private_rescue_fraction']:.3f} | {r['stacked_BA']:.3f} | {r['stacked_gain_BA']:.3f} |")
    report += ["", "## Three-cell completion summary", "", "| Cell | Dim | TRAIN w/fold | k50 | k90 | r_eff | E1 | E2 | E4 | Label-null E4 p95 | Low-rank? | Session R² | OUTER coverage | OUTER cosine | Random coverage p95 | Shared BA | Private BA | Full BA | Private rescue | Stacked BA | Stacked gain | Low-rank label | Unseen label | Private label |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|"]
    for model, task in CELLS:
        rows = [r for r in primary if (r["backbone"], r["task"]) == (model, task)]
        cell = cell_results[f"{model}/{task}"]
        avg = lambda key: float(np.mean([r[key] for r in rows]))
        report.append(f"| {model}/{task} | {rows[0]['embedding_dim']} | {rows[0]['train_decision_vectors']} | {avg('k50'):.2f} | {avg('k90'):.2f} | {avg('effective_rank'):.2f} | {avg('E1'):.3f} | {avg('E2'):.3f} | {avg('E4'):.3f} | {avg('label_null_E4_p95'):.3f} | {cell['low_rank_supported']} | {avg('cross_session_canonical_R2'):.3f} | {avg('outer_mean_coverage'):.3f} | {avg('outer_mean_reconstruction_cosine'):.3f} | {avg('random_basis_coverage_p95'):.3f} | {avg('shared_only_BA'):.3f} | {avg('private_only_BA'):.3f} | {avg('full_linear_BA'):.3f} | {avg('private_rescue_fraction'):.3f} | {avg('stacked_BA'):.3f} | {avg('stacked_gain_BA'):.3f} | {cell['low_rank_supported']} | {cell['unseen_subject_supported']} | {cell['private_complement_supported']} |")
    report += ["", f"Primary overall interpretation: `{interpretation}`.", "", f"Exact next action: `{action}`.", "", "Formal final-heldout EEG reads: `0`.", ""]
    with (outdir / "FINAL_REPORT.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(report) + "\n")
    print(f"AGGREGATED {interpretation} {action}", flush=True)


if __name__ == "__main__":
    main()
