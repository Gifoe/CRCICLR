"""Frozen P/C functional-coupling audit, seed 0, outer-development only."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

EXP = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("PERSIST_SOURCE_REPO", str(EXP.parents[1]))).resolve()
RUNTIME = Path(os.environ.get("COUPLING_RUNTIME", str(ROOT.parent / "protected_complement_coupling_runtime")))
OUT, PROTOCOL = EXP / "outputs", EXP / "protocol"
MODELS, TASKS, FOLDS = ("EEGNet", "EEGConformer"), ("OpenBMI_MI", "OpenBMI_SSVEP"), tuple(range(5))
EPS, PAIRS_PER_SUBJECT_CLASS, RANDOM_DRAWS, SURROGATE_RANDOM_DRAWS = 1e-12, 1, 100, 20
BOOTSTRAPS = 2000
MODULE_PATHS = {
    "EEGNet": {
        "temporal_bn": "model.temporal -> model.bn1",
        "spatial_elu_pool1": "model.spatial -> model.bn2 -> elu -> model.pool1 -> model.drop1",
        "depth_point_elu_pool2": "model.depth -> model.point -> model.bn3 -> elu -> model.pool2 -> model.drop2",
        "embedding_64d": "model.embedding",
        "classifier_input": "frozen native head input",
    },
    "EEGConformer": {
        "patch_tokenizer": "model.patch",
        "conformer_block2": "model.encoder[1]",
        "conformer_block4": "model.encoder[3]",
        "conformer_block6": "model.encoder[5]",
        "classifier_256_elu": "model.classifier[0:2]",
        "classifier_32_elu": "model.classifier[2:5]",
        "classifier_input": "model.classifier[5] -> frozen native head input",
    },
}


def import_file(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AR = import_file("coupling_previous_arbitration", ROOT / "experiments" / "persist_eeg_protected_arbitration_reliability_seed0_v1" / "code" / "run_arbitration.py")
BR, PW, UP = AR.BR, AR.PW, AR.UP


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seed(*parts: object) -> int:
    return PW.stable_seed("coupling-v1", *parts)


def clean(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(clean(value), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: clean(row.get(key, "")) for key in columns} for row in rows])
    os.replace(temporary, path)


def cell_path(model: str, task: str, fold: int) -> Path:
    return RUNTIME / "cells" / model.lower() / task.lower() / f"fold{fold}_seed0.json"


def prior_paths(model: str, task: str, fold: int) -> dict[str, Path]:
    root = ROOT.parent
    return {
        "native": UP.target_path(model, task, fold),
        "pathway": PW.path(model, task, fold),
        "emergence": BR.target_path(model, task, fold),
        "arbitration": AR.tpath(model, task, fold),
    }


def prior_record(model: str, task: str, fold: int) -> dict[str, Any]:
    record, checkpoint, stored, native_provenance = UP.provenance_row(model, task, fold)
    paths = prior_paths(model, task, fold)
    rows = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    if any(row.get("status") != "COMPLETE" for row in rows.values()):
        raise RuntimeError("a frozen upstream cell is not COMPLETE")
    data = UP.outer_data(task, fold)
    normalizer_sha = data["normalizer"]["mean_std_sha256"]
    if record["normalizer"]["mean_std_sha256"] != normalizer_sha:
        raise RuntimeError("TRAIN normalizer differs from frozen checkpoint recipe")
    if rows["pathway"]["basis_sha256"] != rows["emergence"]["basis_sha256"] != rows["arbitration"]["basis_sha256"]:
        raise RuntimeError("upstream canonical basis hashes disagree")
    randoms = UP.random_dims(int(stored["rank"]), len(stored["protected_blocks"]), model, task, fold, "native-equal-rank-random")
    random_sha = hashlib.sha256(np.concatenate(randoms).tobytes()).hexdigest()
    if rows["arbitration"].get("random_subsets_sha256") != random_sha:
        raise RuntimeError("equal-rank random-subset hash mismatch")
    return {
        "record": record, "checkpoint": checkpoint, "stored": stored, "data": data,
        "rows": rows, "paths": paths, "randoms": randoms,
        "hashes": {
            "checkpoint_sha256": digest(checkpoint), "normalizer_sha256": normalizer_sha,
            "basis_sha256": rows["pathway"]["basis_sha256"], "split_sha256": data["split_sha256"],
            "random_subsets_sha256": random_sha,
            **{f"previous_{name}_sha256": digest(path) for name, path in paths.items()},
        },
    }


def lock() -> None:
    if (PROTOCOL / "PROVENANCE.json").exists():
        raise RuntimeError("protocol already exists; do not overwrite a frozen lock")
    cells = []
    for model in MODELS:
        for task in TASKS:
            for fold in FOLDS:
                source = prior_record(model, task, fold)
                cells.append({"model": model, "task": task, "fold": fold, **source["hashes"]})
                print("LOCK_SOURCE", model, task, fold, flush=True)
    value = {
        "schema": "PERSIST_EEG_PROTECTED_COMPLEMENT_COUPLING_SEED0_V1",
        "models": MODELS, "tasks": TASKS, "folds": FOLDS, "seed": 0,
        "pairs_per_subject_class": PAIRS_PER_SUBJECT_CLASS,
        "random_partitions": RANDOM_DRAWS, "surrogate_random_partitions": SURROGATE_RANDOM_DRAWS,
        "mapping": "frozen corrected TRAIN-only centroid-to-final-coordinate ridge; raw-space orthonormalized coefficient columns",
        "pair_matching": "deterministic recipient cap and same-subject/class-compatible donor nearest activation norm; no outcome-based layer selection",
        "local_sensitivity": "exact centered-logit VJP projected into P basis and deterministic TRAIN-only C principal directions",
        "surrogate_complement_PCA_dimensions": [16, 32],
        "surrogate_bilinear_ranks": [1, 2, 4, 8],
        "surrogate_target": "frozen centered native logits only",
        "statistical_unit": "biological subject",
        "final_heldout_accessed": False, "backbone_training": False,
        "native_head_refit": False, "protected_reselection": False,
        "cells": cells,
    }
    write_json(PROTOCOL / "PROVENANCE.json", value)
    (PROTOCOL / "PROVENANCE.sha256").write_text(digest(PROTOCOL / "PROVENANCE.json") + "\n", encoding="utf-8")
    print("PROTOCOL_LOCKED", len(cells), flush=True)


def assert_lock(model: str, task: str, fold: int, source: dict[str, Any]) -> str:
    path = PROTOCOL / "PROVENANCE.json"
    expected = (PROTOCOL / "PROVENANCE.sha256").read_text(encoding="utf-8").strip()
    if digest(path) != expected:
        raise RuntimeError("protocol lock hash mismatch")
    locked = json.loads(path.read_text(encoding="utf-8"))
    row = next(item for item in locked["cells"] if (item["model"], item["task"], item["fold"]) == (model, task, fold))
    for name, value in source["hashes"].items():
        if row.get(name) != value:
            raise RuntimeError(f"frozen {name} mismatch")
    return expected


def centered(logits: np.ndarray) -> np.ndarray:
    return logits - logits.mean(axis=-1, keepdims=True)


def norm(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values, axis=-1)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=-1) / np.maximum(norm(a) * norm(b), EPS)


def margin(logits: np.ndarray, label: np.ndarray) -> np.ndarray:
    own = logits[np.arange(len(label)), label]
    others = logits.copy()
    others[np.arange(len(label)), label] = -np.inf
    return own - others.max(axis=1)


def raw_q(fit: Any, canonical_targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover prior ridge column space, then express it in raw activation units."""
    standardized_q, _ = fit.q(np.ascontiguousarray(canonical_targets, np.float32))
    coefficient = standardized_q.astype(np.float64) / fit.std[:, None].astype(np.float64)
    q, _ = np.linalg.qr(coefficient, mode="reduced")
    return np.ascontiguousarray(q.astype(np.float32)), np.ascontiguousarray(fit.mean.astype(np.float32))


def final_q(spec: dict[str, Any], dims: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = UP.raw_base(spec)[dims].astype(np.float64).T
    q, _ = np.linalg.qr(raw, mode="reduced")
    return np.ascontiguousarray(q.astype(np.float32)), np.ascontiguousarray(spec["mean"].astype(np.float32))


def check_partition(activations: np.ndarray, q: np.ndarray, mean: np.ndarray) -> float:
    centered_a = activations - mean
    p = (centered_a @ q) @ q.T
    c = centered_a - p
    error = float(np.max(np.abs(activations.astype(np.float64) - (mean + p + c).astype(np.float64))))
    if error >= 1e-5:
        raise RuntimeError(f"intermediate activation reconstruction failed: {error}")
    return error


def pair_indices(a: np.ndarray, y: np.ndarray, subjects: np.ndarray, sessions: np.ndarray,
                 model: str, task: str, fold: int, stage: str, kind: str,
                 recipient_cap: int | None = PAIRS_PER_SUBJECT_CLASS) -> list[tuple[int, int]]:
    """Outcome-independent norm matching within fixed class/session eligibility."""
    result = []
    magnitude = np.linalg.norm(a, axis=1)
    for subject in PW.natural(subjects):
        for session in sorted(set(map(int, sessions[subjects == subject]))):
            for label in sorted(set(map(int, y[(subjects == subject) & (sessions == session)]))):
                eligible = np.flatnonzero((subjects == subject) & (sessions == session) & (y == label))
                if not len(eligible):
                    continue
                rng = np.random.default_rng(seed("pairs", model, task, fold, stage, kind, subject, session, label))
                recipients = (eligible if recipient_cap is None else
                              np.sort(rng.choice(eligible, min(recipient_cap, len(eligible)), replace=False)))
                for recipient in recipients:
                    if kind == "cross_label":
                        for donor_label in sorted(set(map(int, y[(subjects == subject) & (sessions == session)])) - {label}):
                            candidates = np.flatnonzero((subjects == subject) & (sessions == session) & (y == donor_label))
                            donor = candidates[np.argmin(np.abs(magnitude[candidates] - magnitude[recipient]))]
                            result.append((int(recipient), int(donor)))
                    else:
                        candidates = np.flatnonzero((subjects == subject) & (sessions != session) & (y == label))
                        if len(candidates):
                            donor = candidates[np.argmin(np.abs(magnitude[candidates] - magnitude[recipient]))]
                            result.append((int(recipient), int(donor)))
    return result


def complement_directions(train_a: np.ndarray, q: np.ndarray, mean: np.ndarray, count: int = 4,
                          projector: tuple[np.ndarray, np.ndarray] | None = None,
                          context: Any = None) -> np.ndarray:
    if context is not None:
        return context.components(q, projector, count).cpu().numpy().astype(np.float32)
    centered_a = train_a - mean
    left, right = (q, q.T) if projector is None else projector
    residual = centered_a - (centered_a @ left) @ right
    rank = min(count, len(residual) - 1, residual.shape[1] - q.shape[1])
    if rank < 1:
        return np.empty((train_a.shape[1], 0), np.float32)
    pca = PCA(n_components=rank, svd_solver="randomized", random_state=seed("complement-pca", len(residual), residual.shape[1]))
    pca.fit(residual)
    basis = pca.components_.T.astype(np.float64)
    null_exclusion = np.linalg.qr(left.astype(np.float64), mode="reduced")[0]
    basis -= null_exclusion @ (null_exclusion.T @ basis)
    return np.ascontiguousarray(np.linalg.qr(basis, mode="reduced")[0].astype(np.float32))


class GramPCA:
    """Exact TRAIN-only residual PCA with shared high-dimensional Gram matrix."""

    def __init__(self, train_a: np.ndarray, outer_a: np.ndarray, mean: np.ndarray,
                 device: torch.device):
        self.device = device
        mu = torch.from_numpy(np.ascontiguousarray(mean)).to(device)
        train = torch.from_numpy(np.ascontiguousarray(train_a)).to(device) - mu
        self.offset = train.mean(dim=0)
        self.centered_train = train - self.offset
        del train
        outer = torch.from_numpy(np.ascontiguousarray(outer_a)).to(device) - mu
        self.centered_outer = outer - self.offset
        del outer
        self.gram = self.centered_train @ self.centered_train.T

    def components(self, q: np.ndarray, projector: tuple[np.ndarray, np.ndarray] | None,
                   requested: int) -> torch.Tensor:
        left_np, right_np = (q, q.T) if projector is None else projector
        left = torch.from_numpy(np.ascontiguousarray(left_np)).to(self.device)
        right = torch.from_numpy(np.ascontiguousarray(right_np)).to(self.device)
        xp = self.centered_train @ left
        cross = right @ self.centered_train.T
        residual_gram = self.gram - xp @ cross - cross.T @ xp.T + (xp @ (right @ right.T)) @ xp.T
        residual_gram = 0.5 * (residual_gram + residual_gram.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(residual_gram)
        count = min(requested, len(self.centered_train) - 1, self.centered_train.shape[1] - q.shape[1])
        if count < 1:
            raise RuntimeError("empty complement PCA")
        selected = eigenvectors[:, -count:].flip(dims=(1,))
        values = eigenvalues[-count:].flip(dims=(0,)).clamp_min(1e-8)
        components = (self.centered_train.T @ selected - right.T @ (xp.T @ selected)) / values.sqrt()[None, :]
        return torch.linalg.qr(components, mode="reduced")[0]

    def coordinates(self, q: np.ndarray, projector: tuple[np.ndarray, np.ndarray] | None,
                    requested: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        left_np, right_np = (q, q.T) if projector is None else projector
        left = torch.from_numpy(np.ascontiguousarray(left_np)).to(self.device)
        right = torch.from_numpy(np.ascontiguousarray(right_np)).to(self.device)
        component = self.components(q, projector, requested)
        with torch.inference_mode():
            pt = self.centered_train @ left + self.offset @ left
            po = self.centered_outer @ left + self.offset @ left
            ct = self.centered_train @ component - (self.centered_train @ left) @ (right @ component)
            co = self.centered_outer @ component - (self.centered_outer @ left) @ (right @ component)
            return tuple(np.ascontiguousarray(value.cpu().numpy().astype(np.float32)) for value in (pt, ct, po, co))


def jacobian_at(runner: Any, stage: str, shape: list[int], activations: torch.Tensor,
                q_p: torch.Tensor, q_c: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    a = activations.detach().requires_grad_(True)
    _, logits = runner.from_stage(stage, a.reshape((len(a), *shape)))
    jp, jc = [], []
    for cls in range(logits.shape[1]):
        grad = torch.autograd.grad(logits[:, cls].sum(), a, retain_graph=cls + 1 < logits.shape[1])[0]
        jp.append((grad @ q_p).detach())
        jc.append((grad @ q_c).detach())
    p = torch.stack(jp, dim=1)
    c = torch.stack(jc, dim=1)
    p = p - p.mean(dim=1, keepdim=True)
    c = c - c.mean(dim=1, keepdim=True)
    return p.cpu().numpy(), c.cpu().numpy()


def functional_pairs(runner: Any, stage: str, shape: list[int], activations: np.ndarray,
                     logits: np.ndarray, y: np.ndarray, subjects: np.ndarray,
                     pairs: list[tuple[int, int]], q: np.ndarray, mean: np.ndarray,
                     c_directions: np.ndarray, sessions: np.ndarray,
                     local: bool = True,
                     projector: tuple[np.ndarray, np.ndarray] | None = None) -> list[dict[str, Any]]:
    if not pairs:
        return []
    device = runner.device
    qt = torch.from_numpy(q).to(device)
    ct = torch.from_numpy(c_directions).to(device)
    mu = torch.from_numpy(mean).to(device)
    exact_left = torch.from_numpy(projector[0]).to(device) if projector is not None else None
    exact_right = torch.from_numpy(projector[1]).to(device) if projector is not None else None
    result: list[dict[str, Any]] = []
    for start in range(0, len(pairs), 8):
        part = pairs[start:start + 8]
        rec = np.asarray([p[0] for p in part], int)
        don = np.asarray([p[1] for p in part], int)
        ai = torch.from_numpy(np.ascontiguousarray(activations[rec])).to(device)
        aj = torch.from_numpy(np.ascontiguousarray(activations[don])).to(device)
        xi, xj = ai - mu, aj - mu
        if projector is None:
            pi, pj = (xi @ qt) @ qt.T, (xj @ qt) @ qt.T
        else:
            pi, pj = (xi @ exact_left) @ exact_right, (xj @ exact_left) @ exact_right
        a01, a10 = mu + pi + xj - pj, mu + pj + xi - pi
        with torch.inference_mode():
            _, z01t = runner.from_stage(stage, a01.reshape((len(part), *shape)))
            _, z10t = runner.from_stage(stage, a10.reshape((len(part), *shape)))
            z01, z10 = z01t.float().cpu().numpy(), z10t.float().cpu().numpy()
        z00, z11 = logits[rec], logits[don]
        e_p_i, e_p_j = centered(z10 - z00), centered(z11 - z01)
        e_c_i, e_c_j = centered(z01 - z00), centered(z11 - z10)
        interaction = centered(z00 - z10 - z01 + z11)
        if not np.allclose(e_p_i - e_p_j, -interaction, rtol=1e-4, atol=1e-5):
            raise RuntimeError("P context-modulation algebra mismatch")
        if not np.allclose(e_c_i - e_c_j, -interaction, rtol=1e-4, atol=1e-5):
            raise RuntimeError("C context-modulation algebra mismatch")
        if local:
            jp00, jc00 = jacobian_at(runner, stage, shape, ai, qt, ct)
            jp01, _ = jacobian_at(runner, stage, shape, a01, qt, ct)
            _, jc10 = jacobian_at(runner, stage, shape, a10, qt, ct)
            djp = np.linalg.norm((jp00 - jp01).reshape(len(part), -1), axis=1) / np.maximum(np.linalg.norm(jp00.reshape(len(part), -1), axis=1), EPS)
            djc = np.linalg.norm((jc00 - jc10).reshape(len(part), -1), axis=1) / np.maximum(np.linalg.norm(jc00.reshape(len(part), -1), axis=1), EPS)
        else:
            djp, djc = np.full(len(part), np.nan), np.full(len(part), np.nan)
        predictions = [z.argmax(1) for z in (z00, z10, z01, z11)]
        runner_up = np.argsort(z00, axis=1)[:, -2]
        for k, (i, j) in enumerate(part):
            true_label, donor_label = int(y[i]), int(y[j])
            true_other = int(np.argsort(z00[k])[-1 if predictions[0][k] != true_label else -2])
            p_margin_i = float(e_p_i[k, donor_label] - e_p_i[k, true_label])
            p_margin_j = float(e_p_j[k, donor_label] - e_p_j[k, true_label])
            c_margin_i = float(e_c_i[k, donor_label] - e_c_i[k, true_label])
            c_margin_j = float(e_c_j[k, donor_label] - e_c_j[k, true_label])
            interaction_correct = float(interaction[k, true_label] - interaction[k, true_other])
            result.append({
                "recipient": int(i), "donor": int(j), "subject_id": str(subjects[i]),
                "session": int(sessions[i]),
                "recipient_label": true_label, "donor_label": donor_label,
                "native_correct": int(predictions[0][k] == true_label),
                "P_effect_norm": float(norm(e_p_i[k])), "C_effect_norm": float(norm(e_c_i[k])),
                "interaction_norm": float(norm(interaction[k])),
                "interaction_main_ratio": float(norm(interaction[k]) / max(norm(e_p_i[k]) + norm(e_c_i[k]), EPS)),
                "P_effect_context_cosine": float(cosine(e_p_i[k], e_p_j[k])),
                "C_effect_context_cosine": float(cosine(e_c_i[k], e_c_j[k])),
                "P_effect_magnitude_ratio": float(norm(e_p_i[k]) / max(norm(e_p_j[k]), EPS)),
                "C_effect_magnitude_ratio": float(norm(e_c_i[k]) / max(norm(e_c_j[k]), EPS)),
                "P_margin_sign_flip": int(p_margin_i * p_margin_j < 0),
                "C_margin_sign_flip": int(c_margin_i * c_margin_j < 0),
                "P_donor_margin_effect": p_margin_i, "C_donor_margin_effect": c_margin_i,
                "P_recipient_margin_effect": -p_margin_i, "C_recipient_margin_effect": -c_margin_i,
                "P_prediction_flip": int(predictions[1][k] != predictions[0][k]),
                "C_prediction_flip": int(predictions[2][k] != predictions[0][k]),
                "P_donor_label_transfer": int(predictions[1][k] == donor_label),
                "C_donor_label_transfer": int(predictions[2][k] == donor_label),
                "interaction_margin_effect": interaction_correct,
                "interaction_donor_margin_effect": float(interaction[k, donor_label] - interaction[k, true_label]),
                "interaction_native_top1_margin_effect": float(interaction[k, predictions[0][k]] - interaction[k, runner_up[k]]),
                "interaction_alignment_with_correct_decision": float(np.sign(interaction_correct)),
                "interaction_helps_correct": int(predictions[0][k] == true_label and interaction_correct > 0),
                "interaction_contributes_error": int(predictions[0][k] != true_label and interaction_correct < 0),
                "interaction_reverses_P": int(p_margin_i < 0 and p_margin_i + float(interaction[k, donor_label] - interaction[k, true_label]) > 0),
                "interaction_enables_C_rescue": int(c_margin_i < 0 and c_margin_i + float(interaction[k, donor_label] - interaction[k, true_label]) > 0),
                "D_JP": float(djp[k]), "D_JC": float(djc[k]),
                "E_P": e_p_i[k], "E_C": e_c_i[k], "I_PC": interaction[k],
                "E_P_donor_context": e_p_j[k], "E_C_donor_context": e_c_j[k],
                "J_P": jp00[k] if local else None, "J_C": jc00[k] if local else None,
                "J_P_donor_context": jp01[k] if local else None,
                "J_C_donor_context": jc10[k] if local else None,
            })
    return result


PAIR_METRICS = (
    "P_effect_norm", "C_effect_norm", "interaction_norm", "interaction_main_ratio",
    "P_effect_context_cosine", "C_effect_context_cosine", "P_effect_magnitude_ratio",
    "C_effect_magnitude_ratio", "P_margin_sign_flip", "C_margin_sign_flip",
    "P_donor_margin_effect", "C_donor_margin_effect", "P_recipient_margin_effect",
    "C_recipient_margin_effect", "P_prediction_flip", "C_prediction_flip",
    "P_donor_label_transfer", "C_donor_label_transfer", "interaction_margin_effect",
    "interaction_donor_margin_effect", "interaction_native_top1_margin_effect",
    "interaction_alignment_with_correct_decision", "interaction_helps_correct",
    "interaction_contributes_error", "interaction_reverses_P", "interaction_enables_C_rescue",
    "D_JP", "D_JC",
)


def subject_pair_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for owner in PW.natural([r["subject_id"] for r in rows]):
        group = [r for r in rows if r["subject_id"] == owner]
        result.append({"subject_id": owner, "pairs": len(group), **{
            key: float(np.nanmean([r[key] for r in group])) for key in PAIR_METRICS
        }})
    return result


FEATURE_KEYS = (
    "P_effect_norm", "C_effect_norm", "interaction_norm", "interaction_main_ratio",
    "P_effect_context_cosine", "C_effect_context_cosine", "P_margin_sign_flip",
    "C_margin_sign_flip", "D_JP", "D_JC",
)


def trial_coupling_features(rows: list[dict[str, Any]], n: int) -> np.ndarray:
    sums = np.zeros((n, len(FEATURE_KEYS)), np.float64)
    counts = np.zeros(n, np.int64)
    for row in rows:
        idx = row["recipient"]
        sums[idx] += np.asarray([row[k] for k in FEATURE_KEYS], np.float64)
        counts[idx] += 1
    sums /= np.maximum(counts[:, None], 1)
    sums[counts == 0] = np.nan
    return sums.astype(np.float32)


def session_stability_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for owner in PW.natural([r["subject_id"] for r in rows]):
        group = [r for r in rows if r["subject_id"] == owner]
        sessions = sorted(set(int(r["session"]) for r in group))
        if len(sessions) != 2:
            continue
        by_key: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            by_key[(row["recipient_label"], row["donor_label"], row["session"])].append(row)
        comparisons = []
        for rec_label, don_label in sorted({(r["recipient_label"], r["donor_label"]) for r in group}):
            left = by_key.get((rec_label, don_label, sessions[0]), [])
            right = by_key.get((rec_label, don_label, sessions[1]), [])
            if left and right:
                comparisons.append((left, right))
        if not comparisons:
            continue
        data = {"subject_id": owner, "matched_class_pairs": len(comparisons)}
        for name in ("E_P", "E_C", "I_PC", "J_P", "J_C"):
            first = np.stack([np.mean(np.stack([r[name] for r in a]), axis=0).ravel() for a, _ in comparisons])
            second = np.stack([np.mean(np.stack([r[name] for r in b]), axis=0).ravel() for _, b in comparisons])
            data[f"{name}_normalized_rms"] = float(np.sqrt(np.mean((first - second) ** 2)) / max(float(np.sqrt(np.mean(first ** 2))), EPS))
            data[f"{name}_cosine"] = float(np.mean(cosine(first, second)))
            data[f"{name}_correlation"] = float(np.corrcoef(first.ravel(), second.ravel())[0, 1]) if first.size > 1 and np.std(first) > 0 and np.std(second) > 0 else None
        result.append(data)
    return result


def same_label_interchange_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for owner in PW.natural([r["subject_id"] for r in rows]):
        group = [r for r in rows if r["subject_id"] == owner]
        data: dict[str, Any] = {"subject_id": owner, "matched_pairs": len(group)}
        for name in ("E_P", "E_C", "J_P", "J_C"):
            first = np.stack([r[name].ravel() for r in group])
            second = np.stack([r[f"{name}_donor_context"].ravel() for r in group])
            data[f"{name}_cosine"] = float(np.mean(cosine(first, second)))
            data[f"{name}_normalized_rms"] = float(np.sqrt(np.mean((first - second) ** 2)) / max(float(np.sqrt(np.mean(first ** 2))), EPS))
            data[f"{name}_correlation"] = float(np.corrcoef(first.ravel(), second.ravel())[0, 1]) if first.size > 1 and np.std(first) > 0 and np.std(second) > 0 else None
        data["I_PC_norm"] = float(np.mean([r["interaction_norm"] for r in group]))
        result.append(data)
    return result


def final_geometry(q: np.ndarray, z0: np.ndarray, zp: np.ndarray, zc: np.ndarray,
                   z: np.ndarray, dims: np.ndarray) -> np.ndarray:
    pz, cz = z0 + zp, z0 + zc
    probabilities = AR.sm(z)
    top = np.partition(z, -2, axis=1)
    pm = np.partition(pz, -2, axis=1)
    cm = np.partition(cz, -2, axis=1)
    p = q[:, dims]
    c = q[:, np.setdiff1d(np.arange(q.shape[1]), dims)]
    pnorm, cnorm = norm(p), norm(c)
    return np.column_stack((
        top[:, -1] - top[:, -2], -np.sum(probabilities * np.log(np.maximum(probabilities, EPS)), axis=1),
        pm[:, -1] - pm[:, -2], cm[:, -1] - cm[:, -2],
        AR.cosine(centered(zp), centered(zc)), pnorm / np.maximum(cnorm, EPS),
        (pz.argmax(1) == cz.argmax(1)).astype(float),
    )).astype(np.float32)


def impute_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    med = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.)
    values = np.where(np.isfinite(x), x, med)
    mean = values.mean(axis=0)
    scale = np.maximum(values.std(axis=0), 1e-6)
    return (values - mean) / scale, med, np.stack((mean, scale))


def impute_apply(x: np.ndarray, med: np.ndarray, stats: np.ndarray) -> np.ndarray:
    return (np.where(np.isfinite(x), x, med) - stats[0]) / stats[1]


def subject_folds(subjects: np.ndarray, folds: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.asarray(PW.natural(subjects))
    result = []
    for held in np.array_split(unique, min(folds, len(unique))):
        valid = np.isin(subjects, held)
        if np.any(valid) and np.any(~valid):
            result.append((~valid, valid))
    return result


def error_targets(y: np.ndarray, z: np.ndarray, zp: np.ndarray, zc: np.ndarray,
                  surface: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    p_correct = zp.argmax(1) == y
    c_correct = zc.argmax(1) == y
    result = {
        "native_error": (z.argmax(1) != y).astype(int),
        "P_correct_C_wrong": (p_correct & ~c_correct).astype(int),
        "P_wrong_C_correct": (~p_correct & c_correct).astype(int),
    }
    if surface is not None:
        result["NOT_REWEIGHT_RECOVERABLE"] = np.where(z.argmax(1) == y, -1, surface.astype(int))
    return result


def probability_metrics(target: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    if len(target) == 0:
        return {"AUROC": None, "AUPRC": None, "Brier": None, "ECE_10bin": None}
    bins = np.minimum((predicted * 10).astype(int), 9)
    ece = sum(float(np.mean(bins == b)) * abs(float(np.mean(predicted[bins == b]) - np.mean(target[bins == b])))
              for b in range(10) if np.any(bins == b))
    return {
        "AUROC": float(roc_auc_score(target, predicted)) if len(np.unique(target)) > 1 else None,
        "AUPRC": float(average_precision_score(target, predicted)) if np.any(target) else None,
        "Brier": float(brier_score_loss(target, predicted)), "ECE_10bin": float(ece),
    }


def fit_error_model(x: np.ndarray, target: np.ndarray, subjects: np.ndarray,
                    tag: tuple[Any, ...]) -> tuple[Any, np.ndarray, np.ndarray, float]:
    scaled, med, stats = impute_fit(x)
    if len(np.unique(target)) < 2:
        return float(np.mean(target)), med, stats, float("nan")
    candidates = (0.01, 0.1, 1., 10.)
    scores = []
    for c in candidates:
        parts = []
        for fit, valid in subject_folds(subjects):
            if len(np.unique(target[fit])) < 2:
                continue
            fold_x, fold_med, fold_stats = impute_fit(x[fit])
            model = LogisticRegression(C=c, penalty="l2", solver="lbfgs", max_iter=500,
                                       random_state=seed(*tag, "inner", c))
            model.fit(fold_x, target[fit])
            p = model.predict_proba(impute_apply(x[valid], fold_med, fold_stats))[:, 1]
            parts.append(float(np.mean((p - target[valid]) ** 2)))
        scores.append(np.mean(parts) if parts else np.inf)
    chosen = float(candidates[int(np.argmin(scores))])
    model = LogisticRegression(C=chosen, penalty="l2", solver="lbfgs", max_iter=500,
                               random_state=seed(*tag, "refit"))
    model.fit(scaled, target)
    return model, med, stats, chosen


def predict_error(model: Any, med: np.ndarray, stats: np.ndarray, x: np.ndarray) -> np.ndarray:
    if isinstance(model, float):
        return np.full(len(x), model, np.float32)
    return model.predict_proba(impute_apply(x, med, stats))[:, 1].astype(np.float32)


def error_predictability(geometry_train: np.ndarray, geometry_outer: np.ndarray,
                         coupling_train: dict[str, np.ndarray], coupling_outer: dict[str, np.ndarray],
                         target_train: dict[str, np.ndarray], target_outer: dict[str, np.ndarray],
                         subjects: np.ndarray, outer_subjects: np.ndarray,
                         tag: tuple[Any, ...]) -> list[dict[str, Any]]:
    result = []
    stage_names = list(coupling_train)
    for target_name in target_train:
        train_mask = target_train[target_name] >= 0
        outer_mask = target_outer[target_name] >= 0
        for family in ("BASE", *stage_names, "ALL_LAYER"):
            if family == "BASE":
                xt, xo = geometry_train, geometry_outer
            elif family == "ALL_LAYER":
                xt = np.column_stack((geometry_train, *[coupling_train[n] for n in stage_names]))
                xo = np.column_stack((geometry_outer, *[coupling_outer[n] for n in stage_names]))
            else:
                xt = np.column_stack((geometry_train, coupling_train[family]))
                xo = np.column_stack((geometry_outer, coupling_outer[family]))
            fitted, med, stats, c = fit_error_model(xt[train_mask], target_train[target_name][train_mask], subjects[train_mask], (*tag, target_name, family))
            p = predict_error(fitted, med, stats, xo[outer_mask])
            outer_sub = outer_subjects[outer_mask]
            outer_target = target_outer[target_name][outer_mask]
            for owner in PW.natural(outer_subjects):
                ix = outer_sub == owner
                result.append({"target": target_name, "family": family, "subject_id": owner,
                               "train_selected_C": c, **probability_metrics(outer_target[ix], p[ix])})
    return result


def surrogate_coordinates(train_a: np.ndarray, outer_a: np.ndarray, q: np.ndarray,
                          mean: np.ndarray, c_dim: int,
                          projector: tuple[np.ndarray, np.ndarray] | None = None,
                          context: Any = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if context is not None:
        return context.coordinates(q, projector, c_dim)
    t = train_a - mean
    o = outer_a - mean
    left, right = (q, q.T) if projector is None else projector
    p_train, p_outer = t @ left, o @ left
    residual = t - p_train @ right
    o_residual = o - p_outer @ right
    n = min(c_dim, residual.shape[0] - 1, residual.shape[1] - q.shape[1])
    if n < 1:
        raise RuntimeError("no complement dimensions for surrogate")
    pca = PCA(n_components=n, svd_solver="randomized", random_state=seed("surrogate-pca", len(t), t.shape[1], n))
    c_train = pca.fit_transform(residual)
    c_outer = pca.transform(o_residual)
    return p_train.astype(np.float32), c_train.astype(np.float32), p_outer.astype(np.float32), c_outer.astype(np.float32)


def interaction_factors(p: np.ndarray, c: np.ndarray, target: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    # Fit full second-order coefficients on TRAIN only; deterministic greedy
    # CP decomposition supplies shared U_P/U_C factors for the rank-r family.
    linear = np.column_stack((p, c))
    products = (p[:, :, None] * c[:, None, :]).reshape(len(p), -1)
    full = Ridge(alpha=1.).fit(np.column_stack((linear, products)), target)
    tensor = full.coef_[:, linear.shape[1]:].T.reshape(p.shape[1], c.shape[1], target.shape[1]).astype(np.float64)
    up, uc = [], []
    for _ in range(rank):
        u, singular, vt = np.linalg.svd(tensor.reshape(p.shape[1], -1), full_matrices=False)
        if not len(singular) or singular[0] < 1e-10:
            break
        v, s2, wt = np.linalg.svd(vt[0].reshape(c.shape[1], target.shape[1]), full_matrices=False)
        left, right = u[:, 0], v[:, 0]
        weight = np.einsum("pck,p,c,k->", tensor, left, right, wt[0])
        tensor -= weight * np.einsum("p,c,k->pck", left, right, wt[0])
        up.append(left)
        uc.append(right)
    if not up:
        return np.zeros((p.shape[1], 1)), np.zeros((c.shape[1], 1))
    return np.stack(up, axis=1).astype(np.float32), np.stack(uc, axis=1).astype(np.float32)


def surrogate_design(p: np.ndarray, c: np.ndarray, family: str,
                     factors: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    if family == "additive":
        return np.column_stack((p, c))
    assert factors is not None
    u, v = factors
    return np.column_stack((p, c, (p @ u) * (c @ v)))


def surrogate_score(native: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    y, z = centered(native), centered(predicted)
    denom = np.sum((y - y.mean(axis=0)) ** 2)
    r2 = 1. - np.sum((y - z) ** 2) / max(denom, EPS)
    true_margin = np.partition(y, -2, axis=1)
    pred_margin = np.partition(z, -2, axis=1)
    pn = AR.sm(native)
    pp = AR.sm(predicted)
    return {
        "centered_logit_R2": float(r2), "logit_cosine": float(np.mean(cosine(y, z))),
        "native_margin_correlation": float(np.corrcoef(true_margin[:, -1] - true_margin[:, -2],
                                                           pred_margin[:, -1] - pred_margin[:, -2])[0, 1]),
        "KL_to_native": float(np.mean(np.sum(pn * (np.log(np.maximum(pn, EPS)) - np.log(np.maximum(pp, EPS))), axis=1))),
        "hard_prediction_agreement": float(np.mean(native.argmax(1) == predicted.argmax(1))),
    }


def fit_bilinear(p: np.ndarray, c: np.ndarray, y: np.ndarray, rank: int) -> tuple[Any, tuple[np.ndarray, np.ndarray]]:
    factors = interaction_factors(p, c, y, rank)
    model = Ridge(alpha=1.).fit(surrogate_design(p, c, "bilinear", factors), y)
    return model, factors


def select_bilinear_rank(p: np.ndarray, c: np.ndarray, z: np.ndarray, subjects: np.ndarray) -> int:
    candidates = (1, 2, 4, 8)
    scores = []
    for rank in candidates:
        losses = []
        for fit, valid in subject_folds(subjects):
            model, factors = fit_bilinear(p[fit], c[fit], z[fit], rank)
            prediction = model.predict(surrogate_design(p[valid], c[valid], "bilinear", factors))
            losses.append(float(np.mean((centered(z[valid]) - centered(prediction)) ** 2)))
        scores.append(float(np.mean(losses)))
    return candidates[int(np.argmin(scores))]


def fit_mlp(x: np.ndarray, y: np.ndarray, rank: int, p_dim: int, c_dim: int,
            tag: tuple[Any, ...]) -> tuple[Any, np.ndarray, np.ndarray]:
    # The nearest hidden width makes parameter count comparable to the chosen
    # low-rank bilinear model. Only frozen centered logits are regression targets.
    output_dim = y.shape[1]
    bilinear_params = (p_dim + c_dim + 1) * output_dim + (p_dim + c_dim + output_dim) * rank
    width = max(1, round((bilinear_params - output_dim) / (x.shape[1] + output_dim + 1)))
    mu, sd = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-6)
    torch.manual_seed(seed(*tag, "mlp"))
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], width), torch.nn.Tanh(),
                                torch.nn.Linear(width, output_dim))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    xt = torch.from_numpy(np.ascontiguousarray((x - mu) / sd, np.float32))
    yt = torch.from_numpy(np.ascontiguousarray(centered(y), np.float32))
    for _ in range(100):
        optimizer.zero_grad()
        prediction = model(xt)
        prediction = prediction - prediction.mean(dim=1, keepdim=True)
        loss = torch.mean((prediction - yt) ** 2)
        loss.backward()
        optimizer.step()
    return model.eval(), mu, sd


def functional_surrogate(train_a: np.ndarray, outer_a: np.ndarray, train_z: np.ndarray,
                         outer_z: np.ndarray, train_subjects: np.ndarray, outer_subjects: np.ndarray,
                         q: np.ndarray, mean: np.ndarray, tag: tuple[Any, ...],
                         include_mlp: bool = True,
                         projector: tuple[np.ndarray, np.ndarray] | None = None,
                         context: Any = None) -> list[dict[str, Any]]:
    rows = []
    for c_dim in (16, 32):
        pt, ct, po, co = surrogate_coordinates(train_a, outer_a, q, mean, c_dim, projector, context)
        p_mu, p_sd = pt.mean(0), np.maximum(pt.std(0), 1e-6)
        c_mu, c_sd = ct.mean(0), np.maximum(ct.std(0), 1e-6)
        pt, po = (pt - p_mu) / p_sd, (po - p_mu) / p_sd
        ct, co = (ct - c_mu) / c_sd, (co - c_mu) / c_sd
        target = centered(train_z)
        additive = Ridge(alpha=1.).fit(surrogate_design(pt, ct, "additive"), target)
        add_pred = additive.predict(surrogate_design(po, co, "additive"))
        rank = select_bilinear_rank(pt, ct, target, train_subjects)
        bilinear, factors = fit_bilinear(pt, ct, target, rank)
        bil_pred = bilinear.predict(surrogate_design(po, co, "bilinear", factors))
        predictions = {"additive": add_pred, "bilinear": bil_pred}
        if include_mlp:
            x_train, x_outer = np.column_stack((pt, ct)), np.column_stack((po, co))
            mlp, mu, sd = fit_mlp(x_train, target, rank, pt.shape[1], ct.shape[1], (*tag, c_dim))
            with torch.inference_mode():
                predictions["matched_MLP"] = mlp(torch.from_numpy(np.ascontiguousarray((x_outer - mu) / sd, np.float32))).numpy()
        for family, prediction in predictions.items():
            for owner in PW.natural(outer_subjects):
                ix = outer_subjects == owner
                rows.append({"PCA_C_dim": c_dim, "family": family, "subject_id": owner,
                             "actual_C_dim": ct.shape[1],
                             "selected_bilinear_rank": rank, **surrogate_score(outer_z[ix], prediction[ix])})
    return rows


def previous_grid_nonrecoverable(y: np.ndarray, z: np.ndarray, z0: np.ndarray,
                               zp: np.ndarray, zc: np.ndarray) -> np.ndarray:
    """Reproduce the preceding fixed 2D response grid only for target D."""
    correct = np.zeros(len(y), bool)
    for beta in AR.SCALE:
        for alpha in AR.SCALE:
            correct |= (z0 + beta * zp + alpha * zc).argmax(1) == y
    return (z.argmax(1) != y) & ~correct


def serializable_pairs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hidden = ("E_P", "E_C", "I_PC", "J_P", "J_C",
              "E_P_donor_context", "E_C_donor_context", "J_P_donor_context", "J_C_donor_context")
    return [{key: value for key, value in row.items() if key not in hidden}
            for row in rows]


def stage_cell(runner: Any, model: str, task: str, fold: int, stage: str,
               shape: list[int], train_centroids: dict[str, Any], canonical_centroids: np.ndarray,
               spec: dict[str, Any], dims: np.ndarray, randoms: list[np.ndarray],
               tx: np.ndarray, ty: np.ndarray, ts: np.ndarray, tse: np.ndarray,
               ox: np.ndarray, oy: np.ndarray, osub: np.ndarray, ose: np.ndarray,
               source_count: int, protocol_sha: str) -> dict[str, Any]:
    tag = (model, task, fold, stage)
    train = PW.outer_stage(runner, tx, stage)
    outer = PW.outer_stage(runner, ox, stage)
    a_train, a_outer = train["a"], outer["a"]
    centroid_a = train_centroids["acts"][stage]
    fit = PW.PathFit(centroid_a) if stage != "classifier_input" else None
    center_targets = canonical_centroids[:, dims]
    if fit is None:
        q, mu = final_q(spec, dims)
        transform = (spec["basis"] / spec["scale"][None, :]) @ spec["directions"][:, dims]
        projector = (np.ascontiguousarray(transform.astype(np.float32)),
                     np.ascontiguousarray(UP.raw_base(spec)[dims].astype(np.float32)))
    else:
        q, mu = raw_q(fit, center_targets)
        projector = None
    error = max(check_partition(a_train[:min(16, len(a_train))], q, mu),
                check_partition(a_outer[:min(16, len(a_outer))], q, mu))
    if projector is not None:
        delta = a_outer[:16] - mu
        p = (delta @ projector[0]) @ projector[1]
        c = delta - p
        error = max(error, float(np.max(np.abs(a_outer[:16] - (mu + p + c)))))
    centroid_pca = GramPCA(centroid_a, centroid_a[:1], mu, runner.device)
    surrogate_pca = GramPCA(a_train, a_outer[source_count:], mu, runner.device)
    cdir = complement_directions(centroid_a, q, mu, projector=projector, context=centroid_pca)
    train_pairs = pair_indices(a_train, ty, ts, tse, model, task, fold, stage, "cross_label", None)
    outer_pairs = pair_indices(a_outer, oy, osub, ose, model, task, fold, stage, "cross_label")
    outer_feature_pairs = pair_indices(a_outer, oy, osub, ose, model, task, fold, stage, "cross_label", None)
    same_label_pairs = pair_indices(a_outer, oy, osub, ose, model, task, fold, stage, "cross_session")
    p_train = functional_pairs(runner, stage, shape, a_train, train["z"], ty, ts,
                               train_pairs, q, mu, cdir, tse, projector=projector)
    p_outer_full = functional_pairs(runner, stage, shape, a_outer, outer["z"], oy, osub,
                                    outer_feature_pairs, q, mu, cdir, ose, projector=projector)
    subset = set(outer_pairs)
    p_outer = [row for row in p_outer_full if (row["recipient"], row["donor"]) in subset]
    if len(p_outer) != len(outer_pairs):
        raise RuntimeError("P pair subset does not match fixed random-control pairs")
    p_same_label = functional_pairs(runner, stage, shape, a_outer, outer["z"], oy, osub,
                                    same_label_pairs, q, mu, cdir, ose, projector=projector)
    future_pairs = [row for row in p_outer if row["session"] == int(ose[source_count])]
    future_feature_pairs = [row for row in p_outer_full if row["session"] == int(ose[source_count])]
    shifted = [{**row, "recipient": row["recipient"] - source_count,
                "donor": row["donor"] - source_count} for row in future_feature_pairs]
    if any(row["recipient"] < 0 or row["donor"] < 0 for row in shifted):
        raise RuntimeError("outer-source pair leaked into future-only error features")
    train_features = trial_coupling_features(p_train, len(tx))
    outer_features = trial_coupling_features(shifted, len(ox) - source_count)
    random_rows, random_surrogates, random_stability = [], [], []
    for draw, subset in enumerate(randoms):
        if fit is None:
            qr, mur = final_q(spec, subset)
            tf = (spec["basis"] / spec["scale"][None, :]) @ spec["directions"][:, subset]
            rproj = (np.ascontiguousarray(tf.astype(np.float32)),
                     np.ascontiguousarray(UP.raw_base(spec)[subset].astype(np.float32)))
        else:
            qr, mur = raw_q(fit, canonical_centroids[:, subset])
            rproj = None
        random_cdir = complement_directions(centroid_a, qr, mur, projector=rproj, context=centroid_pca)
        random_effects = functional_pairs(runner, stage, shape, a_outer, outer["z"], oy, osub,
                                          outer_pairs, qr, mur, random_cdir, ose, projector=rproj)
        random_same_label = functional_pairs(runner, stage, shape, a_outer, outer["z"], oy, osub,
                                             same_label_pairs, qr, mur, random_cdir, ose, projector=rproj)
        future_random = [row for row in random_effects if row["session"] == int(ose[source_count])]
        for row in subject_pair_rows(future_random):
            random_rows.append({"draw": draw, **row})
        for row in session_stability_rows(random_effects):
            random_stability.append({"draw": draw, "pair_type": "cross_label_matched_across_sessions", **row})
        for row in same_label_interchange_rows(random_same_label):
            random_stability.append({"draw": draw, "pair_type": "same_label_cross_session_swap", **row})
        if draw < SURROGATE_RANDOM_DRAWS:
            for row in functional_surrogate(a_train, a_outer[source_count:], train["z"],
                                            outer["z"][source_count:], ts, osub[source_count:],
                                            qr, mur, (*tag, "random", draw), projector=rproj,
                                            context=surrogate_pca):
                random_surrogates.append({"draw": draw, **row})
        if (draw + 1) % 10 == 0:
            print("RANDOM_PARTITION_DONE", *tag, draw + 1, flush=True)
    surrogates = functional_surrogate(a_train, a_outer[source_count:], train["z"],
                                      outer["z"][source_count:], ts, osub[source_count:],
                                      q, mu, tag, projector=projector, context=surrogate_pca)
    result = {
        "stage": stage, "shape": shape, "P_rank": int(q.shape[1]),
        "module_path": MODULE_PATHS[model][stage], "mapping_sha256": AR.array_digest(q, mu),
        "reconstruction_max_abs": error,
        "train_pairs": len(p_train), "outer_pairs": len(p_outer),
        "outer_feature_pairs": len(p_outer_full),
        "train_coupling_features": train_features,
        "outer_coupling_features": outer_features,
        "main_effects": subject_pair_rows(future_pairs),
        "interaction": serializable_pairs(future_feature_pairs),
        "random_subject_effects": random_rows,
        "session_stability": ([{"pair_type": "cross_label_matched_across_sessions", **row}
                                for row in session_stability_rows(p_outer)] +
                               [{"pair_type": "same_label_cross_session_swap", **row}
                                for row in same_label_interchange_rows(p_same_label)]),
        "random_session_stability": random_stability,
        "surrogate": surrogates,
        "random_surrogate": random_surrogates,
        "protocol_sha256": protocol_sha,
        "implementation_sha256": digest(Path(__file__)),
    }
    return result


def cell(model: str, task: str, fold: int) -> None:
    destination = cell_path(model, task, fold)
    if destination.exists():
        print("CELL_CACHED", model, task, fold, flush=True)
        return
    base = {"model": model, "task": task, "fold": fold, "seed": 0,
            "backbone_training": False, "native_head_refit": False,
            "protected_reselection": False, "final_heldout_accessed": False}
    try:
        source = prior_record(model, task, fold)
        protocol_sha = assert_lock(model, task, fold, source)
        record, stored, checkpoint, data = source["record"], source["stored"], source["checkpoint"], source["data"]
        if source["hashes"]["checkpoint_sha256"] != source["rows"]["arbitration"]["checkpoint_sha256"]:
            raise RuntimeError("checkpoint differs from prior arbitration cell")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net, head = UP.helper(model).build_model({
            "Model": model, "Task": task, "fold": fold, "seed": 0,
            "channels": int(record.get("channels") or 62), "samples": int(record.get("samples") or 1000),
            "classes": int(record["classes"]), "checkpoint_path": str(checkpoint),
            "recipe_name": record.get("recipe", {}).get("name"),
            "trainable_parameters": int(record.get("trainable_parameters", record.get("parameters", 0))),
        }, device)
        runner = PW.Stages(net, head, model, device)
        tx, ty, ts, tse = BR.trial_train(data, model, task, fold)
        PW.stage_check(runner, tx)
        train_centroids = PW.train_centroids(runner, data, model, task, fold)
        bh, _, _ = UP.hook_representations(net, head, UP.capped_train(data, task, model, fold)[0], model, device)
        spec = UP.helper(model).spectrum(bh, *UP.capped_train(data, task, model, fold)[1:], task, model, fold)
        dims = np.asarray(stored["protected_blocks"], int)
        basis_sha = UP.array_sha(spec["mean"], spec["basis"], spec["scale"], spec["directions"])
        if basis_sha != source["hashes"]["basis_sha256"] or int(spec["rank"]) != int(stored["rank"]):
            raise RuntimeError("frozen canonical basis/rank mismatch")
        canonical_centroids = PW.canonical(train_centroids["h"], spec)
        sx, sy, ss = PW.capped_outer(data, model, task, fold, "source")
        fx, fy, fs = PW.capped_outer(data, model, task, fold, "future")
        ox, oy = np.concatenate((sx, fx)), np.concatenate((sy, fy))
        osub = np.concatenate((ss, fs)).astype(str)
        ose = np.concatenate((np.full(len(sx), int(data["source_session"])),
                              np.full(len(fx), int(data["future_session"])))).astype(int)
        th, tz, _ = UP.hook_representations(net, head, tx, model, device)
        oh, oz, _ = UP.hook_representations(net, head, ox, model, device)
        tq, tz0, tzp, tzc = BR.decompose(th, tz, spec, dims, head)
        oq, oz0, ozp, ozc = BR.decompose(oh, oz, spec, dims, head)
        exact = max(float(np.max(np.abs(tz - (tz0 + tzp + tzc)))),
                    float(np.max(np.abs(oz - (oz0 + ozp + ozc)))))
        raw_basis = UP.raw_base(spec)
        tp = tq[:, dims] @ raw_basis[dims]
        op = oq[:, dims] @ raw_basis[dims]
        representation_error = max(
            float(np.max(np.abs(th - (spec["mean"] + tp + (th - spec["mean"] - tp))))),
            float(np.max(np.abs(oh - (spec["mean"] + op + (oh - spec["mean"] - op))))),
        )
        if exact >= 1e-5:
            raise RuntimeError(f"final P+C logit decomposition error {exact}")
        if representation_error >= 1e-5:
            raise RuntimeError(f"final P+C representation error {representation_error}")
        randoms = source["randoms"]
        stages = []
        coupling_train, coupling_outer = {}, {}
        for stage in runner.names:
            stage_file = RUNTIME / "stage_cache" / model / task / f"fold{fold}" / f"{stage}.json"
            if stage_file.exists():
                row = json.loads(stage_file.read_text(encoding="utf-8"))
                if row.get("protocol_sha256") != protocol_sha or row.get("implementation_sha256") != digest(Path(__file__)):
                    raise RuntimeError(f"stale stage cache {stage}")
                print("STAGE_CACHED", model, task, fold, stage, flush=True)
            else:
                print("STAGE_START", model, task, fold, stage, flush=True)
                row = stage_cell(runner, model, task, fold, stage, train_centroids["shapes"][stage], train_centroids, canonical_centroids,
                                 spec, dims, randoms, tx, ty, ts, tse, ox, oy, osub, ose,
                                 len(sx), protocol_sha)
                write_json(stage_file, row)
                print("STAGE_COMPLETE", model, task, fold, stage, flush=True)
            coupling_train[stage] = np.asarray(row["train_coupling_features"], np.float32)
            coupling_outer[stage] = np.asarray(row["outer_coupling_features"], np.float32)
            stages.append({key: value for key, value in row.items() if key not in
                           ("train_coupling_features", "outer_coupling_features")})
        geom_train = final_geometry(tq, tz0, tzp, tzc, tz, dims)
        geom_outer = final_geometry(oq[len(sx):], oz0, ozp[len(sx):], ozc[len(sx):], oz[len(sx):], dims)
        targets_train = error_targets(ty, tz, tz0 + tzp, tz0 + tzc,
                                      previous_grid_nonrecoverable(ty, tz, tz0, tzp, tzc))
        targets_outer = error_targets(fy, oz[len(sx):], oz0 + ozp[len(sx):], oz0 + ozc[len(sx):],
                                      previous_grid_nonrecoverable(fy, oz[len(sx):], oz0, ozp[len(sx):], ozc[len(sx):]))
        predictability = error_predictability(geom_train, geom_outer, coupling_train, coupling_outer,
                                               targets_train, targets_outer, ts, fs, (model, task, fold))
        write_json(destination, {
            **base, "status": "COMPLETE", **source["hashes"], "protocol_sha256": protocol_sha,
            "implementation_sha256": digest(Path(__file__)), "final_decomposition_max_abs": exact,
            "final_representation_max_abs": representation_error,
            "protected_dims": dims, "stages": stages, "error_predictability": predictability,
        })
        print("CELL_COMPLETE", model, task, fold, flush=True)
    except Exception as error:
        write_json(destination, {**base, "status": "FAIL_CLOSED", "reason": f"{type(error).__name__}: {error}",
                                 "implementation_sha256": digest(Path(__file__))})
        print("CELL_FAIL_CLOSED", model, task, fold, repr(error), flush=True)
        traceback.print_exc()
        raise


def subject_bootstrap(values: dict[str, list[float]], label: tuple[Any, ...]) -> tuple[float, float, float]:
    per_subject = np.asarray([np.mean(v) for _, v in sorted(values.items()) if v], np.float64)
    if len(per_subject) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed(*label, "subject-bootstrap"))
    sampled = per_subject[rng.integers(0, len(per_subject), size=(BOOTSTRAPS, len(per_subject)))].mean(axis=1)
    return float(per_subject.mean()), float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))


def aggregate() -> None:
    expected = (PROTOCOL / "PROVENANCE.sha256").read_text(encoding="utf-8").strip()
    if digest(PROTOCOL / "PROVENANCE.json") != expected:
        raise RuntimeError("protocol lock mismatch")
    cells = []
    for model in MODELS:
        for task in TASKS:
            for fold in FOLDS:
                path = cell_path(model, task, fold)
                if not path.exists():
                    raise RuntimeError(f"cell missing: {model}/{task}/{fold}")
                row = json.loads(path.read_text(encoding="utf-8"))
                if row.get("status") != "COMPLETE" or row.get("protocol_sha256") != expected:
                    raise RuntimeError(f"cell not valid COMPLETE: {model}/{task}/{fold}")
                if row.get("final_heldout_accessed") or row.get("final_decomposition_max_abs", 1.) >= 1e-5:
                    raise RuntimeError(f"heldout/exact decomposition failure: {model}/{task}/{fold}")
                if len(row.get("stages", [])) != (5 if model == "EEGNet" else 7):
                    raise RuntimeError(f"stage count mismatch: {model}/{task}/{fold}")
                for stage in row["stages"]:
                    if stage["reconstruction_max_abs"] >= 1e-5:
                        raise RuntimeError("layer P+C reconstruction failed")
                    if len({r["draw"] for r in stage["random_subject_effects"]}) != RANDOM_DRAWS:
                        raise RuntimeError("100 fixed random partition controls missing")
                    if len({r["draw"] for r in stage["random_surrogate"]}) != SURROGATE_RANDOM_DRAWS:
                        raise RuntimeError("20 random surrogate controls missing")
                cells.append(row)
    tables: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell_row in cells:
        base = {k: cell_row[k] for k in ("model", "task", "fold", "seed")}
        tables["FINAL_PC_DECOMPOSITION_AUDIT"].append({**base, "status": "COMPLETE",
            "max_abs_z_error": cell_row["final_decomposition_max_abs"],
            "max_abs_h_error": cell_row["final_representation_max_abs"],
            "final_heldout_accessed": False, "basis_sha256": cell_row["basis_sha256"]})
        tables["PROVENANCE"].append({**base, **{k: cell_row[k] for k in cell_row if k.endswith("sha256")}})
        tables["PC_ERROR_PREDICTABILITY"].extend({**base, **entry} for entry in cell_row["error_predictability"])
        for stage in cell_row["stages"]:
            prefix = {**base, "layer_name": stage["stage"]}
            tables["LAYER_PC_DECOMPOSITION_AUDIT"].append({**prefix, "module_path": stage["module_path"],
                "activation_shape": stage["shape"], "P_path_rank": stage["P_rank"],
                "mapping_sha256": stage["mapping_sha256"], "max_abs_reconstruction": stage["reconstruction_max_abs"]})
            for entry in stage["main_effects"]:
                tables["PC_MAIN_EFFECTS"].append({**prefix, **{k: entry[k] for k in entry if k.startswith(("P_", "C_")) or k in ("subject_id", "pairs")}})
                tables["PC_LOCAL_CONDITIONAL_SENSITIVITY"].append({**prefix, "subject_id": entry["subject_id"],
                                                                     "D_JP": entry["D_JP"], "D_JC": entry["D_JC"]})
            for entry in stage["interaction"]:
                tables["PC_INTERACTION"].append({**prefix, **entry})
                tables["PC_ERROR_LINKAGE"].append({**prefix, "subject_id": entry["subject_id"],
                    "native_correct": entry["native_correct"],
                    "interaction_norm": entry["interaction_norm"],
                    "interaction_margin_effect": entry["interaction_margin_effect"],
                    "interaction_helps_correct": entry["interaction_helps_correct"],
                    "interaction_contributes_error": entry["interaction_contributes_error"],
                    "interaction_reverses_P": entry["interaction_reverses_P"],
                    "interaction_enables_C_rescue": entry["interaction_enables_C_rescue"]})
            for entry in stage["surrogate"]:
                tables["PC_FUNCTIONAL_SURROGATE"].append({**prefix, **entry})
            for entry in stage["random_surrogate"]:
                tables["PC_RANDOM_SURROGATE_CONTROL"].append({**prefix, **entry})
            for entry in stage["session_stability"]:
                tables["PC_SESSION_STABILITY"].append({**prefix, "partition": "P", **entry})
            for entry in stage["random_session_stability"]:
                tables["PC_SESSION_STABILITY"].append({**prefix, "partition": "random", **entry})
    specificity = []
    for model in MODELS:
        for task in TASKS:
            subset = [c for c in cells if (c["model"], c["task"]) == (model, task)]
            for stage_name in subset[0]["stages"]:
                stage = stage_name["stage"]
                excess_by_subject: dict[str, list[float]] = defaultdict(list)
                p_values, random_draw_values = [], defaultdict(list)
                for cell_row in subset:
                    item = next(s for s in cell_row["stages"] if s["stage"] == stage)
                    random_by_subject = defaultdict(list)
                    for entry in item["random_subject_effects"]:
                        random_by_subject[entry["subject_id"]].append(entry["interaction_norm"])
                        random_draw_values[entry["draw"]].append(entry["interaction_norm"])
                    for entry in item["main_effects"]:
                        owner = entry["subject_id"]
                        if len(random_by_subject[owner]) != RANDOM_DRAWS:
                            raise RuntimeError("subject-level random controls incomplete")
                        excess_by_subject[owner].append(entry["interaction_norm"] - float(np.mean(random_by_subject[owner])))
                        p_values.append(entry["interaction_norm"])
                point, low, high = subject_bootstrap(excess_by_subject, (model, task, stage, "interaction-excess"))
                p_mean = float(np.mean(p_values))
                random_means = np.asarray([np.mean(random_draw_values[j]) for j in range(RANDOM_DRAWS)])
                specificity.append({"model": model, "task": task, "layer_name": stage,
                    "P_interaction_norm": p_mean, "random_interaction_mean": float(random_means.mean()),
                    "P_minus_random": point, "subject_CI_low": low, "subject_CI_high": high,
                    "random_percentile": float(np.mean(random_means <= p_mean)),
                    "empirical_p": float((1 + np.sum(random_means >= p_mean)) / (1 + RANDOM_DRAWS)),
                    "positive_folds": int(sum(np.mean([e["interaction_norm"] for e in next(s for s in c["stages"] if s["stage"] == stage)["main_effects"]]) >
                                             np.mean([e["interaction_norm"] for e in next(s for s in c["stages"] if s["stage"] == stage)["random_subject_effects"]])
                                             for c in subset)),
                    "bootstrap_unit": "biological_subject"})
    tables["PC_RANDOM_SPECIFICITY"] = specificity
    base_predictions = {
        (r["model"], r["task"], r["fold"], r["target"], r["subject_id"]): r
        for r in tables["PC_ERROR_PREDICTABILITY"] if r["family"] == "BASE"
    }
    for row in tables["PC_ERROR_PREDICTABILITY"]:
        baseline = base_predictions[(row["model"], row["task"], row["fold"], row["target"], row["subject_id"])]
        for metric in ("AUROC", "AUPRC", "Brier", "ECE_10bin"):
            value, original = row[metric], baseline[metric]
            row[f"delta_{metric}_over_BASE"] = (value - original) if value is not None and original is not None else None
    surrogate_groups = defaultdict(dict)
    for row in tables["PC_FUNCTIONAL_SURROGATE"]:
        key = tuple(row[k] for k in ("model", "task", "fold", "layer_name", "PCA_C_dim", "subject_id"))
        surrogate_groups[key][row["family"]] = row
    random_groups = defaultdict(dict)
    for row in tables["PC_RANDOM_SURROGATE_CONTROL"]:
        key = tuple(row[k] for k in ("model", "task", "fold", "layer_name", "PCA_C_dim", "subject_id", "draw"))
        random_groups[key][row["family"]] = row
    for key, family_rows in surrogate_groups.items():
        additive = family_rows["additive"]["centered_logit_R2"]
        family_rows["bilinear"]["delta_R2_over_additive"] = family_rows["bilinear"]["centered_logit_R2"] - additive
        family_rows["matched_MLP"]["delta_R2_over_additive"] = family_rows["matched_MLP"]["centered_logit_R2"] - additive
    for key, family_rows in random_groups.items():
        additive = family_rows["additive"]["centered_logit_R2"]
        gain = family_rows["bilinear"]["centered_logit_R2"] - additive
        family_rows["bilinear"]["delta_R2_over_additive"] = gain
        p_gain = surrogate_groups[key[:-1]]["bilinear"]["delta_R2_over_additive"]
        family_rows["bilinear"]["P_delta_R2"] = p_gain
        family_rows["bilinear"]["P_minus_random_delta_R2"] = p_gain - gain
    summary, report = [], [
        "# Frozen Protected/complement coupling audit", "",
        "At the classifier input, P plus complement reconstructs the complete representation. “Not reweight-recoverable” therefore means failure of the restricted scalar intervention family, not missing representation outside P and C.",
        "", "All mapping, pairing, PCA, surrogate fitting, and regularization choices use TRAIN only. Outer-development is evaluation only. Statistical bootstrap unit: biological subject. Final-heldout access: NO.", "",
    ]
    for model in MODELS:
        for task in TASKS:
            specific = [s for s in specificity if (s["model"], s["task"]) == (model, task)]
            peak = max(specific, key=lambda r: r["P_interaction_norm"])
            positive = [s for s in specific if s["subject_CI_low"] > 0 and s["positive_folds"] >= 4]
            surrogate = [r for r in tables["PC_FUNCTIONAL_SURROGATE"] if
                         (r["model"], r["task"], r["PCA_C_dim"]) == (model, task, 16)]
            sur_by = defaultdict(list)
            for row in surrogate:
                sur_by[(row["layer_name"], row["family"])].append(row["centered_logit_R2"])
            bilinear_delta = {stage: float(np.mean(sur_by[(stage, "bilinear")]) - np.mean(sur_by[(stage, "additive")]))
                              for stage in (s["layer_name"] for s in specific)}
            bilinear_peak = max(bilinear_delta, key=bilinear_delta.get)
            pred = [r for r in tables["PC_ERROR_PREDICTABILITY"] if
                    (r["model"], r["task"], r["target"]) == (model, task, "native_error")]
            pred_by = defaultdict(list)
            for row in pred:
                if row["AUROC"] is not None:
                    pred_by[row["family"]].append(row["AUROC"])
            base_auc = float(np.mean(pred_by["BASE"])) if pred_by["BASE"] else float("nan")
            all_auc = float(np.mean(pred_by["ALL_LAYER"])) if pred_by["ALL_LAYER"] else float("nan")
            error_deltas: dict[str, list[float]] = defaultdict(list)
            for row in pred:
                value = row.get("delta_AUROC_over_BASE")
                if row["family"] == "ALL_LAYER" and value is not None:
                    error_deltas[row["subject_id"]].append(value)
            delta_auc, delta_low, delta_high = subject_bootstrap(error_deltas, (model, task, "native-error", "all-layer"))
            peak_random = [r for r in tables["PC_RANDOM_SURROGATE_CONTROL"] if
                           (r["model"], r["task"], r["layer_name"], r["PCA_C_dim"], r["family"]) ==
                           (model, task, bilinear_peak, 16, "bilinear")]
            random_excess = defaultdict(list)
            for row in peak_random:
                random_excess[row["subject_id"]].append(row["P_minus_random_delta_R2"])
            surrogate_excess, surrogate_low, surrogate_high = subject_bootstrap(
                random_excess, (model, task, bilinear_peak, "surrogate-random-excess"))
            mlp_r2 = float(np.mean(sur_by[(bilinear_peak, "matched_MLP")]))
            additive_r2 = float(np.mean(sur_by[(bilinear_peak, "additive")]))
            bilinear_r2 = float(np.mean(sur_by[(bilinear_peak, "bilinear")]))
            summary.append({"model": model, "task": task, "cells_complete": 5,
                            "max_final_logit_error": max(c["final_decomposition_max_abs"] for c in cells if (c["model"], c["task"]) == (model, task)),
                            "max_interaction_layer": peak["layer_name"],
                            "max_interaction_norm": peak["P_interaction_norm"],
                            "specific_positive_layers": len(positive),
                            "max_bilinear_delta_R2_layer": bilinear_peak,
                            "max_bilinear_delta_R2": bilinear_delta[bilinear_peak],
                            "native_error_AUROC_BASE": base_auc,
                            "native_error_AUROC_ALL_LAYER": all_auc,
                            "native_error_delta_AUROC": delta_auc,
                            "native_error_delta_AUROC_CI_low": delta_low,
                            "native_error_delta_AUROC_CI_high": delta_high,
                            "bilinear_peak_additive_R2": additive_r2,
                            "bilinear_peak_bilinear_R2": bilinear_r2,
                            "bilinear_peak_matched_MLP_R2": mlp_r2,
                            "bilinear_P_minus_random_gain": surrogate_excess,
                            "bilinear_P_minus_random_CI_low": surrogate_low,
                            "bilinear_P_minus_random_CI_high": surrogate_high,
                            "final_heldout_accessed": False})
            report += [f"## {model} / {task}", "",
                f"1. Final reconstruction: max centered-logit error {summary[-1]['max_final_logit_error']:.3g}; P+C is complete at classifier input.",
                "2. A NOT_REWEIGHT_RECOVERABLE error does not imply a third representation block; the scalar (beta, alpha) family is restricted.",
                f"3. Layer P and C effects: see PC_MAIN_EFFECTS.csv; peak P/C interaction is {peak['layer_name']}.",
                f"4. P dependence on C: peak interaction norm {peak['P_interaction_norm']:.6g}; context cosine and sign changes are in PC_INTERACTION.csv.",
                "5. C dependence on P: the same interaction vector governs the symmetric conditional contrast, with opposite modulation sign.",
                f"6. Strongest interaction: {peak['layer_name']}.",
                f"7. Random specificity: {len(positive)} layer(s) have subject CI above zero and at least 4/5 positive folds; full random percentiles and empirical p values are in PC_RANDOM_SPECIFICITY.csv.",
                "8. Correct/error direction: PC_ERROR_LINKAGE.csv reports the true-margin sign, correct/error strata, P reversal, and C rescue without inferring benefit from magnitude alone.",
                f"9. Future-session native-error AUROC: geometry BASE {base_auc:.6g}, ALL_LAYER {all_auc:.6g}; paired subject delta {delta_auc:.6g} (95% CI [{delta_low:.6g}, {delta_high:.6g}]). Per-subject and other targets are in PC_ERROR_PREDICTABILITY.csv.",
                f"10. Frozen-logit surrogates at {bilinear_peak}: additive R2 {additive_r2:.6g}, bilinear R2 {bilinear_r2:.6g}, delta {bilinear_delta[bilinear_peak]:.6g}; full layer profiles are in PC_FUNCTIONAL_SURROGATE.csv.",
                f"11. Generic capacity control: matched one-hidden-layer MLP R2 at that layer was {mlp_r2:.6g}; its capacity was matched to the TRAIN-selected bilinear rank.",
                f"12. Random surrogate specificity: P minus the 20 equal-rank random bilinear gains was {surrogate_excess:.6g} (subject 95% CI [{surrogate_low:.6g}, {surrogate_high:.6g}]) at that layer; all random draws appear in PC_RANDOM_SURROGATE_CONTROL.csv.",
                "13. Cross-session stability: matched subject/class intervention-vector comparisons and their random controls are in PC_SESSION_STABILITY.csv.",
                "14. Architecture contrast: compare this section's layer profile and summary with the other architecture; no architecture-specific mechanism is asserted from an uncorrected peak alone.", ""]
    tables["MODEL_TASK_SUMMARY"] = summary
    for name in ("FINAL_PC_DECOMPOSITION_AUDIT", "LAYER_PC_DECOMPOSITION_AUDIT", "PC_MAIN_EFFECTS",
                 "PC_INTERACTION", "PC_LOCAL_CONDITIONAL_SENSITIVITY", "PC_RANDOM_SPECIFICITY",
                 "PC_ERROR_LINKAGE", "PC_ERROR_PREDICTABILITY", "PC_FUNCTIONAL_SURROGATE",
                 "PC_RANDOM_SURROGATE_CONTROL", "PC_SESSION_STABILITY", "MODEL_TASK_SUMMARY"):
        write_csv(OUT / f"{name}.csv", tables[name])
    write_json(OUT / "PROVENANCE.json", {"schema": "PERSIST_EEG_PROTECTED_COMPLEMENT_COUPLING_SEED0_V1",
        "protocol_sha256": expected, "cells_complete": len(cells), "final_heldout_accessed": False,
        "cell_sha256": {f"{c['model']}/{c['task']}/{c['fold']}": digest(cell_path(c["model"], c["task"], c["fold"])) for c in cells}})
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("AGGREGATE_COMPLETE", len(cells), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=("lock", "cell", "aggregate"))
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--fold", type=int, choices=FOLDS)
    args = parser.parse_args()
    if args.mode == "lock":
        lock()
    elif args.mode == "cell":
        if args.model is None or args.task is None or args.fold is None:
            parser.error("cell requires --model, --task and --fold")
        cell(args.model, args.task, args.fold)
    else:
        aggregate()


if __name__ == "__main__":
    main()
