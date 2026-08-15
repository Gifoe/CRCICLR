"""PERSIST-EEG P5.1 nested TRAIN-only tuning and conditional P6 readout audit.

This file intentionally keeps the P5 scientific family frozen.  P5.1 only
completes the predeclared TRAIN-only hyperparameter protocol.  P6 does not
modify the representation: it tests a frozen, scalar Protected-geometry
readout and prespecified controls.

No outer-test samples, embeddings, labels, or predictions are loaded.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score


REPO_ROOT = Path(__file__).resolve().parents[3]
P5_ROOT = REPO_ROOT / "experiments" / "persist_eeg_p5_icg"
P5_OUT = P5_ROOT / "outputs"
OUT = REPO_ROOT / "experiments" / "persist_eeg_p5_1_p6" / "outputs"
MANIFEST = REPO_ROOT / "outputs" / "persist_eeg_stage0" / "manifests" / "openbmi_trials.parquet"
FOLDS = (0, 1, 2)
SEEDS = (0, 1)
VERSIONS = ("V0", "V1", "V2")
TASK = "mi"
TASK_CLASSES = {"mi": 2, "erp": 2, "ssvep": 4}
MAX_EPOCHS = 24
INNER_FOLDS = 5
ALPHA_GRID = (0.0, 0.10, 0.25, 0.50, 1.0, 2.0, 4.0)
RANDOM_DRAWS = 100
IMPLEMENTATION_ID = "p5_1_nested_train_only_v1_p6_readout_v1"


def clean(value: Any) -> Any:
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_seed(*parts: Any) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)


def seed_all(seed: int) -> None:
    import random

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


def import_p5():
    import importlib.util
    import sys

    path = P5_ROOT / "code" / "p5_icg.py"
    spec = importlib.util.spec_from_file_location("persist_p5_icg_frozen", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


P5 = import_p5()


@dataclass(frozen=True)
class Candidate:
    version: str
    stage: str
    lambda_geometry: float
    lambda_drift: float
    learning_rate: float
    bottleneck: int = 8
    max_epochs: int = MAX_EPOCHS
    subjects_per_batch: int = 6
    trials_per_class: int = 4
    gradient_clip: float = 2.0
    weight_decay: float = 1e-3
    early_stopping_patience: int = 5

    @property
    def key(self) -> str:
        return (f"{self.version}__lg{self.lambda_geometry:g}__ld{self.lambda_drift:g}__"
                f"lr{self.learning_rate:g}__b{self.bottleneck}")


@dataclass
class RunData:
    fold: int
    seed: int
    meta: pd.DataFrame
    h: np.ndarray
    q: np.ndarray
    art: Any
    split: dict[str, list[str]]
    train_pos: np.ndarray
    val_pos: np.ndarray
    targets: Any


def subject_inner_folds(subjects: Sequence[str], fold: int, seed: int) -> list[list[str]]:
    ordered = sorted({str(x) for x in subjects}, key=lambda x: int(x))
    rng = np.random.default_rng(stable_seed("p5.1-inner-folds", fold, seed, ordered))
    perm = [ordered[i] for i in rng.permutation(len(ordered))]
    return [perm[i::INNER_FOLDS] for i in range(INNER_FOLDS)]


def subset_positions(meta: pd.DataFrame, subjects: Sequence[str]) -> np.ndarray:
    return np.flatnonzero(meta.subject.astype(str).isin([str(s) for s in subjects]).to_numpy())


def load_run_data(fold: int, seed: int, meta: pd.DataFrame, cache_namespace: str = "main") -> RunData:
    split = P5.load_split(fold)
    cache = P5_OUT / "cache" / f"fold-{fold}" / f"seed-{seed}"
    h_path = cache / "h0.npy"
    if not h_path.exists():
        raise FileNotFoundError(f"P5 frozen h0 cache missing: {h_path}")
    h = np.asarray(np.load(h_path, mmap_mode="r"), dtype=np.float32)
    if h.shape != (len(meta), 128) or not np.isfinite(h).all():
        raise RuntimeError(f"Invalid h0 cache shape/content: {h_path} {h.shape}")
    art = P5.load_artifacts(fold, seed)
    q = P5.q_from_h(h, art)
    train_pos = subset_positions(meta, split["train_subjects"])
    val_pos = subset_positions(meta, split["validation_subjects"])
    target_path = OUT / "cache" / cache_namespace / f"fold-{fold}" / f"seed-{seed}" / "OUTER_GEOMETRY_TARGETS.npz"
    targets = P5.build_geometry_targets(meta, q, train_pos, art, target_path)
    return RunData(fold, seed, meta, h, q, art, split, train_pos, val_pos, targets)


def labels(meta: pd.DataFrame, positions: np.ndarray) -> np.ndarray:
    return np.array(meta.iloc[positions].label.to_numpy(dtype=np.int64), dtype=np.int64, copy=True)


def historical_head(run: RunData):
    from persist_eeg_stage0.models import build_shared_model

    ckpt_path, _, _ = P5.historical_checkpoint(run.fold, run.seed)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_shared_model("eegnet", int(run.meta.n_channels.iloc[0]), 128, TASK_CLASSES)
    model.load_state_dict(checkpoint["model"])
    return model.heads[TASK], ckpt_path


def candidate_weights(candidate: Candidate, art: Any) -> dict[int, float]:
    if candidate.version == "V0":
        return {int(b): 1.0 for b in art.protected_blocks}
    return {int(b): float(art.weights[b]) for b in art.protected_blocks}


def fast_geometry_loss(q_adj: torch.Tensor, batch_idx: np.ndarray, meta: pd.DataFrame,
                       targets: Any, art: Any, weights: Mapping[int, float]) -> torch.Tensor:
    """Tensor-equivalent form of the frozen P5 structured contrast loss.

    StructuredSampler emits rows in subject/session/class/trial order.  The
    original implementation recomputes each group in Python; reshaping that
    exact order produces the same contrasts and targets with much lower
    interpreter overhead.  This is an engineering acceleration only.
    """
    sessions = sorted(meta.session.astype(str).unique().tolist())
    n_sessions = len(sessions)
    subject_values = meta.subject.astype(str).to_numpy()[batch_idx]
    # The first row of each subject group identifies sampler subject order.
    per_subject = n_sessions * 2
    unique_subjects = []
    for value in subject_values:
        if not unique_subjects or unique_subjects[-1] != value:
            if value not in unique_subjects:
                unique_subjects.append(str(value))
    n_subjects = len(unique_subjects)
    denom_group = n_subjects * n_sessions * 2
    if n_subjects == 0 or len(batch_idx) % denom_group != 0:
        raise RuntimeError("Structured batch does not have subject/session/class/trial layout")
    k = len(batch_idx) // denom_group
    arranged = q_adj.reshape(n_subjects, n_sessions, 2, k, q_adj.shape[1])
    contrast = arranged[:, :, 1].mean(dim=2) - arranged[:, :, 0].mean(dim=2)
    values: list[torch.Tensor] = []
    for block in art.protected_blocks:
        weight = float(weights.get(block, 0.0))
        if weight == 0:
            continue
        dims = torch.as_tensor(art.blocks[block], dtype=torch.long, device=q_adj.device)
        current = contrast.index_select(2, dims)
        same_np = np.stack([[targets.same[(s, session, block)] for session in sessions]
                            for s in unique_subjects])
        cross_np = np.stack([[targets.cross[(s, session, block)] for session in sessions]
                             for s in unique_subjects])
        same = torch.as_tensor(same_np, dtype=q_adj.dtype, device=q_adj.device)
        cross = torch.as_tensor(cross_np, dtype=q_adj.dtype, device=q_adj.device)
        valid = (torch.linalg.vector_norm(same, dim=2) >= 1e-8) & (torch.linalg.vector_norm(cross, dim=2) >= 1e-8)
        term = 0.5 * (1.0 - F.cosine_similarity(current, same, dim=2, eps=1e-8))
        term = term + 0.5 * (1.0 - F.cosine_similarity(current, cross, dim=2, eps=1e-8))
        if bool(valid.any()):
            values.append(weight * term[valid].sum())
    if not values:
        return q_adj.sum() * 0.0
    denom = max(float(sum(weights.get(b, 0.0) for b in art.protected_blocks)), 1e-8)
    return torch.stack(values).sum() / denom


def eval_model(model: Any, h: np.ndarray, q: np.ndarray, y: np.ndarray, device: torch.device) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ht = torch.as_tensor(np.asarray(h, dtype=np.float32), device=device)
    qt = torch.as_tensor(np.asarray(q, dtype=np.float32), device=device)
    logits: list[np.ndarray] = []
    qs: list[np.ndarray] = []
    ds: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(ht), 2048):
            out, qa, delta = model(ht[start:start + 2048], qt[start:start + 2048])
            logits.append(out.detach().cpu().numpy())
            qs.append(qa.detach().cpu().numpy())
            ds.append(delta.detach().cpu().numpy())
    lo = np.concatenate(logits)
    qa = np.concatenate(qs)
    delta = np.concatenate(ds)
    ba = float(balanced_accuracy_score(y, lo.argmax(1)))
    return ba, lo, qa, delta


def train_pair(
    run: RunData,
    candidate: Candidate,
    train_pos: np.ndarray,
    eval_pos: np.ndarray | None,
    targets: Any,
    method_epochs: int | None,
    control_epochs: int | None,
    device: torch.device,
    stream_tag: str,
    return_models: bool = False,
) -> dict[str, Any]:
    """Train a matched method/control pair with shared streams.

    If method_epochs/control_epochs are None, both use max_epochs and the
    best inner-held-out checkpoint is selected using the same BA criterion.
    If durations are supplied, no evaluation labels are touched during fit.
    """
    from persist_eeg_stage0.models import build_shared_model

    # Candidate hyperparameters must not be confounded with a different
    # initialisation or sampler stream.  V0/V1/V2 and all candidate settings
    # share the same fold/seed/split stochastic stream whenever shapes permit.
    seed_all(stable_seed("p5.1-init", IMPLEMENTATION_ID, run.fold, run.seed, stream_tag))
    train_meta = run.meta.iloc[train_pos].reset_index(drop=True)
    h_train = np.array(run.h[train_pos], dtype=np.float32, copy=True)
    q_train = np.array(run.q[train_pos], dtype=np.float32, copy=True)
    y_train = labels(run.meta, train_pos)
    has_eval = eval_pos is not None
    if has_eval:
        eval_meta = run.meta.iloc[eval_pos].reset_index(drop=True)
        h_eval = np.array(run.h[eval_pos], dtype=np.float32, copy=True)
        q_eval = np.array(run.q[eval_pos], dtype=np.float32, copy=True)
        y_eval = labels(run.meta, eval_pos)
    else:
        eval_meta = None
        h_eval = q_eval = y_eval = None
    ckpt_path, _, _ = P5.historical_checkpoint(run.fold, run.seed)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    base = build_shared_model("eegnet", int(run.meta.n_channels.iloc[0]), 128, TASK_CLASSES)
    base.load_state_dict(checkpoint["model"])
    head = base.heads[TASK]
    method = P5.ICGModel(head, run.art, candidate.version, targets, candidate.bottleneck).to(device)
    control = P5.ICGModel(head, run.art, candidate.version, targets, candidate.bottleneck).to(device)
    control.load_state_dict(copy.deepcopy(method.state_dict()))
    opt_m = torch.optim.AdamW(method.parameters(), lr=candidate.learning_rate,
                              weight_decay=candidate.weight_decay)
    opt_c = torch.optim.AdamW(control.parameters(), lr=candidate.learning_rate,
                              weight_decay=candidate.weight_decay)
    htr = torch.as_tensor(h_train, dtype=torch.float32, device=device)
    qtr = torch.as_tensor(q_train, dtype=torch.float32, device=device)
    ytr = torch.as_tensor(y_train, dtype=torch.long, device=device)
    sampler = P5.StructuredSampler(train_meta, train_meta.subject.astype(str).unique().tolist(),
                                   subjects_per_batch=candidate.subjects_per_batch,
                                   trials_per_class=candidate.trials_per_class)
    weights = candidate_weights(candidate, run.art)
    curves: list[dict[str, Any]] = []
    best_m = -np.inf
    best_c = -np.inf
    best_m_state: dict[str, Any] | None = None
    best_c_state: dict[str, Any] | None = None
    best_m_epoch = candidate.max_epochs - 1
    best_c_epoch = candidate.max_epochs - 1
    no_improve_m = 0
    no_improve_c = 0
    start = time.time()
    if method_epochs is not None:
        max_loop = max(int(method_epochs), int(control_epochs or method_epochs))
    else:
        max_loop = candidate.max_epochs
    for epoch in range(max_loop):
        method.train(); control.train()
        batches = sampler.batches(epoch, stable_seed("p5.1-sampler", IMPLEMENTATION_ID,
                                                     run.fold, run.seed, stream_tag))
        loss_m_sum = loss_c_sum = geo_sum = drift_sum = 0.0
        for batch in batches:
            idx = torch.as_tensor(batch, dtype=torch.long, device=device)
            if method_epochs is None or epoch < int(method_epochs):
                opt_m.zero_grad(set_to_none=True)
                logits, qa, delta = method(htr.index_select(0, idx), qtr.index_select(0, idx))
                task = F.cross_entropy(logits, ytr.index_select(0, idx))
                geo = fast_geometry_loss(qa, batch, train_meta, targets, run.art, weights)
                drift = P5.drift_loss(delta, run.art)
                total = task + candidate.lambda_geometry * geo + candidate.lambda_drift * drift
                total.backward()
                torch.nn.utils.clip_grad_norm_(method.parameters(), candidate.gradient_clip)
                opt_m.step()
                loss_m_sum += float(total.detach())
                geo_sum += float(geo.detach())
                drift_sum += float(drift.detach())
            if control_epochs is None or epoch < int(control_epochs):
                opt_c.zero_grad(set_to_none=True)
                clogits, cqa, cdelta = control(htr.index_select(0, idx), qtr.index_select(0, idx))
                ctask = F.cross_entropy(clogits, ytr.index_select(0, idx))
                cdrift = P5.drift_loss(cdelta, run.art)
                ctotal = ctask + candidate.lambda_drift * cdrift
                ctotal.backward()
                torch.nn.utils.clip_grad_norm_(control.parameters(), candidate.gradient_clip)
                opt_c.step()
                loss_c_sum += float(ctotal.detach())
        if has_eval:
            ba_m, _, _, _ = eval_model(method, h_eval, q_eval, y_eval, device)
            ba_c, _, _, _ = eval_model(control, h_eval, q_eval, y_eval, device)
            curves.append({"epoch": epoch, "method_BA": ba_m, "control_BA": ba_c,
                           "method_loss": loss_m_sum / max(len(batches), 1),
                           "control_loss": loss_c_sum / max(len(batches), 1),
                           "geometry_loss": geo_sum / max(len(batches), 1),
                           "drift_loss": drift_sum / max(len(batches), 1)})
            if ba_m > best_m + 1e-12:
                best_m = ba_m; best_m_epoch = epoch; best_m_state = copy.deepcopy(method.state_dict())
                no_improve_m = 0
            else:
                no_improve_m += 1
            if ba_c > best_c + 1e-12:
                best_c = ba_c; best_c_epoch = epoch; best_c_state = copy.deepcopy(control.state_dict())
                no_improve_c = 0
            else:
                no_improve_c += 1
            if (method_epochs is None and epoch + 1 >= candidate.early_stopping_patience + 1
                    and no_improve_m >= candidate.early_stopping_patience
                    and no_improve_c >= candidate.early_stopping_patience):
                break
    if has_eval:
        if best_m_state is None or best_c_state is None:
            raise RuntimeError("Inner evaluation produced no checkpoint")
        method.load_state_dict(best_m_state); control.load_state_dict(best_c_state)
        ba_m, logits_m, q_m, delta_m = eval_model(method, h_eval, q_eval, y_eval, device)
        ba_c, logits_c, q_c, delta_c = eval_model(control, h_eval, q_eval, y_eval, device)
    else:
        # The full-TRAIN retrain uses fixed durations selected from inner CV.
        best_m_epoch = int(method_epochs or candidate.max_epochs) - 1
        best_c_epoch = int(control_epochs or candidate.max_epochs) - 1
        ba_m = ba_c = float("nan")
        logits_m = logits_c = q_m = q_c = delta_m = delta_c = None
    result: dict[str, Any] = {
        "method_BA": float(ba_m), "control_BA": float(ba_c),
        "delta_BA": float(ba_m - ba_c) if has_eval else None,
        "method_epoch": int(best_m_epoch), "control_epoch": int(best_c_epoch),
        "curves": curves, "elapsed_seconds": time.time() - start,
    }
    if return_models:
        result.update({"method": method, "control": control, "method_logits": logits_m,
                       "control_logits": logits_c, "method_q": q_m, "control_q": q_c,
                       "method_delta": delta_m, "control_delta": delta_c,
                       "h_train": h_train, "q_train": q_train, "y_train": y_train,
                       "h_eval": h_eval, "q_eval": q_eval, "y_eval": y_eval})
    return result


def candidate_grid(version: str) -> list[Candidate]:
    stage1 = [Candidate(version, "stage1", lg, 0.10, 3e-4) for lg in (0.03, 0.10, 0.30, 1.00)]
    # Stage 2 is populated after Stage 1 ranking.  The function is kept
    # separate so the exact 4+8 protocol is explicit in the output.
    return stage1


def rank_candidate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]):
        return (-float(row["mean_delta_BA"]), -float(row["mean_method_BA"]),
                -int(row["positive_inner_folds"]), float(row["delta_std"]),
                float(row["lambda_geometry"]), float(row["lambda_drift"]),
                float(row["learning_rate"]))
    return sorted(rows, key=key)


def run_candidate_inner(run: RunData, candidate: Candidate, inner_folds: Sequence[Sequence[str]],
                        device: torch.device, run_out: Path) -> dict[str, Any]:
    fold_rows: list[dict[str, Any]] = []
    for inner_idx, held_subjects in enumerate(inner_folds):
        held_pos = subset_positions(run.meta, held_subjects)
        train_subjects = [s for s in run.split["train_subjects"] if str(s) not in set(map(str, held_subjects))]
        train_pos = subset_positions(run.meta, train_subjects)
        inner_target_path = run_out / "INNER_TARGETS" / f"inner-{inner_idx}.npz"
        targets = P5.build_geometry_targets(run.meta, run.q, train_pos, run.art, inner_target_path)
        result = train_pair(run, candidate, train_pos, held_pos, targets, None, None, device,
                            f"inner-{inner_idx}")
        fold_rows.append({"version": candidate.version, "candidate": candidate.key,
                          "stage": candidate.stage, "inner_fold": inner_idx,
                          "held_subjects": list(map(str, held_subjects)),
                          "n_train_subjects": int(len(train_subjects)),
                          "n_held_subjects": int(len(held_subjects)),
                          "method_BA": result["method_BA"], "control_BA": result["control_BA"],
                          "delta_BA": result["delta_BA"],
                          "method_epoch": result["method_epoch"],
                          "control_epoch": result["control_epoch"],
                          "elapsed_seconds": result["elapsed_seconds"]})
    frame = pd.DataFrame(fold_rows)
    row = {"version": candidate.version, "candidate": candidate.key, "stage": candidate.stage,
           "lambda_geometry": candidate.lambda_geometry, "lambda_drift": candidate.lambda_drift,
           "learning_rate": candidate.learning_rate, "bottleneck": candidate.bottleneck,
           "max_epochs": candidate.max_epochs,
           "early_stopping_patience": candidate.early_stopping_patience,
           "mean_method_BA": float(frame.method_BA.mean()),
           "mean_control_BA": float(frame.control_BA.mean()),
           "mean_delta_BA": float(frame.delta_BA.mean()),
           "delta_std": float(frame.delta_BA.std(ddof=0)),
           "positive_inner_folds": int((frame.delta_BA > 0).sum()),
           "median_method_epoch": int(round(float(frame.method_epoch.median()))),
           "median_control_epoch": int(round(float(frame.control_epoch.median()))),
           "median_pair_epoch": int(round(float(np.median(frame[["method_epoch", "control_epoch"]].to_numpy().reshape(-1))))),
           "inner_rows": fold_rows}
    return row


def write_candidate_row(row: dict[str, Any], path: Path) -> None:
    payload = {k: v for k, v in row.items() if k != "inner_rows"}
    write_json(path, {**payload, "inner_rows": row["inner_rows"]})


def select_for_run(run: RunData, version: str, device: torch.device, out: Path) -> dict[str, Any]:
    run_out = out / f"fold-{run.fold}" / f"seed-{run.seed}"
    run_out.mkdir(parents=True, exist_ok=True)
    splits = subject_inner_folds(run.split["train_subjects"], run.fold, run.seed)
    write_json(run_out / "INNER_SUBJECT_FOLDS.json", {"fold": run.fold, "seed": run.seed,
                                                       "folds": splits, "outer_test_used": False})
    candidates: list[dict[str, Any]] = []
    stage1 = candidate_grid(version)
    for i, candidate in enumerate(stage1):
        row = run_candidate_inner(run, candidate, splits, device, run_out / f"candidate-{i:02d}-{candidate.key}")
        row["stage1_rank"] = None
        candidates.append(row)
    stage1_ranked = rank_candidate_rows(candidates)
    for rank, row in enumerate(stage1_ranked, 1):
        row["stage1_rank"] = rank
    retained = stage1_ranked[:2]
    stage2: list[Candidate] = []
    for row in retained:
        lg = float(row["lambda_geometry"])
        for ld in (0.01, 0.10):
            for lr in (1e-4, 3e-4):
                stage2.append(Candidate(version, "stage2", lg, ld, lr))
    for i, candidate in enumerate(stage2, len(stage1)):
        row = run_candidate_inner(run, candidate, splits, device, run_out / f"candidate-{i:02d}-{candidate.key}")
        row["stage1_rank"] = next(int(x["stage1_rank"]) for x in retained if abs(float(x["lambda_geometry"]) - candidate.lambda_geometry) < 1e-12)
        candidates.append(row)
    pd.DataFrame([{k: v for k, v in row.items() if k != "inner_rows"} for row in candidates]).to_csv(
        run_out / "CANDIDATES.csv", index=False)
    pd.DataFrame([inner for row in candidates for inner in row["inner_rows"]]).to_csv(
        run_out / "INNER_CV_RESULTS.csv", index=False)
    ranked2 = rank_candidate_rows([x for x in candidates if x["stage"] == "stage2"])
    selected = ranked2[0]
    selected["selected"] = True
    write_json(run_out / "SELECTED_CONFIG.json", {k: v for k, v in selected.items() if k != "inner_rows"})
    return {"fold": run.fold, "seed": run.seed, "version": version, "candidates": candidates,
            "selected": selected, "inner_folds": splits}


def median_epoch(value: Any) -> int:
    return max(1, min(MAX_EPOCHS, int(round(float(value))) + 1))


def subject_rows(logits_m: np.ndarray, logits_c: np.ndarray, meta: pd.DataFrame) -> list[dict[str, Any]]:
    pm = logits_m.argmax(1); pc = logits_c.argmax(1)
    rows: list[dict[str, Any]] = []
    for subject, group in meta.groupby("subject", sort=True):
        idx = group.index.to_numpy(dtype=np.int64)
        y = meta.label.to_numpy(dtype=np.int64)[idx]
        rows.append({"subject": str(subject), "method_BA": float(balanced_accuracy_score(y, pm[idx])),
                     "control_BA": float(balanced_accuracy_score(y, pc[idx])),
                     "delta_BA": float(balanced_accuracy_score(y, pm[idx]) - balanced_accuracy_score(y, pc[idx]))})
    return rows


def outer_evaluate(run: RunData, selected: dict[str, Any], device: torch.device, out: Path) -> dict[str, Any]:
    cfg = Candidate(selected["version"], "selected", float(selected["lambda_geometry"]),
                    float(selected["lambda_drift"]), float(selected["learning_rate"]),
                    int(selected["bottleneck"]))
    pair_epoch = selected.get("median_pair_epoch")
    if pair_epoch is None:
        pair_epoch = int(round(float(np.median([selected["median_method_epoch"], selected["median_control_epoch"]]))))
    m_epochs = c_epochs = median_epoch(pair_epoch)
    result = train_pair(run, cfg, run.train_pos, None, run.targets, m_epochs, c_epochs, device,
                        "outer-full-train", return_models=True)
    _, hist_ckpt = historical_head(run)
    head, _ = historical_head(run)
    h_val = np.array(run.h[run.val_pos], dtype=np.float32, copy=True)
    q_val = np.array(run.q[run.val_pos], dtype=np.float32, copy=True)
    y_val = labels(run.meta, run.val_pos)
    ba_m, logits_m, q_m, delta_m = eval_model(result["method"], h_val, q_val, y_val, device)
    ba_c, logits_c, q_c, delta_c = eval_model(result["control"], h_val, q_val, y_val, device)
    hist_head = copy.deepcopy(head).to(device).eval()
    with torch.inference_mode():
        hist_logits = hist_head(torch.as_tensor(h_val, dtype=torch.float32, device=device)).cpu().numpy()
    hist_ba = float(balanced_accuracy_score(y_val, hist_logits.argmax(1)))
    val_meta = run.meta.iloc[run.val_pos].reset_index(drop=True)
    train_meta = run.meta.iloc[run.train_pos].reset_index(drop=True)
    geo_m = P5.geometry_diagnostics(q_val, q_m, val_meta, train_meta,
                                    np.asarray(run.q[run.train_pos]), run.art, run.targets)
    geo_c = P5.geometry_diagnostics(q_val, q_c, val_meta, train_meta,
                                    np.asarray(run.q[run.train_pos]), run.art, run.targets)
    drift_m = P5.drift_diagnostics(delta_m, run.art, run.targets)
    drift_c = P5.drift_diagnostics(delta_c, run.art, run.targets)
    int_m = P5.intervention_diagnostics(result["method"], h_val, q_val, y_val, run.art,
                                        stable_seed("p5.1-intervention", run.fold, run.seed, cfg.key), device)
    int_c = P5.intervention_diagnostics(result["control"], h_val, q_val, y_val, run.art,
                                        stable_seed("p5.1-intervention", run.fold, run.seed, cfg.key, "control"), device)
    run_out = out / f"fold-{run.fold}" / f"seed-{run.seed}"
    run_out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": result["method"].state_dict(), "version": cfg.version,
                "fold": run.fold, "seed": run.seed, "outer_test_used": False,
                "method_epochs": m_epochs, "historical_checkpoint": str(hist_ckpt)}, run_out / "best_method.pt")
    torch.save({"model": result["control"].state_dict(), "version": cfg.version,
                "fold": run.fold, "seed": run.seed, "outer_test_used": False,
                "control_epochs": c_epochs, "historical_checkpoint": str(hist_ckpt)}, run_out / "best_control.pt")
    pd.DataFrame(subject_rows(logits_m, logits_c, val_meta)).to_csv(run_out / "SUBJECT_RESULTS.csv", index=False)
    result_out = {"status": "RUN_COMPLETE", "implementation_id": IMPLEMENTATION_ID,
                  "version": cfg.version, "fold": run.fold, "seed": run.seed,
                  "outer_test_used": False, "historical_strict_inductive_BA": hist_ba,
                  "method_strict_inductive_BA": ba_m, "control_strict_inductive_BA": ba_c,
                  "delta_BA": ba_m - ba_c, "method_epochs": m_epochs, "control_epochs": c_epochs,
                  "selected_config": {k: v for k, v in selected.items() if k != "inner_rows"},
                  "geometry_method": geo_m, "geometry_control": geo_c,
                  "geometry_delta": {k: geo_m[k] - geo_c[k] for k in geo_m},
                  "drift_method": drift_m, "drift_control": drift_c,
                  "intervention_method": int_m, "intervention_control": int_c,
                  "n_validation_subjects": int(val_meta.subject.nunique()),
                  "n_train_subjects": int(train_meta.subject.nunique()),
                  "canonical_spectrum_sha256": run.art.spectrum_sha256,
                  "protected_blocks": run.art.protected_blocks,
                  "protected_weights": candidate_weights(cfg, run.art),
                  "checkpoint_paths": {"method": str((run_out / "best_method.pt").relative_to(OUT)).replace("\\", "/"),
                                       "control": str((run_out / "best_control.pt").relative_to(OUT)).replace("\\", "/")}}
    write_json(run_out / "RUN_RESULT.json", result_out)
    return {"summary": result_out, "selected": selected, "run": run,
            "method": result["method"], "control": result["control"],
            "logits_method": logits_m, "logits_control": logits_c,
            "h_val": h_val, "q_val": q_val, "y_val": y_val,
            "h_train": np.asarray(run.h[run.train_pos]), "q_train": np.asarray(run.q[run.train_pos]),
            "y_train": labels(run.meta, run.train_pos)}


def paired_subject_bootstrap(subject_rows_all: Sequence[dict[str, Any]], draws: int = 10000,
                             seed: int = 20260815) -> dict[str, Any]:
    grouped: dict[str, list[float]] = {}
    for row in subject_rows_all:
        grouped.setdefault(str(row["run"]), []).append(float(row["delta_BA"]))
    runs = sorted(grouped)
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        selected = rng.choice(runs, size=len(runs), replace=True)
        values[i] = np.mean([np.mean(rng.choice(grouped[r], size=len(grouped[r]), replace=True)) for r in selected])
    return {"mean": float(values.mean()), "ci95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
            "sign_probability": float(np.mean(values > 0)), "draws": draws,
            "n_runs": len(runs), "n_subject_values": len(subject_rows_all)}


def aggregate_p5(version: str, results: Sequence[dict[str, Any]], out: Path) -> dict[str, Any]:
    summaries = [x["summary"] for x in results]
    rows = []
    subject_all = []
    for s in summaries:
        rows.append({"version": version, "fold": s["fold"], "seed": s["seed"],
                     "status": s["status"], "outer_test_used": False,
                     "historical_strict_inductive_BA": s["historical_strict_inductive_BA"],
                     "method_strict_inductive_BA": s["method_strict_inductive_BA"],
                     "control_strict_inductive_BA": s["control_strict_inductive_BA"],
                     "delta_BA": s["delta_BA"], "method_epochs": s["method_epochs"],
                     "control_epochs": s["control_epochs"]})
        p = out / version / f"fold-{s['fold']}" / f"seed-{s['seed']}" / "SUBJECT_RESULTS.csv"
        sf = pd.read_csv(p)
        for _, r in sf.iterrows():
            subject_all.append({"run": f"fold-{s['fold']}/seed-{s['seed']}", "subject": str(r.subject),
                                "method_BA": float(r.method_BA), "control_BA": float(r.control_BA),
                                "delta_BA": float(r.delta_BA)})
    frame = pd.DataFrame(rows)
    pd.DataFrame([summaries[i] for i in range(len(summaries))]).to_json(out / version / "RUN_SUMMARIES.json", orient="records", indent=2)
    pd.DataFrame(subject_all).to_csv(out / version / "SUBJECT_LEVEL_RESULTS.csv", index=False)
    frame.to_csv(out / version / "STRICT_INDUCTIVE_RESULTS.csv", index=False)
    mean_delta = float(frame.delta_BA.mean())
    positive = int((frame.delta_BA > 0).sum())
    boot = paired_subject_bootstrap(subject_all, seed=stable_seed("p5.1-bootstrap", version))
    geo = {k: float(np.mean([s["geometry_delta"][k] for s in summaries])) for k in summaries[0]["geometry_delta"]}
    random_superiority = float(np.mean([s["intervention_method"]["protected_drop"] - s["intervention_method"]["random_drop_mean"] for s in summaries])) > 0
    # Keep the original P5 gate definition: the hierarchical paired
    # subject/run bootstrap mean, not a newly substituted run-level mean.
    gates = {"mean_delta_ge_0.005": boot["mean"] >= 0.005, "positive_runs_ge_4": positive >= 4,
             "ci_lower_gt_0": boot["ci95"][0] > 0, "catastrophic_runs_le_1": int((frame.delta_BA < -0.005).sum()) <= 1,
             "geometry_support": any(geo.get(k, 0.0) > 0 for k in ("alignment_same", "alignment_cross", "geometry_margin")),
             "random_not_same": random_superiority}
    viable = all(gates.values())
    status = "PERSIST_ICG_P5_1_VIABLE" if viable else "PERSIST_ICG_REPRESENTATION_ONLY" if gates["geometry_support"] else "PERSIST_ICG_OPTIMIZATION_NOT_SUPPORTED"
    report = {"status": status, "version": version, "primary": {"mean_delta_BA": mean_delta,
              "positive_runs": positive, "n_runs": len(summaries), "hierarchical_subject_run_bootstrap": boot,
              "geometry_delta": geo, "random_intervention": {"method_protected_drop": float(np.mean([s["intervention_method"]["protected_drop"] for s in summaries])),
              "method_random_drop": float(np.mean([s["intervention_method"]["random_drop_mean"] for s in summaries]))}},
              "rules": {"viable": gates}, "outer_test_used": False,
              "selection_scope": "TRAIN-only five-fold subject-disjoint nested CV; development validation evaluated once after selection"}
    write_json(out / version / "VERSION_REPORT.json", report)
    (out / version / "VERSION_REPORT.md").write_text(
        f"# PERSIST-ICG P5.1 {version}\n\nStatus: `{status}`\n\n"
        f"Mean Delta_BA: `{mean_delta:.6f}`; positive runs: `{positive}/6`; CI: `{boot['ci95']}`.\n\n"
        "Hyperparameters and training duration were selected using TRAIN-only nested subject CV.\n\n"
        "Outer-test used: `false`.\n", encoding="utf-8")
    return report


def make_not_authorized_v3(out: Path, v2_report: dict[str, Any]) -> dict[str, Any]:
    status = "NOT_AUTHORIZED_FIXED_RULE"
    v3 = out / "V3"
    for name in ("CONFIGS", "TRAIN_LOGS", "RUN_RESULTS"):
        (v3 / name).mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "version": "V3", "authorized": False,
               "reason": "P5.1 V2 did not satisfy the original V3 rule.",
               "evidence": v2_report, "outer_test_used": False}
    write_json(v3 / "VERSION_REPORT.json", payload)
    (v3 / "VERSION_REPORT.md").write_text("# PERSIST-ICG P5.1 V3\n\nStatus: `NOT_AUTHORIZED_FIXED_RULE`\n\nOuter-test used: `false`.\n", encoding="utf-8")
    return payload


def train_score(logits: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Scale the base logit difference without moving its decision boundary.

    The protocol defines S_base = logit_1 - logit_0.  Subtracting the TRAIN
    mean would silently replace the ordinary decoder's zero threshold and
    make BA_base differ from the selected matched control.  A TRAIN-derived
    positive scale makes fusion magnitudes comparable while preserving every
    base prediction exactly.
    """
    raw = np.asarray(logits[:, 1] - logits[:, 0], dtype=np.float64)
    std = max(float(raw.std(ddof=0)), 1e-6)
    return raw / std, 0.0, std


def score_base(logits: np.ndarray, train_mean: float, train_std: float) -> np.ndarray:
    raw = np.asarray(logits[:, 1] - logits[:, 0], dtype=np.float64)
    return (raw - train_mean) / max(train_std, 1e-6)


def pooled_scale(z: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    z = np.asarray(z, dtype=np.float64); y = np.asarray(y, dtype=np.int64)
    m0 = float(z[y == 0].mean()); m1 = float(z[y == 1].mean())
    centered = np.concatenate([z[y == 0] - m0, z[y == 1] - m1])
    sigma = float(centered.std(ddof=0))
    return m0, m1, max(sigma, 1e-6)


def protected_block_scores(q_train: np.ndarray, y_train: np.ndarray, q_eval: np.ndarray,
                           art: Any, targets: Any, weights: Mapping[int, float]) -> tuple[np.ndarray, dict[str, Any]]:
    block_scores_train: list[np.ndarray] = []; block_scores_eval: list[np.ndarray] = []; stats: dict[str, Any] = {}
    for block in art.protected_blocks:
        dims = np.asarray(art.blocks[block], dtype=np.int64)
        g = np.asarray(targets.global_direction[block], dtype=np.float64)
        ztr = q_train[:, dims] @ g; zev = q_eval[:, dims] @ g
        m0, m1, sigma = pooled_scale(ztr, y_train)
        orient = 1.0 if m1 >= m0 else -1.0
        block_scores_train.append(orient * (ztr - (m0 + m1) / 2.0) / sigma)
        block_scores_eval.append(orient * (zev - (m0 + m1) / 2.0) / sigma)
        stats[str(block)] = {"mu0": m0, "mu1": m1, "sigma": sigma, "orientation": orient,
                             "weight": float(weights.get(block, 0.0)), "dims": dims.tolist()}
    denom = max(float(sum(weights.get(b, 0.0) for b in art.protected_blocks)), 1e-8)
    train_score_arr = sum(float(weights.get(b, 0.0)) * x for b, x in zip(art.protected_blocks, block_scores_train)) / denom
    eval_score_arr = sum(float(weights.get(b, 0.0)) * x for b, x in zip(art.protected_blocks, block_scores_eval)) / denom
    return np.asarray(train_score_arr), {"eval": np.asarray(eval_score_arr), "blocks": stats}


def subspace_score(q_train: np.ndarray, y_train: np.ndarray, q_eval: np.ndarray,
                   dims: Sequence[int], standardize_features: bool = False) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    dims_arr = np.asarray(sorted(set(int(x) for x in dims)), dtype=np.int64)
    xtr = np.asarray(q_train[:, dims_arr], dtype=np.float64)
    xev = np.asarray(q_eval[:, dims_arr], dtype=np.float64)
    feature_mean = np.zeros(xtr.shape[1], dtype=np.float64)
    feature_std = np.ones(xtr.shape[1], dtype=np.float64)
    if standardize_features:
        feature_mean = xtr.mean(0)
        feature_std = np.maximum(xtr.std(0, ddof=0), 1e-6)
        xtr = (xtr - feature_mean) / feature_std
        xev = (xev - feature_mean) / feature_std
    contrast = xtr[y_train == 1].mean(0) - xtr[y_train == 0].mean(0)
    norm = float(np.linalg.norm(contrast))
    direction = contrast / norm if norm > 1e-12 else np.ones(len(dims_arr), dtype=np.float64) / max(1, len(dims_arr))
    ztr = xtr @ direction; zev = xev @ direction
    m0, m1, sigma = pooled_scale(ztr, y_train)
    orient = 1.0 if m1 >= m0 else -1.0
    return (orient * (ztr - (m0 + m1) / 2.0) / sigma,
            orient * (zev - (m0 + m1) / 2.0) / sigma,
            {"dims": dims_arr.tolist(), "direction": direction.tolist(), "mu0": m0, "mu1": m1,
             "sigma": sigma, "orientation": orient, "standardize_features": standardize_features,
             "feature_mean": feature_mean.tolist() if standardize_features else None,
             "feature_std": feature_std.tolist() if standardize_features else None})


def persistence_supported_dims(run: RunData) -> tuple[list[int], dict[str, Any]]:
    """Return the run-frozen canonical coordinates with persistence support.

    This control is deliberately different from both the Protected assignment
    (task utility) and the full canonical representation.  Support is read
    from the already-frozen Signed-V3.1 TRAIN-only audit and is never inferred
    from P5.1 development-validation outcomes.
    """
    path = (P5.V31_ROOT / f"fold-{run.fold}" / f"seed-{run.seed}" /
            "SIGNED_UTILITY_V3_1.csv")
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    task_rows = frame[frame.task.astype(str).str.lower() == TASK].copy()
    if len(task_rows) != len(run.art.blocks):
        raise RuntimeError(
            f"Expected one Signed-V3.1 MI row per canonical block, got {len(task_rows)}"
        )
    support: dict[int, bool] = {}
    for _, row in task_rows.iterrows():
        value = row["persistence_supported"]
        supported = value if isinstance(value, (bool, np.bool_)) else str(value).strip().lower() == "true"
        support[int(row["block"])] = bool(supported)
    supported_blocks = sorted(b for b, flag in support.items() if flag)
    dims = sorted({int(d) for b in supported_blocks for d in run.art.blocks[b]})
    if not dims:
        raise RuntimeError("Signed-V3.1 contains no persistence-supported MI coordinates")
    return dims, {
        "source": str(path),
        "supported_blocks": supported_blocks,
        "unsupported_blocks": sorted(b for b, flag in support.items() if not flag),
        "dims": dims,
        "rank": len(dims),
        "selection_data": "frozen Signed-V3.1 TRAIN-only audit",
    }


def choose_alpha(base_train: np.ndarray, geo_train: np.ndarray, y_train: np.ndarray,
                 base_eval: np.ndarray, geo_eval: np.ndarray, y_eval: np.ndarray) -> tuple[float, dict[str, Any]]:
    scores = []
    for alpha in ALPHA_GRID:
        b = balanced_accuracy_score(y_train, (base_train + alpha * geo_train > 0).astype(np.int64))
        scores.append((float(alpha), float(b)))
    best = sorted(scores, key=lambda x: (-x[1], abs(x[0])))[0]
    final = balanced_accuracy_score(y_eval, (base_eval + best[0] * geo_eval > 0).astype(np.int64))
    return best[0], {"grid_train_BA": scores, "selected_train_BA": best[1], "heldout_BA": float(final)}


def complementarity(base: np.ndarray, geo: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    pb = base > 0; pg = geo > 0; yy = np.asarray(y, dtype=np.int64)
    bcorrect = pb == yy; gcorrect = pg == yy
    return {"BA_base": float(balanced_accuracy_score(yy, pb)),
            "BA_geometry_only": float(balanced_accuracy_score(yy, pg)),
            "base_wrong_geometry_right": int(np.sum((~bcorrect) & gcorrect)),
            "base_right_geometry_wrong": int(np.sum(bcorrect & (~gcorrect))),
            "prediction_disagreement_rate": float(np.mean(pb != pg)),
            "oracle_union_BA": float(balanced_accuracy_score(yy, np.where(bcorrect | gcorrect, yy, pb))),
            "oracle_gain_over_base": float(balanced_accuracy_score(yy, np.where(bcorrect | gcorrect, yy, pb)) - balanced_accuracy_score(yy, pb))}


def p6_inner_payload(run: RunData, base_version: str, selected: dict[str, Any],
                     splits: Sequence[Sequence[str]], device: torch.device) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    cfg = Candidate(base_version, "selected", float(selected["lambda_geometry"]), float(selected["lambda_drift"]),
                    float(selected["learning_rate"]), int(selected["bottleneck"]))
    pair_epoch = selected.get("median_pair_epoch")
    if pair_epoch is None:
        pair_epoch = int(round(float(np.median([selected["median_method_epoch"], selected["median_control_epoch"]]))))
    m_epochs = c_epochs = median_epoch(pair_epoch)
    for inner_idx, held in enumerate(splits):
        held_pos = subset_positions(run.meta, held)
        train_subjects = [s for s in run.split["train_subjects"] if str(s) not in set(map(str, held))]
        train_pos = subset_positions(run.meta, train_subjects)
        targets = P5.build_geometry_targets(run.meta, run.q, train_pos, run.art,
                                            OUT / "P6" / f"fold-{run.fold}" / f"seed-{run.seed}" / "inner-targets-{inner_idx}.npz")
        fitted = train_pair(run, cfg, train_pos, held_pos, targets, m_epochs, c_epochs, device,
                            f"p6-alpha-inner-{inner_idx}", return_models=True)
        logits_tr = eval_model(fitted["control"], run.h[train_pos], run.q[train_pos], labels(run.meta, train_pos), device)[1]
        logits_ev = fitted["control_logits"]
        payload.append({"train_q": np.asarray(run.q[train_pos]), "eval_q": np.asarray(run.q[held_pos]),
                        "train_h": np.asarray(run.h[train_pos]), "eval_h": np.asarray(run.h[held_pos]),
                        "train_y": labels(run.meta, train_pos), "eval_y": labels(run.meta, held_pos),
                        "train_logits": logits_tr, "eval_logits": logits_ev, "targets": targets,
                        "held_subjects": list(map(str, held)), "inner_fold": inner_idx})
    return payload


def alpha_cv_protected(payload: Sequence[dict[str, Any]], art: Any, weights: Mapping[int, float]) -> tuple[float, dict[str, Any]]:
    grid_scores = {float(a): [] for a in ALPHA_GRID}
    for item in payload:
        base_tr, bm, bs = train_score(item["train_logits"], item["train_y"])
        base_ev = score_base(item["eval_logits"], bm, bs)
        geo_tr, geo_info = protected_block_scores(item["train_q"], item["train_y"], item["eval_q"], art, item["targets"], weights)
        geo_ev = geo_info["eval"]
        for a in ALPHA_GRID:
            grid_scores[float(a)].append(float(balanced_accuracy_score(item["eval_y"], (base_ev + a * geo_ev > 0).astype(np.int64))))
    aggregate = [(a, float(np.mean(v))) for a, v in grid_scores.items()]
    best = sorted(aggregate, key=lambda x: (-x[1], abs(x[0])))[0]
    return best[0], {"fold_BA": {str(a): v for a, v in grid_scores.items()}, "mean_inner_BA": aggregate, "selected_alpha": best[0]}


def alpha_cv_subspace(payload: Sequence[dict[str, Any]], art: Any, dims: Sequence[int], label: str,
                      feature_space: str = "q", standardize_features: bool = False) -> tuple[float, dict[str, Any]]:
    grid_scores = {float(a): [] for a in ALPHA_GRID}
    train_key = "train_q" if feature_space == "q" else "train_h"
    eval_key = "eval_q" if feature_space == "q" else "eval_h"
    for item in payload:
        base_tr, bm, bs = train_score(item["train_logits"], item["train_y"])
        base_ev = score_base(item["eval_logits"], bm, bs)
        _, geo_ev, _ = subspace_score(item[train_key], item["train_y"], item[eval_key], dims,
                                      standardize_features=standardize_features)
        for a in ALPHA_GRID:
            grid_scores[float(a)].append(float(balanced_accuracy_score(item["eval_y"], (base_ev + a * geo_ev > 0).astype(np.int64))))
    aggregate = [(a, float(np.mean(v))) for a, v in grid_scores.items()]
    best = sorted(aggregate, key=lambda x: (-x[1], abs(x[0])))[0]
    return best[0], {"label": label, "feature_space": feature_space,
                    "standardize_features": standardize_features,
                    "fold_BA": {str(a): v for a, v in grid_scores.items()}, "mean_inner_BA": aggregate,
                    "selected_alpha": best[0]}


def p6_run(run_result: dict[str, Any], base_version: str, device: torch.device, out: Path) -> dict[str, Any]:
    run = run_result["run"]
    selected = run_result["selected"]
    splits = subject_inner_folds(run.split["train_subjects"], run.fold, run.seed)
    payload = p6_inner_payload(run, base_version, selected, splits, device)
    train_q = np.asarray(run.q[run.train_pos]); val_q = np.asarray(run.q[run.val_pos])
    y_train = labels(run.meta, run.train_pos); y_val = labels(run.meta, run.val_pos)
    control = run_result["control"]
    logits_train = eval_model(control, run.h[run.train_pos], train_q, y_train, device)[1]
    logits_val = run_result["logits_control"]
    base_train, base_mean, base_std = train_score(logits_train, y_train)
    base_val = score_base(logits_val, base_mean, base_std)
    art = run.art; targets = run.targets
    protected_weights = {int(b): float(art.weights[b]) for b in art.protected_blocks}
    uniform_weights = {int(b): 1.0 for b in art.protected_blocks}
    shuffle_rng = np.random.default_rng(stable_seed("p6-shuffled-protected-weights", run.fold, run.seed))
    shuffled_values = list(protected_weights.values())
    if len(shuffled_values) > 1:
        order = shuffle_rng.permutation(len(shuffled_values))
        if np.array_equal(order, np.arange(len(shuffled_values))):
            order = np.roll(order, 1)
        shuffled_values = [shuffled_values[int(i)] for i in order]
    shuffled_weights = {int(b): float(shuffled_values[i]) for i, b in enumerate(art.protected_blocks)}
    alpha_prot, alpha_detail = alpha_cv_protected(payload, art, protected_weights)
    alpha_uniform, alpha_uniform_detail = alpha_cv_protected(payload, art, uniform_weights)
    alpha_shuffle, alpha_shuffle_detail = alpha_cv_protected(payload, art, shuffled_weights)
    geo_train, geo_info = protected_block_scores(train_q, y_train, train_q, art, targets, protected_weights)
    geo_val = geo_info["eval"]
    uni_train, uni_info = protected_block_scores(train_q, y_train, train_q, art, targets, uniform_weights)
    uni_val = uni_info["eval"]
    shuf_train, shuf_info = protected_block_scores(train_q, y_train, train_q, art, targets, shuffled_weights)
    shuf_val = shuf_info["eval"]
    # Recompute eval scores with the correct eval q; the first call above uses
    # train q to obtain TRAIN statistics by design.
    _, geo_eval_info = protected_block_scores(train_q, y_train, val_q, art, targets, protected_weights)
    _, uni_eval_info = protected_block_scores(train_q, y_train, val_q, art, targets, uniform_weights)
    _, shuf_eval_info = protected_block_scores(train_q, y_train, val_q, art, targets, shuffled_weights)
    geo_val = geo_eval_info["eval"]; uni_val = uni_eval_info["eval"]; shuf_val = shuf_eval_info["eval"]
    fused = base_val + alpha_prot * geo_val
    fused_uni = base_val + alpha_uniform * uni_val
    fused_shuf = base_val + alpha_shuffle * shuf_val
    # Same-rank random directions and all/full canonical linear scores.
    protected_dims = list(sorted({d for b in art.protected_blocks for d in art.blocks[b]}))
    rng = np.random.default_rng(stable_seed("p6-random-draws", run.fold, run.seed))
    random_rows: list[dict[str, Any]] = []
    for draw in range(RANDOM_DRAWS):
        dims = rng.choice(np.arange(art.q_dim), size=min(len(protected_dims), art.q_dim), replace=False).tolist()
        alpha, detail = alpha_cv_subspace(payload, art, dims, f"random-{draw}",
                                          standardize_features=True)
        _, random_val_geo, stats = subspace_score(train_q, y_train, val_q, dims,
                                                  standardize_features=True)
        random_fused = base_val + alpha * random_val_geo
        random_rows.append({"draw": draw, "alpha": alpha, "fused_BA": float(balanced_accuracy_score(y_val, random_fused > 0)),
                            "base_BA": float(balanced_accuracy_score(y_val, base_val > 0)),
                            "delta_BA": float(balanced_accuracy_score(y_val, random_fused > 0) - balanced_accuracy_score(y_val, base_val > 0)),
                            "dims": stats["dims"]})
    all_dims, persistence_info = persistence_supported_dims(run)
    alpha_all, detail_all = alpha_cv_subspace(payload, art, all_dims, "all_persistence_supported",
                                              standardize_features=True)
    _, all_val_geo, all_stats = subspace_score(train_q, y_train, val_q, all_dims,
                                               standardize_features=True)
    fused_all = base_val + alpha_all * all_val_geo
    # "Full canonical representation" means every frozen q0 coordinate, not
    # the upstream 128-D h0 encoder feature.  Both this and the all-persistence
    # control use one TRAIN-fitted scalar linear direction, so capacity is
    # matched while their coordinate support differs.
    full_dims = list(range(art.q_dim))
    alpha_full, detail_full = alpha_cv_subspace(payload, art, full_dims, "full_canonical_linear",
                                                feature_space="q", standardize_features=True)
    _, full_val_geo, full_stats = subspace_score(train_q, y_train, val_q, full_dims,
                                                 standardize_features=True)
    fused_full = base_val + alpha_full * full_val_geo
    run_out = out / f"fold-{run.fold}" / f"seed-{run.seed}"; run_out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(random_rows).to_csv(run_out / "RANDOM_100_DRAWS.csv", index=False)
    base_comp = complementarity(base_val, geo_val, y_val)
    def record(name: str, score: np.ndarray, alpha: float, comp: dict[str, Any]) -> dict[str, Any]:
        ba = float(balanced_accuracy_score(y_val, score > 0))
        return {"name": name, "alpha": float(alpha), "base_BA": float(balanced_accuracy_score(y_val, base_val > 0)),
                "fused_BA": ba, "delta_BA": ba - float(balanced_accuracy_score(y_val, base_val > 0)),
                "complementarity": comp}
    records = [record("protected_intervention_weighted", fused, alpha_prot, base_comp),
               record("protected_uniform", fused_uni, alpha_uniform, complementarity(base_val, uni_val, y_val)),
               record("protected_shuffled_weights", fused_shuf, alpha_shuffle, complementarity(base_val, shuf_val, y_val)),
               record("all_persistence", fused_all, alpha_all, complementarity(base_val, all_val_geo, y_val)),
               record("full_canonical", fused_full, alpha_full, complementarity(base_val, full_val_geo, y_val))]
    pd.DataFrame(records).to_json(run_out / "READOUT_VARIANTS.json", orient="records", indent=2)
    sf = pd.DataFrame({"subject": run.meta.iloc[run.val_pos].subject.astype(str).to_numpy(),
                       "y": y_val, "base_score": base_val, "protected_score": geo_val, "fused_score": fused})
    sf.to_csv(run_out / "SUBJECT_SCORES.csv", index=False)
    result = {"status": "RUN_COMPLETE", "implementation_id": IMPLEMENTATION_ID,
              "fold": run.fold, "seed": run.seed, "outer_test_used": False, "base_version": base_version,
              "base_BA": float(balanced_accuracy_score(y_val, base_val > 0)),
              "historical_BA": float(run_result["summary"]["historical_strict_inductive_BA"]),
              "protected": records[0], "uniform": records[1], "shuffled": records[2],
              "all_persistence": records[3], "full_canonical": records[4],
              "random_draw_mean_BA": float(np.mean([x["fused_BA"] for x in random_rows])),
              "random_draw_mean_delta_BA": float(np.mean([x["delta_BA"] for x in random_rows])),
              "random_draw_positive_fraction": float(np.mean([x["delta_BA"] > 0 for x in random_rows])),
              "random_draws": RANDOM_DRAWS,
              "alpha_cv": {"protected": alpha_detail, "uniform": alpha_uniform_detail, "shuffled": alpha_shuffle_detail,
                           "all_persistence": detail_all, "full_canonical": detail_full},
              "control_geometry": {"all_persistence_supported": persistence_info,
                                   "full_canonical_rank": len(full_dims),
                                   "protected_rank": len(protected_dims),
                                   "linear_capacity": "one frozen TRAIN-fitted scalar direction per control"},
              "protected_score_stats": geo_eval_info["blocks"],
              "outer_train_subjects": int(run.meta.iloc[run.train_pos].subject.nunique()),
              "validation_subjects": int(run.meta.iloc[run.val_pos].subject.nunique())}
    write_json(run_out / "RUN_RESULT.json", result)
    return result


def aggregate_p6(p6_results: Sequence[dict[str, Any]], out: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for r in p6_results:
        rows.append({"fold": r["fold"], "seed": r["seed"], "base_version": r["base_version"],
                     "base_BA": r["base_BA"], "protected_fused_BA": r["protected"]["fused_BA"],
                     "protected_delta_BA": r["protected"]["delta_BA"],
                     "random_mean_delta_BA": r["random_draw_mean_delta_BA"],
                     "uniform_delta_BA": r["uniform"]["delta_BA"],
                     "shuffled_delta_BA": r["shuffled"]["delta_BA"],
                     "all_persistence_delta_BA": r["all_persistence"]["delta_BA"],
                     "full_canonical_delta_BA": r["full_canonical"]["delta_BA"]})
    frame = pd.DataFrame(rows); frame.to_csv(out / "P6_READOUT_RESULTS.csv", index=False)
    # Reconstruct subject BA deltas directly to avoid treating 0/1 correctness
    # differences as balanced accuracy; use grouped predictions and labels.
    subj: list[dict[str, Any]] = []
    for r in p6_results:
        sf = pd.read_csv(out / f"fold-{r['fold']}" / f"seed-{r['seed']}" / "SUBJECT_SCORES.csv")
        for subject, g in sf.groupby("subject"):
            y = g.y.to_numpy(dtype=np.int64); base = (g.base_score.to_numpy() > 0).astype(np.int64); fused = (g.fused_score.to_numpy() > 0).astype(np.int64)
            subj.append({"run": f"fold-{r['fold']}/seed-{r['seed']}", "subject": str(subject),
                         "base_BA": float(balanced_accuracy_score(y, base)),
                         "fused_BA": float(balanced_accuracy_score(y, fused)),
                         "delta_BA": float(balanced_accuracy_score(y, fused) - balanced_accuracy_score(y, base))})
    pd.DataFrame(subj).to_csv(out / "P6_SUBJECT_LEVEL_RESULTS.csv", index=False)
    boot = paired_subject_bootstrap(subj, seed=stable_seed("p6-bootstrap"))
    mean_delta = float(frame.protected_delta_BA.mean()); positive = int((frame.protected_delta_BA > 0).sum())
    random_beats = int((frame.protected_delta_BA.to_numpy() > frame.random_mean_delta_BA.to_numpy()).sum())
    control_means = {
        "same_rank_random": float(frame.random_mean_delta_BA.mean()),
        "protected_uniform": float(frame.uniform_delta_BA.mean()),
        "shuffled_weights": float(frame.shuffled_delta_BA.mean()),
        "all_persistence_supported": float(frame.all_persistence_delta_BA.mean()),
        "full_canonical": float(frame.full_canonical_delta_BA.mean()),
    }
    beats_relevant = {name: bool(mean_delta > value) for name, value in control_means.items()}
    comp = [r["protected"]["complementarity"] for r in p6_results]
    complementary = bool(np.mean([x["base_wrong_geometry_right"] for x in comp]) > 0 and
                         np.mean([x["prediction_disagreement_rate"] for x in comp]) > 0)
    gates = {"mean_delta_ge_0.005": mean_delta >= 0.005, "positive_runs_ge_4": positive >= 4,
             "ci_lower_gt_0": boot["ci95"][0] > 0, "catastrophic_runs_le_1": int((frame.protected_delta_BA < -0.005).sum()) <= 1,
             "protected_beats_random": bool(beats_relevant["same_rank_random"] and random_beats >= 4),
             "protected_beats_relevant_controls": all(beats_relevant.values()),
             "genuine_complementarity": complementary,
             "no_target_centering_or_adaptation": True}
    strong = mean_delta >= 0.010 and positive >= 5 and boot["ci95"][0] > 0
    status = "PERSIST_GEOMETRY_READOUT_STRONG" if all(gates.values()) and strong else "GO_PERSIST_GEOMETRY_READOUT" if all(gates.values()) else "PERSISTENCE_GEOMETRY_HAS_NO_DECISION_FUSION_HEADROOM"
    report = {"status": status, "primary": {"mean_protected_fusion_delta_BA": mean_delta,
              "positive_runs": positive, "n_runs": len(p6_results), "hierarchical_subject_run_bootstrap": boot,
              "random_mean_delta_BA": float(frame.random_mean_delta_BA.mean()),
              "protected_beats_random_runs": random_beats,
              "control_mean_delta_BA": control_means,
              "protected_beats_control": beats_relevant,
              "mean_complementarity": {k: float(np.mean([x[k] for x in comp])) for k in comp[0]}},
              "gates": gates, "outer_test_used": False,
              "interpretation": "Frozen Protected geometry was tested as a scalar readout; no representation alignment or target adaptation was used."}
    write_json(out / "P6_FINAL_REPORT.json", report)
    return report


def write_p5_final(reports: Mapping[str, Any], v3: dict[str, Any], out: Path) -> dict[str, Any]:
    any_viable = any(r["status"] == "PERSIST_ICG_P5_1_VIABLE" for r in reports.values())
    status = "PERSIST_ICG_P5_1_VIABLE" if any_viable else "PERSIST_ICG_REPRESENTATION_ONLY" if any(any(float(v) > 0 for v in r["primary"]["geometry_delta"].values()) for r in reports.values()) else "PERSIST_ICG_OPTIMIZATION_NOT_SUPPORTED"
    final = {"status": status, "implementation_id": IMPLEMENTATION_ID, "versions": reports,
             "v3": v3, "outer_test_used": False,
             "p5_1_protocol": "TRAIN-only 5-fold subject-disjoint nested CV; development validation evaluated once after frozen selection",
             "p6_authorized": not any_viable}
    write_json(out / "P5_1_FINAL_REPORT.json", final)
    return final


def verify_inputs() -> None:
    if not MANIFEST.exists() or not (P5_OUT / "protocol" / "P5_INPUT_VERIFICATION.json").exists():
        raise FileNotFoundError("P5 frozen manifest/input verification is missing")
    meta = P5.load_mi_manifest()
    if len(meta) != 10800 or meta.subject.nunique() != 54:
        raise RuntimeError("Unexpected MI manifest")
    write_json(OUT / "protocol" / "P5_1_INPUT_VERIFICATION.json", {"manifest": str(MANIFEST),
        "manifest_sha256": sha256(MANIFEST), "n_mi_rows": len(meta), "n_subjects": int(meta.subject.nunique()),
        "folds": list(FOLDS), "seeds": list(SEEDS), "inner_folds": INNER_FOLDS, "outer_test_used": False,
        "p5_implementation_id": P5.IMPLEMENTATION_ID, "implementation_id": IMPLEMENTATION_ID})
    write_json(OUT / "protocol" / "P5_1_P6_PROTOCOL.json", {
        "implementation_id": IMPLEMENTATION_ID, "primary_task": "MI", "outer_test_used": False,
        "p5_1": {"versions": list(VERSIONS), "inner_subject_folds": INNER_FOLDS,
                  "stage1_lambda_geometry": [0.03, 0.10, 0.30, 1.00],
                  "stage2_lambda_drift": [0.01, 0.10], "stage2_learning_rate": [1e-4, 3e-4],
                  "bottleneck": 8, "max_serious_configurations": 12,
                  "selection_data": "outer TRAIN subjects only",
                  "development_validation": "evaluated once after configuration and duration freeze"},
        "p6": {"alpha_grid": list(ALPHA_GRID), "random_same_rank_draws": RANDOM_DRAWS,
               "representation_modified": False, "target_centering_or_adaptation": False,
               "controls": ["same_rank_random", "all_persistence_supported", "full_canonical_linear",
                            "shuffled_protected_weights", "protected_uniform"]}})


def load_selected_records() -> dict[str, list[dict[str, Any]]]:
    records: dict[str, list[dict[str, Any]]] = {v: [] for v in VERSIONS}
    for version in VERSIONS:
        for fold in FOLDS:
            for seed in SEEDS:
                path = OUT / version / f"fold-{fold}" / f"seed-{seed}" / "SELECTED_CONFIG.json"
                if not path.exists():
                    raise FileNotFoundError(path)
                records[version].append({"fold": fold, "seed": seed,
                                         "selected": json.loads(path.read_text(encoding="utf-8"))})
    return records


def select_p6_base(selected_by_version: Mapping[str, Sequence[dict[str, Any]]]) -> str:
    base_scores: dict[str, float] = {}
    for version in VERSIONS:
        vals = [float(x["selected"]["mean_control_BA"]) for x in selected_by_version[version]]
        base_scores[version] = float(np.mean(vals))
    base_version = sorted(VERSIONS, key=lambda v: (-base_scores[v], VERSIONS.index(v)))[0]
    write_json(OUT / "protocol" / "P6_BASE_VERSION_SELECTION.json", {"base_version": base_version,
               "rule": "highest mean selected inner-CV control BA; simpler version breaks ties",
               "inner_control_BA": base_scores, "outer_test_used": False})
    return base_version


def finalize_p5_workers() -> dict[str, Any]:
    reports: dict[str, Any] = {}
    global_selected: list[pd.DataFrame] = []
    global_candidates: list[pd.DataFrame] = []
    global_inner: list[pd.DataFrame] = []
    for version in VERSIONS:
        version_out = OUT / version
        selected_rows: list[dict[str, Any]] = []
        candidate_frames: list[pd.DataFrame] = []
        inner_frames: list[pd.DataFrame] = []
        aggregate_rows: list[dict[str, Any]] = []
        for fold in FOLDS:
            for seed in SEEDS:
                run_out = version_out / f"fold-{fold}" / f"seed-{seed}"
                required = [run_out / "WORKER_COMPLETE.json", run_out / "OUTER_WORKER_COMPLETE.json",
                            run_out / "SELECTED_CONFIG.json",
                            run_out / "CANDIDATES.csv", run_out / "INNER_CV_RESULTS.csv",
                            run_out / "RUN_RESULT.json", run_out / "SUBJECT_RESULTS.csv"]
                for path in required:
                    if not path.exists():
                        raise FileNotFoundError(path)
                selected = json.loads((run_out / "SELECTED_CONFIG.json").read_text(encoding="utf-8"))
                selected.setdefault("max_epochs", MAX_EPOCHS)
                selected.setdefault("early_stopping_patience", 5)
                cf = pd.read_csv(run_out / "CANDIDATES.csv"); cf["fold"] = fold; cf["seed"] = seed
                if "max_epochs" not in cf:
                    cf["max_epochs"] = MAX_EPOCHS
                if "early_stopping_patience" not in cf:
                    cf["early_stopping_patience"] = 5
                inf = pd.read_csv(run_out / "INNER_CV_RESULTS.csv"); inf["fold"] = fold; inf["seed"] = seed
                chosen_inner = inf[(inf.candidate.astype(str) == str(selected["candidate"])) &
                                   (inf.stage.astype(str) == str(selected["stage"]))]
                if len(chosen_inner) != INNER_FOLDS:
                    raise RuntimeError(f"Expected {INNER_FOLDS} selected inner rows, got {len(chosen_inner)}")
                selected["median_pair_epoch"] = int(round(float(np.median(
                    chosen_inner[["method_epoch", "control_epoch"]].to_numpy().reshape(-1)))))
                write_json(run_out / "SELECTED_CONFIG.json", selected)
                selected_rows.append({**selected, "fold": fold, "seed": seed})
                candidate_frames.append(cf); inner_frames.append(inf)
                aggregate_rows.append({"summary": json.loads((run_out / "RUN_RESULT.json").read_text(encoding="utf-8"))})
        selected_frame = pd.DataFrame(selected_rows)
        candidate_frame = pd.concat(candidate_frames, ignore_index=True)
        inner_frame = pd.concat(inner_frames, ignore_index=True)
        selected_frame.to_csv(version_out / "P5_1_SELECTED_CONFIGS.csv", index=False)
        candidate_frame.to_csv(version_out / "P5_1_HPARAM_CANDIDATES.csv", index=False)
        inner_frame.to_csv(version_out / "P5_1_INNER_CV_RESULTS.csv", index=False)
        global_selected.append(selected_frame); global_candidates.append(candidate_frame); global_inner.append(inner_frame)
        reports[version] = aggregate_p5(version, aggregate_rows, OUT)
    pd.concat(global_selected, ignore_index=True).to_csv(OUT / "P5_1_SELECTED_CONFIGS.csv", index=False)
    pd.concat(global_candidates, ignore_index=True).to_csv(OUT / "P5_1_HPARAM_CANDIDATES.csv", index=False)
    pd.concat(global_inner, ignore_index=True).to_csv(OUT / "P5_1_INNER_CV_RESULTS.csv", index=False)
    v2 = reports["V2"]
    authorized = bool(v2["primary"]["mean_delta_BA"] > 0 and v2["primary"]["positive_runs"] >= 4 and
                      any(v > 0 for v in v2["primary"]["geometry_delta"].values()) and
                      v2["status"] != "PERSIST_ICG_P5_1_VIABLE")
    v3: dict[str, Any] = {"authorized": authorized,
                          "rule": "V2 mean > 0, >=4/6 positive, active geometry, and not STRONG",
                          "outer_test_used": False}
    if authorized:
        write_json(OUT / "protocol" / "V3_PROGRESSION_DECISION.json", v3)
        raise RuntimeError("P5.1 authorized V3; refusing to skip it or proceed to P6")
    v3 = make_not_authorized_v3(OUT, v2)
    write_json(OUT / "protocol" / "V3_PROGRESSION_DECISION.json", v3)
    return write_p5_final(reports, v3, OUT)


def finalize_p6_workers() -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for fold in FOLDS:
        for seed in SEEDS:
            path = OUT / "P6" / f"fold-{fold}" / f"seed-{seed}" / "RUN_RESULT.json"
            if not path.exists():
                raise FileNotFoundError(path)
            results.append(json.loads(path.read_text(encoding="utf-8")))
    report = aggregate_p6(results, OUT / "P6")
    p5_final = json.loads((OUT / "P5_1_FINAL_REPORT.json").read_text(encoding="utf-8"))
    p6_frame = pd.read_csv(OUT / "P6" / "P6_READOUT_RESULTS.csv")
    best_p5 = max(p5_final["versions"].items(), key=lambda item: float(item[1]["primary"]["mean_delta_BA"]))
    control_means = {"same_rank_random": float(p6_frame.random_mean_delta_BA.mean()),
                     "protected_uniform": float(p6_frame.uniform_delta_BA.mean()),
                     "shuffled_weights": float(p6_frame.shuffled_delta_BA.mean()),
                     "all_persistence": float(p6_frame.all_persistence_delta_BA.mean()),
                     "full_canonical": float(p6_frame.full_canonical_delta_BA.mean())}
    protected_mean = float(report["primary"]["mean_protected_fusion_delta_BA"])
    answers = {
        "1_train_only_tuning_rescued_icg": p5_final["status"] == "PERSIST_ICG_P5_1_VIABLE",
        "1_best_p5_1_version": best_p5[0],
        "1_best_p5_1_mean_delta_BA": float(best_p5[1]["primary"]["mean_delta_BA"]),
        "2_v3_authorized": bool(p5_final.get("v3", {}).get("authorized", False)),
        "3_geometry_alignment_improved_decoding": any(v["status"] == "PERSIST_ICG_P5_1_VIABLE" for v in p5_final["versions"].values()),
        "4_protected_geometry_complementary": bool(report["gates"]["genuine_complementarity"]),
        "5_train_only_fusion_improved_BA": bool(report["gates"]["mean_delta_ge_0.005"] and report["gates"]["positive_runs_ge_4"] and report["gates"]["ci_lower_gt_0"]),
        "5_mean_fusion_delta_BA": float(report["primary"]["mean_protected_fusion_delta_BA"]),
        "6_control_mean_deltas": control_means,
        "6_controls_explain_gain": any(value >= protected_mean - 1e-12 for value in control_means.values()),
        "7_terminal_label": report["status"],
    }
    final = {"status": report["status"], "p5": p5_final, "p6": report,
             "answers": answers, "outer_test_used": False, "implementation_id": IMPLEMENTATION_ID}
    write_json(OUT / "P5_1_P6_FINAL_REPORT.json", final)
    control_lines = "\n".join(f"- {k}: mean Delta_BA `{v:.6f}`" for k, v in answers["6_control_mean_deltas"].items())
    (OUT / "P5_1_P6_FINAL_REPORT.md").write_text(
        f"# PERSIST-EEG P5.1 + P6\n\nStatus: `{report['status']}`\n\n"
        "## Required answers\n\n"
        f"1. TRAIN-only tuning rescued ICG: `{answers['1_train_only_tuning_rescued_icg']}`. Best version: `{answers['1_best_p5_1_version']}`, mean Delta_BA `{answers['1_best_p5_1_mean_delta_BA']:.6f}`.\n\n"
        f"2. V3 authorized: `{answers['2_v3_authorized']}`.\n\n"
        f"3. Stronger geometry alignment improved decoding under the frozen viability gates: `{answers['3_geometry_alignment_improved_decoding']}`.\n\n"
        f"4. Protected geometry showed complementary errors: `{answers['4_protected_geometry_complementary']}`.\n\n"
        f"5. TRAIN-only calibrated Protected fusion passed the strict-inductive gain requirements: `{answers['5_train_only_fusion_improved_BA']}`. Mean Delta_BA `{answers['5_mean_fusion_delta_BA']:.6f}`; positive runs `{report['primary']['positive_runs']}/6`; CI `{report['primary']['hierarchical_subject_run_bootstrap']['ci95']}`.\n\n"
        "6. Control mean effects:\n\n" + control_lines + "\n\n"
        f"At least one mandatory control matches or exceeds the Protected mean gain: `{answers['6_controls_explain_gain']}`.\n\n"
        f"7. Terminal label: `{answers['7_terminal_label']}`.\n\n"
        "Outer-test used: `false`.\n", encoding="utf-8")
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("verify", "smoke", "worker_p5", "worker_outer", "finalize_p5",
                                             "worker_p6", "finalize_p6", "p5_1", "p6", "all"),
                        default="all")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--version", choices=VERSIONS)
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "protocol").mkdir(parents=True, exist_ok=True)
    if args.phase not in ("worker_p5", "worker_outer", "worker_p6"):
        verify_inputs()
    if args.phase == "verify":
        return
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This protocol runner requires the existing GPU environment")
    meta = P5.load_mi_manifest()
    if args.phase == "worker_p5":
        if args.version is None or args.fold is None or args.seed is None:
            raise ValueError("worker_p5 requires --version, --fold and --seed")
        run = load_run_data(args.fold, args.seed, meta, args.version)
        version_out = OUT / args.version
        record = select_for_run(run, args.version, device, version_out)
        outer_evaluate(run, record["selected"], device, version_out)
        write_json(version_out / f"fold-{args.fold}" / f"seed-{args.seed}" / "WORKER_COMPLETE.json",
                   {"status": "COMPLETE", "version": args.version, "fold": args.fold,
                    "seed": args.seed, "outer_test_used": False,
                    "implementation_id": IMPLEMENTATION_ID})
        return
    if args.phase == "worker_outer":
        if args.version is None or args.fold is None or args.seed is None:
            raise ValueError("worker_outer requires --version, --fold and --seed")
        run_out = OUT / args.version / f"fold-{args.fold}" / f"seed-{args.seed}"
        selected = json.loads((run_out / "SELECTED_CONFIG.json").read_text(encoding="utf-8"))
        inner = pd.read_csv(run_out / "INNER_CV_RESULTS.csv")
        chosen = inner[(inner.candidate.astype(str) == str(selected["candidate"])) &
                       (inner.stage.astype(str) == str(selected["stage"]))]
        if len(chosen) != INNER_FOLDS:
            raise RuntimeError(f"Expected {INNER_FOLDS} selected inner rows, got {len(chosen)}")
        selected["median_pair_epoch"] = int(round(float(np.median(
            chosen[["method_epoch", "control_epoch"]].to_numpy().reshape(-1)))))
        selected.setdefault("max_epochs", MAX_EPOCHS)
        selected.setdefault("early_stopping_patience", 5)
        write_json(run_out / "SELECTED_CONFIG.json", selected)
        run = load_run_data(args.fold, args.seed, meta, args.version + "-outer")
        outer_evaluate(run, selected, device, OUT / args.version)
        write_json(run_out / "OUTER_WORKER_COMPLETE.json",
                   {"status": "COMPLETE", "version": args.version, "fold": args.fold,
                    "seed": args.seed, "pair_training_duration": median_epoch(selected["median_pair_epoch"]),
                    "outer_test_used": False, "implementation_id": IMPLEMENTATION_ID})
        return
    if args.phase == "finalize_p5":
        final = finalize_p5_workers()
        if final["status"] != "PERSIST_ICG_P5_1_VIABLE":
            select_p6_base(load_selected_records())
        return
    if args.phase == "worker_p6":
        if args.fold is None or args.seed is None:
            raise ValueError("worker_p6 requires --fold and --seed")
        selection_info = json.loads((OUT / "protocol" / "P6_BASE_VERSION_SELECTION.json").read_text(encoding="utf-8"))
        base_version = str(selection_info["base_version"])
        selected = json.loads((OUT / base_version / f"fold-{args.fold}" / f"seed-{args.seed}" / "SELECTED_CONFIG.json").read_text(encoding="utf-8"))
        run = load_run_data(args.fold, args.seed, meta, "P6")
        rr = outer_evaluate(run, selected, device, OUT / base_version)
        p6_run(rr, base_version, device, OUT / "P6")
        write_json(OUT / "P6" / f"fold-{args.fold}" / f"seed-{args.seed}" / "WORKER_COMPLETE.json",
                   {"status": "COMPLETE", "fold": args.fold, "seed": args.seed,
                    "base_version": base_version, "outer_test_used": False,
                    "implementation_id": IMPLEMENTATION_ID})
        return
    if args.phase == "finalize_p6":
        finalize_p6_workers()
        return
    run_data: dict[tuple[int, int], RunData] = {}
    for fold in FOLDS:
        for seed in SEEDS:
            run_data[(fold, seed)] = load_run_data(fold, seed, meta, "sequential")
    if args.phase == "smoke":
        run = run_data[(0, 0)]
        held = subject_inner_folds(run.split["train_subjects"], 0, 0)[0]
        held_pos = subset_positions(run.meta, held)
        train_subjects = [s for s in run.split["train_subjects"] if str(s) not in set(map(str, held))]
        train_pos = subset_positions(run.meta, train_subjects)
        targets = P5.build_geometry_targets(run.meta, run.q, train_pos, run.art,
                                            OUT / "smoke" / "INNER_TARGETS.npz")
        candidate = Candidate("V0", "smoke", 0.03, 0.10, 3e-4, max_epochs=2)
        smoke_meta = run.meta.iloc[train_pos].reset_index(drop=True)
        sampler = P5.StructuredSampler(smoke_meta, train_subjects,
                                       subjects_per_batch=candidate.subjects_per_batch,
                                       trials_per_class=candidate.trials_per_class)
        batch = sampler.batches(0, stable_seed("smoke-equivalence"))[0]
        q_batch = torch.as_tensor(np.asarray(run.q[train_pos][batch]), dtype=torch.float32, device=device)
        smoke_weights = candidate_weights(candidate, run.art)
        old_loss = P5.geometry_loss(q_batch, batch, smoke_meta, targets, run.art, smoke_weights)
        new_loss = fast_geometry_loss(q_batch, batch, smoke_meta, targets, run.art, smoke_weights)
        loss_difference = float(abs(old_loss.detach() - new_loss.detach()).cpu())
        if loss_difference > 1e-5:
            raise RuntimeError(f"Fast geometry loss is not equivalent: {old_loss} vs {new_loss}")
        result = train_pair(run, candidate, train_pos, held_pos, targets, None, None, device,
                            "smoke")
        write_json(OUT / "smoke" / "SMOKE_RESULT.json", {**{k: v for k, v in result.items() if k != "curves"},
                   "old_geometry_loss": float(old_loss.detach().cpu()),
                   "fast_geometry_loss": float(new_loss.detach().cpu()),
                   "absolute_difference": loss_difference})
        return
    p5_reports: dict[str, Any] = {}
    outer_results: dict[str, list[dict[str, Any]]] = {v: [] for v in VERSIONS}
    selected_by_version: dict[str, list[dict[str, Any]]] = {v: [] for v in VERSIONS}
    if args.phase in ("p5_1", "all"):
        for version in VERSIONS:
            version_out = OUT / version
            version_out.mkdir(parents=True, exist_ok=True)
            selection_records = []
            for key, run in run_data.items():
                record = select_for_run(run, version, device, version_out)
                selection_records.append(record)
                selected_by_version[version].append(record)
            pd.DataFrame([{**{k: v for k, v in r["selected"].items() if k != "inner_rows"}, "fold": r["fold"], "seed": r["seed"]} for r in selection_records]).to_csv(version_out / "P5_1_SELECTED_CONFIGS.csv", index=False)
            all_candidates = []
            all_inner = []
            for r in selection_records:
                for c in r["candidates"]:
                    all_candidates.append({k: v for k, v in c.items() if k != "inner_rows"})
                    all_inner.extend(c["inner_rows"])
            pd.DataFrame(all_candidates).to_csv(version_out / "P5_1_HPARAM_CANDIDATES.csv", index=False)
            pd.DataFrame(all_inner).to_csv(version_out / "P5_1_INNER_CV_RESULTS.csv", index=False)
            for r in selection_records:
                run = run_data[(r["fold"], r["seed"])]
                outer_results[version].append(outer_evaluate(run, r["selected"], device, version_out))
            p5_reports[version] = aggregate_p5(version, outer_results[version], OUT)
        v2 = p5_reports["V2"]
        authorized = bool(v2["primary"]["mean_delta_BA"] > 0 and v2["primary"]["positive_runs"] >= 4 and any(v > 0 for v in v2["primary"]["geometry_delta"].values()) and v2["status"] != "PERSIST_ICG_P5_1_VIABLE")
        v3 = {"authorized": authorized, "rule": "V2 mean > 0, >=4/6 positive, active geometry, and not STRONG", "outer_test_used": False}
        if authorized:
            write_json(OUT / "protocol" / "V3_PROGRESSION_DECISION.json", v3)
            raise RuntimeError("P5.1 authorized V3; refusing to skip it or proceed to P6")
        v3 = make_not_authorized_v3(OUT, v2)
        write_json(OUT / "protocol" / "V3_PROGRESSION_DECISION.json", v3)
        p5_final = write_p5_final(p5_reports, v3, OUT)
        if p5_final["status"] == "PERSIST_ICG_P5_1_VIABLE":
            write_json(OUT / "P5_1_P6_FINAL_REPORT.json", {"status": p5_final["status"], "p5": p5_final,
                       "p6": {"status": "NOT_RUN_P5_1_VIABLE"}, "outer_test_used": False})
            return
    else:
        p5_final = json.loads((OUT / "P5_1_FINAL_REPORT.json").read_text(encoding="utf-8"))
        p5_reports = p5_final["versions"]
        for version in VERSIONS:
            selected_by_version[version] = []
            for fold in FOLDS:
                for seed in SEEDS:
                    selected_path = OUT / version / f"fold-{fold}" / f"seed-{seed}" / "SELECTED_CONFIG.json"
                    selected_by_version[version].append({"fold": fold, "seed": seed,
                        "selected": json.loads(selected_path.read_text(encoding="utf-8"))})
    if args.phase in ("p6", "all"):
        # Predeclared base selection uses only the P5.1 inner-CV control BA,
        # never development-validation BA.  Simplicity breaks near ties.
        base_scores = {}
        for v in VERSIONS:
            vals = [float(x["selected"]["mean_control_BA"]) for x in selected_by_version[v]]
            base_scores[v] = float(np.mean(vals))
        base_version = sorted(VERSIONS, key=lambda v: (-base_scores[v], VERSIONS.index(v)))[0]
        write_json(OUT / "protocol" / "P6_BASE_VERSION_SELECTION.json", {"base_version": base_version,
                   "rule": "highest mean selected inner-CV control BA; simpler version breaks ties",
                   "inner_control_BA": base_scores, "outer_test_used": False})
        p6_results = []
        for record in selected_by_version[base_version]:
            run = run_data[(record["fold"], record["seed"])]
            # If resumed in p6-only mode, materialise the selected outer model
            # from its checkpoint rather than selecting on development labels.
            if outer_results.get(base_version):
                rr = next(x for x in outer_results[base_version] if x["summary"]["fold"] == record["fold"] and x["summary"]["seed"] == record["seed"])
            else:
                rr = outer_evaluate(run, record["selected"], device, OUT / base_version)
            p6_results.append(p6_run(rr, base_version, device, OUT / "P6"))
        p6_report = aggregate_p6(p6_results, OUT / "P6")
        p5_final = json.loads((OUT / "P5_1_FINAL_REPORT.json").read_text(encoding="utf-8"))
        final = {"status": p6_report["status"], "p5": p5_final, "p6": p6_report,
                 "outer_test_used": False, "implementation_id": IMPLEMENTATION_ID}
        write_json(OUT / "P5_1_P6_FINAL_REPORT.json", final)
        (OUT / "P5_1_P6_FINAL_REPORT.md").write_text(
            f"# PERSIST-EEG P5.1 + P6\n\nStatus: `{p6_report['status']}`\n\n"
            f"P5.1 completed TRAIN-only nested selection for V0/V1/V2. P6 protected-fusion mean delta: `{p6_report['primary']['mean_protected_fusion_delta_BA']:.6f}`; positive runs: `{p6_report['primary']['positive_runs']}/6`.\n\n"
            "Outer-test used: `false`.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
