"""Run the explicitly amended RAPID_TRIAGE student-only screen.

This module is intentionally separate from the frozen GeoSR decision runner.
It keeps the canonical data loader, model, optimizer, minibatch order and
initial-selection cross-fit helpers, but implements the protocol amendment:
only fold 0 is trained, only SUBJECT_BALANCED_ERM and GEOSR are compared,
only the completed initial-selection cross-fit weights are used, and the
discovery-selected student checkpoint is retained directly (no final-refit
teacher or student stage).  Outcome data are never touched here.

Per-epoch progress is atomically checkpointed, including optimizer and RNG
state, so an interrupted triage worker resumes at the next epoch without
replaying completed training.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

import audit_primitives as ap
import run_geosr as g


METHODS = ("SUBJECT_BALANCED_ERM", "GEOSR")
SEED = 0
FOLD = 0
DATASETS = ("OpenBMI", "WBCIC")
EXP = Path(__file__).resolve().parents[1]
_PROGRESS_CACHE: g.FoldCache | None = None


def clean(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [clean(x) for x in v]
    if isinstance(v, np.ndarray):
        return clean(v.tolist())
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        x = float(v)
        return x if math.isfinite(x) else None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def amendment() -> tuple[dict[str, Any], str]:
    path = EXP / "RAPID_TRIAGE_PROTOCOL_AMENDMENT.json"
    lock_path = EXP / "RAPID_TRIAGE_LOCK.json"
    if not path.is_file() or not lock_path.is_file():
        raise RuntimeError("RAPID_TRIAGE amendment/hash lock is missing")
    # The amendment lock was initially materialised from Windows PowerShell,
    # which may prepend a UTF-8 BOM.  Accepting utf-8-sig is byte-preserving
    # for normal UTF-8 and avoids a spurious orchestration failure.
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    digest = file_sha(path)
    lock = json.loads(lock_path.read_text(encoding="utf-8-sig"))
    if lock.get("amendment_sha256") != digest:
        raise RuntimeError("RAPID_TRIAGE amendment hash mismatch")
    if lock.get("status") != "RAPID_TRIAGE_LOCKED" or lock.get("outcome_labels_read") is not False:
        raise RuntimeError("RAPID_TRIAGE lock is not pre-outcome")
    if lock.get("methods") != list(METHODS) or lock.get("outer_folds") != [FOLD]:
        raise RuntimeError("RAPID_TRIAGE scope mismatch")
    return payload, digest


def copy_initial_cache(source_root: Path, dest_root: Path) -> None:
    """Copy only initial-selection caches; never move/delete source caches."""
    src = source_root / "runtime" / "seed-0" / "cache"
    dst = dest_root / "runtime" / "seed-0" / "cache"
    if not src.is_dir():
        raise FileNotFoundError(f"source cache directory missing: {src}")
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        if "initial_selection" not in rel.parts:
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            shutil.copy2(path, out)


def _restore_rng(payload: Mapping[str, Any]) -> None:
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and payload.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])


def _optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    if device.type != "cuda":
        return
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def progress_expected(dataset: str, method: str, train_rows: np.ndarray, val_rows: np.ndarray,
                      weights: np.ndarray, state: Mapping[str, torch.Tensor], mean: np.ndarray,
                      std: np.ndarray, amendment_sha: str) -> dict[str, Any]:
    return {
        "schema": "PERSIST_EEG_GEOSR_RAPID_PROGRESS_V1",
        "code_fingerprint": g.code_fingerprint(), "amendment_sha256": amendment_sha,
        "dataset": dataset, "fold": FOLD, "seed": SEED, "method": method,
        "train_rows_sha256": g.array_sha(train_rows), "val_rows_sha256": g.array_sha(val_rows),
        "weights_sha256": g.array_sha(np.asarray(weights, dtype=np.float32)),
        "state_sha256": g.state_hash(state), "mean_sha256": g.bytes_sha(np.asarray(mean).tobytes()),
        "std_sha256": g.bytes_sha(np.asarray(std).tobytes()),
    }


def load_progress(path: Path, expected: Mapping[str, Any], device: torch.device) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        meta = payload.get("cache_meta", {})
        if not all(meta.get(k) == v for k, v in expected.items()):
            return None
        if _PROGRESS_CACHE is None:
            raise RuntimeError("progress cache context is not initialized")
        model = g.make_model(_PROGRESS_CACHE, device)
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=g.LR, weight_decay=g.WEIGHT_DECAY)
        optimizer.load_state_dict(payload["optimizer_state"])
        _optimizer_state_to_device(optimizer, device)
        _restore_rng(payload)
        payload["_model"] = model
        payload["_optimizer"] = optimizer
        return payload
    except Exception as exc:
        print(f"[rapid] ignoring unreadable progress {path.name}: {exc}", flush=True)
        return None


def save_progress(path: Path, expected: Mapping[str, Any], model: torch.nn.Module,
                  optimizer: torch.optim.Optimizer, epoch: int, best_epoch: int,
                  best_ba: float, best_nll: float, stale: int,
                  best_state: Mapping[str, torch.Tensor], history: list[dict[str, Any]]) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    payload = {
        "cache_meta": dict(expected), "epoch": int(epoch), "best_epoch": int(best_epoch),
        "best_ba": float(best_ba), "best_nll": float(best_nll), "stale": int(stale),
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "best_state": {k: v.detach().cpu() for k, v in best_state.items()},
        "optimizer_state": copy.deepcopy(optimizer.state_dict()), "history": history,
        "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    g.atomic_torch_save(payload, path)


def save_rapid_checkpoint(path: Path, model_state: Mapping[str, torch.Tensor], mean: np.ndarray,
                          std: np.ndarray, dataset: str, method: str, selected_epoch: int,
                          init_sha: str, expected: Mapping[str, Any]) -> str:
    payload = {
        "model_state": {k: v.detach().cpu() for k, v in model_state.items()},
        "mean": np.asarray(mean), "std": np.asarray(std), "dataset": dataset,
        "fold": FOLD, "seed": SEED, "method": method, "selected_epoch": int(selected_epoch),
        "initial_state_sha256": init_sha, "protocol": "GeoSR_rapid_triage_v1",
        "training": {"initial_selection_only": True, "exact_refit": False,
                      "discovery_selected": True},
    }
    g.atomic_torch_save(payload, path)
    write_json(g.checkpoint_meta_path(path), dict(expected, stage="rapid_triage_initial_selection"))
    return file_sha(path)


def select_keep_best(cache: g.FoldCache, train_rows: np.ndarray, val_rows: np.ndarray,
                     mean: np.ndarray, std: np.ndarray, weights: np.ndarray,
                     state: Mapping[str, torch.Tensor], dataset: str, method: str,
                     device: torch.device, progress_path: Path, ckpt_path: Path,
                     amendment_sha: str) -> tuple[int, list[dict[str, Any]], float, bool, bool, str]:
    """Canonical discovery selection with best model retained, plus resume."""
    global _PROGRESS_CACHE
    _PROGRESS_CACHE = cache
    expected = progress_expected(dataset, method, train_rows, val_rows, weights, state, mean, std, amendment_sha)
    ck_meta = g.checkpoint_meta_path(ckpt_path)
    if ckpt_path.is_file() and ck_meta.is_file():
        try:
            meta = json.loads(ck_meta.read_text(encoding="utf-8"))
            if all(meta.get(k) == v for k, v in expected.items()) and meta.get("stage") == "rapid_triage_initial_selection":
                payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                history = []
                if progress_path.is_file():
                    saved = torch.load(progress_path, map_location="cpu", weights_only=False)
                    history = list(saved.get("history", []))
                return int(payload["selected_epoch"]), history, 0.0, True, True, file_sha(ckpt_path)
        except Exception:
            pass

    g.seed_everything(SEED)
    model = g.make_model(cache, device)
    model.load_state_dict(state, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=g.LR, weight_decay=g.WEIGHT_DECAY)
    row_weight_lookup = np.zeros(len(cache.meta), dtype=np.float32)
    row_weight_lookup[np.asarray(train_rows, dtype=np.int64)] = np.asarray(weights, dtype=np.float32)
    weight_device = torch.from_numpy(row_weight_lookup).to(device, non_blocking=True) if device.type == "cuda" else None
    best_ba, best_nll, best_epoch, stale = -math.inf, math.inf, 1, 0
    best_state: dict[str, torch.Tensor] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history: list[dict[str, Any]] = []
    start_epoch = 0
    resumed = False
    loaded = load_progress(progress_path, expected, device)
    if loaded is not None:
        model = loaded.pop("_model")
        optimizer = loaded.pop("_optimizer")
        start_epoch = int(loaded.get("epoch", 0))
        best_epoch = int(loaded.get("best_epoch", 1)); best_ba = float(loaded.get("best_ba", -math.inf))
        best_nll = float(loaded.get("best_nll", math.inf)); stale = int(loaded.get("stale", 0))
        best_state = {k: v.detach().cpu().clone() for k, v in loaded.get("best_state", {}).items()}
        history = list(loaded.get("history", [])); resumed = True
        print(f"[rapid] resume {dataset} fold={FOLD} method={method} epoch={start_epoch}", flush=True)

    t0 = time.perf_counter()
    for epoch in range(start_epoch + 1, g.MAX_EPOCHS + 1):
        if epoch - 1 >= g.MIN_EPOCHS and stale >= g.PATIENCE:
            break
        epoch_t0 = time.perf_counter()
        order = g.order_for(train_rows, dataset, FOLD, SEED, "select", "student-common", epoch)
        loss = g.train_epoch(model, cache, train_rows, mean, std, weights, optimizer, order, device,
                             row_weight_lookup=row_weight_lookup, weight_device=weight_device)
        val = g.eval_rows(cache, model, val_rows, mean, std, device)
        ba = float(val.BA.mean()) if len(val) else -math.inf
        nll = float(val.NLL.mean()) if len(val) else math.inf
        improved = ba > best_ba + 1e-12 or (abs(ba - best_ba) <= 1e-12 and nll < best_nll - 1e-12)
        if improved:
            best_ba, best_nll, best_epoch, stale = ba, nll, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        elapsed = time.perf_counter() - epoch_t0
        history.append({"epoch": epoch, "train_loss": loss, "val_BA": ba, "val_NLL": nll, "sec": elapsed})
        print(f"[rapid] {dataset} fold={FOLD} method={method} epoch={epoch} BA={ba:.4f} best={best_epoch} sec={elapsed:.3f}", flush=True)
        save_progress(progress_path, expected, model, optimizer, epoch, best_epoch, best_ba, best_nll, stale, best_state, history)
        if epoch >= g.MIN_EPOCHS and stale >= g.PATIENCE:
            break
    total = time.perf_counter() - t0
    ck_sha = save_rapid_checkpoint(ckpt_path, best_state, mean, std, dataset, method, best_epoch, g.state_hash(state), expected)
    # Retain progress and its complete history for audit and interruption recovery.
    del model, optimizer
    gc.collect()
    return int(best_epoch), history, total, resumed, False, ck_sha


def run(dataset: str, root: Path, source_root: Path, device: torch.device, amendment_sha: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    copy_initial_cache(source_root.resolve(), root)
    roles, _, _ = ap.load_roles(dataset)
    role = roles[FOLD]
    source = g.subj_sort(role["model_fit"])
    source_all = g.subj_sort(set(role["model_fit"]) | set(role["discovery"]))
    cache = g.FoldCache(dataset, source_all, SEED, FOLD)
    fit_rows = cache.rows(source, g.sessions_for(dataset))
    discovery_rows = cache.rows(role["discovery"], (g.SESSION_DISCOVERY[dataset],))
    fit_mean, fit_std = cache.normalizer(fit_rows)
    cache.normalize(fit_mean, fit_std)
    cache_root = root / "runtime" / "seed-0" / "cache"
    print(f"[rapid] start {dataset} fold={FOLD}; initial-selection teachers only", flush=True)
    risk, assignments, teachers = g.crossfit_scalars(cache, source, dataset, FOLD, SEED,
                                                      "initial_selection", device, cache_root=cache_root)
    weights, weight_audit = g.source_weights(risk, dataset, FOLD, SEED, methods=METHODS)
    state, init_seed, init_sha = g.initial_state(cache, dataset, FOLD, SEED, "student")
    training_rows: list[dict[str, Any]] = []
    selected: dict[str, int] = {}
    checkpoint_info: dict[str, Any] = {}
    histories: dict[str, Any] = {}
    for method in METHODS:
        wvec = g.weight_vector(cache, fit_rows, weights[method], method)
        out_dir = root / "runtime" / "seed-0" / dataset / f"fold-{FOLD}"
        out_dir.mkdir(parents=True, exist_ok=True)
        progress_path = out_dir / f"{method}.progress.pt"
        ckpt_path = out_dir / f"{method}.pt"
        ep, hist, sec, resumed, hit, ck_sha = select_keep_best(
            cache, fit_rows, discovery_rows, fit_mean, fit_std, wvec, state, dataset, method,
            device, progress_path, ckpt_path, amendment_sha)
        selected[method] = ep; histories[method] = hist
        checkpoint_info[method] = {"path": str(ckpt_path), "sha256": ck_sha, "selected_epoch": ep,
                                   "initial_selection_only": True, "exact_refit": False}
        training_rows.append({"dataset": dataset, "fold": FOLD, "seed": SEED,
                              "stage": "rapid_triage_initial_selection", "method": method,
                              "selected_epoch": ep, "initial_state_sha256": init_sha,
                              "normalizer_mean_sha256": g.bytes_sha(fit_mean.tobytes()),
                              "normalizer_std_sha256": g.bytes_sha(fit_std.tobytes()),
                              "training_subjects": len(source), "discovery_subjects": len(role["discovery"]),
                              "weight_mean": float(wvec.mean()), "weight_min": float(wvec.min()),
                              "weight_max": float(wvec.max()), "fit_sec": float(sec),
                              "fit_sec_per_epoch": float(np.mean([h["sec"] for h in hist])) if hist else None,
                              "completed_epochs": len(hist),
                              "total_epoch_sec_including_resumed": float(sum(h["sec"] for h in hist)),
                              "resumed": bool(resumed),
                              "checkpoint_cache_hit": bool(hit)})

    results = root / "results"
    write_csv(results / "CROSS_FIT_ASSIGNMENTS.csv", assignments)
    write_csv(results / "CROSSFIT_TEACHER_AUDIT.csv", teachers)
    write_csv(results / "SOURCE_GEOMETRY_RISK.csv", [{**r, "seed": SEED} for _, r in risk.iterrows()])
    write_csv(results / "SOURCE_WEIGHT_AUDIT.csv", [{**r, "stage": "initial_selection"} for r in weight_audit["rows"]])
    write_csv(results / "TRAINING_SUMMARY.csv", training_rows)
    write_json(results / "INITIAL_STATE_HASHES.json", {f"{dataset}/fold-{FOLD}/seed-{SEED}": {"initial_state_sha256": init_sha, "initial_seed": init_seed}})
    key = f"{dataset}/fold-{FOLD}/seed-{SEED}"
    manifest = {
        key: {"dataset": dataset, "fold": FOLD, "seed": SEED, "model_fit_subjects": source,
              "discovery_subjects": g.subj_sort(role["discovery"]),
              "outcome_subjects_hash": g.bytes_sha("|".join(g.subj_sort(role["outcome"])).encode()),
              "role_hash": g.role_hash(role), "selected_epochs": selected,
              "checkpoints": checkpoint_info, "initial_normalizer_mean_sha256": g.bytes_sha(fit_mean.tobytes()),
              "initial_normalizer_std_sha256": g.bytes_sha(fit_std.tobytes()),
              "source_initial_weight_lock": weight_audit["lock"], "initial_histories": histories,
              "rapid_triage": True, "initial_selection_weights_reused": True,
              "exact_refit": False, "final_refit_teacher": False}
    }
    write_json(root / "runtime" / "seed-0" / "PREFLIGHT_MANIFEST.json", manifest)
    write_json(root / "DATA_SUPPORT_LOCK.json", {"descriptor_cap": g.CAP, "seed": SEED, "rapid_triage": True,
                                                    "outcome_labels_read": False})
    lock = {
        "schema": "PERSIST_EEG_GEOSR_RAPID_TRIAGE_WORKER_LOCK_V1",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "amendment_sha256": amendment_sha, "dataset": dataset, "fold": FOLD, "seed": SEED,
        "methods": list(METHODS), "inner_crossfit_k": g.INNER_K, "descriptor_cap": g.CAP,
        "role_hash": g.role_hash(role), "code_sha256": file_sha(Path(g.__file__)),
        "audit_primitives_sha256": file_sha(Path(ap.__file__)),
        "crossfit_stage_used": "initial_selection_only", "initial_selection_weights_reused": True,
        "wbcic_final_refit_teacher": False, "exact_refit": False, "student_discovery_selected": True,
        "outcome_labels_read": False, "WBCIC_outer_10_opened": False,
        "OpenBMI_sealed_holdout_opened": False, "scientific_definition_changed": True,
        "final_claim_authorized": False, "manifest_sha256": file_sha(root / "runtime" / "seed-0" / "PREFLIGHT_MANIFEST.json"),
    }
    write_json(root / "PRE_OUTCOME_RAPID_TRIAGE_WORKER_LOCK.json", lock)
    write_json(root / "DATA_LEGALITY_AUDIT.json", {
        "schema": "PERSIST_EEG_GEOSR_RAPID_TRIAGE_LEGALITY_V1", "seed": SEED,
        "dataset": dataset, "folds": [FOLD], "methods": list(METHODS),
        "WBCIC_outer_10_opened": False, "OpenBMI_sealed_holdout_opened": False,
        "canonical_outcome_indices_materialized": False, "canonical_outcome_labels_read": False,
        "outcome_labels_read_before_lock": False, "outcome_labels_read_after_lock": False,
        "amendment_sha256": amendment_sha, "lock_sha256": file_sha(root / "PRE_OUTCOME_RAPID_TRIAGE_WORKER_LOCK.json"),
    })
    write_json(root / "RAPID_TRIAGE_WORKER_COMPLETE.json", {
        "schema": "PERSIST_EEG_GEOSR_RAPID_TRIAGE_WORKER_COMPLETE_V1", "dataset": dataset,
        "fold": FOLD, "seed": SEED, "methods": list(METHODS), "amendment_sha256": amendment_sha,
        "code_fingerprint": g.code_fingerprint(), "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    print(f"[rapid] complete {dataset} fold={FOLD}", flush=True)
    del cache
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--fold", type=int, default=FOLD)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.device == "cuda":
        args.device = f"cuda:{torch.cuda.current_device()}"
    if args.fold != FOLD:
        raise SystemExit("RAPID_TRIAGE is fixed to outer fold 0")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    _, amendment_sha = amendment()
    run(args.dataset, args.root.resolve(), args.source_root.resolve(), torch.device(args.device), amendment_sha)


if __name__ == "__main__":
    main()
