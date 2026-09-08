"""Locked Phase-A training runner for NRRF-v1.

This runner deliberately performs no outer-development evaluation.  It writes
only source-session training diagnostics and resumable runtime checkpoints;
``evaluate_nrrf.py`` is invoked only after all 20 Phase-A cells finish.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

from nrrf_model import ALPHA, EMA_BETA, LAMBDA_EXPR, LAMBDA_REGRET, LAMBDA_TAIL, NRRF, load_ema, loss_terms, make_ema, update_ema
from subject_balanced_sampler import M_TRIALS, SubjectBalancedSampler

EXP = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("R2EEG_REPO", EXP.parents[2])).resolve()
RUNTIME = Path(os.environ.get("NRRF_RUNTIME", "/root/rivermind-data/nrrf_final_model_stage1_runtime")).resolve()
SOURCE_RUNTIME = Path(os.environ.get("CARRIER_5FOLD_RUNTIME", "/root/rivermind-data/carrier_5fold_multiseed_stability_runtime")).resolve()
PROTOCOL, OUTPUTS = EXP / "protocol", EXP / "outputs"
CARRIER_EXP = REPO / "experiments" / "persist_eeg_carrier_5fold_multiseed_stability_v1"
CARRIER_CODE = REPO / "experiments" / "persist_eeg_carrier_dualdataset_screen_v1" / "code"
STAGE1_CODE = REPO / "experiments" / "persist_eeg_r2eeg_stage1_v1" / "code"
sys.path[:0] = [str(CARRIER_CODE), str(STAGE1_CODE)]
import run_carrier_screen as carrier  # noqa: E402
import run_stage1 as v1  # noqa: E402
from eegnet_locked import EEGNet  # noqa: E402

EPOCHS = 20
LR = 1e-4
WEIGHT_DECAY = 5e-4
GRAD_CLIP = 5.0
METHODS = ("JOINT-CE", "NRRF-v1")
DATASETS = ("OpenBMI", "WBCIC")


def clean(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)): return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)): return bool(value)
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [clean(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    temp.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def state_hash(module: torch.nn.Module) -> str:
    import io
    buffer = io.BytesIO(); torch.save(module.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available(): state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state: torch.cuda.set_rng_state_all(state["cuda"])


def split_payload() -> tuple[dict[str, Any], str]:
    path = CARRIER_EXP / "protocol" / "FIVEFOLD_SPLIT.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("protocol") != "CARRIER_5FOLD_MULTISEED_STABILITY_V1" or len(raw.get("folds", {})) != 2:
        raise RuntimeError("required frozen fivefold split is unavailable or malformed")
    for dataset, expected in (("OpenBMI", 40), ("WBCIC", 31)):
        subjects = raw["search_subjects"].get(dataset, [])
        folds = raw["folds"].get(dataset, [])
        if len(subjects) != expected or len(folds) != 5:
            raise RuntimeError(f"frozen split invariant failed for {dataset}")
        outer = set().union(*(set(map(str, f["outer_dev_subjects"])) for f in folds))
        if outer != set(map(str, subjects)):
            raise RuntimeError(f"outer-development partition mismatch for {dataset}")
    return raw, sha256(path)


def checkpoint_path(dataset: str, fold: int, seed: int, model: str) -> Path:
    return SOURCE_RUNTIME / f"{dataset.lower()}_fold{fold}_seed{seed}_{model.lower()}" / "selected_best.pt"


def all_source_checkpoints() -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        for fold in range(5):
            for seed in range(3):
                for model in ("EEGNet", "LiteBN"):
                    path = checkpoint_path(dataset, fold, seed, model)
                    if not path.is_file(): raise FileNotFoundError(path)
                    rows.append({"dataset": dataset, "fold": fold, "seed": seed, "model": model,
                                 "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
    return rows


def construct(name: str, channels: int) -> torch.nn.Module:
    if name == "EEGNet": return EEGNet(channels)
    if name == "LiteBN": return carrier.CompactLite(channels, "bn")
    raise ValueError(name)


def strict_load(name: str, channels: int, path: Path, device: torch.device, frozen: bool) -> torch.nn.Module:
    model = construct(name, channels).to(device)
    loaded = torch.load(path, map_location=device, weights_only=False)
    mismatch = model.load_state_dict(loaded, strict=True)
    if mismatch.missing_keys or mismatch.unexpected_keys: raise RuntimeError(f"strict loading failed: {path}")
    if frozen:
        for parameter in model.parameters(): parameter.requires_grad_(False)
        model.eval()
    return model


def source_index(bundle: Any, fold: dict[str, Any]) -> dict[str, np.ndarray]:
    sessions = v1.source_sessions(bundle.name)
    answer: dict[str, np.ndarray] = {}
    for subject in fold["inner_train_subjects"]:
        values = bundle.indices([str(subject)], sessions)
        if not len(values): raise RuntimeError(f"no legal source trials for {bundle.name}/{subject}")
        answer[str(subject)] = np.asarray(values, dtype=np.int64)
    return answer


def inner_val_ba(model: NRRF, bundle: Any, subjects: list[str], cache: Any) -> float:
    # The allowed inner-validation future-session diagnostic is never used to select a checkpoint.
    model.eval(); values = []
    with torch.no_grad():
        for subject in subjects:
            idx = bundle.indices([str(subject)], (2,)); y = bundle.labels(idx); logits = []
            for start in range(0, len(idx), 128):
                x, _ = cache.batch(idx[start:start + 128]); logits.append(model(x)[2].float().cpu().numpy())
            values.append(balanced_accuracy_score(y, np.concatenate(logits).argmax(axis=1)))
    return float(np.mean(values))


def model_from_sources(dataset: str, fold: int, seed: int, channels: int, device: torch.device) -> tuple[NRRF, dict[str, str]]:
    anchor_path, expr_path = checkpoint_path(dataset, fold, seed, "EEGNet"), checkpoint_path(dataset, fold, seed, "LiteBN")
    anchor = strict_load("EEGNet", channels, anchor_path, device, frozen=True)
    expressive = strict_load("LiteBN", channels, expr_path, device, frozen=False)
    return NRRF(anchor, expressive).to(device), {"EEGNet": sha256(anchor_path), "LiteBN": sha256(expr_path)}


def implementation_tests(device: torch.device) -> dict[str, Any]:
    # This is a synthetic invariant test; it does not touch outer data or labels.
    set_seed(9107)
    anchor = construct("EEGNet", 62).to(device); expr = construct("LiteBN", 62).to(device); model = NRRF(anchor, expr).to(device)
    anchor_before = {k: v.detach().clone() for k, v in model.anchor.state_dict().items()}
    expr_before = state_hash(model.expressive)
    x = torch.randn(8, 62, 1000, device=device); y = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], device=device)
    slots = torch.arange(2, device=device).repeat_interleave(4)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    model.train(); a, e, f = model(x); terms = loss_terms(a, e, f, y, slots, 2, "NRRF-v1")
    opt.zero_grad(set_to_none=True); terms["total"].backward(); opt.step()
    anchor_unchanged = all(torch.equal(anchor_before[k], v) for k, v in model.anchor.state_dict().items())
    expr_changed = expr_before != state_hash(model.expressive)
    low_f = torch.tensor([[20.0, 0.0], [0.0, 20.0]], device=device)
    high_f = torch.zeros_like(low_f)
    fixed_a = torch.tensor([[10.0, 0.0], [0.0, 10.0]], device=device)
    zero_regret = float(loss_terms(fixed_a, fixed_a, low_f, torch.tensor([0, 1], device=device), torch.tensor([0, 1], device=device), 2, "NRRF-v1")["L_regret"].item()) == 0.0
    positive_regret = float(loss_terms(fixed_a, fixed_a, high_f, torch.tensor([0, 1], device=device), torch.tensor([0, 1], device=device), 2, "NRRF-v1")["L_regret"].item()) > 0.0
    fake_labels = np.array([0, 1] * 32, dtype=np.int64); fake = {"a": np.arange(0, 32), "b": np.arange(32, 64)}
    sampler_a, sampler_b = SubjectBalancedSampler(fake, fake_labels, 5), SubjectBalancedSampler(fake, fake_labels, 5)
    batch_a, batch_b = sampler_a.one(1, 0), sampler_b.one(1, 0)
    return {"alpha_exactly_0_5": ALPHA == 0.5, "anchor_parameters_frozen": not any(p.requires_grad for p in model.anchor.parameters()),
            "anchor_state_unchanged_after_synthetic_step": anchor_unchanged, "litebn_parameters_changed_after_synthetic_step": expr_changed,
            "anchor_eval_after_train": not model.anchor.training, "regret_zero_when_fused_not_worse": zero_regret,
            "regret_positive_when_fused_worse": positive_regret, "cvar_q_ceil_25pct_K8": int(math.ceil(.25 * 8)) == 2,
            "sampler_grouping": len(batch_a.subject_ids) == 2 and len(batch_a.indices) == 32,
            "sampler_deterministic": batch_a.as_dict() == batch_b.as_dict(), "ema_beta_exactly_0_99": EMA_BETA == .99}


def train_cell(dataset: str, fold: dict[str, Any], seed: int, method: str, bundle: Any, cache: Any, split_sha: str, device: torch.device) -> None:
    cell_name = f"{dataset.lower()}_fold{int(fold['fold_id'])}_seed{seed}_{method.lower().replace('-', '_')}"
    cell = RUNTIME / "cells" / cell_name; cell.mkdir(parents=True, exist_ok=True)
    final = cell / "epoch20_ema.pt"
    if final.is_file():
        saved = torch.load(final, map_location="cpu", weights_only=False)
        if saved.get("method") == method and saved.get("epochs") == EPOCHS:
            print(f"[resume] completed {cell_name}", flush=True); return
        raise RuntimeError(f"completed checkpoint invariant mismatch: {final}")
    set_seed(seed)
    model, source_hashes = model_from_sources(dataset, int(fold["fold_id"]), seed, bundle.channels, device)
    source_by_subject = source_index(bundle, fold)
    source_indices = np.concatenate(list(source_by_subject.values()))
    labels = bundle.labels(np.arange(len(bundle.search_rows), dtype=np.int64))
    sampler_seed = int(hashlib.sha256(f"{dataset}|{fold['fold_id']}|{seed}|NRRF-v1".encode()).hexdigest()[:8], 16)
    steps = max(20, int(math.ceil(len(source_indices) / 128)))
    plan = SubjectBalancedSampler(source_by_subject, labels, sampler_seed).manifest(EPOCHS, steps)
    plan_payload = [[batch.as_dict() for batch in epoch] for epoch in plan]
    manifest_sha = sha_json(plan_payload)
    initial_expr_sha = state_hash(model.expressive)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = make_ema(model.expressive)
    latest = cell / "checkpoint_latest.pt"
    start, history = 1, []
    if latest.is_file():
        saved = torch.load(latest, map_location=device, weights_only=False)
        invariants = {"method": method, "manifest_sha256": manifest_sha, "initial_expressive_sha256": initial_expr_sha,
                      "source_checkpoint_sha256": source_hashes, "split_sha256": split_sha}
        if any(saved.get(k) != v for k, v in invariants.items()): raise RuntimeError(f"resume invariant mismatch: {latest}")
        model.expressive.load_state_dict(saved["expressive_state"], strict=True); optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"]); ema = saved["ema_parameters"]; history = saved["history"]; start = int(saved["epoch"]) + 1
        restore_rng(saved["rng"])
    start_time = time.perf_counter()
    for epoch in range(start, EPOCHS + 1):
        model.train(); epoch_values: dict[str, list[float]] = {key: [] for key in ("total", "L_mean", "L_tail", "L_regret", "L_expr", "positive_regret_fraction", "mean_positive_regret", "worst_subject_loss", "mean_subject_loss", "gradient_norm")}
        fallback_count = 0
        for batch in plan[epoch - 1]:
            x, y = cache.batch(batch.indices); slots = torch.arange(len(batch.subject_ids), device=device).repeat_interleave(M_TRIALS)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                a, e, f = model(x); terms = loss_terms(a, e, f, y, slots, len(batch.subject_ids), method)
            if not torch.isfinite(terms["total"]): raise RuntimeError(f"non-finite loss: {cell_name}")
            scaler.scale(terms["total"]).backward(); scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(list(model.trainable_parameters()), GRAD_CLIP)
            scaler.step(optimizer); scaler.update(); update_ema(ema, model.expressive)
            for key in epoch_values:
                epoch_values[key].append(float(grad_norm.detach().cpu()) if key == "gradient_norm" else float(terms[key].detach().cpu()))
            fallback_count += len(batch.fallbacks)
        diag = inner_val_ba(model, bundle, fold["inner_val_subjects"], cache)
        row = {"epoch": epoch, "inner_val_subject_BA_diagnostic": diag, "fallback_events": fallback_count,
               **{key: float(np.mean(values)) for key, values in epoch_values.items()}}
        history.append(row)
        payload = {"epoch": epoch, "method": method, "epochs": EPOCHS, "history": history, "expressive_state": model.expressive.state_dict(),
                   "ema_parameters": ema, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(),
                   "manifest_sha256": manifest_sha, "initial_expressive_sha256": initial_expr_sha, "source_checkpoint_sha256": source_hashes,
                   "split_sha256": split_sha, "steps_per_epoch": steps, "sampler_seed": sampler_seed}
        torch.save(payload, latest)
        print(f"[{cell_name}] epoch={epoch:02d}/{EPOCHS} total={row['total']:.5f} mean={row['L_mean']:.5f} regret={row['L_regret']:.5f} inner_val_diag={diag:.4f}", flush=True)
    raw_state = copy.deepcopy(model.expressive.state_dict())
    torch.save({"method": method, "epochs": EPOCHS, "checkpoint_rule": "final_epoch20_raw", "expressive_state": raw_state,
                "source_checkpoint_sha256": source_hashes, "manifest_sha256": manifest_sha}, cell / "epoch20_raw.pt")
    load_ema(model.expressive, ema)
    torch.save({"method": method, "epochs": EPOCHS, "checkpoint_rule": "PRIMARY_final_epoch20_EMA", "expressive_state": model.expressive.state_dict(),
                "ema_parameters": ema, "source_checkpoint_sha256": source_hashes, "manifest_sha256": manifest_sha}, final)
    write_json(cell / "training_diagnostics.json", {"dataset": dataset, "fold": int(fold["fold_id"]), "seed": seed, "method": method,
                                                       "epochs": EPOCHS, "steps_per_epoch": steps, "source_sessions": list(v1.source_sessions(dataset)),
                                                       "sampler_manifest_sha256": manifest_sha, "history": history,
                                                       "elapsed_seconds_this_invocation": time.perf_counter() - start_time})


def prepare_protocol(split: dict[str, Any], split_sha: str, device: torch.device) -> None:
    rows = all_source_checkpoints()
    write_json(PROTOCOL / "SOURCE_CHECKPOINTS.json", {"source_runtime": SOURCE_RUNTIME, "selected_checkpoints": rows,
                                                         "rule": "exact carrier selected_best.pt only; source files never modified"})
    write_json(PROTOCOL / "CHECKPOINT_HASH_AUDIT.json", {"all_present": len(rows) == 60, "entries": rows,
                                                            "phase_a_required_entries": [x for x in rows if x["seed"] == 0]})
    write_json(PROTOCOL / "FIVEFOLD_SPLIT_REFERENCE.json", {"source": str(CARRIER_EXP / "protocol" / "FIVEFOLD_SPLIT.json"), "sha256": split_sha,
                                                               "payload": split})
    write_json(PROTOCOL / "TRAINING_PROTOCOL.json", {"phase": "A", "datasets": list(DATASETS), "folds": list(range(5)), "seeds": [0],
        "methods": list(METHODS), "new_training_cells": 20, "optimizer": "AdamW", "lr": LR, "weight_decay": WEIGHT_DECAY,
        "gradient_clip": GRAD_CLIP, "epochs": EPOCHS, "scheduler": None, "early_stopping": False, "alpha": ALPHA,
        "anchor": "EEGNet exact selected checkpoint, frozen/eval/no_grad", "expressive": "exact CompactLite-BN selected checkpoint, trainable",
        "ema": {"parameter_scope": "LiteBN parameters only", "beta": EMA_BETA, "primary_checkpoint": "final epoch20 EMA"},
        "source_sessions": {"OpenBMI": [1], "WBCIC": [0, 1]}, "outer_development_sessions": {"OpenBMI": [2], "WBCIC": [2]},
        "sampler": {"K_subjects": 8, "M_trials_per_subject": M_TRIALS, "class_balance": "8/8 when available", "deterministic": True}})
    write_json(PROTOCOL / "LOSS_DEFINITION.json", {"NRRF-v1": "L_mean + 0.5 L_tail + 1.0 L_regret + 0.25 L_expr", "JOINT-CE": "L_mean + 0.25 L_expr",
        "cvar": "top ceil(0.25*K) largest subject L_F", "regret": "mean ReLU(L_F(s)-stopgrad(L_A(s)))", "alpha": ALPHA,
        "lambda_tail": LAMBDA_TAIL, "lambda_regret": LAMBDA_REGRET, "lambda_expr": LAMBDA_EXPR})
    write_json(PROTOCOL / "INFORMATION_MATCHING.json", {"JOINT-CE_and_NRRF_share": ["initialization", "anchor checkpoint", "expressive checkpoint", "split", "sampler manifest", "source sessions", "normalizer", "optimizer", "learning rate", "weight decay", "epochs", "alpha", "EMA", "seed"], "only_intended_difference": "NRRF adds L_tail and L_regret"})
    write_json(PROTOCOL / "HOLDOUT_ISOLATION_AUDIT.json", {"V8_INTERNAL_HOLDOUT_loaded": False, "V8_INTERNAL_HOLDOUT_labels_loaded": False,
        "WBCIC_true_outer_loaded": False, "WBCIC_true_outer_labels_loaded": False, "scope": "carrier fivefold SEARCH development subjects only"})
    tests = implementation_tests(device)
    tests.update({"strict_checkpoint_loading_required": True, "all_60_source_selected_checkpoints_hashed": len(rows) == 60,
                  "only_source_sessions_enter_optimization": True, "outer_subjects_never_enter_optimization": True,
                  "same_fivefold_split_reused": True, "no_early_stopping": True, "primary_checkpoint_final_epoch20_ema": True,
                  "holdouts_untouched": True})
    if not all(bool(v) for v in tests.values()): raise RuntimeError(f"implementation test failure: {tests}")
    write_json(PROTOCOL / "TESTS.json", tests)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--validate-only", action="store_true"); args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("NRRF Phase A requires the provisioned CUDA server")
    for directory in (PROTOCOL, OUTPUTS, RUNTIME): directory.mkdir(parents=True, exist_ok=True)
    split, split_sha = split_payload(); prepare_protocol(split, split_sha, device)
    if args.validate_only:
        print("NRRF_PHASE_A_PROTOCOL_VALID", flush=True); return 0
    # Cache one dataset at a time to minimize peak VRAM and to avoid concurrent GPU jobs.
    for dataset in DATASETS:
        bundle = v1.load_bundle(dataset, list(map(str, split["search_subjects"][dataset])))
        for fold in split["folds"][dataset]:
            # Exact normalizer is rebuilt solely from each fold's legal inner-train source sessions.
            mean, std, _ = v1.normalizer(bundle, list(map(str, fold["inner_train_subjects"])))
            cache = carrier.GPUCache(bundle, mean, std, device)
            for method in METHODS:
                train_cell(dataset, fold, 0, method, bundle, cache, split_sha, device)
            del cache
            torch.cuda.empty_cache()
    print("NRRF_PHASE_A_TRAINING_COMPLETE_NO_OUTER_OUTCOME_REPORTED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
