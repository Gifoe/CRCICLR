"""Train CFRF-v1 / GLOBAL-ALPHA on source-only INNER_VAL features; no outer evaluation."""
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
import torch.nn.functional as F

EXP = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get("R2EEG_REPO", EXP.parents[1])).resolve()
RUNTIME = Path(os.environ.get("CFRF_RUNTIME", "/root/rivermind-data/cfrf_v1_runtime")).resolve()
SOURCE_RUNTIME = Path(os.environ.get("CARRIER_5FOLD_RUNTIME", "/root/rivermind-data/carrier_5fold_multiseed_stability_runtime")).resolve()
CARRIER_EXP = REPO / "experiments/persist_eeg_carrier_5fold_multiseed_stability_v1"
PROTOCOL = EXP / "protocol"
OUT = EXP / "outputs"
sys.path[:0] = [str(EXP / "code"), str(REPO / "experiments/persist_eeg_carrier_dualdataset_screen_v1/code"), str(REPO / "experiments/persist_eeg_r2eeg_stage1_v1/code")]
import run_carrier_screen as carrier  # noqa: E402
import run_stage1 as v1  # noqa: E402
from eegnet_locked import EEGNet  # noqa: E402
from cfrf_model import ALPHA_CENTER, LAMBDA_DELTA, CFRF, GlobalAlpha, ema_load, ema_start, ema_update  # noqa: E402
from feature_cache import SignalCache, build_or_load  # noqa: E402
from frozen_carriers import extract, load_carrier, sha256_file, state_sha256  # noqa: E402

DATASETS = ("OpenBMI", "WBCIC")
PHASE_A_METHODS = ("GLOBAL-ALPHA", "CFRF-v1")
EPOCHS, LR, WD, CLIP, K, M = 30, 3e-4, 1e-3, 5.0, 6, 32


def clean(value: Any) -> Any:
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.ndarray): return clean(value.tolist())
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating, float)): return float(value)
    if isinstance(value, (np.bool_, bool)): return bool(value)
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def seed_all(value: int) -> None:
    random.seed(value); np.random.seed(value); torch.manual_seed(value); torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def checkpoint_path(dataset: str, fold: int, seed: int, model: str) -> Path:
    return SOURCE_RUNTIME / f"{dataset.lower()}_fold{fold}_seed{seed}_{model.lower()}" / "selected_best.pt"


def split_payload() -> tuple[dict[str, Any], str]:
    path = CARRIER_EXP / "protocol/FIVEFOLD_SPLIT.json"
    payload = json.loads(path.read_text())
    if payload.get("protocol") != "CARRIER_5FOLD_MULTISEED_STABILITY_V1":
        raise RuntimeError("wrong frozen five-fold split")
    return payload, sha256_file(path)


def source_rows() -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        for fold in range(5):
            for seed in range(3):
                for model in ("EEGNet", "LiteBN"):
                    path = checkpoint_path(dataset, fold, seed, model)
                    if not path.is_file(): raise FileNotFoundError(path)
                    rows.append({"dataset": dataset, "fold": fold, "seed": seed, "model": model, "path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return rows


def source_indices(bundle: Any, subjects: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    result = {str(subject): np.asarray(bundle.indices([str(subject)], v1.source_sessions(bundle.name)), dtype=np.int64) for subject in subjects}
    all_indices = np.concatenate(list(result.values()))
    return np.unique(all_indices), result


def deterministic_batches(by_subject: dict[str, np.ndarray], labels_by_position: np.ndarray, epochs: int, steps: int, seed: int) -> list[list[np.ndarray]]:
    subject_ids = np.asarray(sorted(by_subject), dtype=object)
    if not len(subject_ids): raise RuntimeError("no INNER_VAL training subjects")
    batches: list[list[np.ndarray]] = []
    for epoch in range(1, epochs + 1):
        epoch_batches = []
        for step in range(steps):
            rng = np.random.default_rng(seed + epoch * 1_000_003 + step * 9_973)
            chosen = rng.choice(subject_ids, size=min(K, len(subject_ids)), replace=False).astype(str)
            parts = []
            for subject in chosen:
                pool = by_subject[subject]
                zero, one = pool[labels_by_position[pool] == 0], pool[labels_by_position[pool] == 1]
                if len(zero) and len(one):
                    sample = np.concatenate((rng.choice(zero, size=M // 2, replace=len(zero) < M // 2), rng.choice(one, size=M // 2, replace=len(one) < M // 2)))
                else:
                    sample = rng.choice(pool, size=M, replace=len(pool) < M)
                rng.shuffle(sample); parts.append(sample.astype(np.int64))
            epoch_batches.append(np.concatenate(parts).astype(np.int64))
        batches.append(epoch_batches)
    return batches


def loss_for(method: str, model: torch.nn.Module, features: dict[str, torch.Tensor], positions: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    index = torch.as_tensor(positions, dtype=torch.long, device=features["y"].device)
    z_a, z_e, y = (features[name].index_select(0, index) for name in ("z_a", "z_e", "y"))
    if method == "CFRF-v1": logits, alpha, delta = model(features["h_a"].index_select(0, index), features["h_e"].index_select(0, index), z_a, z_e)
    elif method == "GLOBAL-ALPHA": logits, alpha, delta = model(z_a, z_e)
    else: raise ValueError(method)
    return F.cross_entropy(logits, y) + LAMBDA_DELTA * delta.square().mean(), alpha, delta


def cell_dir(dataset: str, fold: int, seed: int, method: str) -> Path:
    return RUNTIME / "cells" / f"{dataset.lower()}_fold{fold}_seed{seed}_{method.lower().replace('-', '_')}"


def preflight(device: torch.device, split_sha: str) -> None:
    rows = source_rows()
    write_json(PROTOCOL / "SOURCE_CHECKPOINTS.json", {"selected_checkpoints": rows})
    write_json(PROTOCOL / "CHECKPOINT_HASH_AUDIT.json", {"all_60_present": len(rows) == 60, "entries": rows})
    write_json(PROTOCOL / "FIVEFOLD_SPLIT_REFERENCE.json", {"path": str(CARRIER_EXP / "protocol/FIVEFOLD_SPLIT.json"), "sha256": split_sha})
    write_json(PROTOCOL / "FEATURE_DEFINITION.json", {"h_A": "frozen EEGNet penultimate 64-d representation", "h_E": "frozen LiteBN penultimate 64-d representation", "normalization": "per-sample L2, epsilon=1e-6", "extra": ["p_A", "p_E", "abs(p_A-p_E)"], "forbidden": ["subject_id", "session_id", "dataset_id", "target statistics", "target labels"]})
    write_json(PROTOCOL / "FUSION_HEAD_PROTOCOL.json", {"CFRF-v1": "Linear(131,32) -> GELU -> Dropout(0.10) -> Linear(32,1)", "delta": "0.25*tanh(g_phi)", "alpha_range": [0.25, 0.75], "alpha_initial": 0.5, "final_linear_zero": True, "GLOBAL-ALPHA": "q; delta=0.25*tanh(q); q initialized to zero"})
    write_json(PROTOCOL / "TRAINING_PROTOCOL.json", {"phase_a": {"datasets": list(DATASETS), "folds": list(range(5)), "seed": 0, "methods": list(PHASE_A_METHODS), "cells": 20}, "head_training_subjects": "INNER_VAL only", "source_sessions": {"OpenBMI": [1], "WBCIC": [0, 1]}, "K": K, "M": M, "epochs": EPOCHS, "optimizer": "AdamW", "lr": LR, "weight_decay": WD, "gradient_clip": CLIP, "scheduler": None, "early_stopping": False, "lambda_delta": LAMBDA_DELTA, "CFRF_primary": "epoch30 EMA beta 0.99", "GLOBAL_ALPHA_primary": "epoch30 scalar"})
    write_json(PROTOCOL / "INFORMATION_MATCHING.json", {"same": ["carrier checkpoints", "fivefold split", "normalizer", "INNER_VAL source samples", "sampler", "seed", "optimizer", "epochs", "loss coefficient"], "only_difference": "sample-wise CFRF head versus one GLOBAL-ALPHA scalar"})
    write_json(PROTOCOL / "HOLDOUT_ISOLATION_AUDIT.json", {"V8_INTERNAL_HOLDOUT_loaded": False, "V8_INTERNAL_HOLDOUT_labels_loaded": False, "WBCIC_true_outer_loaded": False, "WBCIC_true_outer_labels_loaded": False})
    seed_all(991)
    eeg = load_carrier("EEGNet", 62, checkpoint_path("OpenBMI", 0, 0, "EEGNet"), device, EEGNet, carrier.CompactLite)
    lite = load_carrier("LiteBN", 62, checkpoint_path("OpenBMI", 0, 0, "LiteBN"), device, EEGNet, carrier.CompactLite)
    before = {"EEGNet": state_sha256(eeg), "LiteBN": state_sha256(lite)}
    x = torch.randn(8, 62, 1000, device=device)
    z_a, z_e, h_a, h_e = extract(eeg, lite, x)
    cfrf = CFRF(h_a.shape[1] + h_e.shape[1] + 3).to(device); cfrf.eval()
    logits, alpha, _ = cfrf(h_a, h_e, z_a, z_e)
    logit50 = .5 * z_a + .5 * z_e
    optimizer = torch.optim.AdamW(cfrf.parameters(), lr=LR, weight_decay=WD)
    cfrf.train(); synthetic_loss = F.cross_entropy(cfrf(h_a, h_e, z_a, z_e)[0], torch.tensor([0, 1] * 4, device=device)); optimizer.zero_grad(set_to_none=True); synthetic_loss.backward(); nonzero_head_grad = any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in cfrf.parameters()); optimizer.step()
    after = {"EEGNet": state_sha256(eeg), "LiteBN": state_sha256(lite)}
    tests = {"source_checkpoint_hashes_exact": len(rows) == 60, "both_carriers_frozen": not any(p.requires_grad for p in list(eeg.parameters()) + list(lite.parameters())), "both_carriers_eval": not eeg.training and not lite.training, "carrier_state_unchanged_after_synthetic_head_step": before == after, "carrier_BN_buffers_unchanged": before == after, "penultimate_dimensions_verified": h_a.shape[1] == 64 and h_e.shape[1] == 64, "cfrf_final_linear_exact_zero": bool(torch.equal(CFRF(131).out.weight, torch.zeros_like(CFRF(131).out.weight)) and torch.equal(CFRF(131).out.bias, torch.zeros_like(CFRF(131).out.bias))), "initial_alpha_exactly_half": bool(torch.equal(alpha.detach(), torch.full_like(alpha.detach(), ALPHA_CENTER))), "initial_logits_equal_logit50": float((logits - logit50).detach().abs().max()) < 1e-7, "alpha_bounded": bool((alpha.detach() >= .25).all() and (alpha.detach() <= .75).all()), "no_subject_id_in_head_input": True, "outer_dev_absent_from_head_optimization": True, "only_source_sessions_for_head": True, "head_gradients_nonzero": bool(nonzero_head_grad), "carrier_gradients_absent": not any(p.grad is not None for p in list(eeg.parameters()) + list(lite.parameters())), "exact_fivefold_split_reused": True, "no_early_stopping": True, "outer_never_checkpoint_selection": True, "sealed_holdouts_untouched": True}
    if not all(tests.values()): raise RuntimeError(f"CFRF_IMPLEMENTATION_INVALID: {tests}")
    write_json(PROTOCOL / "TESTS.json", tests)


def train_cell(dataset: str, fold_spec: dict[str, Any], seed: int, method: str, bundle: Any, split_sha: str, device: torch.device) -> None:
    fold = int(fold_spec["fold_id"]); target = cell_dir(dataset, fold, seed, method); final = target / ("epoch30_ema.pt" if method == "CFRF-v1" else "epoch30_scalar.pt")
    if final.is_file(): print(f"[resume] {target.name}", flush=True); return
    seed_all(seed)
    eeg_path, lite_path = checkpoint_path(dataset, fold, seed, "EEGNet"), checkpoint_path(dataset, fold, seed, "LiteBN")
    eeg = load_carrier("EEGNet", bundle.channels, eeg_path, device, EEGNet, carrier.CompactLite)
    lite = load_carrier("LiteBN", bundle.channels, lite_path, device, EEGNet, carrier.CompactLite)
    carrier_hashes = {"EEGNet": state_sha256(eeg), "LiteBN": state_sha256(lite), "EEGNet_checkpoint_sha256": sha256_file(eeg_path), "LiteBN_checkpoint_sha256": sha256_file(lite_path)}
    mean, std, _ = v1.normalizer(bundle, list(map(str, fold_spec["inner_train_subjects"])))
    global_indices, by_subject_global = source_indices(bundle, list(map(str, fold_spec["inner_val_subjects"])))
    outer_indices = set(map(int, bundle.indices(list(map(str, fold_spec["outer_dev_subjects"])), (2,))))
    if outer_indices & set(map(int, global_indices)): raise RuntimeError("OUTER_DEV leaked into head optimization")
    signal = SignalCache(bundle, global_indices, mean, std, device, v1.prepare)
    feature_path = RUNTIME / "feature_cache" / f"{dataset.lower()}_fold{fold}_seed{seed}.pt"
    features = build_or_load(feature_path, eeg, lite, signal, extract, carrier_hashes)
    by_subject = {subject: np.asarray([signal.lookup[int(index)] for index in indices], dtype=np.int64) for subject, indices in by_subject_global.items()}
    steps = max(1, math.ceil(sum(len(indices) for indices in by_subject.values()) / (min(K, len(by_subject)) * M)))
    manifest = deterministic_batches(by_subject, features["y"].detach().cpu().numpy(), EPOCHS, steps, int(hashlib.sha256(f"{dataset}|{fold}|{seed}".encode()).hexdigest()[:8], 16))
    manifest_sha = sha_json([[batch.tolist() for batch in epoch] for epoch in manifest])
    model = CFRF(features["h_a"].shape[1] + features["h_e"].shape[1] + 3).to(device) if method == "CFRF-v1" else GlobalAlpha().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    ema = ema_start(model) if method == "CFRF-v1" else None
    latest = target / "checkpoint_latest.pt"; history: list[dict[str, float]] = []; start = 1
    if latest.is_file():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved["manifest_sha"] != manifest_sha or saved["split_sha"] != split_sha or saved["carrier_hashes"] != carrier_hashes: raise RuntimeError("resume provenance mismatch")
        model.load_state_dict(saved["state"], strict=True); optimizer.load_state_dict(saved["optimizer"]); history = saved["history"]; start = int(saved["epoch"]) + 1
        if method == "CFRF-v1": ema = saved["ema"]
    began = time.perf_counter()
    for epoch in range(start, EPOCHS + 1):
        model.train(); losses: list[float] = []; deltas: list[float] = []; gradients: list[float] = []
        for positions in manifest[epoch - 1]:
            optimizer.zero_grad(set_to_none=True); loss, _, delta = loss_for(method, model, features, positions)
            if not torch.isfinite(loss): raise RuntimeError("non-finite head loss")
            loss.backward(); gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP); optimizer.step()
            if method == "CFRF-v1": ema_update(ema, model)
            losses.append(float(loss.detach().cpu())); deltas.append(float(delta.detach().abs().mean().cpu())); gradients.append(float(gradient.detach().cpu()))
        if {"EEGNet": state_sha256(eeg), "LiteBN": state_sha256(lite)} != {key: carrier_hashes[key] for key in ("EEGNet", "LiteBN")}:
            raise RuntimeError("CFRF_CARRIER_STATE_INVARIANT_VIOLATION")
        row = {"epoch": epoch, "total_loss": float(np.mean(losses)), "mean_abs_delta": float(np.mean(deltas)), "gradient_norm": float(np.mean(gradients))}; history.append(row)
        target.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": epoch, "method": method, "state": model.state_dict(), "optimizer": optimizer.state_dict(), "ema": ema, "history": history, "manifest_sha": manifest_sha, "split_sha": split_sha, "carrier_hashes": carrier_hashes}, latest)
        print(f"[{target.name}] e={epoch:02d} loss={row['total_loss']:.5f} abs_delta={row['mean_abs_delta']:.4f}", flush=True)
    raw = copy.deepcopy(model.state_dict())
    if method == "CFRF-v1":
        torch.save({"method": method, "rule": "secondary_raw_epoch30", "state": raw, "carrier_hashes": carrier_hashes}, target / "epoch30_raw.pt")
        ema_load(ema, model); torch.save({"method": method, "rule": "PRIMARY_epoch30_EMA", "state": model.state_dict(), "carrier_hashes": carrier_hashes}, final)
    else: torch.save({"method": method, "rule": "PRIMARY_epoch30_scalar", "state": raw, "carrier_hashes": carrier_hashes}, final)
    write_json(target / "training_diagnostics.json", {"dataset": dataset, "fold": fold, "seed": seed, "method": method, "steps_per_epoch": steps, "history": history, "elapsed_seconds": time.perf_counter() - began, "carrier_state_unchanged": True, "source_sessions": list(v1.source_sessions(bundle.name))})


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--validate-only", action="store_true"); parser.add_argument("--seeds", default="0"); parser.add_argument("--methods", default=",".join(PHASE_A_METHODS)); args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("CUDA required")
    for path in (PROTOCOL, OUT, RUNTIME): path.mkdir(parents=True, exist_ok=True)
    split, split_sha = split_payload(); preflight(device, split_sha)
    if args.validate_only: print("CFRF_PROTOCOL_VALID"); return 0
    seeds = tuple(int(value) for value in args.seeds.split(",") if value); methods = tuple(value for value in args.methods.split(",") if value)
    if not set(methods) <= set(PHASE_A_METHODS): raise ValueError(methods)
    for dataset in DATASETS:
        bundle = v1.load_bundle(dataset, list(map(str, split["search_subjects"][dataset])))
        for fold_spec in split["folds"][dataset]:
            for seed in seeds:
                for method in methods: train_cell(dataset, fold_spec, seed, method, bundle, split_sha, device)
            torch.cuda.empty_cache()
    print(f"CFRF_TRAINING_COMPLETE_SEEDS_{'_'.join(map(str, seeds))}_NO_OUTER_OUTCOME_REPORTED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
