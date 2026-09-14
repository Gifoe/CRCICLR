#!/usr/bin/env python3
"""CHF Stage A: F0 reuse plus cached-transform F1/F2 seed-0 five-fold screen."""
from __future__ import annotations

import copy
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from litebn_chf import LiteBNCHF

REPO = Path(os.environ.get("CHF_REPO", "/root/rivermind-data/CRCICLR_CHF_WORK")).resolve()
EXP = REPO / "experiments/persist_eeg_litebn_complex_harmonic_former_v1"
OUT, PROTOCOL, RUNTIME = EXP / "outputs", EXP / "protocol", EXP / "runtime"
BASE_RUNNER = REPO / "experiments/persist_eeg_linux_w0r0_screen_v1/code/run_w0r0_seed0.py"
F0_EXP = Path("/root/rivermind-data/CRCICLR_TFF_WORK/experiments/persist_eeg_litebn_tfformer_v1")
TASK, FS, EPOCHS, SEED = "OpenBMI_SSVEP", 250, 60, 0
VARIANTS = ("F1", "F2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


def seed_everything(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def load_base():
    os.environ["W0R0_REPO"] = str(REPO)
    spec = importlib.util.spec_from_file_location("chf_base_runner", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load base runner")
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    base, _ = runner.load_modules()
    base.RUNTIME = RUNTIME
    return base, runner


base, runner = load_base()


def bn_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    result = {}
    for module_name, module in model.base.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            for key, value in module.state_dict().items():
                result[f"{module_name}.{key}"] = value.detach().cpu().clone()
    return result


def make_model(fold_id: int, variant: str, device: torch.device) -> tuple[LiteBNCHF, Path]:
    checkpoint = runner.baseline_path(TASK, fold_id)
    backbone = base.build_model("LiteBN_BASELINE", TASK)
    backbone.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False), strict=True)
    seed_everything(SEED)
    model = LiteBNCHF(backbone, fs=FS, harmonic=variant == "F2").to(device)
    return model, checkpoint


class DeterministicTransformCache:
    """GPU cache for fold-normalized, parameter-free STFT/rFFT transforms."""
    def __init__(self, raw, mean: np.ndarray, std: np.ndarray, probe: LiteBNCHF, device: torch.device):
        self.raw, self.device, self.n = raw, device, int(raw.x.shape[0])
        self.mean, self.std = mean, std
        self.local: list[torch.Tensor] | None = None
        self.fine: torch.Tensor | None = None
        self.frequency: torch.Tensor | None = None
        self.n_fft: int | None = None
        started = time.perf_counter()
        with torch.no_grad():
            for start in range(0, self.n, 32):
                indices = np.arange(start, min(start + 32, self.n), dtype=np.int64)
                value, _ = raw.batch(indices, mean, std)
                local = probe.spectral.transform(value)
                fine, frequency, n_fft = probe.fine.transform(value)
                if self.local is None:
                    self.local = [torch.empty((self.n, *part.shape[1:]), dtype=part.dtype, device=device) for part in local]
                    self.fine = torch.empty((self.n, *fine.shape[1:]), dtype=fine.dtype, device=device)
                    self.frequency, self.n_fft = frequency, int(n_fft)
                for destination, part in zip(self.local, local):
                    destination[start:start + len(indices)].copy_(part)
                self.fine[start:start + len(indices)].copy_(fine)
        assert self.local is not None and self.fine is not None and self.frequency is not None and self.n_fft is not None
        self.elapsed = time.perf_counter() - started
        self.bytes = sum(x.numel() * x.element_size() for x in self.local) + self.fine.numel() * self.fine.element_size()

    def batch(self, indices: np.ndarray):
        value, labels = self.raw.batch(indices, self.mean, self.std)
        index = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        local = [part.index_select(0, index) for part in self.local]
        fine = (self.fine.index_select(0, index), self.frequency, self.n_fft)
        return value, labels, (local, fine)


def subject_equal_ba(model, bundle, cache: DeterministicTransformCache, subjects: list[str]) -> tuple[float, dict[str, float]]:
    model.eval()
    rows: dict[str, float] = {}
    with torch.no_grad():
        for subject in subjects:
            indices = bundle.indices([subject], (int(base.TASKS[TASK]["future_session"]),))
            logits = []
            for start in range(0, len(indices), 128):
                value, _, transformed = cache.batch(indices[start:start + 128])
                logits.append(model(value, transformed)[0].float().cpu().numpy())
            labels = bundle.labels(indices)
            rows[str(subject)] = float(base.classification_metrics(labels, np.concatenate(logits))["BA"])
    return float(np.mean(list(rows.values()))), rows


def parameter_groups(model: LiteBNCHF):
    new, backend, stem = [], [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if not name.startswith("base."):
            new.append(parameter)
        elif name.startswith(("base.temporal.", "base.temporal_norm.", "base.spatial.", "base.spatial_norm.")):
            stem.append(parameter)
        else:
            backend.append(parameter)
    ids = [id(p) for group in (new, backend, stem) for p in group]
    if len(ids) != len(set(ids)) or len(ids) != sum(p.requires_grad for p in model.parameters()):
        raise RuntimeError("parameter groups are not a disjoint exhaustive partition")
    return new, backend, stem


def train_variant(variant: str, fold: dict[str, Any], bundle, cache, device, normalizer_sha: str) -> dict[str, Any]:
    fold_id = int(fold["fold_id"])
    directory = RUNTIME / "checkpoints" / TASK / f"fold{fold_id}_{variant.lower()}"
    latest, selected = directory / "checkpoint_latest.pt", directory / "selected_best.pt"
    model, baseline_checkpoint = make_model(fold_id, variant, device)
    new, backend, stem = parameter_groups(model)
    invariants = {
        "task": TASK, "fold": fold_id, "variant": variant, "seed": SEED,
        "effective_fs": FS, "epochs": EPOCHS, "normalizer_sha256": normalizer_sha,
        "baseline_checkpoint_sha256": sha256(baseline_checkpoint),
        "source_sha256": sha256(Path(__file__).with_name("litebn_chf.py")),
        "label_smoothing": 0.05,
    }
    optimizer = torch.optim.AdamW([
        {"params": new, "lr": 3e-4, "name": "new_modules"},
        {"params": backend, "lr": 5e-5, "name": "LiteBN_backend_readout"},
        {"params": stem, "lr": 1e-5, "name": "LiteBN_stem"},
    ], weight_decay=5e-4)
    start, history, best, best_epoch, best_state = 1, [], -float("inf"), None, None
    initial_bn = bn_snapshot(model)
    first_gradients: dict[str, float] = {}
    if latest.is_file():
        saved = torch.load(latest, map_location=device, weights_only=False)
        if saved["invariants"] != invariants:
            raise RuntimeError(f"resume invariant mismatch: {latest}")
        model.load_state_dict(saved["current_state"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"])
        start, history = int(saved["epoch"]) + 1, saved["history"]
        best, best_epoch, best_state = float(saved["best"]), saved["best_epoch"], saved["best_state"]
        first_gradients = saved.get("first_gradients", {})
    train_indices = bundle.indices(fold["inner_train_subjects"], base.TASKS[TASK]["source_sessions"])
    started = time.perf_counter()
    for epoch in range(start, EPOCHS + 1):
        model.train()
        factor = epoch / 5 if epoch <= 5 else 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (epoch - 5) / 55))
        for group, initial_lr in zip(optimizer.param_groups, (3e-4, 5e-5, 1e-5)):
            group["lr"] = initial_lr * factor
        losses, gradients = [], []
        batches = base.task_epoch_batches(train_indices, TASK, fold_id, epoch)
        for batch_number, indices in enumerate(batches):
            value, labels, transformed = cache.batch(np.asarray(indices, dtype=np.int64))
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(value, transformed)
            loss = F.cross_entropy(logits, labels, label_smoothing=0.05)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss: {variant}/fold{fold_id}/epoch{epoch}")
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(new + backend + stem, 5.0))
            if epoch == 1 and batch_number == 0:
                for name in ("fine.wr", "finecross.q.weight", "spectral.proj.0.weight", "cross.q.weight"):
                    parameter = dict(model.named_parameters())[name]
                    first_gradients[name] = 0.0 if parameter.grad is None else float(parameter.grad.detach().norm())
                if variant == "F2":
                    first_gradients["fine.h2.weight"] = float(model.fine.h2.weight.grad.detach().norm())
            optimizer.step()
            losses.append(float(loss.detach()))
            gradients.append(grad_norm)
        validation_ba, subject_rows = subject_equal_ba(model, bundle, cache, fold["inner_val_subjects"])
        chose = validation_ba > best + 1e-12
        if chose:
            best, best_epoch = validation_ba, epoch
            best_state = copy.deepcopy(model.state_dict())
        history.append({
            "epoch": epoch, "train_loss": float(np.mean(losses)), "inner_val_BA": validation_ba,
            "inner_val_subject_BA": subject_rows, "selected": chose,
            "lr_N": optimizer.param_groups[0]["lr"], "lr_B": optimizer.param_groups[1]["lr"],
            "lr_S": optimizer.param_groups[2]["lr"], "grad_norm": float(np.mean(gradients)),
        })
        if epoch == 1 or epoch % 5 == 0 or chose:
            print(f"CHF {variant} fold={fold_id} epoch={epoch:02d} loss={np.mean(losses):.4f} valBA={validation_ba:.6f}", flush=True)
        if epoch % 5 == 0 or epoch == EPOCHS:
            directory.mkdir(parents=True, exist_ok=True)
            tmp = latest.with_suffix(".pt.part")
            torch.save({
                "epoch": epoch, "history": history, "best": best, "best_epoch": best_epoch,
                "best_state": best_state, "current_state": model.state_dict(), "optimizer": optimizer.state_dict(),
                "rng": rng_state(), "invariants": invariants, "first_gradients": first_gradients,
            }, tmp)
            os.replace(tmp, latest)
    if best_state is None or best_epoch is None:
        raise RuntimeError("no selected checkpoint")
    model.load_state_dict(best_state, strict=True)
    final_ba, final_subjects = subject_equal_ba(model, bundle, cache, fold["inner_val_subjects"])
    if abs(final_ba - best) > 1e-12:
        raise RuntimeError("selected checkpoint replay mismatch")
    final_bn = bn_snapshot(model)
    bn_unchanged = set(initial_bn) == set(final_bn) and all(torch.equal(initial_bn[k], final_bn[k]) for k in initial_bn)
    if not bn_unchanged:
        raise RuntimeError("historical BatchNorm state changed")
    tmp = selected.with_suffix(".pt.part")
    torch.save({"state_dict": model.state_dict(), "epoch": best_epoch, "invariants": invariants}, tmp)
    os.replace(tmp, selected)
    return {
        "task": TASK, "variant": variant, "fold": fold_id, "seed": SEED,
        "selected_epoch": best_epoch, "inner_val_BA": best, "inner_val_subject_BA": final_subjects,
        "checkpoint": str(selected), "checkpoint_sha256": sha256(selected),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "first_gradients": first_gradients, "BN_eval_locked_and_unchanged": bn_unchanged,
        "history": history, "elapsed_seconds": time.perf_counter() - started,
    }


def load_f0() -> tuple[dict[int, dict[str, float]], dict[int, float], list[dict[str, Any]]]:
    selection_path = F0_EXP / "outputs/CHECKPOINT_SELECTION.csv"
    audit_path = F0_EXP / "outputs/INITIALIZATION_AUDIT.csv"
    selection, audit = pd.read_csv(selection_path), pd.read_csv(audit_path)
    selection = selection[selection.task == TASK].sort_values("fold")
    audit = audit[audit.task == TASK].sort_values("fold")
    if list(selection.fold) != list(range(5)) or list(audit.fold) != list(range(5)):
        raise RuntimeError("F0 provenance does not contain exactly five matching folds")
    if selection.B0_fallback.astype(str).str.lower().isin(("true", "1")).any():
        raise RuntimeError("F0 provenance contains a B0 fallback fold")
    f0, litebn, provenance = {}, {}, []
    for row, arow in zip(selection.itertuples(), audit.itertuples()):
        checkpoint = Path(row.checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        f0[int(row.fold)] = {"BA": float(row.inner_val_BA), "epoch": int(row.selected_epoch)}
        litebn[int(row.fold)] = float(arow.initial_B0_BA)
        provenance.append({
            "fold": int(row.fold), "F0_inner_val_BA": float(row.inner_val_BA),
            "F0_selected_epoch": int(row.selected_epoch), "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint), "normalizer_sha256": str(arow.normalizer_sha256),
            "source_selection_csv": str(selection_path), "source_audit_csv": str(audit_path),
        })
    return f0, litebn, provenance


def preflight(device: torch.device, folds: list[dict[str, Any]], f0_provenance: list[dict[str, Any]]) -> None:
    for path in (OUT, PROTOCOL, RUNTIME):
        path.mkdir(parents=True, exist_ok=True)
    atomic_csv(OUT / "SAMPLING_RATE_AUDIT.csv", [{
        "task": TASK, "effective_fs": FS,
        "source": "frozen runtime convention inherited from LiteBN-TFFormer v1",
        "metadata_file": "not required",
    }])
    atomic_csv(OUT / "F0_REUSE_PROVENANCE.csv", f0_provenance)
    probe, _ = make_model(0, "F2", device)
    probe.train()
    value = torch.zeros(2, int(base.TASKS[TASK]["channels"]), int(base.TASKS[TASK]["samples"]), device=device)
    logits, representation = probe(value)
    n_fft, delta_f, frequency = probe.fine.grid
    if logits.shape != (2, int(base.TASKS[TASK]["classes"])) or representation.shape != (2, 64):
        raise RuntimeError("CHF output shape mismatch")
    if n_fft != 1024 or abs(delta_f - 250 / 1024) > 1e-12 or len(frequency) != 180:
        raise RuntimeError("fine spectrum grid mismatch")
    if any(module.training for module in probe.base.modules() if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)):
        raise RuntimeError("historical BatchNorm is not eval-locked")
    atomic_json(PROTOCOL / "STAGE_A_PREFLIGHT.json", {
        "pass": True, "task": TASK, "seed": SEED, "folds": len(folds), "effective_fs": FS,
        "n_fft": n_fft, "delta_f_hz": delta_f, "retained_native_bins": len(frequency),
        "frequency_min_hz": float(frequency.min()), "frequency_max_hz": float(frequency.max()),
        "F0_reused": True, "F0_provenance_exact": True,
        "inner_train_validation_only": True, "internal_heldout_accessed": False,
    })
    del probe, value
    torch.cuda.empty_cache()


def finalize(records: dict[tuple[str, int], dict[str, Any]], f0, litebn) -> dict[str, Any]:
    fold_rows = []
    for fold in range(5):
        for variant in ("F0", "F1", "F2"):
            if variant == "F0":
                ba, epoch = f0[fold]["BA"], f0[fold]["epoch"]
            else:
                row = records[(variant, fold)]
                ba, epoch = float(row["inner_val_BA"]), int(row["selected_epoch"])
            fold_rows.append({
                "task": TASK, "fold": fold, "variant": variant, "seed": SEED,
                "selected_epoch": epoch, "inner_val_BA": ba,
                "matched_LiteBN_BA": litebn[fold], "delta_vs_F0_pp": 100 * (ba - f0[fold]["BA"]),
                "delta_vs_matched_LiteBN_pp": 100 * (ba - litebn[fold]),
            })
    summary = []
    for variant in ("F0", "F1", "F2"):
        rows = [x for x in fold_rows if x["variant"] == variant]
        summary.append({
            "task": TASK, "variant": variant,
            "mean_inner_val_BA": float(np.mean([x["inner_val_BA"] for x in rows])),
            "mean_delta_vs_F0_pp": float(np.mean([x["delta_vs_F0_pp"] for x in rows])),
            "mean_delta_vs_matched_LiteBN_pp": float(np.mean([x["delta_vs_matched_LiteBN_pp"] for x in rows])),
            "nonnegative_folds_vs_F0": int(sum(x["delta_vs_F0_pp"] >= -1e-12 for x in rows)),
        })
    candidates = [x for x in summary if x["variant"] in VARIANTS]
    selected = sorted(candidates, key=lambda x: (-x["mean_inner_val_BA"], x["variant"]))[0]
    passed = selected["mean_delta_vs_F0_pp"] >= 0.20 - 1e-12 and selected["nonnegative_folds_vs_F0"] >= 3
    decision = {
        "stage": "A", "selected_architecture": selected["variant"],
        "selected_mean_inner_val_BA": selected["mean_inner_val_BA"],
        "delta_vs_F0_pp": selected["mean_delta_vs_F0_pp"],
        "nonnegative_folds_vs_F0": selected["nonnegative_folds_vs_F0"],
        "required_delta_pp": 0.20, "required_nonnegative_folds": 3,
        "PASS": passed, "next": "STAGE_B_LOSS_SCREEN" if passed else "STOP_CHF_DIRECTION",
        "internal_heldout_accessed": False,
    }
    atomic_csv(OUT / "STAGE_A_FOLD_RESULTS.csv", fold_rows)
    atomic_csv(OUT / "STAGE_A_SUMMARY.csv", summary)
    atomic_json(OUT / "STAGE_A_DECISION.json", decision)
    return decision


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    _, all_folds, split_hash = base.load_folds()
    folds = all_folds["OpenBMI"]
    f0, litebn, f0_provenance = load_f0()
    preflight(device, folds, f0_provenance)
    atomic_json(PROTOCOL / "DATA_SCOPE_AUDIT.json", {
        "stage": "A", "task": TASK, "allowed_scope": "canonical inner-train and inner-validation only",
        "fivefold_split_sha256": split_hash, "internal_heldout_accessed": False,
        "outer_development_accessed": False,
    })
    records_path = RUNTIME / "STAGE_A_TRAINING_LOGS.json"
    previous = json.loads(records_path.read_text(encoding="utf-8")) if records_path.is_file() else []
    records = {(x["variant"], int(x["fold"])): x for x in previous}
    cache_audit = []
    for fold in folds:
        fold_id = int(fold["fold_id"])
        missing = [variant for variant in VARIANTS if (variant, fold_id) not in records]
        if not missing:
            continue
        allowed = fold["inner_train_subjects"] + fold["inner_val_subjects"]
        bundle = base.build_bundle(TASK, allowed)
        normalizer_path = runner.normalizer_source(TASK, fold_id)
        mean, std, metadata = base.load_tensor_pair(normalizer_path)
        if metadata["mean_std_sha256"] != f0_provenance[fold_id]["normalizer_sha256"]:
            raise RuntimeError(f"F0 normalizer mismatch in fold {fold_id}")
        raw = base.RawGPUCache(bundle, device)
        probe, _ = make_model(fold_id, "F1", device)
        torch.cuda.reset_peak_memory_stats()
        transformed = DeterministicTransformCache(raw, mean, std, probe, device)
        cache_audit.append({
            "task": TASK, "fold": fold_id, "trials": transformed.n,
            "cache_bytes": transformed.bytes, "cache_GiB": transformed.bytes / 2**30,
            "build_seconds": transformed.elapsed, "dtype": "float32/complex64",
            "normalizer_sha256": metadata["mean_std_sha256"],
            "cached_operations": "parameter-free local STFT log-magnitude and full-trial Hann rFFT",
            "learned_operations_cached": False,
        })
        atomic_csv(OUT / "DETERMINISTIC_TRANSFORM_CACHE_AUDIT.csv", cache_audit)
        del probe
        torch.cuda.empty_cache()
        print(f"CHF_CACHE fold={fold_id} trials={transformed.n} GiB={transformed.bytes/2**30:.3f} seconds={transformed.elapsed:.1f}", flush=True)
        for variant in missing:
            result = train_variant(variant, fold, bundle, transformed, device, metadata["mean_std_sha256"])
            records[(variant, fold_id)] = result
            atomic_json(records_path, list(records.values()))
            print(f"CHF_FOLD_DONE {variant} fold={fold_id} BA={result['inner_val_BA']:.6f} epoch={result['selected_epoch']} seconds={result['elapsed_seconds']:.1f}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
        del transformed, raw, bundle
        gc.collect()
        torch.cuda.empty_cache()
    if len(records) != 10:
        raise RuntimeError(f"incomplete Stage A grid: {len(records)}/10")
    decision = finalize(records, f0, litebn)
    print("CHF_STAGE_A_DONE " + json.dumps(decision, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
