#!/usr/bin/env python3
"""Stage-0 gradient-transfer audit in the exact LiteBN C/M parameter space."""
from __future__ import annotations

import copy
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

REPO = Path(os.environ.get("CM_GA_REPO", "/root/rivermind-data/CRCICLR_CM_GA_WORK")).resolve()
EXP = REPO / "experiments/persist_eeg_litebn_cm_ga_final_candidate_v1"
CODE = EXP / "code"
OUT = EXP / "outputs"
PROTOCOL = EXP / "protocol"
RUNTIME = Path(os.environ.get("CM_GA_RUNTIME", "/root/rivermind-data/litebn_cm_ga_final_candidate_runtime")).resolve()
BASE_RUNNER = REPO / "experiments/persist_eeg_linux_w0r0_screen_v1/code/run_w0r0_seed0.py"
SEED = 0
PAIR_ROUNDS = 2
TRIALS_PER_CLASS = 8
BOOTSTRAPS = 10_000
LR = 3e-4
WEIGHT_DECAY = 5e-4
CLIP = 5.0
EPS = 1e-12


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


os.environ["W0R0_REPO"] = str(REPO)
runner = load_module("cm_ga_base_runner", BASE_RUNNER)
base, _unused_w0r0 = runner.load_modules()
models = load_module("cm_ga_models", CODE / "models_cm.py")


def stable_seed(*parts: object) -> int:
    return int.from_bytes(
        hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:4], "little"
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    temporary = path.with_suffix(path.suffix + ".part")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def subject_sort(values: Iterable[str], dataset: str) -> list[str]:
    return base.subject_sort(values, dataset)


def build_source_bundle(task: str, subjects: Iterable[str]):
    spec = base.TASKS[task]
    dataset = spec["dataset"]
    canonical = subject_sort(subjects, dataset)
    rows = []
    if dataset == "OpenBMI":
        for subject in canonical:
            for session in spec["source_sessions"]:
                signal = base.openbmi_path(task, subject, session, "signal")
                labels = base.openbmi_path(task, subject, session, "label")
                x = np.load(signal, mmap_mode="r", allow_pickle=False)
                y = np.load(labels, mmap_mode="r", allow_pickle=False)
                for index, code in enumerate(y):
                    rows.append(base.Row(str(subject), int(session), str(signal), index, int(spec["raw_codes"][int(code)])))
    else:
        root = base.CACHE / "wbcic/wbcic_epochs"
        for subject in canonical:
            for session in spec["source_sessions"]:
                signal = root / subject / f"ses-{session}_epochs.npy"
                labels = root / subject / f"ses-{session}_labels.npy"
                x = np.load(signal, mmap_mode="r", allow_pickle=False)
                y = np.load(labels, mmap_mode="r", allow_pickle=False)
                for index, code in enumerate(y):
                    rows.append(base.Row(str(subject), int(session), str(signal), index, int(code)))
    if not rows:
        raise RuntimeError(f"empty source bundle: {task}")
    return base.SignalBundle(task, canonical, rows)


def make_model(task: str, fold: int, device: torch.device):
    set_seed(SEED)
    b0 = base.build_model("LiteBN_BASELINE", task)
    checkpoint = runner.baseline_path(task, fold)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    b0.load_state_dict(state, strict=True)
    set_seed(SEED)
    model = models.LiteBNCMGA(b0).to(device)
    model.base.eval()
    return model, checkpoint


def metric(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    prediction = logits.argmax(axis=1)
    return {
        "BA": float(balanced_accuracy_score(labels, prediction)),
        "macro_F1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, prediction)),
    }


def balanced_indices(bundle, subjects: list[str], per_class: int = 32) -> np.ndarray:
    subject = subjects[0]
    candidates = bundle.indices([subject], base.TASKS[bundle.task]["source_sessions"])
    labels = bundle.labels(candidates)
    selected = []
    for cls in range(bundle.classes):
        pool = candidates[labels == cls]
        selected.extend(pool[: min(per_class, len(pool))].tolist())
    return np.asarray(selected, dtype=np.int64)


def identity_preflight(device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    _search, folds, split_hash = base.load_folds()
    identity_rows = []
    frozen_rows = []
    for task in base.TASK_ORDER:
        dataset = base.TASKS[task]["dataset"]
        for fold in folds[dataset]:
            fold_id = int(fold["fold_id"])
            model, checkpoint = make_model(task, fold_id, device)
            frozen_before = models.frozen_base_sha256(model)
            bundle = build_source_bundle(task, fold["inner_train_subjects"])
            mean, std, metadata = base.load_tensor_pair(runner.normalizer_source(task, fold_id))
            cache = base.RawGPUCache(bundle, device)
            indices = balanced_indices(bundle, fold["inner_train_subjects"])
            value, _ = cache.batch(indices, mean, std)
            labels = bundle.labels(indices)
            model.eval()
            with torch.inference_mode():
                base_logits, _ = model.base(value)
                cm_logits, _ = model(value)
            base_np = base_logits.float().cpu().numpy()
            cm_np = cm_logits.float().cpu().numpy()
            maximum = float(np.max(np.abs(base_np - cm_np)))
            b_metrics, c_metrics = metric(labels, base_np), metric(labels, cm_np)
            exact_predictions = bool(np.array_equal(base_np.argmax(1), cm_np.argmax(1)))
            metrics_equal = bool(b_metrics == c_metrics)
            lambda_zero = float(model.lambda_channel.detach().cpu()) == 0.0
            gamma_zero = all(float(block.gamma.detach().cpu()) == 0.0 for block in model.mixers)
            passed = bool(maximum <= 1e-7 and exact_predictions and metrics_equal and lambda_zero and gamma_zero)
            identity_rows.append({
                "task": task, "dataset": dataset, "fold": fold_id, "seed": 0,
                "checkpoint_path": str(checkpoint), "checkpoint_sha256": runner.sha256(checkpoint),
                "normalizer_sha256": metadata["mean_std_sha256"], "audited_trials": len(indices),
                "max_abs_logit_difference": maximum, "exact_predictions": exact_predictions,
                "BA_equal": b_metrics["BA"] == c_metrics["BA"],
                "macro_F1_equal": b_metrics["macro_F1"] == c_metrics["macro_F1"],
                "accuracy_equal": b_metrics["accuracy"] == c_metrics["accuracy"],
                "lambda_channel_zero": lambda_zero, "all_mixer_gamma_zero": gamma_zero,
                "pass": passed,
            })
            frozen_after = models.frozen_base_sha256(model)
            frozen_rows.append({
                "stage": "epoch0_identity", "task": task, "fold": fold_id,
                "base_sha256_before": frozen_before, "base_sha256_after": frozen_after,
                "bitwise_unchanged": frozen_before == frozen_after,
            })
            del model, cache, bundle, value
            gc.collect()
            if device.type == "cuda": torch.cuda.empty_cache()
    identity = pd.DataFrame(identity_rows)
    frozen = pd.DataFrame(frozen_rows)
    write_csv(OUT / "EPOCH0_IDENTITY_AUDIT.csv", identity)
    write_csv(OUT / "FROZEN_BASE_AUDIT.csv", frozen)
    if len(identity) != 20 or not identity["pass"].all() or not frozen["bitwise_unchanged"].all():
        write_json(OUT / "GRADIENT_TRANSFER_DECISION.json", {
            "terminal": "PROTOCOL_FAIL", "reason": "epoch0 identity or frozen-base audit failed",
            "new_sealed_test_accessed": False,
        })
        raise RuntimeError("PROTOCOL_FAIL: identity audit")
    write_json(PROTOCOL / "SOURCE_PROVENANCE.json", {
        "fivefold_split_sha256": split_hash,
        "historical_litebn_source": str(base.CARRIER_CODE / "run_carrier_screen.py"),
        "historical_litebn_source_sha256": base.sha256_file(base.CARRIER_CODE / "run_carrier_screen.py"),
        "historical_gradient_references": [
            "persist_eeg_prospective_gradient_signal_audit_v1",
            "persist_eeg_cross_batch_subject_harm_audit_v1",
        ],
        "candidate_weights_reused_from_gradient_references": False,
    })
    return identity, frozen


def pools(bundle, subjects: list[str]) -> dict[str, dict[int, np.ndarray]]:
    result = {}
    sessions = base.TASKS[bundle.task]["source_sessions"]
    for subject in subjects:
        indices = bundle.indices([subject], sessions)
        labels = bundle.labels(indices)
        result[subject] = {cls: indices[labels == cls] for cls in range(bundle.classes)}
        for cls, values in result[subject].items():
            if len(values) < 2 * TRIALS_PER_CLASS:
                raise RuntimeError(f"insufficient Stage0 trials: {bundle.task}/{subject}/{cls}")
    return result


def block(subject_pools, subject: str, dataset: str, fold: int, round_id: int, role: str) -> np.ndarray:
    selected = []
    for cls, values in subject_pools[subject].items():
        rng = np.random.default_rng(stable_seed("cm-ga-stage0", dataset, fold, round_id, subject, cls))
        order = values[rng.permutation(len(values))]
        if role == "guard": part = order[:TRIALS_PER_CLASS]
        elif role == "future": part = order[TRIALS_PER_CLASS:2 * TRIALS_PER_CLASS]
        elif role == "grad": part = order[:TRIALS_PER_CLASS]
        else: raise ValueError(role)
        selected.extend(part.tolist())
    result = np.asarray(selected, dtype=np.int64)
    if len(set(result.tolist())) != len(result):
        raise RuntimeError("duplicate trial in Stage0 block")
    return result


def flatten(grads, params) -> torch.Tensor:
    return torch.cat([
        (gradient if gradient is not None else torch.zeros_like(parameter)).reshape(-1).float()
        for gradient, parameter in zip(grads, params)
    ])


def chunks(vector: torch.Tensor, params) -> list[torch.Tensor]:
    output, offset = [], 0
    for parameter in params:
        count = parameter.numel()
        output.append(vector[offset:offset + count].reshape_as(parameter))
        offset += count
    if offset != vector.numel(): raise RuntimeError("gradient vector size mismatch")
    return output


def cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second) + EPS
    return float((torch.dot(first, second) / denominator).detach().cpu())


def gradient(model, params, value, labels, rng_seed: int) -> tuple[torch.Tensor, float]:
    model.train(True)
    set_seed(rng_seed)
    logits, _ = model(value)
    loss = F.cross_entropy(logits, labels)
    values = torch.autograd.grad(loss, params, allow_unused=True, create_graph=False)
    vector = flatten(values, params).detach()
    return vector, float(loss.detach().cpu())


def deterministic_loss(model, value, labels) -> float:
    model.eval()
    with torch.inference_mode():
        return float(F.cross_entropy(model(value)[0], labels).detach().cpu())


def candidate_harm(model, params, gradient_a: torch.Tensor, future_x, future_y) -> tuple[float, float, float]:
    before = deterministic_loss(model, future_x, future_y)
    snapshots = [parameter.detach().clone() for parameter in params]
    residual_before = models.tensor_mapping_sha256({str(i): p for i, p in enumerate(params)})
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    norm = float(torch.linalg.vector_norm(gradient_a).detach().cpu())
    clipped = gradient_a * min(1.0, CLIP / max(norm, EPS))
    optimizer.zero_grad(set_to_none=True)
    for parameter, value in zip(params, chunks(clipped, params)):
        parameter.grad = value.detach().clone()
    optimizer.step()
    after = deterministic_loss(model, future_x, future_y)
    with torch.no_grad():
        for parameter, snapshot in zip(params, snapshots): parameter.copy_(snapshot)
    residual_after = models.tensor_mapping_sha256({str(i): p for i, p in enumerate(params)})
    if residual_before != residual_after:
        raise RuntimeError("stored residual model changed during in-memory candidate update")
    return after - before, before, after


def stage0_observations(device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    _search, folds, _split_hash = base.load_folds()
    all_rows = []
    frozen_rows = []
    for task in ("OpenBMI_MI", "WBCIC_MI"):
        dataset = base.TASKS[task]["dataset"]
        for fold in folds[dataset]:
            fold_id = int(fold["fold_id"])
            subjects = subject_sort(fold["inner_train_subjects"], dataset)
            bundle = build_source_bundle(task, subjects)
            mean, std, metadata = base.load_tensor_pair(runner.normalizer_source(task, fold_id))
            cache = base.RawGPUCache(bundle, device)
            subject_pools = pools(bundle, subjects)
            model, checkpoint = make_model(task, fold_id, device)
            params = model.residual_parameters()
            frozen_before = models.frozen_base_sha256(model)
            residual_initial = models.tensor_mapping_sha256({str(i): p for i, p in enumerate(params)})
            for round_id in range(PAIR_ROUNDS):
                for index, subject_b in enumerate(subjects):
                    shift_a = 1 + round_id
                    subject_a = subjects[(index + shift_a) % len(subjects)]
                    if subject_a == subject_b: raise RuntimeError("A/B subject collision")
                    shift_c = shift_a + 1
                    subject_c = subjects[(index + shift_c) % len(subjects)]
                    while subject_c in (subject_a, subject_b):
                        shift_c += 1
                        subject_c = subjects[(index + shift_c) % len(subjects)]
                    a_idx = block(subject_pools, subject_a, dataset, fold_id, round_id, "grad")
                    b_guard_idx = block(subject_pools, subject_b, dataset, fold_id, round_id, "guard")
                    b_future_idx = block(subject_pools, subject_b, dataset, fold_id, round_id, "future")
                    c_idx = block(subject_pools, subject_c, dataset, fold_id, round_id, "guard")
                    if set(b_guard_idx.tolist()) & set(b_future_idx.tolist()):
                        raise RuntimeError("B_guard/B_future overlap")
                    xa, ya = cache.batch(a_idx, mean, std)
                    xb, yb = cache.batch(b_guard_idx, mean, std)
                    xf, yf = cache.batch(b_future_idx, mean, std)
                    xc, yc = cache.batch(c_idx, mean, std)
                    rng_seed = stable_seed("cm-ga-paired-dropout", dataset, fold_id, round_id, subject_b)
                    ga, loss_a = gradient(model, params, xa, ya, rng_seed)
                    gb, loss_b = gradient(model, params, xb, yb, rng_seed)
                    gc_control, _ = gradient(model, params, xc, yc, rng_seed)
                    agreement = cosine(ga, gb)
                    control_agreement = cosine(ga, gc_control)
                    harm, before, after = candidate_harm(model, params, ga, xf, yf)
                    all_rows.append({
                        "task": task, "dataset": dataset, "fold": fold_id, "seed": 0,
                        "round": round_id, "subject_A": subject_a, "subject_B": subject_b,
                        "control_subject": subject_c, "A_B_disjoint": subject_a != subject_b,
                        "B_guard_future_disjoint": True, "trials_per_class": TRIALS_PER_CLASS,
                        "agreement": agreement, "conflict_score": -agreement,
                        "control_agreement": control_agreement,
                        "control_conflict_score": -control_agreement,
                        "future_harm": harm, "harm_label": int(harm > 0.0),
                        "loss_A": loss_a, "loss_B_guard": loss_b,
                        "loss_B_future_before": before, "loss_B_future_after": after,
                        "g_A_norm": float(torch.linalg.vector_norm(ga).cpu()),
                        "g_B_norm": float(torch.linalg.vector_norm(gb).cpu()),
                        "paired_dropout_rng": True,
                        "checkpoint_sha256": runner.sha256(checkpoint),
                        "normalizer_sha256": metadata["mean_std_sha256"],
                    })
                    del xa, ya, xb, yb, xf, yf, xc, yc, ga, gb, gc_control
            residual_final = models.tensor_mapping_sha256({str(i): p for i, p in enumerate(params)})
            frozen_after = models.frozen_base_sha256(model)
            frozen_rows.append({
                "stage": "gradient_transfer", "task": task, "fold": fold_id,
                "base_sha256_before": frozen_before, "base_sha256_after": frozen_after,
                "bitwise_unchanged": frozen_before == frozen_after,
                "residual_sha256_before": residual_initial,
                "residual_sha256_after": residual_final,
                "stored_residual_bitwise_unchanged": residual_initial == residual_final,
            })
            print(f"STAGE0_FOLD_COMPLETE {task} fold={fold_id} observations={PAIR_ROUNDS * len(subjects)}", flush=True)
            del model, cache, bundle
            gc.collect()
            if device.type == "cuda": torch.cuda.empty_cache()
    observations = pd.DataFrame(all_rows)
    frozen = pd.DataFrame(frozen_rows)
    write_csv(OUT / "GRADIENT_TRANSFER_OBSERVATIONS.csv", observations)
    return observations, frozen


def statistics(frame: pd.DataFrame, score: str) -> dict[str, Any]:
    labels = frame.harm_label.to_numpy(int)
    values = frame[score].to_numpy(float)
    harm = frame.future_harm.to_numpy(float)
    auroc = float(roc_auc_score(labels, values)) if len(np.unique(labels)) == 2 else None
    rho = float(spearmanr(values, harm).statistic)
    return {"auroc": auroc, "spearman": rho, "observations": len(frame), "subjects": int(frame.subject_B.nunique()), "harm_rate": float(labels.mean())}


def cluster_bootstrap(frame: pd.DataFrame, dataset: str) -> tuple[dict[str, Any], pd.DataFrame]:
    subjects = sorted(frame.subject_B.unique().tolist(), key=lambda x: int(str(x).replace("sub-", "")))
    groups = {subject: frame[frame.subject_B == subject] for subject in subjects}
    rng = np.random.default_rng(stable_seed("cm-ga-bootstrap", dataset, SEED))
    rows = []
    for draw in range(BOOTSTRAPS):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        sample = pd.concat([groups[subject] for subject in sampled], ignore_index=True)
        real = statistics(sample, "conflict_score")
        control = statistics(sample, "control_conflict_score")
        if real["auroc"] is None or control["auroc"] is None: continue
        rows.append({
            "dataset": dataset, "draw": draw,
            "real_auroc": real["auroc"], "control_auroc": control["auroc"],
            "auroc_advantage": real["auroc"] - control["auroc"],
            "real_spearman": real["spearman"], "control_spearman": control["spearman"],
            "spearman_advantage": real["spearman"] - control["spearman"],
        })
    draws = pd.DataFrame(rows)
    if len(draws) < int(BOOTSTRAPS * 0.95):
        raise RuntimeError(f"too many undefined bootstrap draws: {dataset}/{len(draws)}")
    def interval(column: str) -> list[float]:
        return [float(draws[column].quantile(.025)), float(draws[column].quantile(.975))]
    return {
        "draws_requested": BOOTSTRAPS, "draws_valid": len(draws),
        "real_auroc_ci95": interval("real_auroc"),
        "real_spearman_ci95": interval("real_spearman"),
        "auroc_advantage_ci95": interval("auroc_advantage"),
        "spearman_advantage_ci95": interval("spearman_advantage"),
    }, draws


def summarize(observations: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    summaries, bootstrap_frames = [], []
    decision = {"datasets": {}}
    for dataset in ("OpenBMI", "WBCIC"):
        frame = observations[observations.dataset == dataset].copy()
        real = statistics(frame, "conflict_score")
        control = statistics(frame, "control_conflict_score")
        boot, draws = cluster_bootstrap(frame, dataset)
        bootstrap_frames.append(draws)
        auc_adv = real["auroc"] - control["auroc"]
        rho_adv = real["spearman"] - control["spearman"]
        passed = bool(
            real["auroc"] >= .60 and boot["real_auroc_ci95"][0] > .50
            and real["spearman"] > 0 and boot["real_spearman_ci95"][0] > 0
            and auc_adv > 0 and boot["auroc_advantage_ci95"][0] > 0
            and rho_adv > 0 and boot["spearman_advantage_ci95"][0] > 0
        )
        row = {
            "dataset": dataset, "auroc": real["auroc"],
            "auroc_ci95_lower": boot["real_auroc_ci95"][0], "auroc_ci95_upper": boot["real_auroc_ci95"][1],
            "spearman": real["spearman"], "spearman_ci95_lower": boot["real_spearman_ci95"][0], "spearman_ci95_upper": boot["real_spearman_ci95"][1],
            "control_auroc": control["auroc"], "auroc_advantage": auc_adv,
            "auroc_advantage_ci95_lower": boot["auroc_advantage_ci95"][0], "auroc_advantage_ci95_upper": boot["auroc_advantage_ci95"][1],
            "control_spearman": control["spearman"], "spearman_advantage": rho_adv,
            "spearman_advantage_ci95_lower": boot["spearman_advantage_ci95"][0], "spearman_advantage_ci95_upper": boot["spearman_advantage_ci95"][1],
            "harm_rate": real["harm_rate"], "subjects": real["subjects"], "observations": real["observations"], "pass": passed,
        }
        summaries.append(row)
        decision["datasets"][dataset] = row
    summary = pd.DataFrame(summaries)
    write_csv(OUT / "GRADIENT_TRANSFER_SUMMARY.csv", summary)
    write_csv(OUT / "GRADIENT_TRANSFER_BOOTSTRAP.csv", pd.concat(bootstrap_frames, ignore_index=True))
    both = bool(summary["pass"].all())
    decision.update({
        "terminal": "CM_GA_GRADIENT_TRANSFER_SUPPORTED" if both else "CM_GA_GRADIENT_TRANSFER_NOT_SUPPORTED",
        "stage1_authorized": both, "seed1_run": False, "seed2_run": False,
        "exposed_benchmark_evaluation_accessed": False,
        "new_sealed_test_accessed": False,
    })
    write_json(OUT / "GRADIENT_TRANSFER_DECISION.json", decision)
    write_json(OUT / "FINAL_CM_GA_DECISION.json", decision)
    lines = ["# LiteBN-CM-GA Stage-0 gradient-transfer report", "", f"Terminal: `{decision['terminal']}`.", "", "| Dataset | AUROC | 95% CI | Spearman | 95% CI | control AUROC | AUROC advantage | advantage 95% CI | pass |", "|---|---:|---|---:|---|---:|---:|---|---|"]
    for _, row in summary.iterrows():
        lines.append(f"| {row['dataset']} | {row['auroc']:.4f} | [{row['auroc_ci95_lower']:.4f}, {row['auroc_ci95_upper']:.4f}] | {row['spearman']:.4f} | [{row['spearman_ci95_lower']:.4f}, {row['spearman_ci95_upper']:.4f}] | {row['control_auroc']:.4f} | {row['auroc_advantage']:+.4f} | [{row['auroc_advantage_ci95_lower']:+.4f}, {row['auroc_advantage_ci95_upper']:+.4f}] | {bool(row['pass'])} |")
    lines += ["", "Stage 0 used inner-training subjects and source-session blocks only. Exposed benchmark evaluation and all new sealed cohorts were not accessed."]
    (OUT / "FINAL_CM_GA_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary, decision


def protocol_documents() -> None:
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    RUNTIME.mkdir(parents=True, exist_ok=True)
    write_json(PROTOCOL / "DATA_SCOPE_AUDIT.json", {
        "stage0_subject_role": "canonical inner-training biological subjects only",
        "stage0_session_role": "source sessions only",
        "development_outer_accessed": False,
        "exposed_benchmark_evaluation_accessed": False,
        "new_sealed_test_accessed": False,
    })
    write_json(PROTOCOL / "RUNTIME_AUDIT.json", {
        "platform": platform.platform(), "python": sys.version, "torch": torch.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    })


def main() -> None:
    started = time.time()
    protocol_documents()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("CUDA is required")
    identity, frozen_identity = identity_preflight(device)
    observations, frozen_stage0 = stage0_observations(device)
    frozen = pd.concat((frozen_identity, frozen_stage0), ignore_index=True)
    write_csv(OUT / "FROZEN_BASE_AUDIT.csv", frozen)
    if not frozen["bitwise_unchanged"].all() or not frozen.get("stored_residual_bitwise_unchanged", pd.Series([True])).fillna(True).all():
        raise RuntimeError("PROTOCOL_FAIL: stored model changed during Stage0")
    summary, decision = summarize(observations)
    print(json.dumps({"terminal": decision["terminal"], "runtime_seconds": time.time() - started}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
