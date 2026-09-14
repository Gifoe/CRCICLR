#!/usr/bin/env python3
"""Run the source-only, optimizer-aligned LiteBN C/M harm audit."""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from construct_guard_blocks import BLOCK_NAMES, make_subject_blocks, stable_seed
from optimizer_displacement import (
    active_gradient_counts,
    deterministic_loss,
    eval_gradient,
    named_space_parameters,
    one_step_displacement,
)

REPO = Path(os.environ.get("OPTALIGN_REPO", "/root/rivermind-data/CRCICLR_OPTALIGN_WORK")).resolve()
EXP = REPO / "experiments/persist_eeg_cm_optimizer_aligned_harm_audit_v1"
CODE, OUT, PROTOCOL = EXP / "code", EXP / "outputs", EXP / "protocol"
BASE_RUNNER = REPO / "experiments/persist_eeg_linux_w0r0_screen_v1/code/run_w0r0_seed0.py"
SEED, PAIR_ROUNDS = 0, 2
TASKS = ("OpenBMI_MI", "WBCIC_MI")
SPACES = ("C", "M", "CM")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


os.environ["W0R0_REPO"] = str(REPO)
runner = load_module("optimizer_aligned_base_runner", BASE_RUNNER)
base, _unused = runner.load_modules()
models = load_module("optimizer_aligned_models", CODE / "models_cm.py")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = value if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
    tmp = path.with_suffix(path.suffix + ".part")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def subject_sort(values: Iterable[str], dataset: str) -> list[str]:
    return base.subject_sort(values, dataset)


def build_source_bundle(task: str, subjects: Iterable[str]):
    """Open only frozen canonical inner-subject source sessions."""
    spec = base.TASKS[task]
    dataset = spec["dataset"]
    canonical = subject_sort(subjects, dataset)
    rows = []
    if dataset == "OpenBMI":
        for subject in canonical:
            for session in spec["source_sessions"]:
                signal = base.openbmi_path(task, subject, session, "signal")
                labels = base.openbmi_path(task, subject, session, "label")
                y = np.load(labels, mmap_mode="r", allow_pickle=False)
                for index, code in enumerate(y):
                    rows.append(base.Row(str(subject), int(session), str(signal), index,
                                         int(spec["raw_codes"][int(code)])))
    else:
        root = base.CACHE / "wbcic/wbcic_epochs"
        for subject in canonical:
            for session in spec["source_sessions"]:
                signal = root / subject / f"ses-{session}_epochs.npy"
                labels = root / subject / f"ses-{session}_labels.npy"
                y = np.load(labels, mmap_mode="r", allow_pickle=False)
                for index, code in enumerate(y):
                    rows.append(base.Row(str(subject), int(session), str(signal), index, int(code)))
    if not rows:
        raise RuntimeError(f"empty legal source bundle: {task}")
    return base.SignalBundle(task, canonical, rows)


def make_model(task: str, fold: int, device: torch.device):
    set_seed(SEED)
    b0 = base.build_model("LiteBN_BASELINE", task)
    checkpoint = runner.baseline_path(task, fold)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    b0.load_state_dict(state, strict=True)
    set_seed(SEED)
    model = models.LiteBNCMGA(b0).to(device)
    model.eval()
    return model, checkpoint


def metric(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    prediction = logits.argmax(axis=1)
    return {
        "BA": float(balanced_accuracy_score(labels, prediction)),
        "macro_F1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(labels, prediction)),
    }


def balanced_indices(bundle, subject: str) -> np.ndarray:
    candidates = bundle.indices([subject], base.TASKS[bundle.task]["source_sessions"])
    labels = bundle.labels(candidates)
    output = []
    for cls in range(bundle.classes):
        pool = candidates[labels == cls]
        output.extend(pool[: min(32, len(pool))].tolist())
    return np.asarray(output, dtype=np.int64)


def subject_pools(bundle, subjects: list[str]) -> dict[str, dict[int, np.ndarray]]:
    pools: dict[str, dict[int, np.ndarray]] = {}
    sessions = base.TASKS[bundle.task]["source_sessions"]
    for subject in subjects:
        indices = bundle.indices([subject], sessions)
        labels = bundle.labels(indices)
        pools[subject] = {cls: indices[labels == cls] for cls in range(bundle.classes)}
    return pools


def residual_hash(model: torch.nn.Module) -> str:
    return models.tensor_mapping_sha256({name: value for name, value in model.state_dict().items()
                                         if not name.startswith("base.")})


def cyclic_controls(subjects: list[str], subject_a: str, subject_b: str, *, dataset: str,
                    fold: int, round_id: int, kind: str, count: int) -> list[str]:
    start = stable_seed("optimizer-aligned-controls", dataset, fold, SEED, round_id, subject_a, subject_b, kind) % len(subjects)
    output = []
    for offset in range(len(subjects)):
        candidate = subjects[(start + offset) % len(subjects)]
        if candidate not in (subject_a, subject_b) and candidate not in output:
            output.append(candidate)
            if len(output) == count:
                return output
    raise RuntimeError(f"unable to select {count} controls excluding A/B")


def identity_preflight(device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    _search, folds, split_hash = base.load_folds()
    historical_model_source = REPO / "experiments/persist_eeg_litebn_cm_ga_final_candidate_v1/code/models_cm.py"
    if runner.sha256(CODE / "models_cm.py") != runner.sha256(historical_model_source):
        raise RuntimeError("PROTOCOL_FAIL: models_cm.py is not the byte-identical historical CM-GA implementation")
    identity_rows, frozen_rows = [], []
    for task in TASKS:
        dataset = base.TASKS[task]["dataset"]
        for fold in folds[dataset]:
            fold_id = int(fold["fold_id"])
            model, checkpoint = make_model(task, fold_id, device)
            before = models.frozen_base_sha256(model)
            bundle = build_source_bundle(task, fold["inner_train_subjects"])
            mean, std, meta = base.load_tensor_pair(runner.normalizer_source(task, fold_id))
            cache = base.RawGPUCache(bundle, device)
            indices = balanced_indices(bundle, subject_sort(fold["inner_train_subjects"], dataset)[0])
            value, _ = cache.batch(indices, mean, std)
            labels = bundle.labels(indices)
            model.eval()
            with torch.inference_mode():
                b0_logits, _ = model.base(value)
                cm_logits, _ = model(value)
            b0_np, cm_np = b0_logits.float().cpu().numpy(), cm_logits.float().cpu().numpy()
            b0_metric, cm_metric = metric(labels, b0_np), metric(labels, cm_np)
            maximum = float(np.max(np.abs(b0_np - cm_np)))
            passed = bool(maximum <= 1e-7 and np.array_equal(b0_np.argmax(1), cm_np.argmax(1))
                          and b0_metric == cm_metric and float(model.lambda_channel) == 0.0
                          and all(float(item.gamma) == 0.0 for item in model.mixers))
            identity_rows.append({
                "task": task, "dataset": dataset, "fold": fold_id, "seed": SEED,
                "checkpoint_path": str(checkpoint), "checkpoint_sha256": runner.sha256(checkpoint),
                "normalizer_sha256": meta["mean_std_sha256"], "audited_trials": len(indices),
                "max_abs_logit_difference": maximum,
                "exact_predictions": bool(np.array_equal(b0_np.argmax(1), cm_np.argmax(1))),
                "BA_equal": b0_metric["BA"] == cm_metric["BA"],
                "macro_F1_equal": b0_metric["macro_F1"] == cm_metric["macro_F1"],
                "accuracy_equal": b0_metric["accuracy"] == cm_metric["accuracy"],
                "lambda_channel_zero": float(model.lambda_channel) == 0.0,
                "all_mixer_gamma_zero": all(float(item.gamma) == 0.0 for item in model.mixers),
                "pass": passed,
            })
            after = models.frozen_base_sha256(model)
            frozen_rows.append({"stage": "identity_preflight", "task": task, "dataset": dataset,
                                "fold": fold_id, "space": "ALL", "base_sha256_before": before,
                                "base_sha256_after": after, "bitwise_unchanged": before == after})
            del model, cache, bundle, value
            gc.collect()
            if device.type == "cuda": torch.cuda.empty_cache()
    identity, frozen = pd.DataFrame(identity_rows), pd.DataFrame(frozen_rows)
    write_csv(OUT / "IDENTITY_AUDIT.csv", identity)
    if len(identity) != 10 or not identity["pass"].all() or not frozen["bitwise_unchanged"].all():
        raise RuntimeError("PROTOCOL_FAIL: identity/frozen LiteBN preflight")
    write_json(PROTOCOL / "SOURCE_PROVENANCE.json", {
        "seed": SEED, "fivefold_split_sha256": split_hash,
        "exact_model_source": str(historical_model_source),
        "exact_model_source_sha256": runner.sha256(historical_model_source),
        "current_model_source_sha256": runner.sha256(CODE / "models_cm.py"),
        "model_source_byte_identical_to_cm_ga": True,
        "historical_litebn_checkpoints": "strictly loaded historical matched 64-feature LiteBN seed0 checkpoints",
        "normalizers": "frozen exact historical source-session normalizers",
        "previous_cm_ga_terminal": "CM_GA_GRADIENT_TRANSFER_NOT_SUPPORTED",
        "previous_raw_cosine": {"OpenBMI": {"auroc": 0.6432259213, "spearman": 0.2599282625},
                                "WBCIC": {"auroc": 0.6034500357, "spearman": 0.2213600083}},
    })
    return identity, frozen


def active_gradient_audit(device: torch.device) -> pd.DataFrame:
    _search, folds, _split = base.load_folds()
    rows = []
    for task in TASKS:
        dataset = base.TASKS[task]["dataset"]
        for fold in folds[dataset]:
            fold_id, subjects = int(fold["fold_id"]), subject_sort(fold["inner_train_subjects"], dataset)
            bundle = build_source_bundle(task, subjects)
            mean, std, meta = base.load_tensor_pair(runner.normalizer_source(task, fold_id))
            cache = base.RawGPUCache(bundle, device)
            blocks = {subject: make_subject_blocks(subject, pools, dataset=dataset, fold=fold_id, seed=SEED)
                      for subject, pools in subject_pools(bundle, subjects).items()}
            x, y = cache.batch(blocks[subjects[0]].blocks["B1"], mean, std)
            for space in SPACES:
                model, checkpoint = make_model(task, fold_id, device)
                named = named_space_parameters(model, space)
                _vector, loss, raw = eval_gradient(model, [value for _, value in named], x, y)
                rows.append({"task": task, "dataset": dataset, "fold": fold_id, "seed": SEED,
                             "space": space, "gradient_mode": "eval_dropout_disabled",
                             "probe_subject": subjects[0], "probe_block": "B1", "loss": loss,
                             "checkpoint_sha256": runner.sha256(checkpoint),
                             "normalizer_sha256": meta["mean_std_sha256"],
                             **active_gradient_counts(raw, named)})
                del model
            del x, y, cache, bundle
            gc.collect()
            if device.type == "cuda": torch.cuda.empty_cache()
    frame = pd.DataFrame(rows)
    write_csv(OUT / "ACTIVE_GRADIENT_DIMENSION.csv", frame)
    return frame


def run_observations(device: torch.device, frozen_preflight: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    _search, folds, _split = base.load_folds()
    rows, displacement_rows = [], []
    frozen_rows = frozen_preflight.to_dict("records")
    for task in TASKS:
        dataset = base.TASKS[task]["dataset"]
        for fold in folds[dataset]:
            fold_id = int(fold["fold_id"])
            subjects = subject_sort(fold["inner_train_subjects"], dataset)
            bundle = build_source_bundle(task, subjects)
            mean, std, meta = base.load_tensor_pair(runner.normalizer_source(task, fold_id))
            cache = base.RawGPUCache(bundle, device)
            blocks = {subject: make_subject_blocks(subject, pools, dataset=dataset, fold=fold_id, seed=SEED)
                      for subject, pools in subject_pools(bundle, subjects).items()}
            for space in SPACES:
                model, checkpoint = make_model(task, fold_id, device)
                named = named_space_parameters(model, space)
                params = [value for _, value in named]
                initial_base, initial_residual = models.frozen_base_sha256(model), residual_hash(model)
                # Cache all independent source-block gradients at identity, in eval mode.
                gradients: dict[tuple[str, str], tuple[torch.Tensor, float]] = {}
                for subject in subjects:
                    for block_name in ("B1", "B2", "B3", "B4"):
                        x, y = cache.batch(blocks[subject].blocks[block_name], mean, std)
                        vector, loss, _raw = eval_gradient(model, params, x, y)
                        gradients[(subject, block_name)] = (vector.detach(), loss)
                        del x, y
                for round_id in range(PAIR_ROUNDS):
                    for number, subject_b in enumerate(subjects):
                        subject_a = subjects[(number + 1 + round_id) % len(subjects)]
                        if subject_a == subject_b:
                            raise RuntimeError("A/B collision")
                        a_x, a_y = cache.batch(blocks[subject_a].blocks["B1"], mean, std)
                        update = one_step_displacement(model, space, a_x, a_y)
                        proposal, delta = update["proposal"], update["delta"]
                        future_x, future_y = cache.batch(blocks[subject_b].blocks["B_future"], mean, std)
                        loss_before = deterministic_loss(model, future_x, future_y)
                        loss_after = deterministic_loss(proposal, future_x, future_y)
                        harm = loss_after - loss_before
                        controls = cyclic_controls(subjects, subject_a, subject_b, dataset=dataset, fold=fold_id,
                                                    round_id=round_id, kind="different", count=4)
                        global_subjects = cyclic_controls(subjects, subject_a, subject_b, dataset=dataset, fold=fold_id,
                                                          round_id=round_id, kind="global", count=4)
                        disp_id = f"{task}:f{fold_id}:{space}:r{round_id}:B={subject_b}:A={subject_a}"
                        displacement_rows.append({
                            "observation_id": disp_id, "task": task, "dataset": dataset, "fold": fold_id,
                            "seed": SEED, "space": space, "round": round_id, "subject_A": subject_a,
                            "subject_B": subject_b, "optimizer": "AdamW", "learning_rate": 3e-4,
                            "weight_decay": 5e-4, "global_grad_clip": 5.0, "gradient_mode": "eval_dropout_disabled",
                            "delta_norm": update["delta_norm"], "g_A_norm": update["g_a_norm"],
                            "g_A_delta_dot": update["g_a_delta_dot"], "g_A_delta_cosine": update["g_a_delta_cosine"],
                            "preclip_global_grad_norm": update["preclip_global_grad_norm"], "loss_A": update["loss_a"],
                            "checkpoint_sha256": runner.sha256(checkpoint), "normalizer_sha256": meta["mean_std_sha256"],
                        })
                        for k in (1, 2, 4):
                            g_same = torch.stack([gradients[(subject_b, f"B{idx}")][0] for idx in range(1, k + 1)]).mean(0)
                            g_diff = torch.stack([gradients[(subject, "B1")][0] for subject in controls[:k]]).mean(0)
                            g_global = torch.stack([gradients[(subject, "B2")][0] for subject in global_subjects[:k]]).mean(0)
                            rows.append({
                                "observation_id": disp_id, "task": task, "dataset": dataset, "fold": fold_id,
                                "seed": SEED, "space": space, "round": round_id, "K": k,
                                "subject_A": subject_a, "subject_B": subject_b,
                                "different_subjects": ";".join(controls[:k]), "global_subjects": ";".join(global_subjects[:k]),
                                "m_per_class": blocks[subject_b].m_per_class,
                                "B_blocks_disjoint": True, "A_B_disjoint": subject_a != subject_b,
                                "controls_exclude_A_B": all(value not in (subject_a, subject_b) for value in controls[:k] + global_subjects[:k]),
                                "gradient_mode": "eval_dropout_disabled",
                                "c_same": float(torch.dot(g_same, delta).detach().cpu()),
                                "c_different": float(torch.dot(g_diff, delta).detach().cpu()),
                                "c_global": float(torch.dot(g_global, delta).detach().cpu()),
                                "g_B_norm": float(torch.linalg.vector_norm(g_same).detach().cpu()),
                                "g_different_norm": float(torch.linalg.vector_norm(g_diff).detach().cpu()),
                                "g_global_norm": float(torch.linalg.vector_norm(g_global).detach().cpu()),
                                "future_harm": harm, "harm_label": int(harm > 0.0),
                                "loss_B_future_before": loss_before, "loss_B_future_after": loss_after,
                                "checkpoint_sha256": runner.sha256(checkpoint), "normalizer_sha256": meta["mean_std_sha256"],
                            })
                        del a_x, a_y, future_x, future_y, proposal, delta
                final_base, final_residual = models.frozen_base_sha256(model), residual_hash(model)
                frozen_rows.append({"stage": "optimizer_aligned_observations", "task": task, "dataset": dataset,
                                    "fold": fold_id, "space": space, "base_sha256_before": initial_base,
                                    "base_sha256_after": final_base, "bitwise_unchanged": initial_base == final_base,
                                    "stored_residual_sha256_before": initial_residual,
                                    "stored_residual_sha256_after": final_residual,
                                    "stored_residual_bitwise_unchanged": initial_residual == final_residual})
                write_csv(OUT / "HARM_OBSERVATIONS.csv", rows)
                write_csv(OUT / "OPTIMIZER_DISPLACEMENT_AUDIT.csv", displacement_rows)
                print(f"OPTALIGN_FOLD_SPACE_COMPLETE {task} fold={fold_id} space={space} rows={len(rows)}", flush=True)
                del gradients, model
                gc.collect()
                if device.type == "cuda": torch.cuda.empty_cache()
            del cache, bundle
            gc.collect()
            if device.type == "cuda": torch.cuda.empty_cache()
    observations, frozen = pd.DataFrame(rows), pd.DataFrame(frozen_rows)
    write_csv(OUT / "HARM_OBSERVATIONS.csv", observations)
    write_csv(OUT / "OPTIMIZER_DISPLACEMENT_AUDIT.csv", pd.DataFrame(displacement_rows))
    write_csv(OUT / "FROZEN_BASE_AUDIT.csv", frozen)
    if not frozen["bitwise_unchanged"].all() or not frozen.get("stored_residual_bitwise_unchanged", pd.Series([True])).fillna(True).all():
        raise RuntimeError("PROTOCOL_FAIL: stored identity model changed")
    return observations, frozen


def protocol_documents() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PROTOCOL.mkdir(parents=True, exist_ok=True)
    write_json(PROTOCOL / "DATA_SCOPE_AUDIT.json", {
        "DEVELOPMENT_OUTER_ACCESSED": "NO", "EXPOSED_BENCHMARK_ACCESSED": "NO",
        "NEW_SEALED_TEST_ACCESSED": "NO", "EEG_MODEL_TRAINED": "NO",
        "tasks": list(TASKS), "seed": SEED, "folds": 5,
        "subjects": "canonical inner-training biological subjects only",
        "sessions": "source sessions only",
    })
    write_json(PROTOCOL / "RUNTIME_AUDIT.json", {
        "platform": platform.platform(), "python": sys.version, "torch": torch.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "primary_gradient_mode": "model.eval; dropout disabled", "optimizer": "AdamW",
        "lr": 3e-4, "weight_decay": 5e-4, "global_grad_clip": 5.0,
    })
    (PROTOCOL / "OPTIMIZER_ALIGNED_AUDIT_PROTOCOL.md").write_text(
        "# Optimizer-aligned source harm audit\n\n"
        "The primary certificate is `g_guard^T Delta_A`, where `Delta_A` is obtained by one real AdamW step on a disposable in-memory identity clone. "
        "Only source sessions from canonical inner-training subjects are read. The base LiteBN is frozen and eval-mode throughout. "
        "Target B has five deterministic, class-balanced, mutually disjoint source blocks; B1..B4 form K=1/2/4 guards and B_future is reserved for actual CE harm.\n",
        encoding="utf-8")
    (EXP / "README.md").write_text(
        "# LiteBN C/M optimizer-aligned harm audit\n\n"
        "This branch audits C, M, and CM identity residual parameter spaces. It does not train a candidate or access development outer, benchmark, heldout, or sealed data. "
        "Run `code/run_optimizer_aligned_audit.py`, then `code/aggregate_optimizer_aligned_audit.py`.\n",
        encoding="utf-8")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the locked Linux audit")
    set_seed(SEED)
    protocol_documents()
    device = torch.device("cuda")
    _identity, frozen = identity_preflight(device)
    active_gradient_audit(device)
    observations, _frozen = run_observations(device, frozen)
    expected = sum(2 * sum(len(fold["inner_train_subjects"]) for fold in base.load_folds()[1][dataset])
                   for dataset in ("OpenBMI", "WBCIC")) * len(SPACES) * 3
    if len(observations) != expected:
        raise RuntimeError(f"observation cardinality mismatch: {len(observations)}/{expected}")
    print(f"OPTIMIZER_ALIGNED_AUDIT_COMPLETE observations={len(observations)}", flush=True)


if __name__ == "__main__":
    main()
