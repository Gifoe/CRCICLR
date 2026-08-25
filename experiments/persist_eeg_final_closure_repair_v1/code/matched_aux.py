"""Strictly nested, exactly matched Phase-B auxiliary-objective experiment."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (
    EXP,
    FIGURES,
    FINAL,
    RESULTS,
    RUNTIME,
    SOURCE,
    append_engineering_log,
    audit_frozen_tables,
    canonical_json_sha,
    clean,
    compute_normalizer_local,
    core,
    diag,
    ensure_dirs,
    expected_subjects,
    historical_run_dir,
    load_authorized_data,
    paired_subject_stats,
    protocol,
    read_json,
    sha256_file,
    state_sha256,
    subject_bootstrap,
    write_csv,
    write_json,
    write_md,
)


METHODS = (
    "Matched-TaskOnly",
    "Random-Aux",
    "Identity-Aux",
    "Full-Teacher-KD-Aux",
    "P-only-Aux",
    "PUD-Aux",
)
TARGET_KIND = {
    "Matched-TaskOnly": "TASK_ONLY",
    "Random-Aux": "RANDOM",
    "Identity-Aux": "IDENTITY",
    "Full-Teacher-KD-Aux": "FULL_KD",
    "P-only-Aux": "P_ONLY",
    "PUD-Aux": "PUD",
}


class AuxNet(core.EEGNetClassifier):
    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self.aux_head = nn.Linear(self.encoder.embedding_dim, 2)

    def forward_with_aux(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(x)
        return self.head(features), self.aux_head(features)


def method_slug(method: str) -> str:
    return method.lower().replace("-", "_").replace(" ", "_")


def load_certificate(stage_dir: Path) -> core.Certificate:
    return diag.load_certificate(stage_dir)


def load_teacher(stage_dir: Path, device: torch.device) -> tuple[core.EEGNetClassifier, dict[str, Any]]:
    metadata = read_json(stage_dir / "TEACHER.json")
    model = core.EEGNetClassifier(dict(metadata["config"]))
    payload = torch.load(stage_dir / "teacher.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    if state_sha256(model) != metadata["trained_state_sha256"]:
        raise RuntimeError(f"teacher state hash mismatch: {stage_dir}")
    return model, metadata


def train_teacher_and_certificate(
    data: core.DevelopmentData,
    fold: int,
    seed: int,
    stage: str,
    subjects: Sequence[str],
    device: torch.device,
) -> tuple[core.EEGNetClassifier, core.Certificate, np.ndarray, np.ndarray, dict[str, Any]]:
    stage_dir = RUNTIME / "matched_runs" / f"fold-{fold}" / f"seed-{seed}" / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    subject_tuple = tuple(core.subject_sort(subjects))
    signature = canonical_json_sha({
        "protocol_sha": sha256_file(EXP / "CLOSURE_REPAIR_PROTOCOL_FROZEN.json"),
        "fold": fold,
        "seed": seed,
        "stage": stage,
        "subjects": subject_tuple,
        "teacher_epochs": 40,
    })
    ready_path = stage_dir / "READY.json"
    normalizer_path = stage_dir / "normalizer.npz"
    if ready_path.is_file():
        ready = read_json(ready_path)
        if ready.get("signature") != signature:
            raise RuntimeError(f"stale nested teacher cache: {stage_dir}")
        mean, std = compute_normalizer_local(data, subject_tuple, normalizer_path)
        teacher, metadata = load_teacher(stage_dir, device)
        certificate = load_certificate(stage_dir)
        if tuple(core.subject_sort(certificate.audit["source_subjects"])) != subject_tuple:
            raise RuntimeError(f"certificate subject mismatch: {stage_dir}")
        return teacher, certificate, mean, std, metadata

    mean, std = compute_normalizer_local(data, subject_tuple, normalizer_path)
    historical_lock = read_json(historical_run_dir(fold, seed) / "RUN_LOCK.json")
    teacher_config = dict(historical_lock["baseline_configuration"])
    teacher_seed = core.stable_seed("closure-repair-teacher", fold, seed, stage)
    teacher = core.EEGNetClassifier(teacher_config).to(device)
    core.deterministic_reinitialize(teacher, teacher_seed)
    initial_sha = state_sha256(teacher)
    train_indices = core.row_indices(data.metadata, subject_tuple, (1, 2))
    teacher, _, history = core.train_single(
        teacher,
        data,
        train_indices,
        None,
        device,
        mean,
        std,
        teacher_seed,
        teacher_config,
        fixed_epochs=40,
    )
    trained_sha = state_sha256(teacher)
    torch.save({"state_dict": {name: value.detach().cpu() for name, value in teacher.state_dict().items()}}, stage_dir / "teacher.pt")
    metadata = {
        "fold": fold,
        "seed": seed,
        "stage": stage,
        "subjects": list(subject_tuple),
        "rows": int(len(train_indices)),
        "sessions": [1, 2],
        "epochs": 40,
        "fixed_epochs": True,
        "validation_subjects_used": False,
        "config": teacher_config,
        "teacher_seed": teacher_seed,
        "initial_state_sha256": initial_sha,
        "trained_state_sha256": trained_sha,
        "last_train_loss": float(history[-1]["train_loss"]),
    }
    write_json(stage_dir / "TEACHER.json", metadata)
    teacher.eval()
    certificate, _ = core.fit_certificate(teacher, data, subject_tuple, device, mean, std, fold, seed)
    core.save_certificate(certificate, stage_dir / "certificate")
    if tuple(core.subject_sort(certificate.audit["source_subjects"])) != subject_tuple:
        raise RuntimeError("new certificate used a subject outside its legal nested subset")
    write_json(ready_path, {
        "signature": signature,
        "teacher_sha256": trained_sha,
        "certificate_sha256": sha256_file(stage_dir / "certificate" / "PUD_CERTIFICATE.npz"),
        "subjects": list(subject_tuple),
        "internal_holdout_accessed": False,
        "WBCIC_outer_accessed": False,
    })
    return teacher, certificate, mean, std, metadata


def build_or_load_targets(
    data: core.DevelopmentData,
    teacher: core.EEGNetClassifier,
    certificate: core.Certificate,
    subjects: Sequence[str],
    mean: np.ndarray,
    std: np.ndarray,
    fold: int,
    seed: int,
    stage: str,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    stage_dir = RUNTIME / "matched_runs" / f"fold-{fold}" / f"seed-{seed}" / stage
    target_path = stage_dir / "targets.npz"
    audit_path = stage_dir / "TARGET_AUDIT.json"
    indices = core.row_indices(data.metadata, subjects, (1, 2))
    if target_path.is_file() and audit_path.is_file():
        values = np.load(target_path, allow_pickle=False)
        targets = {name: values[name].astype(np.float32) for name in ("TASK_ONLY", "RANDOM", "IDENTITY", "FULL_KD", "P_ONLY", "PUD")}
        audit = read_json(audit_path)
        if audit.get("source_indices_sha256") != canonical_json_sha(indices.tolist()):
            raise RuntimeError(f"stale target cache: {target_path}")
        return targets, audit

    evaluation = core.evaluate_single(teacher, data, indices, device, mean, std, include_features=True, batch_size=512)
    size = len(data.metadata)

    def full_array(values: np.ndarray) -> np.ndarray:
        out = np.zeros((size, 2), dtype=np.float32)
        out[evaluation.indices] = np.asarray(values, dtype=np.float32)
        return out

    full_kd = core.centered_logits_np(evaluation.logits).astype(np.float32)
    pud = np.asarray(core.teacher_targets(teacher, certificate, evaluation, "PUD")["protected"], dtype=np.float32)
    p_only = np.asarray(core.teacher_targets(teacher, certificate, evaluation, "P")["protected"], dtype=np.float32)
    identity = np.asarray(core.teacher_targets(teacher, certificate, evaluation, "IDENTITY")["protected"], dtype=np.float32)
    rng = np.random.default_rng(core.stable_seed("closure-repair-centered-random-target", fold, seed, stage))
    random = rng.normal(size=pud.shape).astype(np.float32)
    random -= random.mean(axis=1, keepdims=True)
    pud_rms = float(np.sqrt(np.mean(np.square(pud))))
    random_rms_before = float(np.sqrt(np.mean(np.square(random))))
    random *= pud_rms / max(random_rms_before, 1e-12)
    random_rms_after = float(np.sqrt(np.mean(np.square(random))))
    targets = {
        "TASK_ONLY": np.zeros((size, 2), dtype=np.float32),
        "RANDOM": full_array(random),
        "IDENTITY": full_array(identity),
        "FULL_KD": full_array(full_kd),
        "P_ONLY": full_array(p_only),
        "PUD": full_array(pud),
    }
    source_mask = np.zeros(size, dtype=bool)
    source_mask[indices] = True
    if any(np.any(values[~source_mask] != 0) for values in targets.values()):
        raise RuntimeError("auxiliary target escaped the legal source subset")
    if float(np.max(np.abs(random.sum(axis=1)))) > 1e-5:
        raise RuntimeError("Random-Aux target was not centered in the class-logit subspace")
    if not math.isclose(random_rms_after, pud_rms, rel_tol=1e-5, abs_tol=1e-7):
        raise RuntimeError("Random-Aux RMS does not match PUD target RMS")
    np.savez_compressed(target_path, **targets)
    audit = {
        "fold": fold,
        "seed": seed,
        "stage": stage,
        "subjects": list(core.subject_sort(subjects)),
        "source_rows": int(len(indices)),
        "source_indices_sha256": canonical_json_sha(indices.tolist()),
        "PUD_rank": int(certificate.audit["PUD_rank"]),
        "P_rank": int(certificate.audit["P_rank"]),
        "identity_rank": int(certificate.audit["identity_rank"]),
        "random_centered_max_abs_class_sum": float(np.max(np.abs(random.sum(axis=1)))),
        "PUD_target_RMS": pud_rms,
        "random_RMS_before_match": random_rms_before,
        "random_RMS_after_match": random_rms_after,
        "targets_use_outcome_rows": False,
        "internal_holdout_accessed": False,
        "WBCIC_outer_accessed": False,
    }
    write_json(audit_path, audit)
    return targets, audit


def train_aux(
    data: core.DevelopmentData,
    config: Mapping[str, Any],
    train_indices: np.ndarray,
    validation_indices: np.ndarray | None,
    target: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    lam: float,
    initialization_seed: int,
    loader_seed: int,
    fixed_epochs: int | None = None,
) -> tuple[AuxNet, dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AuxNet(config).to(device)
    core.deterministic_reinitialize(model, initialization_seed)
    initial_full_sha = state_sha256(model)
    initial_main_sha = state_sha256(model, prefixes=("encoder.", "head."))
    cfg = protocol()["phase_b"]["aux_training"]
    maximum = int(fixed_epochs if fixed_epochs is not None else cfg["max_epochs"])
    minimum = int(cfg["minimum_epochs"])
    patience = int(cfg["patience"])
    loader = core.make_loader(data, train_indices, int(cfg["batch_size"]), True, loader_seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scaler = core._scaler(device)
    mean_t = torch.as_tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.as_tensor(std, dtype=torch.float32, device=device)
    target_scale = max(float(np.sqrt(np.mean(np.square(target[train_indices])))), 1e-4)
    best_state: dict[str, torch.Tensor] | None = None
    best_key: tuple[float, float, int] | None = None
    best_score = math.nan
    best_epoch = maximum
    stale = 0
    history: list[dict[str, float]] = []
    started = time.time()
    for epoch in range(maximum):
        model.train()
        total = 0.0
        seen = 0
        for x, y, index, _ in loader:
            x = core.normalize_tensor(x.to(device, non_blocking=True), mean_t, std_t)
            y = y.to(device, non_blocking=True)
            aux_target = torch.as_tensor(target[index.numpy()], dtype=torch.float32, device=device)
            optimizer.zero_grad(set_to_none=True)
            with core._amp_context(device):
                task_logits, aux_logits = model.forward_with_aux(x)
                task_loss = F.cross_entropy(task_logits, y)
                aux_loss = F.mse_loss(core.centered_logits(aux_logits).float() / target_scale, aux_target / target_scale)
                loss = task_loss + float(lam) * aux_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach().cpu()) * len(y)
            seen += len(y)
        row = {"epoch": float(epoch + 1), "train_loss": total / max(seen, 1)}
        if validation_indices is not None:
            evaluation = core.evaluate_single(model, data, validation_indices, device, mean, std, include_features=False, batch_size=512)
            score = float(core.subject_mean_ba(evaluation.labels, evaluation.logits, evaluation.subjects))
            row["validation_mean_subject_BA"] = score
            key = (score, -row["train_loss"], -(epoch + 1))
            if best_key is None or key > best_key:
                best_key = key
                best_score = score
                best_epoch = epoch + 1
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
        history.append(row)
        if validation_indices is not None and epoch + 1 >= minimum and stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    metadata = {
        "lambda_aux": float(lam),
        "initialization_seed": int(initialization_seed),
        "loader_seed": int(loader_seed),
        "initial_full_state_sha256": initial_full_sha,
        "initial_main_state_sha256": initial_main_sha,
        "best_epoch": int(best_epoch),
        "best_validation_BA": float(best_score),
        "epochs_executed": int(len(history)),
        "target_scale": target_scale,
        "final_train_loss": float(history[-1]["train_loss"]),
        "elapsed_seconds": float(time.time() - started),
    }
    return model, metadata


def write_phase_b_protocol() -> None:
    cfg = protocol()["phase_b"]
    write_md(
        EXP / "PHASE_B_MATCHED_PROTOCOL.md",
        "Phase B exact matched protocol",
        "The primary comparison is PUD-Aux versus Matched-TaskOnly. Every method uses the same single-path EEGNet task network and training-only linear auxiliary head; Matched-TaskOnly sets lambda to zero. Within fold/seed/stage, complete initialization and task-network SHA, source rows, normalizer, minibatch order, optimizer, task loss, validation metric, patience, and epoch budget are identical.\n\n"
        f"Lambda grid: {cfg['lambda_grid']}. Inner teachers and certificates use inner_train only. Final teachers/certificates are rebuilt on all outer-source subjects only after `SELECTION_FROZEN.json` is written. Random targets are centered over classes before RMS matching to the legal PUD target. Outcome Session 2 is evaluated only after all selection artifacts are frozen.\n\n"
        "OpenBMI internal holdout and WBCIC outer access are forbidden and guarded.",
    )


def run_context(fold: int, seed: int) -> Path:
    return RUNTIME / "matched_runs" / f"fold-{fold}" / f"seed-{seed}"


def run_one(fold: int, seed: int) -> None:
    if fold not in range(5) or seed not in range(3):
        raise ValueError("fold must be 0..4 and seed must be 0..2")
    ensure_dirs()
    write_phase_b_protocol()
    data = load_authorized_data()
    roles = core.outer_folds(data.search_subjects)
    role = roles[fold]
    frozen_fold = next(row for row in protocol()["dataset"]["folds"] if int(row["fold"]) == fold)
    if tuple(core.subject_sort(role["outcome"])) != tuple(core.subject_sort(frozen_fold["outcome"])):
        raise RuntimeError("runtime outer fold differs from repair protocol")
    if tuple(core.subject_sort(role["inner_validation"])) != tuple(core.subject_sort(frozen_fold["inner_validation"])):
        raise RuntimeError("runtime inner validation differs from repair protocol")
    context = run_context(fold, seed)
    context.mkdir(parents=True, exist_ok=True)
    complete = context / "RUN_COMPLETE.json"
    if complete.is_file():
        payload = read_json(complete)
        if payload.get("pass") is True:
            print(f"[matched-aux] fold={fold} seed={seed} already complete", flush=True)
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = diag.eegnet_f8_config()
    inner_train = core.row_indices(data.metadata, role["inner_train"], (1, 2))
    inner_validation = core.row_indices(data.metadata, role["inner_validation"], (1, 2))
    inner_teacher, inner_certificate, inner_mean, inner_std, _ = train_teacher_and_certificate(
        data, fold, seed, "inner", role["inner_train"], device
    )
    inner_targets, inner_target_audit = build_or_load_targets(
        data, inner_teacher, inner_certificate, role["inner_train"], inner_mean, inner_std, fold, seed, "inner", device
    )
    del inner_teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()

    candidate_dir = context / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    lambdas = [float(value) for value in protocol()["phase_b"]["lambda_grid"]]
    init_seed = core.stable_seed("closure-repair-aux-init", fold, seed, "inner")
    loader_seed = core.stable_seed("closure-repair-minibatch", fold, seed, "inner")
    candidates: list[dict[str, Any]] = []
    for method in METHODS:
        method_lambdas = [0.0] if method == "Matched-TaskOnly" else lambdas
        for lam in method_lambdas:
            path = candidate_dir / f"{method_slug(method)}__lambda-{lam:.2f}.json"
            if path.is_file():
                record = read_json(path)
            else:
                model, metadata = train_aux(
                    data,
                    config,
                    inner_train,
                    inner_validation,
                    inner_targets[TARGET_KIND[method]],
                    inner_mean,
                    inner_std,
                    lam,
                    init_seed,
                    loader_seed,
                )
                record = {
                    "fold": fold,
                    "seed": seed,
                    "method": method,
                    "target_kind": TARGET_KIND[method],
                    "nested_target_subjects": list(core.subject_sort(role["inner_train"])),
                    "inner_validation_subjects": list(core.subject_sort(role["inner_validation"])),
                    "outcome_subjects_used": False,
                    **metadata,
                }
                write_json(path, record)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            candidates.append(record)
            print(f"[matched-aux] fold={fold} seed={seed} candidate={method} lambda={lam:.2f} BA={record['best_validation_BA']:.6f}", flush=True)

    candidate_frame = pd.DataFrame(candidates)
    if candidate_frame.initial_full_state_sha256.nunique() != 1 or candidate_frame.initial_main_state_sha256.nunique() != 1:
        raise RuntimeError("inner candidate initial states are not exactly matched")
    if candidate_frame.loader_seed.nunique() != 1 or candidate_frame.initialization_seed.nunique() != 1:
        raise RuntimeError("inner candidate minibatch/initialization seeds are not matched")
    selected: dict[str, dict[str, Any]] = {}
    for method, group in candidate_frame.groupby("method", sort=False):
        chosen = group.sort_values(["best_validation_BA", "lambda_aux", "best_epoch"], ascending=[False, True, True]).iloc[0].to_dict()
        selected[method] = clean(chosen)
    selection_payload = {
        "fold": fold,
        "seed": seed,
        "selection_frozen_before_outer_teacher_or_outcome_evaluation": True,
        "selection_metric": "inner-validation mean subject BA, task head only",
        "inner_train_subjects": list(core.subject_sort(role["inner_train"])),
        "inner_validation_subjects": list(core.subject_sort(role["inner_validation"])),
        "outcome_session_2_labels_used": False,
        "initial_full_state_sha256": candidate_frame.initial_full_state_sha256.iloc[0],
        "initial_main_state_sha256": candidate_frame.initial_main_state_sha256.iloc[0],
        "loader_seed": int(candidate_frame.loader_seed.iloc[0]),
        "target_audit": inner_target_audit,
        "selected": selected,
    }
    write_json(context / "SELECTION_FROZEN.json", selection_payload)

    outer_teacher, outer_certificate, outer_mean, outer_std, _ = train_teacher_and_certificate(
        data, fold, seed, "outer", role["source"], device
    )
    outer_targets, outer_target_audit = build_or_load_targets(
        data, outer_teacher, outer_certificate, role["source"], outer_mean, outer_std, fold, seed, "outer", device
    )
    del outer_teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()

    outer_train = core.row_indices(data.metadata, role["source"], (1, 2))
    outcome = core.row_indices(data.metadata, role["outcome"], (2,))
    observed_outcome_subjects = set(data.metadata.iloc[outcome].subject_id.astype(str))
    if observed_outcome_subjects != set(role["outcome"]):
        raise RuntimeError("outcome row construction changed")
    final_dir = context / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    outer_init_seed = core.stable_seed("closure-repair-aux-init", fold, seed, "outer")
    outer_loader_seed = core.stable_seed("closure-repair-minibatch", fold, seed, "outer")
    per_method = []
    final_meta = []
    for method in METHODS:
        result_path = final_dir / f"{method_slug(method)}.csv"
        meta_path = final_dir / f"{method_slug(method)}.json"
        if result_path.is_file() and meta_path.is_file():
            frame = pd.read_csv(result_path)
            metadata = read_json(meta_path)
        else:
            chosen = selected[method]
            model, metadata = train_aux(
                data,
                config,
                outer_train,
                None,
                outer_targets[TARGET_KIND[method]],
                outer_mean,
                outer_std,
                float(chosen["lambda_aux"]),
                outer_init_seed,
                outer_loader_seed,
                fixed_epochs=int(chosen["best_epoch"]),
            )
            evaluation = core.evaluate_single(model, data, outcome, device, outer_mean, outer_std, include_features=False, batch_size=512)
            frame = core.per_subject_metrics(evaluation.labels, evaluation.logits, evaluation.subjects)
            frame.insert(0, "method", method)
            frame.insert(1, "fold", fold)
            frame.insert(2, "seed", seed)
            frame["lambda_aux"] = float(chosen["lambda_aux"])
            frame["selected_epoch"] = int(chosen["best_epoch"])
            frame["target_kind"] = TARGET_KIND[method]
            if set(frame.subject_id.astype(str)) != set(role["outcome"]) or len(frame) != 8:
                raise RuntimeError(f"final outcome subjects are invalid for {method}")
            torch.save({"state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()}}, final_dir / f"{method_slug(method)}.pt")
            metadata.update({
                "fold": fold,
                "seed": seed,
                "method": method,
                "target_kind": TARGET_KIND[method],
                "selected_lambda": float(chosen["lambda_aux"]),
                "selected_epoch_from_inner_validation": int(chosen["best_epoch"]),
                "outer_source_subjects": list(core.subject_sort(role["source"])),
                "outcome_labels_used_for_selection": False,
                "inference_uses_aux_head": False,
                "trained_state_sha256": state_sha256(model),
            })
            write_csv(result_path, frame)
            write_json(meta_path, metadata)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        per_method.append(frame)
        final_meta.append(metadata)
        print(f"[matched-aux] fold={fold} seed={seed} final={method} BA={frame.BA.mean():.6f}", flush=True)

    final_frame = pd.concat(per_method, ignore_index=True)
    metadata_frame = pd.DataFrame(final_meta)
    if len(final_frame) != 48 or final_frame.method.value_counts().to_dict() != {method: 8 for method in METHODS}:
        raise RuntimeError("run-level final result cardinality failure")
    if metadata_frame.initial_full_state_sha256.nunique() != 1 or metadata_frame.initial_main_state_sha256.nunique() != 1:
        raise RuntimeError("outer final initial states are not exactly matched")
    if metadata_frame.loader_seed.nunique() != 1:
        raise RuntimeError("outer final minibatch order is not exactly matched")
    write_csv(context / "per_subject.csv", final_frame)
    complete_payload = {
        "pass": True,
        "fold": fold,
        "seed": seed,
        "rows": len(final_frame),
        "methods": list(METHODS),
        "outer_initial_full_state_sha256": metadata_frame.initial_full_state_sha256.iloc[0],
        "outer_initial_main_state_sha256": metadata_frame.initial_main_state_sha256.iloc[0],
        "outer_loader_seed": int(metadata_frame.loader_seed.iloc[0]),
        "outer_target_audit": outer_target_audit,
        "internal_holdout_accessed": False,
        "WBCIC_outer_accessed": False,
    }
    write_json(complete, complete_payload)
    print(f"[matched-aux] fold={fold} seed={seed} COMPLETE", flush=True)


def make_phase_b_figures(main: pd.DataFrame, subject_delta: pd.Series) -> None:
    order = list(METHODS)
    values = main.set_index("method").loc[order]
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    colors = ["#707070", "#6baed6", "#74c476", "#fd8d3c", "#9e9ac8", "#cb181d"]
    ax.bar(np.arange(len(order)), values.BA, color=colors)
    ax.axhline(0.7861667, color="black", ls="--", lw=1.0, label="historical Vanilla")
    ax.set_xticks(np.arange(len(order)), order, rotation=22, ha="right")
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(max(0.65, float(values.BA.min()) - 0.03), min(1.0, float(values.BA.max()) + 0.03))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "matched_aux_main.png", dpi=220)
    fig.savefig(FIGURES / "matched_aux_main.pdf")
    plt.close(fig)

    ordered = subject_delta.sort_values()
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    colors = np.where(ordered.to_numpy() > 0, "#3182bd", np.where(ordered.to_numpy() < 0, "#de2d26", "#969696"))
    ax.bar(np.arange(len(ordered)), ordered.to_numpy(), color=colors, width=0.85)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(xlabel="Subjects sorted by delta", ylabel="PUD-Aux − Matched-TaskOnly BA")
    fig.tight_layout()
    fig.savefig(FIGURES / "matched_aux_subject_delta.png", dpi=220)
    fig.savefig(FIGURES / "matched_aux_subject_delta.pdf")
    plt.close(fig)


def aggregate() -> dict[str, Any]:
    ensure_dirs()
    write_phase_b_protocol()
    frames = []
    selections = []
    completions = []
    final_metadata = []
    for fold in range(5):
        for seed in range(3):
            context = run_context(fold, seed)
            complete = read_json(context / "RUN_COMPLETE.json")
            if complete.get("pass") is not True:
                raise RuntimeError(f"incomplete matched run fold={fold} seed={seed}")
            completions.append(complete)
            frames.append(pd.read_csv(context / "per_subject.csv"))
            selection = read_json(context / "SELECTION_FROZEN.json")
            selections.append(selection)
            for method in METHODS:
                final_metadata.append(read_json(context / "final" / f"{method_slug(method)}.json"))
    per = pd.concat(frames, ignore_index=True)
    per.subject_id = per.subject_id.astype(str)
    expected = set(expected_subjects())
    if len(per) != 720:
        raise RuntimeError(f"expected 720 final rows, got {len(per)}")
    if per.duplicated(["method", "fold", "seed", "subject_id"]).any():
        raise RuntimeError("duplicate final method/fold/seed/subject key")
    if per.method.value_counts().to_dict() != {method: 120 for method in METHODS}:
        raise RuntimeError(f"method row count failure: {per.method.value_counts().to_dict()}")
    if not set(per.subject_id).issubset(expected):
        raise RuntimeError("restricted subject appeared in matched results")
    write_csv(RESULTS / "matched_aux_per_subject.csv", per)

    fold_frame = per.groupby(["method", "fold"], as_index=False).agg(BA=("BA", "mean"), macro_f1=("macro_f1", "mean"), subjects=("subject_id", "nunique"))
    seed_frame = per.groupby(["method", "seed"], as_index=False).agg(BA=("BA", "mean"), macro_f1=("macro_f1", "mean"), subjects=("subject_id", "nunique"))
    write_csv(RESULTS / "matched_aux_per_fold.csv", fold_frame)
    write_csv(RESULTS / "matched_aux_per_seed.csv", seed_frame)
    subject = per.groupby(["method", "subject_id"], as_index=False).agg(BA=("BA", "mean"), macro_f1=("macro_f1", "mean"), seeds=("seed", "nunique"))
    if not bool((subject.seeds == 3).all()):
        raise RuntimeError("subject inference did not average exactly three seeds")
    pivot = subject.pivot(index="subject_id", columns="method", values="BA")
    primary_stats = paired_subject_stats(pivot["Matched-TaskOnly"], pivot["PUD-Aux"], seed=9173)
    primary_delta = pivot["PUD-Aux"] - pivot["Matched-TaskOnly"]

    replay = pd.read_csv(SOURCE / "results" / "replay_per_subject.csv")
    replay.subject_id = replay.subject_id.astype(str)
    old = replay[replay.method.eq("B0_VANILLA_EEGNET")].groupby("subject_id").BA.mean().reindex(pivot.index)
    historical_stats = paired_subject_stats(old, pivot["PUD-Aux"], seed=9174)
    historical_mean = float(old.mean())
    if not math.isclose(historical_mean, 0.7861667, abs_tol=1e-7):
        raise RuntimeError(f"historical Vanilla mean changed: {historical_mean}")

    main = subject.groupby("method", as_index=False).agg(BA=("BA", "mean"), macro_f1=("macro_f1", "mean"))
    matched_ba = float(main.loc[main.method.eq("Matched-TaskOnly"), "BA"].iloc[0])
    main["delta_vs_matched_task_only"] = main.BA - matched_ba
    main["delta_vs_historical_vanilla"] = main.BA - historical_mean
    write_csv(RESULTS / "matched_aux_main.csv", main)

    paired_methods = {}
    for offset, method in enumerate(METHODS):
        paired_methods[method] = paired_subject_stats(pivot["Matched-TaskOnly"], pivot[method], seed=9300 + offset)
    fold_pivot = fold_frame.pivot(index="fold", columns="method", values="BA")
    seed_pivot = seed_frame.pivot(index="seed", columns="method", values="BA")
    positive_folds = int((fold_pivot["PUD-Aux"] > fold_pivot["Matched-TaskOnly"]).sum())
    positive_seeds = int((seed_pivot["PUD-Aux"] > seed_pivot["Matched-TaskOnly"]).sum())
    pud_ba = float(main.loc[main.method.eq("PUD-Aux"), "BA"].iloc[0])
    random_ba = float(main.loc[main.method.eq("Random-Aux"), "BA"].iloc[0])
    identity_ba = float(main.loc[main.method.eq("Identity-Aux"), "BA"].iloc[0])

    initial_audit = pd.DataFrame(final_metadata)
    exact_outer = True
    for (fold, seed), group in initial_audit.groupby(["fold", "seed"]):
        exact_outer &= group.initial_full_state_sha256.nunique() == 1
        exact_outer &= group.initial_main_state_sha256.nunique() == 1
        exact_outer &= group.loader_seed.nunique() == 1
        exact_outer &= group.initialization_seed.nunique() == 1
    exact_inner = all(
        len({entry["initial_full_state_sha256"] for entry in selection["selected"].values()}) == 1
        and len({entry["initial_main_state_sha256"] for entry in selection["selected"].values()}) == 1
        for selection in selections
    )
    frozen_audit = audit_frozen_tables()
    purity = bool(
        exact_outer
        and exact_inner
        and frozen_audit["pass"]
        and all(not item.get("internal_holdout_accessed") and not item.get("WBCIC_outer_accessed") for item in completions)
    )
    gates = {
        "M1_delta_at_least_0.005": bool(primary_stats["mean"] >= 0.005),
        "M2_subject_CI_lower_positive": bool(primary_stats["ci95_l"] > 0),
        "M3_at_least_4_of_5_folds_positive": bool(positive_folds >= 4),
        "M4_at_least_2_of_3_seeds_positive": bool(positive_seeds >= 2),
        "M5_beats_random_and_identity": bool(pud_ba > random_ba and pud_ba > identity_ba),
        "M6_purity_and_integrity": purity,
    }
    matched_success = all(gates.values())
    historical_delta = pud_ba - historical_mean
    historical_success = bool(historical_delta >= 0.005 and historical_stats["ci95_l"] > 0)
    if not matched_success:
        terminal = "PUD_AUX_MATCHED_NOT_SUPPORTED"
    elif not historical_success:
        terminal = "PUD_AUX_RELATIVE_BENEFIT_ONLY_RECIPE_TAX_REMAINS"
    else:
        terminal = "PUD_AUX_CLEAN_CONSTRUCTIVE_SUPPORTED"

    ledger_rows = []
    for selection in selections:
        for method, chosen in selection["selected"].items():
            ledger_rows.append({"fold": selection["fold"], "seed": selection["seed"], "method": method, "selected_lambda": chosen["lambda_aux"], "selected_epoch": chosen["best_epoch"], "inner_validation_BA": chosen["best_validation_BA"]})
    write_csv(RESULTS / "matched_aux_training_ledger.csv", pd.DataFrame(ledger_rows))
    integrity = {
        "pass": purity,
        "rows": len(per),
        "method_rows": per.method.value_counts().to_dict(),
        "duplicate_keys": int(per.duplicated(["method", "fold", "seed", "subject_id"]).sum()),
        "exact_inner_initialization_and_order": exact_inner,
        "exact_outer_initialization_and_order": exact_outer,
        "observed_subjects": sorted(set(per.subject_id), key=int),
        "internal_holdout_accessed": False,
        "WBCIC_outer_accessed": False,
    }
    write_json(RESULTS / "matched_aux_integrity.json", integrity)
    statistics = {
        "terminal": terminal,
        "primary": {
            "comparison": "PUD-Aux minus Matched-TaskOnly",
            **primary_stats,
            "positive_folds": positive_folds,
            "positive_seeds": positive_seeds,
        },
        "historical": {
            "historical_vanilla_BA": historical_mean,
            "PUD_Aux_minus_historical_Vanilla": historical_delta,
            **historical_stats,
            "causal_control": False,
        },
        "method_means": main.to_dict("records"),
        "paired_vs_matched": paired_methods,
        "gates": gates,
        "matched_success": matched_success,
        "historical_success": historical_success,
        "integrity": integrity,
    }
    write_json(RESULTS / "matched_aux_statistics.json", statistics)
    make_phase_b_figures(main, primary_delta)

    gate_lines = "\n".join(f"- {name}: **{'PASS' if passed else 'FAIL'}**" for name, passed in gates.items())
    method_lines = "\n".join(f"| {row['method']} | {row['BA']:.6f} | {row['macro_f1']:.6f} | {row['delta_vs_matched_task_only']:+.6f} |" for row in main.to_dict("records"))
    write_md(
        EXP / "PHASE_B_MATCHED_FINAL.md",
        "Phase B exact matched final",
        f"| method | BA | Macro-F1 | Δ vs Matched-TaskOnly |\n|---|---:|---:|---:|\n{method_lines}\n\n"
        f"Primary PUD-Aux−Matched-TaskOnly: {primary_stats['mean']:+.6f}, median {primary_stats['median']:+.6f}, subject-bootstrap 95% CI [{primary_stats['ci95_l']:+.6f}, {primary_stats['ci95_u']:+.6f}], positive/negative/tied subjects {primary_stats['positive']}/{primary_stats['negative']}/{primary_stats['tied']}, positive folds {positive_folds}/5, positive seeds {positive_seeds}/3.\n\n"
        f"Historical Vanilla BA: {historical_mean:.7f}; PUD-Aux−old Vanilla: {historical_delta:+.6f}, paired 40-subject CI [{historical_stats['ci95_l']:+.6f}, {historical_stats['ci95_u']:+.6f}]. The old pipeline is an external reference, not the matched causal control.\n\n"
        f"{gate_lines}\n\nTerminal: **{terminal}**.",
    )

    phase_a = read_json(RESULTS / "phase_a_repaired_summary.json")
    closure = {
        "terminal": terminal,
        "phase_a": phase_a,
        "phase_b": statistics,
        "OpenBMI_internal_holdout_accessed": False,
        "WBCIC_outer_accessed": False,
        "constructive_PUD_development_closed": terminal != "PUD_AUX_CLEAN_CONSTRUCTIVE_SUPPORTED",
    }
    write_json(RESULTS / "closure_repair_summary.json", closure)
    strongest = (
        "Task-consequential persistent structure can be identified reliably, but the PUD auxiliary objective does not improve future-session generalization over an exactly matched task-only training pipeline."
        if terminal == "PUD_AUX_MATCHED_NOT_SUPPORTED"
        else "PUD improves relative to the matched Phase-B recipe, but the recipe remains inferior to the historical standard baseline; this is not a successful generalization method."
        if terminal == "PUD_AUX_RELATIVE_BENEFIT_ONLY_RECIPE_TAX_REMAINS"
        else "PUD-Aux satisfies the pre-frozen matched and historical-reference gates in this development-only protocol; sealed-data confirmation remains unavailable in this closure task."
    )
    write_md(
        EXP / "FINAL_CLOSURE_REPAIRED.md",
        "Final constructive closure — repaired",
        f"Terminal state: **{terminal}**.\n\n{strongest}\n\n"
        f"Phase-A H3: **{phase_a['H3']}**. Direction consequence transfer: **{phase_a['direction_consequence_transfer']}**. Certificate-score transfer: **{phase_a['certificate_score_transfer']}**.\n\n"
        "The B0 join and certificate-direction join are repaired; source-score ranking is not inferred from positive mean future harm; historical Vanilla is not mislabeled as a matched causal control. No sealed OpenBMI holdout or WBCIC outer data was accessed. No PUD V2/V3 or new constructive family is authorized after this closure.",
    )
    append_engineering_log(
        "Phase B matched repair",
        "Added strict inner_train-only teacher/certificate construction, post-selection outer rebuild, class-centered RMS-matched random targets, exact initialization/minibatch SHA audits, resumable fold/seed caches, and a newly trained Matched-TaskOnly control. Outcome Session-2 labels were evaluated only after each run's selection artifact was frozen.",
    )
    print(json.dumps(clean(statistics), indent=2), flush=True)
    return statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    protocol()
    if args.aggregate:
        aggregate()
    elif args.fold is not None and args.seed is not None:
        run_one(args.fold, args.seed)
    else:
        parser.error("provide --fold and --seed, or --aggregate")


if __name__ == "__main__":
    main()
