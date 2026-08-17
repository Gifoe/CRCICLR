"""EEGNet-only competence and prospective WBCIC actionability pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from core import (
    ACTION_LOCK_PATH,
    BLOCKS,
    BOOTSTRAP_DRAWS,
    CACHE,
    EPS,
    EXP_ROOT,
    FROZEN_PATH,
    IMPLEMENTATION_ID,
    MODEL,
    OUT,
    PROTOCOL,
    RANDOM_DRAWS,
    REPO_ROOT,
    RESULTS,
    EEGNet,
    audit_roles,
    balanced_accuracy_score,
    bootstrap_mean,
    ce_rows,
    centered_rms,
    clean,
    exact_matched_delta,
    holm,
    infer,
    load_model,
    model_state_sha256,
    random_bases,
    require_development_protocol,
    sha256_file,
    sha_lines,
    signflip_p,
    softmax,
    stable_seed,
    train_model,
    true_margin,
    write_csv,
    write_json,
)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, timeout=5
        ).strip()
    except Exception:
        return None


def write_once(path: Path, payload: Any) -> None:
    text = json.dumps(clean(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"Frozen artifact mismatch; refusing overwrite: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def device_from_argument(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def candidate_configs() -> list[dict[str, Any]]:
    lock = json.loads((PROTOCOL / "REPRESENTATION_CANDIDATE_LOCK.json").read_text(encoding="utf-8"))
    if lock.get("candidate_pool") != ["EEGNet"] or lock.get("selection_is_task_only") is not True:
        raise RuntimeError("Representation candidate lock is not EEGNet-only/task-only")
    configs = [dict(value) for value in lock["configs"]]
    if [value["id"] for value in configs] != ["EEGNET_STD", "EEGNET_STABLE"]:
        raise RuntimeError("Unexpected candidate configuration set")
    return configs


def get_or_train(
    checkpoint: Path,
    train_subjects: Sequence[str],
    config: Mapping[str, Any],
    fold_label: str,
    device: torch.device,
    workers: int,
) -> tuple[EEGNet, dict[str, Any]]:
    if checkpoint.exists():
        model, payload = load_model(checkpoint, device)
        if (
            payload.get("implementation_id") != IMPLEMENTATION_ID
            or payload.get("fold_label") != fold_label
            or payload.get("config") != dict(config)
            or payload.get("train_subjects") != list(train_subjects)
            or payload.get("train_sessions") != [0, 1]
        ):
            raise RuntimeError(f"Existing checkpoint does not match frozen job: {checkpoint}")
        payload["checkpoint_sha256"] = sha256_file(checkpoint)
        print(f"[resume] {checkpoint}", flush=True)
        return model, payload
    return train_model(train_subjects, [0, 1], config, checkpoint, fold_label, device, workers)


def competence(device: torch.device, workers: int) -> dict[str, Any]:
    scope, _ = require_development_protocol()
    subjects = list(map(str, scope["allowed_subjects"]))
    folds = {int(key[1:]): list(map(str, value)) for key, value in scope["folds"].items()}
    configs = candidate_configs()
    subject_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    started = time.time()
    for config in configs:
        for fold in range(5):
            outcome = folds[fold]
            train_subjects = [subject for subject in subjects if subject not in set(outcome)]
            checkpoint = MODEL / "competence" / f"{config['id']}_fold-{fold}.pt"
            model, payload = get_or_train(
                checkpoint, train_subjects, config, f"competence-fold-{fold}", device, workers
            )
            arrays = infer(model, outcome, [2], device, workers)
            prediction = arrays["logits"].argmax(1)
            ce = ce_rows(arrays["logits"], arrays["labels"])
            for index, subject in enumerate(outcome):
                mask = arrays["subject_index"] == index
                subject_rows.append(
                    {
                        "config": config["id"],
                        "fold": fold,
                        "subject": subject,
                        "n_S3_trials": int(mask.sum()),
                        "balanced_accuracy": balanced_accuracy_score(
                            arrays["labels"][mask], prediction[mask]
                        ),
                        "cross_entropy": float(ce[mask].mean()),
                    }
                )
            checkpoint_rows.append(
                {
                    "config": config["id"],
                    "fold": fold,
                    "checkpoint": str(checkpoint.relative_to(REPO_ROOT)).replace("\\", "/"),
                    "checkpoint_sha256": payload["checkpoint_sha256"],
                    "model_state_sha256": payload["model_state_sha256"],
                    "train_subjects_hash": payload["train_subjects_hash"],
                }
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    aggregate_rows: list[dict[str, Any]] = []
    aggregate_by_config: dict[str, dict[str, Any]] = {}
    for config in configs:
        values = np.asarray(
            [row["balanced_accuracy"] for row in subject_rows if row["config"] == config["id"]],
            dtype=np.float64,
        )
        if len(values) != 41:
            raise RuntimeError(f"Competence did not produce one outcome per development subject: {config['id']}")
        mean, lcb, ucb = bootstrap_mean(values, stable_seed("competence", config["id"]))
        fraction = float(np.mean(values > 0.5))
        passed = bool(mean >= 0.60 and lcb > 0.55 and fraction >= 0.70)
        row = {
            "config": config["id"],
            "subject_count": len(values),
            "mean_subject_BA": mean,
            "subject_bootstrap_CI95_L": lcb,
            "subject_bootstrap_CI95_U": ucb,
            "fraction_subject_BA_gt_0p5": fraction,
            "competence_gate_pass": passed,
        }
        aggregate_rows.append(row)
        aggregate_by_config[config["id"]] = row
    winner_config = max(configs, key=lambda value: (aggregate_by_config[value["id"]]["mean_subject_BA"], value["id"]))
    winner = aggregate_by_config[winner_config["id"]]
    gate_pass = bool(winner["competence_gate_pass"])
    write_csv(RESULTS / "REPRESENTATION_COMPETENCE_SUBJECT_RESULTS.csv", subject_rows)
    write_csv(RESULTS / "REPRESENTATION_COMPETENCE_RESULTS.csv", aggregate_rows)
    result = {
        "terminal_state": "REPRESENTATION_COMPETENCE_PASS" if gate_pass else "REPRESENTATION_COMPETENCE_FAIL",
        "selection_basis": "task-only five-fold unseen-subject S3 mean subject balanced accuracy",
        "winner": winner_config["id"],
        "winner_metrics": winner,
        "gate": "mean>=0.60 AND bootstrap LCB95>0.55 AND fraction(BA_i>0.5)>=0.70",
        "gate_pass": gate_pass,
        "subject_count": 41,
        "outer_test_used": False,
        "actionability_used_for_selection": False,
        "checkpoints": checkpoint_rows,
        "elapsed_seconds": time.time() - started,
    }
    write_json(RESULTS / "REPRESENTATION_COMPETENCE_RESULT.json", result)
    if not gate_pass:
        final = {
            "terminal_state": "REPRESENTATION_COMPETENCE_FAIL",
            "scientific_conclusion": "EEGNet did not meet the prospectively frozen competence gate; H1-H5 and AGDI were not run.",
            "agdi_training_authorized": False,
            "outer_test_used": False,
            "winner": winner_config["id"],
            "winner_metrics": winner,
        }
        write_json(OUT / "FINAL_DECISION.json", final)
        write_report(final)
    else:
        frozen = {
            "status": "REPRESENTATION_FROZEN",
            "implementation_id": IMPLEMENTATION_ID,
            "git_commit": git_commit(),
            "architecture": "EEGNet",
            "config": winner_config,
            "selection_is_task_only": True,
            "selection_metrics": winner,
            "competence_checkpoint_set": [
                value for value in checkpoint_rows if value["config"] == winner_config["id"]
            ],
            "actionability_based_reselection_forbidden": True,
            "outer_test_state": "OUTER_TEST_LOCKED",
        }
        write_once(FROZEN_PATH, frozen)
    print(json.dumps(clean(result), indent=2))
    return result


def discovered_basis(arrays: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    subjects = arrays["subjects"].astype(str).tolist()
    h = arrays["embeddings"].astype(np.float64)
    sid = arrays["subject_index"].astype(int)
    session = arrays["session"].astype(int)
    centroids: dict[tuple[int, int], np.ndarray] = {}
    session_means: dict[int, np.ndarray] = {}
    for ses in (0, 1):
        values = []
        for index in range(len(subjects)):
            mask = (sid == index) & (session == ses)
            if not np.any(mask):
                raise RuntimeError(f"Discovery subject lacks ses-{ses}: {subjects[index]}")
            centroids[(index, ses)] = h[mask].mean(axis=0)
            values.append(centroids[(index, ses)])
        session_means[ses] = np.mean(values, axis=0)
    a = np.stack([centroids[(index, 0)] - session_means[0] for index in range(len(subjects))])
    b = np.stack([centroids[(index, 1)] - session_means[1] for index in range(len(subjects))])
    cross = (a.T @ b + b.T @ a) / (2 * max(len(subjects) - 1, 1))
    eigenvalues, eigenvectors = np.linalg.eigh((cross + cross.T) / 2)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    basis = eigenvectors[:, order].astype(np.float64)
    error = float(np.linalg.norm(basis.T @ basis - np.eye(32), ord="fro"))
    if error > 1e-8 or not np.isfinite(basis).all():
        raise RuntimeError(f"Invalid persistent basis: orthogonality error {error}")
    center = h.mean(axis=0)
    return basis, eigenvalues, center, session_means


def persistence_values(
    arrays: Mapping[str, np.ndarray],
    basis: np.ndarray,
    random: Sequence[np.ndarray],
    session_means: Mapping[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    subjects = arrays["subjects"].astype(str).tolist()
    h = arrays["embeddings"].astype(np.float64)
    sid = arrays["subject_index"].astype(int)
    session = arrays["session"].astype(int)
    centroids: dict[tuple[int, int], np.ndarray] = {}
    for index in range(len(subjects)):
        for ses in (0, 1):
            mask = (sid == index) & (session == ses)
            centroids[(index, ses)] = h[mask].mean(axis=0) - session_means[ses]

    def advantage(vectors: Mapping[tuple[int, int], np.ndarray]) -> np.ndarray:
        values = np.empty(len(subjects), dtype=np.float64)
        for index in range(len(subjects)):
            anchor = vectors[(index, 0)]
            genuine = float(np.sum((anchor - vectors[(index, 1)]) ** 2))
            impostor = [
                float(np.sum((anchor - vectors[(other, 1)]) ** 2))
                for other in range(len(subjects))
                if other != index
            ]
            values[index] = float(np.mean(impostor) - genuine)
        return values

    keys = sorted(centroids)
    residual = np.stack([centroids[key] for key in keys])
    target = residual @ basis @ basis.T
    candidate = advantage({key: target[index] for index, key in enumerate(keys)})
    controls = []
    for random_basis in random:
        delta = exact_matched_delta(residual, target, random_basis)
        controls.append(advantage({key: delta[index] for index, key in enumerate(keys)}))
    return candidate, np.stack(controls)


def per_subject_finite(
    base_logits: np.ndarray,
    candidate_logits: np.ndarray,
    random_logits: Sequence[np.ndarray],
    subject_index: np.ndarray,
    count: int,
) -> np.ndarray:
    values = np.empty(count, dtype=np.float64)
    for index in range(count):
        mask = subject_index == index
        candidate = centered_rms(candidate_logits[mask] - base_logits[mask])
        control = float(
            np.mean([centered_rms(value[mask] - base_logits[mask]) for value in random_logits])
        )
        values[index] = candidate / max(control, EPS)
    return values


def audit(device: torch.device, workers: int) -> dict[str, Any]:
    scope, _ = require_development_protocol()
    if not FROZEN_PATH.is_file():
        raise RuntimeError("Representation is not frozen; run competence first")
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    config = dict(frozen["config"])
    subjects = list(map(str, scope["allowed_subjects"]))
    h1_subject: dict[str, list[float]] = {name: [] for name, _, _ in BLOCKS}
    h1_subject_ids: dict[str, list[str]] = {name: [] for name, _, _ in BLOCKS}
    finite_subject: dict[str, list[float]] = {name: [] for name, _, _ in BLOCKS}
    finite_subject_ids: dict[str, list[str]] = {name: [] for name, _, _ in BLOCKS}
    outcome_values: dict[str, dict[str, list[Any]]] = {
        name: {key: [] for key in ("subject", "fold", "u_abs", "u_spec", "ba_delta", "ba_random", "ba_specific")}
        for name, _, _ in BLOCKS
    }
    local_fold_values: dict[str, list[dict[str, Any]]] = {name: [] for name, _, _ in BLOCKS}
    h1_random_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    fold_checkpoints: list[dict[str, Any]] = []
    started = time.time()

    for fold in range(5):
        outcome, discovery, model_fit = audit_roles(scope, fold)
        checkpoint = MODEL / "audit" / f"EEGNet_fold-{fold}.pt"
        model, payload = get_or_train(
            checkpoint, model_fit, config, f"audit-fold-{fold}", device, workers
        )
        fold_checkpoints.append(
            {
                "fold": fold,
                "checkpoint": str(checkpoint.relative_to(REPO_ROOT)).replace("\\", "/"),
                "checkpoint_sha256": payload["checkpoint_sha256"],
                "model_state_sha256": payload["model_state_sha256"],
                "model_fit_subjects_hash": sha_lines(model_fit),
            }
        )
        discovery_arrays = infer(model, discovery, [0, 1], device, workers)
        outcome_arrays = infer(model, outcome, [2], device, workers)
        embedding_path = OUT / "cache" / "fold_embeddings" / f"fold-{fold}.npz"
        save_npz(
            embedding_path,
            discovery_embeddings=discovery_arrays["embeddings"].astype(np.float16),
            discovery_logits=discovery_arrays["logits"].astype(np.float32),
            discovery_labels=discovery_arrays["labels"],
            discovery_subject_index=discovery_arrays["subject_index"],
            discovery_session=discovery_arrays["session"],
            discovery_subjects=discovery_arrays["subjects"],
            outcome_embeddings=outcome_arrays["embeddings"].astype(np.float16),
            outcome_logits=outcome_arrays["logits"].astype(np.float32),
            outcome_labels=outcome_arrays["labels"],
            outcome_subject_index=outcome_arrays["subject_index"],
            outcome_session=outcome_arrays["session"],
            outcome_subjects=outcome_arrays["subjects"],
        )
        basis_all, eigenvalues, center, session_means = discovered_basis(discovery_arrays)
        basis_path = MODEL / "audit" / f"persistent_basis_fold-{fold}.npz"
        save_npz(
            basis_path,
            basis=basis_all.astype(np.float32),
            eigenvalues=eigenvalues.astype(np.float64),
            center=center.astype(np.float32),
            session_means=np.stack([session_means[0], session_means[1]]).astype(np.float32),
            discovery_subjects=np.asarray(discovery),
        )
        weight = model.head.weight.detach().cpu().numpy().astype(np.float64)
        centered_weight = weight - weight.mean(axis=0, keepdims=True)
        outcome_h = outcome_arrays["embeddings"].astype(np.float64)
        outcome_z = outcome_arrays["logits"].astype(np.float64)
        outcome_y = outcome_arrays["labels"].astype(int)
        outcome_sid = outcome_arrays["subject_index"].astype(int)
        # The finite erasure is exactly the alpha=1 AGDI readout W(I-P)h+b.
        # The discovery center is used for basis estimation, not silently
        # added back as an intervention-specific bias.
        outcome_residual = outcome_h
        base_ce = ce_rows(outcome_z, outcome_y)
        base_pred = outcome_z.argmax(1)
        base_prob = softmax(outcome_z)
        discovery_h = discovery_arrays["embeddings"].astype(np.float64)
        discovery_z = discovery_arrays["logits"].astype(np.float64)
        discovery_sid = discovery_arrays["subject_index"].astype(int)
        discovery_residual = discovery_h
        for index, subject in enumerate(outcome):
            mask = outcome_sid == index
            baseline_rows.append(
                {
                    "fold": fold,
                    "subject": subject,
                    "n_S3_trials": int(mask.sum()),
                    "baseline_BA": balanced_accuracy_score(outcome_y[mask], base_pred[mask]),
                    "baseline_CE": float(base_ce[mask].mean()),
                }
            )

        for block_name, start, end in BLOCKS:
            block = basis_all[:, start:end]
            rank = end - start
            random = random_bases(rank, fold, block_name)
            candidate_persistence, random_persistence = persistence_values(
                discovery_arrays, block, random, session_means
            )
            persistence_specific = candidate_persistence - random_persistence.mean(axis=0)
            h1_subject[block_name].extend(persistence_specific.tolist())
            h1_subject_ids[block_name].extend(discovery)
            for subject_index, subject in enumerate(discovery):
                h1_random_rows.append(
                    {
                        "fold": fold,
                        "block": block_name,
                        "subject": subject,
                        "candidate_persistence": candidate_persistence[subject_index],
                        "random_persistence_mean": random_persistence[:, subject_index].mean(),
                        "persistence_specific": persistence_specific[subject_index],
                    }
                )

            discovery_delta = discovery_residual @ block @ block.T
            discovery_candidate_z = discovery_z - discovery_delta @ weight.T
            discovery_random_z = []
            for random_basis in random:
                delta = exact_matched_delta(discovery_residual, discovery_delta, random_basis)
                discovery_random_z.append(discovery_z - delta @ weight.T)
            finite = per_subject_finite(
                discovery_z,
                discovery_candidate_z,
                discovery_random_z,
                discovery_sid,
                len(discovery),
            )
            finite_subject[block_name].extend(finite.tolist())
            finite_subject_ids[block_name].extend(discovery)
            candidate_local = float(np.sum((centered_weight @ block) ** 2) / rank)
            random_local = np.asarray(
                [np.sum((centered_weight @ value) ** 2) / rank for value in random], dtype=np.float64
            )
            local_fold_values[block_name].append(
                {
                    "fold": fold,
                    "candidate": candidate_local,
                    "random": random_local,
                    "ratio": candidate_local / max(float(random_local.mean()), EPS),
                }
            )

            target_delta = outcome_residual @ block @ block.T
            candidate_z = outcome_z - target_delta @ weight.T
            candidate_ce = ce_rows(candidate_z, outcome_y)
            candidate_pred = candidate_z.argmax(1)
            candidate_prob = softmax(candidate_z)
            random_z: list[np.ndarray] = []
            random_ce: list[np.ndarray] = []
            for random_basis in random:
                delta = exact_matched_delta(outcome_residual, target_delta, random_basis)
                z = outcome_z - delta @ weight.T
                random_z.append(z)
                random_ce.append(ce_rows(z, outcome_y))
            random_ce_array = np.stack(random_ce)
            for subject_index, subject in enumerate(outcome):
                mask = outcome_sid == subject_index
                base_ba = balanced_accuracy_score(outcome_y[mask], base_pred[mask])
                candidate_ba = balanced_accuracy_score(outcome_y[mask], candidate_pred[mask])
                random_ba = np.asarray(
                    [balanced_accuracy_score(outcome_y[mask], z.argmax(1)[mask]) for z in random_z]
                )
                u_abs = float(np.mean(candidate_ce[mask] - base_ce[mask]))
                u_random = float(np.mean(random_ce_array[:, mask] - base_ce[None, mask]))
                ba_delta = candidate_ba - base_ba
                ba_random = float(random_ba.mean() - base_ba)
                ba_specific = ba_delta - ba_random
                values = outcome_values[block_name]
                values["subject"].append(subject)
                values["fold"].append(fold)
                values["u_abs"].append(u_abs)
                values["u_spec"].append(u_abs - u_random)
                values["ba_delta"].append(ba_delta)
                values["ba_random"].append(ba_random)
                values["ba_specific"].append(ba_specific)
                subject_rows.append(
                    {
                        "fold": fold,
                        "block": block_name,
                        "subject": subject,
                        "n_S3_trials": int(mask.sum()),
                        "base_BA": base_ba,
                        "candidate_BA": candidate_ba,
                        "random_BA_mean": float(random_ba.mean()),
                        "u_abs": u_abs,
                        "u_random": u_random,
                        "u_spec": u_abs - u_random,
                        "delta_BA": ba_delta,
                        "delta_BA_random": ba_random,
                        "delta_BA_specific": ba_specific,
                        "logit_RMS": centered_rms(candidate_z[mask] - outcome_z[mask]),
                        "margin_displacement": float(
                            np.mean(
                                np.abs(
                                    true_margin(candidate_z[mask], outcome_y[mask])
                                    - true_margin(outcome_z[mask], outcome_y[mask])
                                )
                            )
                        ),
                        "prediction_flip_rate": float(np.mean(candidate_pred[mask] != base_pred[mask])),
                        "total_variation": float(
                            np.mean(0.5 * np.sum(np.abs(candidate_prob[mask] - base_prob[mask]), axis=1))
                        ),
                    }
                )
                for draw in range(RANDOM_DRAWS):
                    random_rows.append(
                        {
                            "fold": fold,
                            "block": block_name,
                            "subject": subject,
                            "draw": draw,
                            "u_random": float(np.mean(random_ce_array[draw, mask] - base_ce[mask])),
                            "delta_BA_random": float(random_ba[draw] - base_ba),
                            "finite_logit_RMS": centered_rms(random_z[draw][mask] - outcome_z[mask]),
                        }
                    )
            basis_rows.append(
                {
                    "fold": fold,
                    "block": block_name,
                    "rank": rank,
                    "eigenvalue_sum": float(eigenvalues[start:end].sum()),
                    "minimum_eigenvalue": float(eigenvalues[start:end].min()),
                    "basis_sha256": sha256_file(basis_path),
                    "embedding_sha256": sha256_file(embedding_path),
                }
            )
            print(f"[audit fold={fold}] {block_name} complete", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if any(len(values) != 41 for values in h1_subject.values()) or any(
        len(values) != 41 for values in finite_subject.values()
    ):
        raise RuntimeError("Cross-fitting did not yield one discovery result per subject/block")
    if any(len(value["subject"]) != 41 for value in outcome_values.values()):
        raise RuntimeError("Cross-fitting did not yield one outcome result per subject/block")

    p_raw = {name: {} for name in ("H1", "H2", "H3_finite", "H4", "protected")}
    persistence_rows: list[dict[str, Any]] = []
    utility_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    actionability_rows: list[dict[str, Any]] = []
    block_cache: dict[str, dict[str, Any]] = {}
    for block_name, start, end in BLOCKS:
        persistence = np.asarray(h1_subject[block_name], dtype=np.float64)
        finite = np.asarray(finite_subject[block_name], dtype=np.float64)
        outcome = outcome_values[block_name]
        u_abs = np.asarray(outcome["u_abs"], dtype=np.float64)
        u_spec = np.asarray(outcome["u_spec"], dtype=np.float64)
        ba_delta = np.asarray(outcome["ba_delta"], dtype=np.float64)
        ba_random = np.asarray(outcome["ba_random"], dtype=np.float64)
        ba_specific = np.asarray(outcome["ba_specific"], dtype=np.float64)
        p_summary = bootstrap_mean(persistence, stable_seed("bootstrap", block_name, "H1"))
        u_abs_summary = bootstrap_mean(u_abs, stable_seed("bootstrap", block_name, "u_abs"))
        u_summary = bootstrap_mean(u_spec, stable_seed("bootstrap", block_name, "H2"))
        finite_summary = bootstrap_mean(finite, stable_seed("bootstrap", block_name, "H3"))
        ba_summary = bootstrap_mean(ba_specific, stable_seed("bootstrap", block_name, "H4"))
        p_raw["H1"][block_name] = signflip_p(persistence, "positive")
        p_raw["H2"][block_name] = signflip_p(u_spec, "negative")
        p_raw["protected"][block_name] = signflip_p(u_spec, "positive")
        p_raw["H3_finite"][block_name] = signflip_p(np.log(np.maximum(finite, EPS)), "positive")
        p_raw["H4"][block_name] = signflip_p(ba_specific, "positive")
        local = local_fold_values[block_name]
        candidate_local = float(np.mean([value["candidate"] for value in local]))
        random_local_by_draw = np.mean(np.stack([value["random"] for value in local]), axis=0)
        local_ratio_draws = candidate_local / np.maximum(random_local_by_draw, EPS)
        local_ratio = candidate_local / max(float(random_local_by_draw.mean()), EPS)
        local_lcb, local_ucb = map(float, np.quantile(local_ratio_draws, [0.025, 0.975]))
        local_p = float((1 + np.sum(random_local_by_draw >= candidate_local)) / (1 + RANDOM_DRAWS))
        folds = np.asarray(outcome["fold"], dtype=int)
        loso = [float(np.delete(ba_specific, index).mean()) for index in range(len(ba_specific))]
        lofo = [float(ba_specific[folds != fold].mean()) for fold in range(5)]
        nonnegative = float(np.mean(ba_specific >= 0))
        stability = bool(min(loso) > 0 and min(lofo) > 0 and nonnegative >= 0.60)
        persistence_rows.append(
            {
                "block": block_name,
                "rank": end - start,
                "mean_specific_advantage": p_summary[0],
                "CI95_L": p_summary[1],
                "CI95_U": p_summary[2],
                "p_raw": p_raw["H1"][block_name],
            }
        )
        utility_rows.append(
            {
                "block": block_name,
                "rank": end - start,
                "u_abs_mean": u_abs_summary[0],
                "u_abs_CI95_L": u_abs_summary[1],
                "u_abs_CI95_U": u_abs_summary[2],
                "u_spec_mean": u_summary[0],
                "u_spec_CI95_L": u_summary[1],
                "u_spec_CI95_U": u_summary[2],
                "p_raw_harmful": p_raw["H2"][block_name],
                "p_raw_protected": p_raw["protected"][block_name],
            }
        )
        decision_rows.append(
            {
                "block": block_name,
                "rank": end - start,
                "local_energy": candidate_local,
                "local_random_mean": float(random_local_by_draw.mean()),
                "local_ratio": local_ratio,
                "local_ratio_CI95_L": local_lcb,
                "local_ratio_CI95_U": local_ucb,
                "local_randomization_p": local_p,
                "finite_ratio_mean": finite_summary[0],
                "finite_ratio_CI95_L": finite_summary[1],
                "finite_ratio_CI95_U": finite_summary[2],
                "finite_p_raw": p_raw["H3_finite"][block_name],
            }
        )
        actionability_rows.append(
            {
                "block": block_name,
                "rank": end - start,
                "delta_BA_mean": float(ba_delta.mean()),
                "delta_BA_random_mean": float(ba_random.mean()),
                "delta_BA_specific_mean": ba_summary[0],
                "delta_BA_specific_CI95_L": ba_summary[1],
                "delta_BA_specific_CI95_U": ba_summary[2],
                "p_raw": p_raw["H4"][block_name],
                "minimum_LOSO_mean": min(loso),
                "minimum_leave_one_fold_out_mean": min(lofo),
                "nonnegative_subject_fraction": nonnegative,
                "median_specific": float(np.median(ba_specific)),
                "worst_subject_specific": float(np.min(ba_specific)),
                "stability_preliminary": stability,
            }
        )
        block_cache[block_name] = {
            "persistence": p_summary,
            "u": u_summary,
            "finite": finite_summary,
            "ba": ba_summary,
            "local": (local_ratio, local_lcb, local_ucb, local_p),
            "stability": stability,
            "nonnegative": nonnegative,
        }

    adjusted = {family: holm(values) for family, values in p_raw.items()}
    for row in persistence_rows:
        row["p_holm"] = adjusted["H1"][row["block"]]
    for row in utility_rows:
        row["p_holm_harmful"] = adjusted["H2"][row["block"]]
        row["p_holm_protected"] = adjusted["protected"][row["block"]]
    for row in decision_rows:
        row["finite_p_holm"] = adjusted["H3_finite"][row["block"]]
    for row in actionability_rows:
        row["p_holm"] = adjusted["H4"][row["block"]]

    assignments: list[dict[str, Any]] = []
    for block_name, start, end in BLOCKS:
        value = block_cache[block_name]
        h1 = bool(value["persistence"][1] > 0 and adjusted["H1"][block_name] < 0.05)
        h2 = bool(value["u"][2] < 0 and adjusted["H2"][block_name] < 0.05)
        h3 = bool(
            value["local"][1] > 1
            and value["local"][3] < 0.05
            and value["finite"][1] > 1
            and adjusted["H3_finite"][block_name] < 0.05
        )
        h4 = bool(value["ba"][1] > 0 and value["ba"][0] >= 0.005 and adjusted["H4"][block_name] < 0.05)
        h5 = bool(value["stability"])
        protected = bool(
            h1
            and h3
            and value["u"][1] > 0
            and adjusted["protected"][block_name] < 0.05
        )
        actionable = bool(h1 and h2 and h3 and h4 and h5)
        if actionable:
            assignment, action = "ACTIONABLE-HARMFUL", "SUPPRESS_CANDIDATE"
        elif protected:
            assignment, action = "PROTECTED", "PRESERVE"
        elif h1 and not h3:
            assignment, action = "DECISION-NULL / WEAKLY ACTIVE", "NO_OP"
        elif h1 and h3:
            assignment, action = "DECISION-ACTIVE NON-ACTIONABLE", "NO_OP"
        else:
            assignment, action = "UNCERTAIN", "NO_OP"
        assignments.append(
            {
                "block": block_name,
                "rank": end - start,
                "H1": h1,
                "H2": h2,
                "H3": h3,
                "H4": h4,
                "H5": h5,
                "all_H1_H5": actionable,
                "protected_utility_gate": protected,
                "assignment": assignment,
                "action": action,
            }
        )

    write_csv(RESULTS / "PERSISTENCE_RESULTS.csv", persistence_rows)
    write_csv(RESULTS / "SIGNED_UTILITY_RESULTS.csv", utility_rows)
    write_csv(RESULTS / "DECISION_DEPENDENCE_RESULTS.csv", decision_rows)
    write_csv(RESULTS / "ACTIONABILITY_RESULTS.csv", actionability_rows)
    write_csv(RESULTS / "BLOCK_ASSIGNMENTS.csv", assignments)
    write_csv(RESULTS / "WBCIC_AUDIT_SUBJECT_RESULTS.csv", subject_rows)
    write_csv(RESULTS / "WBCIC_AUDIT_RANDOM_SUBJECT_RESULTS.csv", random_rows)
    write_csv(RESULTS / "WBCIC_PERSISTENCE_SUBJECT_RESULTS.csv", h1_random_rows)
    write_csv(RESULTS / "WBCIC_AUDIT_BASELINE_SUBJECT_RESULTS.csv", baseline_rows)
    write_csv(RESULTS / "PERSISTENCE_BASIS_RESULTS.csv", basis_rows)
    found = [row["block"] for row in assignments if row["all_H1_H5"]]
    final = {
        "terminal_state": "AGDI_AUTHORIZED" if found else "WBCIC_AUDIT_NO_ACTIONABLE_HARMFUL",
        "scientific_conclusion": (
            "At least one prospectively tested WBCIC block passed H1-H5."
            if found
            else "No prospectively tested WBCIC block passed all H1-H5."
        ),
        "next_action": "AGDI_TRAINING_AUTHORIZED" if found else "STOP_AGDI_NO_ACTIONABLE_TARGET",
        "agdi_training_authorized": bool(found),
        "actionable_harmful_blocks": found,
        "protected_blocks": [row["block"] for row in assignments if row["assignment"] == "PROTECTED"],
        "outer_test_state": "OUTER_TEST_LOCKED",
        "outer_test_used": False,
        "development_subjects": 41,
        "cross_fitting": "five subject-disjoint audit folds",
        "random_draws": RANDOM_DRAWS,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "multiplicity": "Holm within each frozen four-block family",
        "baseline_mean_subject_BA": float(np.mean([row["baseline_BA"] for row in baseline_rows])),
        "fold_checkpoints": fold_checkpoints,
        "elapsed_seconds": time.time() - started,
        "assignments": assignments,
        "limitations": [
            "Each fold-specific cross-session basis is estimated from only 8-9 discovery subjects; components beyond the empirical positive cross-covariance rank are therefore weakly identified and must pass H1 rather than being presumed persistent.",
            "The primary protocol has one future session (S3); H5 uses leave-one-subject and leave-one-audit-fold stability, not a second independent future-session outcome.",
        ],
    }
    write_json(OUT / "FINAL_DECISION.json", final)
    write_report(final)
    print(json.dumps(clean(final), indent=2))
    return final


def block_union(basis: np.ndarray, names: Sequence[str]) -> np.ndarray:
    columns = [basis[:, start:end] for name, start, end in BLOCKS if name in set(names)]
    return np.concatenate(columns, axis=1) if columns else np.empty((32, 0), dtype=np.float64)


def residual_harmful(protected: np.ndarray, harmful: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    protected = np.asarray(protected, dtype=np.float64)
    harmful = np.asarray(harmful, dtype=np.float64)
    if harmful.size == 0:
        return np.empty((32, 0), dtype=np.float64), {
            "raw_harmful_rank": 0,
            "protected_rank": int(protected.shape[1]),
            "overlap_energy_fraction": 0.0,
            "residual_harmful_rank": 0,
            "residual_energy_fraction": 0.0,
        }
    if protected.size:
        protected, _ = np.linalg.qr(protected)
        protected = protected[:, : np.asarray(protected).shape[1]]
        raw_residual = harmful - protected @ (protected.T @ harmful)
        overlap = harmful - raw_residual
    else:
        raw_residual = harmful.copy()
        overlap = np.zeros_like(harmful)
    u, singular, _ = np.linalg.svd(raw_residual, full_matrices=False)
    threshold = max(float(singular[0]) if len(singular) else 0.0, 1.0) * 1e-10
    keep = singular > threshold
    residual = u[:, keep]
    raw_energy = float(np.sum(harmful * harmful))
    value = {
        "raw_harmful_rank": int(harmful.shape[1]),
        "protected_rank": int(protected.shape[1]),
        "overlap_energy_fraction": float(np.sum(overlap * overlap) / max(raw_energy, EPS)),
        "residual_harmful_rank": int(residual.shape[1]),
        "residual_energy_fraction": float(np.sum(raw_residual * raw_residual) / max(raw_energy, EPS)),
        "singular_values": singular,
    }
    return residual, value


def fold_artifacts(fold: int, device: torch.device) -> tuple[EEGNet, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    checkpoint = MODEL / "audit" / f"EEGNet_fold-{fold}.pt"
    model, _ = load_model(checkpoint, device)
    with np.load(OUT / "cache" / "fold_embeddings" / f"fold-{fold}.npz", allow_pickle=False) as item:
        arrays = {
            "embeddings": item["outcome_embeddings"].astype(np.float64),
            "logits": item["outcome_logits"].astype(np.float64),
            "labels": item["outcome_labels"].astype(int),
            "subject_index": item["outcome_subject_index"].astype(int),
            "subjects": item["outcome_subjects"].astype(str),
        }
    with np.load(MODEL / "audit" / f"persistent_basis_fold-{fold}.npz", allow_pickle=False) as item:
        basis = item["basis"].astype(np.float64)
        center = item["center"].astype(np.float64)
    return model, arrays, basis, center


def evaluate_union(
    names: Sequence[str], alpha: float, device: torch.device
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in range(5):
        model, arrays, basis, _ = fold_artifacts(fold, device)
        union = block_union(basis, names)
        if union.size:
            union, _ = np.linalg.qr(union)
            union = union[:, : sum(end - start for name, start, end in BLOCKS if name in set(names))]
        weight = model.head.weight.detach().cpu().numpy().astype(np.float64)
        bias = model.head.bias.detach().cpu().numpy().astype(np.float64)
        after_weight = (
            weight
            if union.size == 0
            else weight @ (np.eye(32) - float(alpha) * union @ union.T)
        )
        after_logits = arrays["embeddings"] @ after_weight.T + bias
        before_prediction = arrays["logits"].argmax(1)
        after_prediction = after_logits.argmax(1)
        for index, subject in enumerate(arrays["subjects"]):
            mask = arrays["subject_index"] == index
            before_ba = balanced_accuracy_score(arrays["labels"][mask], before_prediction[mask])
            after_ba = balanced_accuracy_score(arrays["labels"][mask], after_prediction[mask])
            rows.append(
                {
                    "fold": fold,
                    "subject": subject,
                    "baseline_BA": before_ba,
                    "intervention_BA": after_ba,
                    "delta_BA": after_ba - before_ba,
                }
            )
        del model
    return rows


def agdi(device: torch.device, workers: int) -> dict[str, Any]:
    del workers  # all required fold embeddings are frozen before this stage
    scope, _ = require_development_protocol()
    final_path = OUT / "FINAL_DECISION.json"
    if not final_path.is_file():
        raise RuntimeError("Run the H1-H5 audit before AGDI")
    previous = json.loads(final_path.read_text(encoding="utf-8"))
    if previous.get("agdi_training_authorized") is not True:
        if previous.get("next_action") != "STOP_AGDI_NO_ACTIONABLE_TARGET":
            raise RuntimeError("AGDI authorization state is malformed")
        print(json.dumps({"terminal_state": previous["terminal_state"], "next_action": previous["next_action"]}, indent=2))
        reproduce()
        return previous

    import pandas as pd

    assignments = pd.read_csv(RESULTS / "BLOCK_ASSIGNMENTS.csv").to_dict("records")
    harmful_names = [str(row["block"]) for row in assignments if bool(row["all_H1_H5"])]
    protected_names = [str(row["block"]) for row in assignments if str(row["assignment"]) == "PROTECTED"]
    if not harmful_names:
        raise RuntimeError("AGDI authorization exists without an actionable harmful block")
    protocol = {
        "status": "AGDI_PROTOCOL_FROZEN",
        "implementation_id": IMPLEMENTATION_ID,
        "git_commit": git_commit(),
        "actionable_harmful_blocks": harmful_names,
        "protected_blocks": protected_names,
        "encoder": "frozen",
        "primary": "W_alpha = W_0 (I - alpha P_H)",
        "alpha_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
        "alpha_selection_order": [
            "protected relative error <= 1e-6",
            "harmful local dependence ratio < 1",
            "random/other local dependence ratio within [0.90,1.10]",
            "specific BA gain bootstrap LCB95 > 0",
            "BA gain bootstrap LCB95 > 0 and mean >= 0.005",
            "highest mean development BA; smaller alpha breaks ties",
        ],
        "protected_hard_tolerance": 1e-6,
        "random_dependence_equivalence_ratio": [0.90, 1.10],
        "dependence_transfer_failure_ratio": 1.25,
        "same_rank_random_controls": 100,
        "adapter_secondary": {"status": "SECONDARY_ONLY_NOT_USED_FOR_PRIMARY_SELECTION"},
        "outer_test_state": "OUTER_TEST_LOCKED",
    }
    write_once(PROTOCOL / "AGDI_PROTOCOL_LOCK.json", protocol)

    alpha_subject_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    dependence_rows: list[dict[str, Any]] = []
    for fold in range(5):
        model, arrays, basis, _ = fold_artifacts(fold, device)
        protected = block_union(basis, protected_names)
        harmful_raw = block_union(basis, harmful_names)
        harmful, overlap = residual_harmful(protected, harmful_raw)
        overlap_rows.append({"fold": fold, **clean(overlap)})
        if harmful.shape[1] == 0:
            final = {
                **previous,
                "terminal_state": "AGDI_NO_OP_PROTECTED_OVERLAP",
                "next_action": "STOP_AGDI_NO_ACTIONABLE_TARGET",
                "agdi_training_authorized": False,
                "outer_test_used": False,
            }
            write_json(final_path, final)
            write_json(RESULTS / "AGDI_PROTECTED_OVERLAP.json", {"folds": overlap_rows})
            write_report(final)
            return final
        excluded = np.concatenate([protected, harmful], axis=1) if protected.size else harmful
        complement_rank = 32 - int(np.linalg.matrix_rank(excluded))
        if complement_rank < harmful.shape[1]:
            final = {
                **previous,
                "terminal_state": "AGDI_FAIL_NO_SPECIFICITY",
                "scientific_conclusion": "No orthogonal same-rank random-control subspace exists after Protected/Harmful exclusion.",
                "agdi_training_authorized": False,
                "outer_test_used": False,
            }
            write_json(final_path, final)
            write_report(final)
            return final
        controls = random_bases(harmful.shape[1], fold, "AGDI", excluded=excluded)
        weight = model.head.weight.detach().cpu().numpy().astype(np.float64)
        bias = model.head.bias.detach().cpu().numpy().astype(np.float64)
        h = arrays["embeddings"]
        y = arrays["labels"]
        sid = arrays["subject_index"]
        base_logits = h @ weight.T + bias
        base_prediction = base_logits.argmax(1)
        centered_weight = weight - weight.mean(axis=0, keepdims=True)
        harmful_before = float(np.sum((centered_weight @ harmful) ** 2))
        for alpha in protocol["alpha_grid"]:
            after_weight = weight @ (np.eye(32) - alpha * harmful @ harmful.T)
            after_centered = after_weight - after_weight.mean(axis=0, keepdims=True)
            candidate_logits = h @ after_weight.T + bias
            candidate_prediction = candidate_logits.argmax(1)
            random_logits = []
            for control in controls:
                random_weight = weight @ (np.eye(32) - alpha * control @ control.T)
                random_logits.append(h @ random_weight.T + bias)
            protected_error = 0.0
            if protected.size:
                protected_error = float(
                    np.linalg.norm(after_weight @ protected - weight @ protected)
                    / max(np.linalg.norm(weight @ protected), EPS)
                )
            harmful_after = float(np.sum((after_centered @ harmful) ** 2))
            random_other_ratios = []
            for control in controls:
                before = float(np.sum((centered_weight @ control) ** 2))
                after = float(np.sum((after_centered @ control) ** 2))
                random_other_ratios.append(after / max(before, EPS))
            dependence_rows.append(
                {
                    "fold": fold,
                    "alpha": alpha,
                    "harmful_ratio": harmful_after / max(harmful_before, EPS),
                    "protected_relative_error": protected_error,
                    "random_other_ratio": float(np.mean(random_other_ratios)),
                }
            )
            for index, subject in enumerate(arrays["subjects"]):
                mask = sid == index
                base_ba = balanced_accuracy_score(y[mask], base_prediction[mask])
                candidate_ba = balanced_accuracy_score(y[mask], candidate_prediction[mask])
                random_ba = np.asarray(
                    [balanced_accuracy_score(y[mask], logits.argmax(1)[mask]) for logits in random_logits]
                )
                alpha_subject_rows.append(
                    {
                        "fold": fold,
                        "subject": subject,
                        "alpha": alpha,
                        "baseline_BA": base_ba,
                        "AGDI_BA": candidate_ba,
                        "random_BA_mean": float(random_ba.mean()),
                        "delta_BA": candidate_ba - base_ba,
                        "delta_BA_random": float(random_ba.mean() - base_ba),
                        "delta_BA_specific": float(candidate_ba - random_ba.mean()),
                    }
                )
        del model

    write_json(
        RESULTS / "AGDI_PROTECTED_OVERLAP.json",
        {
            "status": "AGDI_PROTECTED_OVERLAP_AUDITED",
            "folds": overlap_rows,
            "protected_priority": True,
        },
    )
    alpha_rows: list[dict[str, Any]] = []
    for alpha in protocol["alpha_grid"]:
        rows = [row for row in alpha_subject_rows if float(row["alpha"]) == alpha]
        delta = np.asarray([row["delta_BA"] for row in rows], dtype=np.float64)
        specific = np.asarray([row["delta_BA_specific"] for row in rows], dtype=np.float64)
        delta_summary = bootstrap_mean(delta, stable_seed("AGDI", alpha, "delta"))
        specific_summary = bootstrap_mean(specific, stable_seed("AGDI", alpha, "specific"))
        dependence = [row for row in dependence_rows if float(row["alpha"]) == alpha]
        harmful_ratio = float(np.mean([row["harmful_ratio"] for row in dependence]))
        protected_error = float(max(row["protected_relative_error"] for row in dependence))
        random_ratio = float(np.mean([row["random_other_ratio"] for row in dependence]))
        eligible = bool(
            alpha > 0
            and protected_error <= protocol["protected_hard_tolerance"]
            and harmful_ratio < 1
            and protocol["random_dependence_equivalence_ratio"][0]
            <= random_ratio
            <= protocol["random_dependence_equivalence_ratio"][1]
            and specific_summary[1] > 0
            and delta_summary[1] > 0
            and delta_summary[0] >= 0.005
        )
        alpha_rows.append(
            {
                "alpha": alpha,
                "subject_count": len(rows),
                "delta_BA_mean": delta_summary[0],
                "delta_BA_CI95_L": delta_summary[1],
                "delta_BA_CI95_U": delta_summary[2],
                "specific_gain_mean": specific_summary[0],
                "specific_gain_CI95_L": specific_summary[1],
                "specific_gain_CI95_U": specific_summary[2],
                "harmful_dependence_ratio": harmful_ratio,
                "protected_relative_error_max": protected_error,
                "random_other_dependence_ratio": random_ratio,
                "eligible": eligible,
            }
        )
    eligible = [row for row in alpha_rows if row["eligible"]]
    selected = max(eligible, key=lambda row: (row["delta_BA_mean"], -row["alpha"])) if eligible else alpha_rows[0]
    for row in alpha_rows:
        row["selected"] = bool(row["alpha"] == selected["alpha"])
    write_csv(RESULTS / "AGDI_ALPHA_SELECTION.csv", alpha_rows)
    write_csv(RESULTS / "AGDI_RESULTS.csv", alpha_rows)
    write_csv(RESULTS / "AGDI_SUBJECT_RESULTS.csv", alpha_subject_rows)
    write_csv(RESULTS / "AGDI_POST_ACTION_AUDIT.csv", dependence_rows)

    selection_sets = {
        "baseline": [],
        "naive_global_persistence": [row["block"] for row in assignments if bool(row["H1"])],
        "persistence_only": [row["block"] for row in assignments if bool(row["H1"])],
        "utility_only": [row["block"] for row in assignments if bool(row["H2"])],
        "decision_only": [row["block"] for row in assignments if bool(row["H3"])],
        "P+U": [row["block"] for row in assignments if bool(row["H1"]) and bool(row["H2"])],
        "P+U+D": [row["block"] for row in assignments if bool(row["H1"]) and bool(row["H2"]) and bool(row["H3"])],
        "P+U+D+A": [row["block"] for row in assignments if bool(row["H1"]) and bool(row["H2"]) and bool(row["H3"]) and bool(row["H4"])],
        "full_AGDI": harmful_names,
    }
    ablation_rows: list[dict[str, Any]] = []
    for name, blocks in selection_sets.items():
        rows = evaluate_union(blocks, float(selected["alpha"]), device)
        values = np.asarray([row["delta_BA"] for row in rows], dtype=np.float64)
        summary = bootstrap_mean(values, stable_seed("AGDI-ablation", name))
        ablation_rows.append(
            {
                "ablation": name,
                "selected_blocks": "|".join(blocks),
                "alpha": selected["alpha"],
                "delta_BA_mean": summary[0],
                "delta_BA_CI95_L": summary[1],
                "delta_BA_CI95_U": summary[2],
                "deployable": name != "oracle",
            }
        )
    ablation_rows.append(
        {
            "ablation": "AGDI-Adapter",
            "selected_blocks": "|".join(harmful_names),
            "alpha": None,
            "delta_BA_mean": None,
            "delta_BA_CI95_L": None,
            "delta_BA_CI95_U": None,
            "deployable": False,
            "status": "SECONDARY_NOT_RUN_UNLESS_PRIMARY_ALPHA_PASSES",
        }
    )
    write_csv(RESULTS / "AGDI_ABLATIONS.csv", ablation_rows)

    if not eligible:
        reason = (
            "AGDI_FAIL_NO_SPECIFICITY"
            if any(row["delta_BA_CI95_L"] > 0 and row["specific_gain_CI95_L"] <= 0 for row in alpha_rows if row["alpha"] > 0)
            else "AGDI_FAIL_NO_GENERALIZATION_GAIN"
        )
        final = {
            **previous,
            "terminal_state": reason,
            "scientific_conclusion": "An actionable block was certified, but no nonzero frozen-grid AGDI alpha passed development efficacy and specificity gates.",
            "selected_alpha": 0.0,
            "agdi_training_authorized": False,
            "outer_test_used": False,
            "next_action": "STOP_AGDI_DEVELOPMENT_FAIL",
        }
        write_json(final_path, final)
        write_report(final)
        reproduce()
        print(json.dumps(clean(final), indent=2))
        return final

    # Freeze a single deployable model using all 41 development subjects only.
    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    all_subjects = list(map(str, scope["allowed_subjects"]))
    checkpoint = MODEL / "final" / "EEGNet_development_all.pt"
    final_model, payload = get_or_train(
        checkpoint,
        all_subjects,
        dict(frozen["config"]),
        "final-development-all",
        device,
        workers=0,
    )
    development_arrays = infer(final_model, all_subjects, [0, 1], device, workers=0)
    final_basis, final_eigenvalues, final_center, _ = discovered_basis(development_arrays)
    final_basis_path = MODEL / "final" / "persistent_basis_development_all.npz"
    save_npz(
        final_basis_path,
        basis=final_basis.astype(np.float32),
        eigenvalues=final_eigenvalues,
        center=final_center.astype(np.float32),
        development_subjects_hash=np.asarray(sha_lines(all_subjects)),
    )
    final_protected = block_union(final_basis, protected_names)
    final_harmful_raw = block_union(final_basis, harmful_names)
    final_harmful, final_overlap = residual_harmful(final_protected, final_harmful_raw)
    if final_harmful.shape[1] == 0:
        final = {
            **previous,
            "terminal_state": "AGDI_NO_OP_PROTECTED_OVERLAP",
            "selected_alpha": 0.0,
            "agdi_training_authorized": False,
            "outer_test_used": False,
        }
        write_json(final_path, final)
        write_report(final)
        return final
    write_json(
        RESULTS / "AGDI_PROTECTED_OVERLAP.json",
        {
            "status": "AGDI_PROTECTED_OVERLAP_AUDITED",
            "folds": overlap_rows,
            "final_development_basis": final_overlap,
        },
    )
    outer_lock = {
        "status": "AGDI_PRIMARY_PASS_OUTER_LOCKED",
        "implementation_id": IMPLEMENTATION_ID,
        "git_commit": git_commit(),
        "development_subjects_hash": sha_lines(all_subjects),
        "outer_subject_count": 10,
        "outer_subject_ids_present": False,
        "model_checkpoint": str(checkpoint.relative_to(REPO_ROOT)).replace("\\", "/"),
        "model_checkpoint_sha256": payload["checkpoint_sha256"],
        "model_state_sha256": payload["model_state_sha256"],
        "basis": str(final_basis_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "basis_sha256": sha256_file(final_basis_path),
        "harmful_blocks": harmful_names,
        "protected_blocks": protected_names,
        "selected_alpha": selected["alpha"],
        "metrics_frozen": [
            "subject_mean balanced accuracy",
            "paired subject delta and bootstrap CI",
            "specific gain versus 100 same-rank random suppressions",
            "subject non-harm fraction",
            "harmful/protected/random dependence ratios",
            "NLL secondary",
        ],
        "protected_tolerance": protocol["protected_hard_tolerance"],
        "random_equivalence_ratio": protocol["random_dependence_equivalence_ratio"],
        "outer_evaluation_authorized_once": True,
        "retraining_after_outer_forbidden": True,
    }
    write_once(PROTOCOL / "FINAL_OUTER_EVALUATION_LOCK.json", outer_lock)
    final = {
        **previous,
        "terminal_state": "AGDI_PRIMARY_PASS_OUTER_LOCKED",
        "scientific_conclusion": "A nonzero minimal AGDI intervention passed development cross-fitting and is frozen for one-time outer evaluation.",
        "selected_alpha": selected["alpha"],
        "development_AGDI_metrics": selected,
        "agdi_training_authorized": True,
        "outer_test_used": False,
        "next_action": "RUN_ONE_TIME_WBCIC_OUTER",
    }
    write_json(final_path, final)
    write_report(final)
    reproduce()
    print(json.dumps(clean(final), indent=2))
    return final


def write_report(final: Mapping[str, Any]) -> None:
    competence_result = RESULTS / "REPRESENTATION_COMPETENCE_RESULT.json"
    competence_value = json.loads(competence_result.read_text(encoding="utf-8")) if competence_result.exists() else {}
    lines = [
        "# PERSIST-EEG WBCIC EEGNet actionability report",
        "",
        f"Terminal state: `{final['terminal_state']}`",
        "",
        "## Frozen scope",
        "",
        "The primary cohort is the 51-subject Yang2025/NEMAR 2C core. Forty-one deterministic development subjects are used for task-only selection and five-fold actionability cross-fitting. Ten outer subjects remain sealed.",
        "",
        "Primary session target: `S1+S2 -> S3`. Backbone route: EEGNet only.",
        "",
        "## Representation competence",
        "",
    ]
    if competence_value:
        metric = competence_value.get("winner_metrics", {})
        lines.extend(
            [
                f"Selected task-only configuration: `{competence_value.get('winner')}`.",
                "",
                f"Mean subject BA: {metric.get('mean_subject_BA', float('nan')):.4f}; "
                f"95% subject-bootstrap CI [{metric.get('subject_bootstrap_CI95_L', float('nan')):.4f}, "
                f"{metric.get('subject_bootstrap_CI95_U', float('nan')):.4f}]; "
                f"fraction above chance: {metric.get('fraction_subject_BA_gt_0p5', float('nan')):.3f}.",
                "",
            ]
        )
    if final["terminal_state"] == "REPRESENTATION_COMPETENCE_FAIL":
        lines.extend(
            [
                "The frozen competence gate failed. Persistence/actionability was not computed, because nuisance claims from an incompetent task model would be uninterpretable.",
                "",
            ]
        )
    elif (RESULTS / "BLOCK_ASSIGNMENTS.csv").exists():
        import pandas as pd

        persistence = pd.read_csv(RESULTS / "PERSISTENCE_RESULTS.csv")
        utility = pd.read_csv(RESULTS / "SIGNED_UTILITY_RESULTS.csv")
        decision = pd.read_csv(RESULTS / "DECISION_DEPENDENCE_RESULTS.csv")
        action = pd.read_csv(RESULTS / "ACTIONABILITY_RESULTS.csv")
        assignment = pd.read_csv(RESULTS / "BLOCK_ASSIGNMENTS.csv")
        merged = persistence.merge(utility, on=["block", "rank"]).merge(decision, on=["block", "rank"])
        merged = merged.merge(action, on=["block", "rank"]).merge(assignment, on=["block", "rank"])
        lines.extend(
            [
                "## Prospective H1-H5 audit",
                "",
                "| Block | H1 | H2 | H3 | H4 | H5 | u_spec | finite ratio | BA specific | Assignment |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for row in merged.itertuples():
            lines.append(
                f"| {row.block} | {bool(row.H1)} | {bool(row.H2)} | {bool(row.H3)} | "
                f"{bool(row.H4)} | {bool(row.H5)} | {row.u_spec_mean:.5f} | "
                f"{row.finite_ratio_mean:.3f} | {row.delta_BA_specific_mean:.4f} | {row.assignment} |"
            )
        lines.extend(["", f"Decision: `{final['next_action']}`", ""])
    lines.extend(
        [
            "## Interpretation",
            "",
            str(final.get("scientific_conclusion", "")),
            "",
            "A negative gate is not converted into a positive claim by changing blocks, thresholds, subjects, preprocessing, or backbone. AGDI is run only when H1-H5 jointly authorize a target.",
            "",
            "## Reproducibility",
            "",
            f"- Git commit recorded at report time: `{git_commit()}`",
            "- Seed: `20260817`",
            "- Random controls per block/fold: `100`",
            "- Subject bootstrap draws: `10000`",
        ]
    )
    (OUT / "scientific_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def reproduce() -> dict[str, Any]:
    packages = {}
    for package in ("numpy", "torch", "pandas", "scipy"):
        try:
            module = __import__(package)
            packages[package] = getattr(module, "__version__", "unknown")
        except Exception as error:
            packages[package] = f"unavailable: {error}"
    value = {
        "implementation_id": IMPLEMENTATION_ID,
        "git_commit": git_commit(),
        "python": sys.version,
        "packages": packages,
        "torch_cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": 20260817,
        "commands": [
            "python code/protocol.py prepare --raw-root <WBCIC_BIDS_ROOT>",
            "python code/cache.py build --raw-root <WBCIC_BIDS_ROOT> --workers 4",
            "python code/pipeline.py competence --device cuda --workers 4",
            "python code/pipeline.py audit --device cuda --workers 4",
            "python code/pipeline.py agdi --device cuda --workers 4",
        ],
        "outer_test_used": json.loads((OUT / "FINAL_DECISION.json").read_text(encoding="utf-8")).get("outer_test_used") if (OUT / "FINAL_DECISION.json").exists() else False,
    }
    write_json(OUT / "REPRODUCIBILITY.json", value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("competence", "audit", "agdi", "reproduce"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    device = device_from_argument(args.device)
    if args.command == "competence":
        competence(device, max(0, args.workers))
    elif args.command == "audit":
        audit(device, max(0, args.workers))
    elif args.command == "agdi":
        agdi(device, max(0, args.workers))
    else:
        print(json.dumps(clean(reproduce()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
