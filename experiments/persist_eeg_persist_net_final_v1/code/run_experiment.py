"""Server runner for PERSIST-Net.

Phases are deliberately explicit so audit/selection locks exist before any
outer development Session-2 outcome is evaluated.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

import core


def device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("The frozen full experiment requires the server CUDA device")
    return torch.device("cuda:0")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=core.REPO,
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except Exception:
        return None


def config_by_id(config_id: str) -> dict[str, Any]:
    candidates = core.protocol()["baseline_candidates"]
    return dict(next(row for row in candidates if row["id"] == config_id))


def select_dual_width(baseline_params: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = []
    eligible = []
    for config in core.protocol()["dual_width_candidates"]:
        model = core.DualPathEEGNet(config)
        params = core.parameter_count(model)
        ratio = params / baseline_params
        row = {**config, "parameters": params, "ratio_vs_B1": ratio, "eligible": ratio <= 1.25}
        rows.append(row)
        if row["eligible"]:
            eligible.append(row)
    if not eligible:
        raise RuntimeError(f"No dual width satisfies the capacity cap: {rows}")
    selected = min(eligible, key=lambda row: (abs(row["parameters"] - baseline_params), row["parameters"]))
    return dict(selected), rows


def preflight(force_cache: bool = False) -> dict[str, Any]:
    started = time.time()
    core.ensure_dirs()
    exact = core.validate_exact_d()
    initialization = core.validate_deterministic_initialization()
    core.write_json(core.PROTOCOL_DIR / "EXACT_DFINITE_VALIDATION.json", exact)
    core.write_json(core.PROTOCOL_DIR / "DETERMINISTIC_INITIALIZATION_VALIDATION.json", initialization)
    paths = core.build_authorized_cache(force=force_cache)
    data = core.load_development_data()
    folds = core.outer_folds(data.search_subjects)
    fold_payload = {
        "protocol_seed": core.protocol()["protocol_seed"],
        "split_unit": "subject_id",
        "folds": [
            {
                "fold": fold,
                "outcome_subjects": row["outcome"],
                "source_subjects": row["source"],
                "inner_validation_subjects": row["inner_validation"],
                "inner_train_subjects": row["inner_train"],
            }
            for fold, row in enumerate(folds)
        ],
        "development_subjects": len(data.search_subjects),
        "sealed_internal_holdout_count": data.holdout_count,
        "holdout_eeg_materialized": False,
        "OUTER_TEST_USED": False,
    }
    core.write_json(core.PROTOCOL_DIR / "DEVELOPMENT_FOLDS.json", fold_payload)
    baseline_models = []
    for config in core.protocol()["baseline_candidates"]:
        model = core.EEGNetClassifier(config)
        baseline_models.append({"id": config["id"], "parameters": core.parameter_count(model)})
    payload = {
        "status": "PREFLIGHT_PASS",
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": os.sys.executable,
        "git_commit_seen_on_server": git_commit(),
        "intended_base_commit": core.protocol()["base_commit"],
        "data_cache": str(paths.x),
        "data_cache_sha256": core.sha256_file(paths.x),
        "metadata_sha256": core.sha256_file(paths.metadata),
        "baseline_parameter_counts": baseline_models,
        "exact_D": exact,
        "deterministic_initialization": initialization,
        "internal_holdout_accessed": False,
        "outer_test_used": False,
        "runtime_s": time.time() - started,
    }
    core.write_json(core.PROTOCOL_DIR / "PREFLIGHT.json", payload)
    print(json.dumps(core.clean(payload), indent=2), flush=True)
    return payload


def _score_single_on_subjects(
    model: core.EEGNetClassifier,
    data: core.DevelopmentData,
    subjects: tuple[str, ...],
    dev: torch.device,
    mean: np.ndarray,
    std: np.ndarray,
) -> float:
    idx = core.row_indices(data.metadata, subjects, (2,))
    result = core.evaluate_single(model, data, idx, dev, mean, std, include_features=False)
    return core.subject_mean_ba(result.labels, result.logits, result.subjects)


def _select_generic(
    model: core.EEGNetClassifier,
    data: core.DevelopmentData,
    validation_subjects: tuple[str, ...],
    dev: torch.device,
    mean: np.ndarray,
    std: np.ndarray,
    fold: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = core.protocol()["generic_adaptation_candidates"]
    rows = []
    for order, config in enumerate(candidates):
        values = []
        times = []
        for subject in validation_subjects:
            hx, hy, _, fy, _, future_indices = core.raw_subject(data, subject)
            adapted, adaptation = core.adapt_single(
                model,
                hx,
                hy,
                config,
                dev,
                mean,
                std,
                core.stable_seed("generic-selection", fold, order, subject),
            )
            evaluation = core.evaluate_single(
                adapted, data, future_indices, dev, mean, std, include_features=False
            )
            values.append(core.subject_mean_ba(evaluation.labels, evaluation.logits, evaluation.subjects))
            times.append(float(adaptation["adaptation_time_s"]))
        rows.append(
            {
                "order": order,
                "config": dict(config),
                "validation_mean_subject_BA": float(np.mean(values)),
                "validation_subject_BA": values,
                "mean_adaptation_time_s": float(np.mean(times)),
            }
        )
    selected = max(rows, key=lambda row: (row["validation_mean_subject_BA"], -row["order"]))
    return dict(selected["config"]), rows


def selection_path(fold: int) -> Path:
    return core.RUNTIME / "selection" / f"FOLD_{fold}.json"


def select_fold(fold: int, force: bool = False) -> dict[str, Any]:
    """Nested source-only architecture/adaptation/epoch selection."""
    path = selection_path(fold)
    if path.is_file() and not force:
        payload = json.loads(path.read_text(encoding="utf-8"))
        print(json.dumps(payload, indent=2), flush=True)
        return payload
    started = time.time()
    dev = device()
    data = core.load_development_data()
    roles = core.outer_folds(data.search_subjects)[fold]
    mean, std, normalizer_path = core.compute_normalizer(
        data, roles["inner_train"], f"FOLD_{fold}_INNER_TRAIN"
    )
    train_indices = core.row_indices(data.metadata, roles["inner_train"], (1, 2))
    validation_indices = core.row_indices(data.metadata, roles["inner_validation"], (2,))
    candidates = []
    trained: dict[str, core.EEGNetClassifier] = {}
    for order, config in enumerate(core.protocol()["baseline_candidates"]):
        model = core.EEGNetClassifier(config)
        model, epoch, history = core.train_single(
            model,
            data,
            train_indices,
            validation_indices,
            dev,
            mean,
            std,
            core.stable_seed("baseline-selection", fold, order),
            config,
        )
        score = _score_single_on_subjects(
            model, data, roles["inner_validation"], dev, mean, std
        )
        candidates.append(
            {
                "order": order,
                "id": config["id"],
                "configuration": dict(config),
                "best_epoch": epoch,
                "validation_mean_subject_BA": score,
                "history": history,
                "parameters": core.parameter_count(model),
            }
        )
        trained[str(config["id"])] = model
        print(f"[select] fold={fold} baseline={config['id']} BA={score:.4f} epoch={epoch}", flush=True)
    selected_baseline = max(candidates, key=lambda row: (row["validation_mean_subject_BA"], -row["order"]))
    teacher = trained[selected_baseline["id"]]
    generic, generic_rows = _select_generic(
        teacher,
        data,
        roles["inner_validation"],
        dev,
        mean,
        std,
        fold,
    )
    width, width_rows = select_dual_width(core.parameter_count(teacher))
    certificate, teacher_eval = core.fit_certificate(
        teacher,
        data,
        roles["inner_train"],
        dev,
        mean,
        std,
        fold,
        0,
    )
    targets = core.teacher_targets(teacher, certificate, teacher_eval, "PUD")
    full_model = core.DualPathEEGNet(width)
    full_model, student_epoch, student_history, diagnostics = core.train_dual(
        full_model,
        data,
        train_indices,
        validation_indices,
        targets,
        dev,
        mean,
        std,
        core.stable_seed("student-epoch-selection", fold),
        fixed_epochs=None,
        task_only=False,
    )
    validation = core.evaluate_dual(full_model, data, validation_indices, dev, mean, std)
    lp_rms = float(np.sqrt(np.mean(core.exact_centered_logit_sq(validation["protected_logits"]))))
    la_rms = float(np.sqrt(np.mean(core.exact_centered_logit_sq(validation["adaptive_logits"]))))
    freeze_subject = roles["inner_validation"][0]
    freeze_hx, freeze_hy, _, _, _, freeze_future = core.raw_subject(data, freeze_subject)
    freeze_before = core.evaluate_dual(full_model, data, freeze_future, dev, mean, std)
    freeze_model, freeze_adaptation = core.adapt_dual(
        full_model,
        freeze_hx,
        freeze_hy,
        generic,
        dev,
        mean,
        std,
        core.stable_seed("freeze-preflight", fold),
        all_adapt=False,
    )
    freeze_after = core.evaluate_dual(freeze_model, data, freeze_future, dev, mean, std)
    freeze_logit_drift = core.exact_d_finite(
        freeze_before["protected_logits"], freeze_after["protected_logits"]
    )
    pathology_checks = {
        "protected_gradient_nonzero": bool((diagnostics["first_batch_gradient_norm"]["protected"] or 0.0) > 0),
        "adaptive_gradient_nonzero": bool((diagnostics["first_batch_gradient_norm"]["adaptive"] or 0.0) > 0),
        "protected_logits_noncollapsed": bool(lp_rms > 1e-4 or certificate.rank == 0),
        "adaptive_logits_noncollapsed": bool(la_rms > 1e-4),
        "finite_student_outputs": bool(
            np.isfinite(validation["protected_logits"]).all()
            and np.isfinite(validation["adaptive_logits"]).all()
        ),
        "target_scale_finite": bool(np.isfinite(float(targets["scale"])) and float(targets["scale"]) > 0),
        "protected_freeze_logit_exact": bool(freeze_logit_drift <= 1e-10),
        "protected_freeze_buffer_exact": bool(freeze_adaptation["protected_buffer_update_l2"] <= 1e-12),
        "adaptive_update_nonzero": bool(freeze_adaptation["adaptive_parameter_update_l2"] > 0.0),
    }
    pathology = not all(pathology_checks.values())
    payload = {
        "fold": fold,
        "selection_scope": "outer-source subjects only",
        "outer_outcome_subjects_absent_from_selection": not bool(
            set(roles["outcome"]) & (set(roles["inner_train"]) | set(roles["inner_validation"]))
        ),
        "inner_train_subjects": roles["inner_train"],
        "inner_validation_subjects": roles["inner_validation"],
        "outer_outcome_subject_count": len(roles["outcome"]),
        "normalizer": str(normalizer_path),
        "normalizer_subjects": roles["inner_train"],
        "baseline_candidates": candidates,
        "selected_baseline": selected_baseline,
        "generic_candidates": generic_rows,
        "selected_generic": generic,
        "dual_width_candidates": width_rows,
        "selected_dual_width": width,
        "student_best_epoch": student_epoch,
        "student_history": student_history,
        "certificate_audit": certificate.audit,
        "protected_logit_rms": lp_rms,
        "adaptive_logit_rms": la_rms,
        "training_diagnostics": diagnostics,
        "freeze_preflight": {"protected_logit_drift": freeze_logit_drift, **freeze_adaptation},
        "pathology_checks": pathology_checks,
        "engineering_pathology_detected": pathology,
        "V1_1_authorized": pathology,
        "target_future_outer_labels_used": False,
        "internal_holdout_accessed": False,
        "outer_test_used": False,
        "runtime_s": time.time() - started,
    }
    core.write_json(path, payload)
    # Lightweight source-only selection summary is copied into results.
    summary_path = core.RESULTS / f"selection_fold_{fold}.json"
    core.write_json(summary_path, payload)
    print(json.dumps(core.clean(payload), indent=2), flush=True)
    return payload


def _checkpoint(
    model: torch.nn.Module,
    directory: Path,
    method: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    path = directory / "checkpoints" / f"{method}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": core.clean(metadata)}, path)
    return {"path": str(path), "sha256": core.sha256_file(path), "parameters": core.parameter_count(model)}


def _metrics_row(
    method: str,
    fold: int,
    seed: int,
    subject: str,
    labels: np.ndarray,
    logits: np.ndarray,
    before_logits: np.ndarray | None,
    adaptation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    after = core.per_subject_metrics(labels, logits, np.asarray([subject] * len(labels))).iloc[0]
    before_ba = None
    adaptation_delta = None
    if before_logits is not None:
        before = core.per_subject_metrics(labels, before_logits, np.asarray([subject] * len(labels))).iloc[0]
        before_ba = float(before.BA)
        adaptation_delta = float(after.BA - before.BA)
    return {
        "method": method,
        "fold": fold,
        "seed": seed,
        "subject_id": subject,
        "BA": float(after.BA),
        "macro_f1": float(after.macro_f1),
        "n_trials": int(after.n_trials),
        "source_model_noadapt_BA": before_ba,
        "adaptation_delta_BA": adaptation_delta,
        "negative_transfer": bool(adaptation_delta is not None and adaptation_delta < -1e-12),
        "adaptation_time_s": float(adaptation["adaptation_time_s"]) if adaptation else 0.0,
        "target_trainable_parameters": int(adaptation["trainable_parameters"]) if adaptation else 0,
        "target_future_labels_used_for_fit": False,
        "internal_holdout_used": False,
        "outer_test_used": False,
    }


def _teacher_target_for_indices(
    teacher: core.EEGNetClassifier,
    certificate: core.Certificate,
    data: core.DevelopmentData,
    indices: np.ndarray,
    dev: torch.device,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    evaluation = core.evaluate_single(teacher, data, indices, dev, mean, std, include_features=True)
    return np.asarray(core.teacher_targets(teacher, certificate, evaluation, "PUD")["protected"])


def run_fold_seed(fold: int, seed: int, force: bool = False) -> dict[str, Any]:
    run_dir = core.RUNTIME_RUNS / f"fold-{fold}" / f"seed-{seed}"
    done_path = run_dir / "DONE.json"
    if done_path.is_file() and not force:
        payload = json.loads(done_path.read_text(encoding="utf-8"))
        print(json.dumps(payload, indent=2), flush=True)
        return payload
    selection = select_fold(fold, force=False)
    if selection.get("engineering_pathology_detected"):
        raise RuntimeError(
            "V0 detected a predeclared engineering pathology. Document and implement the one allowed V1.1 repair before outcomes."
        )
    started = time.time()
    dev = device()
    data = core.load_development_data()
    roles = core.outer_folds(data.search_subjects)[fold]
    mean, std, normalizer_path = core.compute_normalizer(data, roles["source"], f"FOLD_{fold}_SOURCE32")
    source_indices = core.row_indices(data.metadata, roles["source"], (1, 2))
    baseline_config = dict(selection["selected_baseline"]["configuration"])
    baseline_epochs = int(selection["selected_baseline"]["best_epoch"])
    generic_config = dict(selection["selected_generic"])
    width = dict(selection["selected_dual_width"])
    student_epochs = int(selection["student_best_epoch"])
    run_dir.mkdir(parents=True, exist_ok=True)

    # All source models are trained before any outcome Session-2 evaluation.
    teacher_seed = core.stable_seed("B1-refit", fold, seed)
    teacher = core.EEGNetClassifier(baseline_config)
    teacher, _, teacher_history = core.train_single(
        teacher,
        data,
        source_indices,
        None,
        dev,
        mean,
        std,
        teacher_seed,
        baseline_config,
        fixed_epochs=baseline_epochs,
    )
    b1_checkpoint = _checkpoint(
        teacher,
        run_dir,
        "B1_STRONG_EEGNET",
        {
            "configuration": baseline_config,
            "epochs": baseline_epochs,
            "seed": teacher_seed,
            "source_subjects": roles["source"],
            "source_sessions": [1, 2],
            "normalizer_sha256": core.sha256_file(normalizer_path),
        },
    )
    b0_config = config_by_id("EEGNET_F8")
    if baseline_config["id"] == "EEGNET_F8":
        b0 = teacher
        b0_history = teacher_history
        b0_checkpoint = b1_checkpoint
        b0_seed = teacher_seed
    else:
        b0_seed = core.stable_seed("B0-refit", fold, seed)
        b0 = core.EEGNetClassifier(b0_config)
        b0, _, b0_history = core.train_single(
            b0,
            data,
            source_indices,
            None,
            dev,
            mean,
            std,
            b0_seed,
            b0_config,
            fixed_epochs=baseline_epochs,
        )
        b0_checkpoint = _checkpoint(
            b0,
            run_dir,
            "B0_VANILLA_EEGNET",
            {
                "configuration": b0_config,
                "epochs": baseline_epochs,
                "seed": b0_seed,
                "source_subjects": roles["source"],
                "source_sessions": [1, 2],
                "normalizer_sha256": core.sha256_file(normalizer_path),
            },
        )

    certificate, teacher_source_eval = core.fit_certificate(
        teacher,
        data,
        roles["source"],
        dev,
        mean,
        std,
        fold,
        seed,
    )
    core.save_certificate(certificate, run_dir / "certificate")
    target_map = {
        basis: core.teacher_targets(teacher, certificate, teacher_source_eval, basis)
        for basis in ("P", "PU", "PD", "PUD", "IDENTITY", "RANDOM", "PCA")
    }

    source_models: dict[str, core.DualPathEEGNet] = {}
    training_records: dict[str, Any] = {}

    # All matched dual-path methods share initialization and source minibatch
    # order within a fold/seed.  Their only intended difference is supervision.
    paired_dual_seed = core.stable_seed("paired-dual-source", fold, seed)
    dual_control = core.DualPathEEGNet(width)
    dual_control, _, history, diagnostics = core.train_dual(
        dual_control,
        data,
        source_indices,
        None,
        None,
        dev,
        mean,
        std,
        paired_dual_seed,
        fixed_epochs=student_epochs,
        task_only=True,
    )
    source_models["A2_DUAL_CONTROL"] = dual_control
    training_records["A2_DUAL_CONTROL"] = {"history": history, "diagnostics": diagnostics}

    basis_methods = {
        "PUD": "PUD_SOURCE",
        "IDENTITY": "A7_IDENTITY_PROTECTED",
        "RANDOM": "A8_RANDOM_PROTECTED",
    }
    if seed == 0:
        basis_methods.update(
            {
                "P": "A3_P_ONLY",
                "PU": "A4_P_PLUS_U",
                "PD": "A5_P_PLUS_D",
                "PCA": "A9_PCA_PROTECTED",
            }
        )
    for basis, method in basis_methods.items():
        model = core.DualPathEEGNet(width)
        model, _, history, diagnostics = core.train_dual(
            model,
            data,
            source_indices,
            None,
            target_map[basis],
            dev,
            mean,
            std,
            paired_dual_seed,
            fixed_epochs=student_epochs,
            task_only=False,
        )
        source_models[method] = model
        training_records[method] = {"history": history, "diagnostics": diagnostics, "basis": basis}
        print(f"[source] fold={fold} seed={seed} method={method}", flush=True)

    checkpoints = {
        "B0_VANILLA_EEGNET": b0_checkpoint,
        "B1_STRONG_EEGNET": b1_checkpoint,
    }
    for method, model in source_models.items():
        checkpoints[method] = _checkpoint(
            model,
            run_dir,
            method,
            {"width": width, "epochs": student_epochs, **training_records[method].get("diagnostics", {})},
        )

    run_lock = {
        "fold": fold,
        "seed": seed,
        "lock_created_before_outer_outcome_evaluation": True,
        "source_subjects": roles["source"],
        "outer_outcome_subject_count": len(roles["outcome"]),
        "outer_outcome_subjects_not_used_for_training": not bool(set(roles["source"]) & set(roles["outcome"])),
        "normalizer": str(normalizer_path),
        "normalizer_sha256": core.sha256_file(normalizer_path),
        "normalizer_subjects": roles["source"],
        "source_training_sessions": [1, 2],
        "target_history_session": 1,
        "future_evaluation_session": 2,
        "baseline_configuration": baseline_config,
        "baseline_epochs": baseline_epochs,
        "teacher_seed": teacher_seed,
        "B0_seed": b0_seed,
        "generic_adaptation": generic_config,
        "dual_width": width,
        "student_epochs": student_epochs,
        "loss": core.protocol()["loss"],
        "certificate_audit": certificate.audit,
        "checkpoint_hashes": checkpoints,
        "implementation_sha256": {
            "core.py": core.sha256_file(Path(core.__file__).resolve()),
            "run_experiment.py": core.sha256_file(Path(__file__).resolve()),
            "PROTOCOL_FROZEN.json": core.sha256_file(core.PROTOCOL_PATH),
        },
        "methods": sorted(source_models),
        "target_future_labels_used": False,
        "internal_holdout_accessed": False,
        "outer_test_used": False,
    }
    core.write_json(run_dir / "RUN_LOCK.json", run_lock)
    core.write_json(run_dir / "TRAINING_RECORDS.json", training_records)

    subject_rows: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []
    adaptation_rows: list[dict[str, Any]] = []

    for subject in roles["outcome"]:
        hx, hy, _, fy, _, future_indices = core.raw_subject(data, subject)
        b0_eval = core.evaluate_single(b0, data, future_indices, dev, mean, std, include_features=False)
        subject_rows.append(
            _metrics_row("B0_VANILLA_EEGNET", fold, seed, subject, fy, b0_eval.logits, None, None)
        )
        b1_eval = core.evaluate_single(teacher, data, future_indices, dev, mean, std, include_features=False)
        subject_rows.append(
            _metrics_row("B1_STRONG_EEGNET", fold, seed, subject, fy, b1_eval.logits, None, None)
        )
        adapted_single, single_adaptation = core.adapt_single(
            teacher,
            hx,
            hy,
            generic_config,
            dev,
            mean,
            std,
            core.stable_seed("B2-adapt", fold, seed, subject),
        )
        b2_eval = core.evaluate_single(
            adapted_single, data, future_indices, dev, mean, std, include_features=False
        )
        subject_rows.append(
            _metrics_row(
                "B2_STRONG_GENERIC",
                fold,
                seed,
                subject,
                fy,
                b2_eval.logits,
                b1_eval.logits,
                single_adaptation,
            )
        )
        adaptation_rows.append({"method": "B2_STRONG_GENERIC", "fold": fold, "seed": seed, "subject_id": subject, **single_adaptation})

        teacher_target = _teacher_target_for_indices(
            teacher, certificate, data, future_indices, dev, mean, std
        )
        deployed = {
            "A2_DUAL_CONTROL": (source_models["A2_DUAL_CONTROL"], False, None),
            "A6_PUD_ALL_ADAPT": (source_models["PUD_SOURCE"], True, teacher_target),
            "A7_IDENTITY_PROTECTED": (source_models["A7_IDENTITY_PROTECTED"], False, None),
            "A8_RANDOM_PROTECTED": (source_models["A8_RANDOM_PROTECTED"], False, None),
            "A10_FULL_PUD_FREEZE": (source_models["PUD_SOURCE"], False, teacher_target),
        }
        if seed == 0:
            deployed.update(
                {
                    "A3_P_ONLY": (source_models["A3_P_ONLY"], False, None),
                    "A4_P_PLUS_U": (source_models["A4_P_PLUS_U"], False, None),
                    "A5_P_PLUS_D": (source_models["A5_P_PLUS_D"], False, None),
                    "A9_PCA_PROTECTED": (source_models["A9_PCA_PROTECTED"], False, None),
                }
            )
        for method, (source_model, all_adapt, functional_target) in deployed.items():
            before = core.evaluate_dual(source_model, data, future_indices, dev, mean, std)
            adapted, adaptation = core.adapt_dual(
                source_model,
                hx,
                hy,
                generic_config,
                dev,
                mean,
                std,
                core.stable_seed(method, "adapt", fold, seed, subject),
                all_adapt=all_adapt,
            )
            after = core.evaluate_dual(adapted, data, future_indices, dev, mean, std)
            before_logits = before["protected_logits"] + before["adaptive_logits"]
            after_logits = after["protected_logits"] + after["adaptive_logits"]
            subject_rows.append(
                _metrics_row(
                    method,
                    fold,
                    seed,
                    subject,
                    fy,
                    after_logits,
                    before_logits,
                    adaptation,
                )
            )
            mechanism = core.branch_mechanism_metrics(before, after, functional_target)
            combined_ba = float(core.per_subject_metrics(fy, after_logits, np.asarray([subject] * len(fy))).iloc[0].BA)
            adaptive_only_ba = float(
                core.per_subject_metrics(fy, after["adaptive_logits"], np.asarray([subject] * len(fy))).iloc[0].BA
            )
            protected_only_ba = float(
                core.per_subject_metrics(fy, after["protected_logits"], np.asarray([subject] * len(fy))).iloc[0].BA
            )
            mechanism_rows.append(
                {
                    "method": method,
                    "fold": fold,
                    "seed": seed,
                    "subject_id": subject,
                    **mechanism,
                    "combined_BA": combined_ba,
                    "protected_branch_erasure_harm_BA": combined_ba - adaptive_only_ba,
                    "adaptive_branch_erasure_harm_BA": combined_ba - protected_only_ba,
                    "protected_parameter_update_l2": adaptation["protected_parameter_update_l2"],
                    "adaptive_parameter_update_l2": adaptation["adaptive_parameter_update_l2"],
                    "protected_buffer_update_l2": adaptation["protected_buffer_update_l2"],
                    "target_future_labels_used_for_fit": False,
                    "internal_holdout_used": False,
                    "outer_test_used": False,
                }
            )
            adaptation_rows.append({"method": method, "fold": fold, "seed": seed, "subject_id": subject, **adaptation})
        print(f"[outcome] fold={fold} seed={seed} subject={subject}", flush=True)

    subject_frame = pd.DataFrame(subject_rows)
    mechanism_frame = pd.DataFrame(mechanism_rows)
    adaptation_frame = pd.DataFrame(adaptation_rows)
    core.write_csv(run_dir / "SUBJECT_RESULTS.csv", subject_frame)
    core.write_csv(run_dir / "MECHANISM_METRICS.csv", mechanism_frame)
    core.write_csv(run_dir / "ADAPTATION_LEDGER.csv", adaptation_frame)

    efficiency = []
    measured_models: dict[str, torch.nn.Module] = {"B0_VANILLA_EEGNET": b0, "B1_STRONG_EEGNET": teacher, **source_models}
    for method, model in measured_models.items():
        model = model.to(dev)
        efficiency.append(
            {
                "source_model": method,
                "fold": fold,
                "seed": seed,
                "parameters": core.parameter_count(model),
                "approximate_MACs": core.approximate_macs(model),
                "capacity_ratio_vs_B1": core.parameter_count(model) / core.parameter_count(teacher),
            }
        )
    core.write_csv(run_dir / "EFFICIENCY.csv", pd.DataFrame(efficiency))

    payload = {
        "status": "RUN_COMPLETE",
        "fold": fold,
        "seed": seed,
        "subjects": len(roles["outcome"]),
        "subject_rows": len(subject_frame),
        "mechanism_rows": len(mechanism_frame),
        "PUD_rank": certificate.rank,
        "baseline_configuration": baseline_config["id"],
        "generic_configuration": generic_config["id"],
        "dual_width": width["id"],
        "internal_holdout_accessed": False,
        "target_future_labels_used_for_fit": False,
        "outer_test_used": False,
        "runtime_s": time.time() - started,
    }
    core.write_json(done_path, payload)
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def status() -> dict[str, Any]:
    rows = []
    for fold in range(5):
        for seed in range(3):
            path = core.RUNTIME_RUNS / f"fold-{fold}" / f"seed-{seed}" / "DONE.json"
            if path.is_file():
                rows.append(json.loads(path.read_text(encoding="utf-8")))
            else:
                rows.append({"fold": fold, "seed": seed, "status": "PENDING"})
    payload = {"runs": rows, "complete": sum(row["status"] == "RUN_COMPLETE" for row in rows), "total": 15}
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "select", "run", "status"))
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.phase == "preflight":
        preflight(force_cache=args.force)
    elif args.phase == "select":
        if args.fold not in range(5):
            raise ValueError("--fold 0..4 required")
        select_fold(args.fold, force=args.force)
    elif args.phase == "run":
        if args.fold not in range(5) or args.seed not in range(3):
            raise ValueError("--fold 0..4 and --seed 0..2 required")
        run_fold_seed(args.fold, args.seed, force=args.force)
    else:
        status()


if __name__ == "__main__":
    main()
