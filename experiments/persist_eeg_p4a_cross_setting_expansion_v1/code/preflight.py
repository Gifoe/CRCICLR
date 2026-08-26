from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import common


def write_md(name: str, title: str, body: str) -> None:
    (common.EXP / name).write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def main() -> None:
    common.ensure_dirs()
    if common.PROTOCOL_PATH.exists():
        raise RuntimeError("refusing to overwrite an existing frozen P4A protocol")
    if common.git_head() != "1ff8edda656372d8d36a2bcdb7d96311f88f8da6":
        raise RuntimeError("P4A must start from the frozen P3 tip before the protocol-freeze commit")

    openbmi_subjects = common.subject_sort(
        set().union(*(set(common.openbmi_roles(f)[role]) for f in common.FOLDS for role in ("model_fit", "validation", "outcome")))
    )
    wbcic_subjects = common.subject_sort(
        set().union(*(set(common.wbcic_roles(f)[role]) for f in common.FOLDS for role in ("model_fit", "validation", "outcome")))
    )
    if len(openbmi_subjects) != 40 or len(wbcic_subjects) != 41:
        raise RuntimeError("authorized subject pool cardinality failure")

    erp = pd.read_parquet(
        common.OPENBMI_MANIFEST,
        filters=[("subject_id", "in", openbmi_subjects), ("paradigm", "==", "erp")],
        engine="pyarrow",
    )
    cells = erp.groupby(["subject_id", "session_id"]).size()
    events = erp.groupby(["subject_id", "session_id", "event_code"]).size()
    if (
        len(erp) != 158400
        or erp.subject_id.nunique() != 40
        or set(erp.session_id.astype(int)) != {1, 2}
        or len(cells) != 80
        or int(cells.min()) != 1980
        or int(cells.max()) != 1980
        or len(events) != 160
    ):
        raise RuntimeError("OpenBMI ERP availability audit failed")

    protocol = {
        "schema": "PERSIST_EEG_P4A_CROSS_SETTING_EXPANSION_V1",
        "repository_start_sha": "1ff8edda656372d8d36a2bcdb7d96311f88f8da6",
        "branch": "codex/persist-eeg-p4a-cross-setting-expansion-v1",
        "frozen_before_any_new_setting_outcome_evaluation": True,
        "future_direction_utility_sealed": True,
        "invariance_outcome_delta_sealed": True,
        "planned_settings": common.SETTINGS,
        "folds": list(common.FOLDS),
        "seeds": list(common.SEEDS),
        "methods": {"ERM": [0.0], "DANN": list(common.LAMBDAS), "CORAL": list(common.LAMBDAS), "MMD": list(common.LAMBDAS)},
        "subject_scopes": {
            "OpenBMI_development_count": 40,
            "OpenBMI_development_hash": common.text_sha256(openbmi_subjects),
            "OpenBMI_sealed_internal_holdout_count": 14,
            "OpenBMI_sealed_internal_holdout_membership_materialized": False,
            "WBCIC_development_count": 41,
            "WBCIC_development_hash": common.text_sha256(wbcic_subjects),
            "WBCIC_sealed_outer_count": 10,
            "WBCIC_sealed_outer_membership_materialized": False,
        },
        "session_rules": {
            "OpenBMI": {"source": ["S1", "S2"], "outcome": "held-subject S2"},
            "WBCIC": {"source": ["S1", "S2"], "outcome": "held-subject S3"},
        },
        "training": {
            "optimizer": "AdamW",
            "max_epochs": 60,
            "minimum_epochs": 20,
            "patience": 12,
            "selection": "source validation mean-subject balanced accuracy; validation NLL tie-break",
            "gradient_clip_norm": 5.0,
            "OpenBMI_ERP_class_weight": "inverse frequency computed on model-fit rows only",
            "backbones": {
                "EEGNet": {"lr": 0.0003, "weight_decay": 0.0005, "batch_size": 512},
                "EEGConformer": {"lr": 0.0003, "weight_decay": 0.0005, "batch_size": 256},
            },
        },
        "architectures": {
            "EEGNet": {"F1": 8, "D": 2, "F2": 16, "temporal_kernel": 64, "dropout": 0.25, "representation_dim": 64},
            "EEGConformer": {"temporal_filters": 40, "kernel": 25, "pool_kernel": 25, "pool_stride": 10, "depth": 2, "heads": 4, "ffn": 160, "representation_dim": 64, "dropout": [0.4, 0.3], "channel_drop": 0.03},
        },
        "source_primitives": {
            "I": "standardized multiclass ridge alpha=1 symmetric S1<->S2 identity skill",
            "I_direction": "I_full minus I_after_rank1_erasure",
            "P": "P2/P3 subject-centroid SVD pool; sort by cross-session correlation then geometry; first 8",
            "D_finite": "RMS class-centered logit displacement",
            "C_src": ["validation CE harm", "validation BA harm", "validation macro-F1 harm"],
            "O_task": "squared projection on class-centered linear classifier-weight span",
        },
        "random_controls": {"per_direction": 100, "rank": 1, "rule": "P3 full-space random subspace with per-trial displacement-norm matching"},
        "competence_gate": {"chance": 0.5, "mean_outcome_BA_strictly_greater_than": 0.60, "minimum_fold_means_above_chance": 4},
        "bootstrap_draws": 10000,
        "forbidden_in_P4A": ["future direction erasure utility", "invariance outcome delta summary", "reliability-condition fitting", "selective-invariance model design"],
        "scientific_protocol_post_outcome_modification_allowed": False,
    }
    common.write_json(common.PROTOCOL_PATH, protocol)

    erp_protocol = {
        "schema": "OPENBMI_ERP_P4A_PREPROCESSING_FROZEN_V1",
        "provenance": str(common.OPENBMI_MANIFEST),
        "authorized_subject_count": 40,
        "authorized_subject_hash": common.text_sha256(openbmi_subjects),
        "sessions": [1, 2],
        "event_mapping": {"1": "NonTarget", "2": "Target"},
        "event_mapping_zero_based_training": {"0": "NonTarget", "1": "Target"},
        "epoch_seconds": [0.0, 1.0],
        "sampling_rate_hz": 250,
        "channels": 62,
        "bandpass_hz": [1, 45],
        "baseline_correction": None,
        "stage0_dtype": "float32",
        "runtime_storage_dtype": "float16 deterministic cast only",
        "training_normalization": "per-channel mean/std from model-fit subjects and S1+S2 only",
        "class_imbalance_handling": "inverse-frequency task-loss weights from model-fit rows only",
        "trial_exclusion": "none",
        "outcome_driven_choice": False,
    }
    common.write_json(common.EXP / "PREPROCESSING_PROTOCOL_ERP_FROZEN.json", erp_protocol)

    source_manifest = {
        "schema": "P4A_SETTING_SOURCE_MANIFEST_V1",
        "settings": {
            "S1": {"source_experiment": str(common.P2_ROOT), "source_status": "READ_ONLY", "source_commit": "historical artifact tree at frozen P3 tip", "outcome_status": "HISTORICALLY_OBSERVED", "representation_dim": 64},
            "S2": {"source_experiment": str(common.P2_ROOT), "source_status": "READ_ONLY", "source_commit": "historical artifact tree at frozen P3 tip", "outcome_status": "HISTORICALLY_OBSERVED", "representation_dim": 64},
            "S3": {"source_experiment": str(common.P3_ROOT), "source_status": "READ_ONLY", "source_commit": "1ff8edda656372d8d36a2bcdb7d96311f88f8da6", "outcome_status": "HISTORICALLY_OBSERVED", "representation_dim": 32},
            "S4": {"source_experiment": str(common.EXP), "source_status": "NEW", "outcome_status": "P4B_DIRECTION_UTILITY_SEALED", "representation_dim": 64},
            "S5": {"source_experiment": str(common.EXP), "source_status": "NEW", "outcome_status": "P4B_DIRECTION_UTILITY_SEALED", "representation_dim": 64},
            "S6": {"source_experiment": str(common.EXP), "source_status": "NEW", "outcome_status": "P4B_DIRECTION_UTILITY_SEALED", "representation_dim": 64},
        },
    }
    common.write_json(common.EXP / "SETTING_SOURCE_MANIFEST.json", source_manifest)
    common.write_json(
        common.EXP / "SETTING_MANIFEST.json",
        {
            "schema": "P4A_SETTING_MANIFEST_V1",
            "settings": common.SETTINGS,
            "folds": list(common.FOLDS),
            "seeds": list(common.SEEDS),
            "configuration_count_per_setting": 150,
            "planned_model_cube_rows": 900,
            "planned_evidence_cube_rows": 720,
            "planned_control_rows": 72000,
        },
    )

    cache_audit = common.prepare_erp_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("mandatory P4A training server GPU is unavailable")
    forward = {}
    penalties = {}
    for setting in ("S4", "S5", "S6"):
        model = common.build_model(setting, common.stable_seed("P4A-preflight", setting)).to(device).train()
        spec = common.SETTINGS[setting]
        channels = 58 if spec["dataset"] == "WBCIC" else 62
        times = 1000 if spec["dataset"] == "WBCIC" else 250
        x = torch.randn(24, channels, times, device=device)
        h = model.forward_features(x)
        logits = model.head(h)
        if h.shape != (24, 64) or logits.shape != (24, 2):
            raise RuntimeError(f"{setting} backbone forward shape failed")
        domain = torch.arange(24, device=device) % 8
        forward[setting] = {"features": list(h.shape), "logits": list(logits.shape)}
        penalties[setting] = {"CORAL": float(common.coral_penalty(h, domain).detach().cpu()), "MMD": float(common.mmd_penalty(h, domain, [2.0, 4.0, 8.0]).detach().cpu())}
        del model, x, h, logits

    feature = torch.randn(16, 7, device=device, requires_grad=True)
    head = torch.nn.Linear(7, 4).to(device)
    label = torch.arange(16, device=device) % 4
    direct_grad = torch.autograd.grad(F.cross_entropy(head(feature), label), feature, retain_graph=True)[0]
    reverse_grad = torch.autograd.grad(F.cross_entropy(head(common.GradientReverse.apply(feature)), label), feature)[0]
    grl_error = float(torch.max(torch.abs(direct_grad + reverse_grad)).detach().cpu())
    if grl_error > 1e-7:
        raise RuntimeError("gradient reversal sign audit failed")

    preflight = {
        "pass": True,
        "current_git_head": common.git_head(),
        "protocol_sha256": common.file_sha256(common.PROTOCOL_PATH),
        "erp_protocol_sha256": common.file_sha256(common.EXP / "PREPROCESSING_PROTOCOL_ERP_FROZEN.json"),
        "OpenBMI_development_subject_count": len(openbmi_subjects),
        "WBCIC_development_subject_count": len(wbcic_subjects),
        "OpenBMI_ERP_rows": len(erp),
        "OpenBMI_ERP_subject_session_cells": len(cells),
        "OpenBMI_ERP_event_cells": len(events),
        "OpenBMI_ERP_cache": cache_audit,
        "GPU": torch.cuda.get_device_name(0),
        "BF16": torch.cuda.is_bf16_supported(),
        "forward_shapes": forward,
        "penalties": penalties,
        "GRL_max_sign_error": grl_error,
        "OpenBMI_sealed_14_accessed": False,
        "OpenBMI_sealed_14_membership_enumerated": False,
        "WBCIC_sealed_10_accessed": False,
        "WBCIC_sealed_10_membership_enumerated": False,
        "new_setting_outcome_labels_accessed": False,
    }
    common.write_json(common.RUNTIME / "PREFLIGHT.json", preflight)

    fold_lines = []
    for setting in ("S4", "S5", "S6"):
        for fold in common.FOLDS:
            roles = common.roles_for(setting, fold)
            fold_lines.append(f"- {setting} fold {fold}: model-fit={len(roles['model_fit'])}, validation={len(roles['validation'])}, outcome={len(roles['outcome'])}")
    write_md("DATA_SCOPE_AUDIT.md", "Data Scope Audit", "Only the frozen 40-subject OpenBMI development pool and 41-subject WBCIC development pool are authorized. No sealed-subject identifier was materialized.\n\n" + "\n".join(fold_lines))
    write_md("OPENBMI_ERP_AVAILABILITY_AUDIT.md", "OpenBMI ERP Availability Audit", f"PASS: 40/40 authorized development subjects have both sessions. There are {len(erp):,} epochs, 80 subject-session cells, exactly 1,980 epochs per cell, 62 channels, 250 samples, and 250 Hz sampling. Event counts are NonTarget={int((erp.event_code == 1).sum()):,}, Target={int((erp.event_code == 2).sum()):,}. No sealed OpenBMI subject was inspected.")
    write_md("CROSS_TASK_PROTOCOL_AUDIT.md", "Cross-Task Protocol Audit", "OpenBMI MI and ERP use the identical frozen 40-subject five-fold assignment. Source representations use both S1 and S2 in model-fit subjects; source consequence uses both sessions in validation subjects; competence uses held-subject S2. The task-specific differences are event extraction, epoch duration, and ERP class weighting, all frozen before outcome access.")
    write_md("PREPROCESSING_AUDIT.md", "Preprocessing Audit", "WBCIC S4 inherits the frozen P3 cache and preprocessing. OpenBMI ERP S5/S6 inherit the Stage-0 manifest pipeline (1–45 Hz, 250 Hz, 0–1 s, 62 channels, no baseline correction). Runtime float16 is a storage-only cast; model-fit-only channel normalization is applied during training.")
    write_md("BACKBONE_PORT_AUDIT.md", "Backbone Port Audit", "S4 and S6 use the frozen V7 Compact EEGConformer with only input-channel/time dimensional adaptation. S5 uses the authoritative Standard EEGNet. Forward-shape and penalty preflight passed.")
    write_md("ENGINEERING_REPAIR_LOG.md", "Engineering Repair Log", "Before outcome access, the initial staging implementation was repaired to use the exact P2/P3 subject-centroid SVD persistence construction and P3 full-space displacement-matched random controls. No dataset, role, seed, method, lambda, competence gate, or outcome rule changed.")
    write_md("OUTCOME_ACCESS_LEDGER.md", "Outcome Access Ledger", "At protocol freeze: no S4/S5/S6 outcome label had been accessed. P4A permits only ERM competence evaluation after the protocol-freeze commit. Invariance outcome deltas and all direction-level future utilities remain sealed.")
    for name, title in (
        ("TRAINING_LEDGER.md", "Training Ledger"),
        ("SOURCE_IDENTITY_AUDIT.md", "Source Identity Audit"),
        ("PERSISTENCE_AUDIT.md", "Persistence Audit"),
        ("DECISION_DEPENDENCE_AUDIT.md", "Decision Dependence Audit"),
        ("SOURCE_CONSEQUENCE_AUDIT.md", "Source Consequence Audit"),
        ("TASK_SUBSPACE_OVERLAP_AUDIT.md", "Task-Subspace Overlap Audit"),
        ("SETTING_COMPETENCE_REPORT.md", "Setting Competence Report"),
        ("HOLDOUT_PURITY_AUDIT.md", "Holdout Purity Audit"),
    ):
        write_md(name, title, "Protocol-frozen placeholder. This report will be populated only from frozen source artifacts (and, for competence alone, authorized ERM outcome evaluation).")

    print("P4A_PREFLIGHT_PASS_PROTOCOL_FREEZE_READY", flush=True)
    print(json.dumps({"GPU": preflight["GPU"], "ERP_rows": len(erp), "protocol_sha256": preflight["protocol_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
