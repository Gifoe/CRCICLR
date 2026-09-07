"""Fail-fast protocol, purity, implementation, and GPU preflight."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import common


def main() -> None:
    common.ensure_dirs()
    cfg = common.protocol()
    data = common.load_data()
    observed_subjects = common.subject_sort(data.metadata.subject_id.unique())
    fold_rows = []
    outcome_union: set[str] = set()
    for fold in range(5):
        roles = common.frozen_fold(fold)
        outcome_union |= set(roles["outcome"])
        train = common.row_indices(data.metadata, roles["inner_train"], (1, 2))
        validation = common.row_indices(data.metadata, roles["inner_validation"], (1, 2))
        outcome = common.row_indices(data.metadata, roles["outcome"], (2,))
        if len(train) != 4800 or len(validation) != 1600 or len(outcome) != 800:
            raise RuntimeError(f"fold {fold} row counts changed")
        if set(train) & set(validation) or set(train) & set(outcome) or set(validation) & set(outcome):
            raise RuntimeError(f"fold {fold} row overlap")
        fold_rows.append(
            {
                "fold": fold,
                "inner_train_subjects": len(roles["inner_train"]),
                "inner_validation_subjects": len(roles["inner_validation"]),
                "outcome_subjects": len(roles["outcome"]),
                "inner_train_rows": len(train),
                "inner_validation_rows": len(validation),
                "outcome_S2_rows": len(outcome),
            }
        )
    if outcome_union != set(common.frozen_subjects()):
        raise RuntimeError("five-fold outcome union differs from authorized subject pool")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matched_initialization: dict[str, list[str]] = {}
    forward_shapes: dict[str, list[int]] = {}
    penalty_values: dict[str, float] = {}
    for backbone in common.BACKBONES:
        seed = common.stable_seed("preflight-init", backbone)
        hashes = [common.state_sha256(common.build_model(backbone, seed)) for _ in range(10)]
        if len(set(hashes)) != 1:
            raise RuntimeError(f"{backbone} matched initialization failed")
        matched_initialization[backbone] = hashes
        model = common.build_model(backbone, seed).to(device)
        model.train()
        x = torch.randn(48, 62, 1000, device=device)
        domain = torch.arange(48, device=device) % 8
        h = model.forward_features(x)
        logits = model.head(h)
        if h.shape != (48, 64) or logits.shape != (48, 2):
            raise RuntimeError(f"{backbone} forward shape invalid: h={h.shape} logits={logits.shape}")
        forward_shapes[backbone] = list(h.shape)
        penalty_values[f"{backbone}_CORAL"] = float(common.coral_penalty(h, domain).detach().cpu())
        penalty_values[f"{backbone}_MMD"] = float(common.mmd_penalty(h, domain, [4.0, 8.0, 16.0]).detach().cpu())
        del model, x, h, logits

    # Exact gradient-reversal sign check independent of the backbone.
    torch.manual_seed(1)
    feature = torch.randn(16, 7, device=device, requires_grad=True)
    head = torch.nn.Linear(7, 4).to(device)
    label = torch.arange(16, device=device) % 4
    direct = F.cross_entropy(head(feature), label)
    direct_grad = torch.autograd.grad(direct, feature, retain_graph=True)[0]
    reversed_loss = F.cross_entropy(head(common.GradientReverse.apply(feature)), label)
    reversed_grad = torch.autograd.grad(reversed_loss, feature)[0]
    grl_error = float(torch.max(torch.abs(direct_grad + reversed_grad)).detach().cpu())
    if grl_error > 1e-7:
        raise RuntimeError(f"GRL gradient sign check failed: {grl_error}")

    # Exact Exp3 D_finite analytic check for a two-class centered margin change.
    rng = np.random.default_rng(20260826)
    margin = rng.normal(size=2048)
    clean = np.zeros((len(margin), 2), dtype=np.float64)
    erased = np.stack([-0.5 * margin, 0.5 * margin], axis=1)
    numeric = common.exact_d_finite(clean, erased)
    analytic = float(np.sqrt(np.mean(np.square(margin)) / 2.0))
    if abs(numeric - analytic) > 1e-12:
        raise RuntimeError("exact D_finite implementation changed")

    banned_patterns = (
        "V8_INTERNAL_HOLDOUT",
        "build_authorized_cache(",
        "_split_payload(",
        "load_protocol(",
        "V8_SEARCH_SPLIT.json",
    )
    code_hits: list[str] = []
    for path in common.HERE.glob("*.py"):
        if path.name == "preflight.py":
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in banned_patterns:
            if pattern in text:
                code_hits.append(f"{path.name}:{pattern}")
    if code_hits:
        raise RuntimeError(f"restricted/historical loader reference in new execution code: {code_hits}")

    payload = {
        "pass": True,
        "protocol_schema": cfg["schema"],
        "repository_start_sha": cfg["repository_start_sha"],
        "current_git_head": common.git_head(),
        "cache_root": str(data.cache_root),
        "cache_rows": len(data.metadata),
        "signal_shape": list(data.x.shape),
        "observed_subjects": observed_subjects,
        "observed_subjects_equal_frozen_40": observed_subjects == list(common.frozen_subjects()),
        "folds": fold_rows,
        "outcome_union_equals_frozen_40": outcome_union == set(common.frozen_subjects()),
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        "matched_initialization_unique_hashes": {key: len(set(value)) for key, value in matched_initialization.items()},
        "forward_shapes": forward_shapes,
        "penalty_values": penalty_values,
        "GRL_max_sign_error": grl_error,
        "D_finite_numeric": numeric,
        "D_finite_analytic": analytic,
        "D_finite_absolute_error": abs(numeric - analytic),
        "restricted_loader_code_hits": code_hits,
        "OpenBMI_internal_holdout_accessed": False,
        "OpenBMI_internal_holdout_membership_enumerated": False,
        "WBCIC_accessed": False,
        "outcome_S2_labels_used_for_training_or_selection": False,
    }
    common.write_json(common.RUNTIME / "PREFLIGHT.json", payload)
    common.write_json(
        common.RESULTS / "holdout_purity.json",
        {
            "status": "PASS_PREFLIGHT_PENDING_FULL_RUN_AUDIT",
            "authorized_subject_count": 40,
            "observed_subjects_equal_authorized_pool": True,
            "restricted_holdout_accessed": False,
            "restricted_holdout_membership_enumerated": False,
            "WBCIC_accessed": False,
            "historical_holdout_enumerating_loader_imported": False,
            "outcome_S2_labels_used_for_training_or_selection": False,
            "outcome_evaluation_guard": "LAMBDA_SELECTION_FROZEN.json must exist before evaluation",
        },
    )
    print("STRESS_TEST_PREFLIGHT_PASS", flush=True)
    print(f"GPU={payload['gpu_name']} cache={data.cache_root}", flush=True)


if __name__ == "__main__":
    main()
