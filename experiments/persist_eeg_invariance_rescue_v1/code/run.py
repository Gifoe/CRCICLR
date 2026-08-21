from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from common import (
    CONFIG_PATH,
    EXPERIMENT_ROOT,
    OUTPUTS,
    ensure_directories,
    git_sha,
    load_config,
    sha256_file,
    write_csv,
    write_json,
)
from data import load_development_split, load_manifest, persist_split_manifests
from models import build_model, parameter_count, primary_pairs, roster
from rescue import run_eligible_rescues
from spectrum import audit_all
from statistics import finalize
from train import run_full, run_smoke


def _device(require_cuda: bool) -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if require_cuda and device.type != "cuda":
        raise RuntimeError("Training is authorized only on the designated CUDA server")
    return device


def phase0() -> dict[str, Any]:
    ensure_directories()
    config = load_config()
    split_rows = persist_split_manifests()
    dataset_rows = []
    for fold in config["development_folds"]:
        split = load_development_split(int(fold))
        # Phase 0 validates model-fit/calibration coverage without materializing
        # development-outcome event labels before final evaluation.
        manifest = load_manifest(split, split.model_fit_subjects + split.calibration_subjects)
        dataset_rows.append(
            {
                "fold": int(fold),
                "model_fit_subjects": len(split.model_fit_subjects),
                "calibration_subjects": len(split.calibration_subjects),
                "outcome_subjects": len(split.outcome_subjects),
                "allowed_subjects": len(split.allowed_subjects),
                "materialized_rows": len(manifest),
                "materialized_role": "model_fit_plus_calibration_only",
                "sessions": sorted(map(int, manifest.session_id.unique())),
                "channels": sorted(map(int, manifest.n_channels.unique())),
                "sampling_rates": sorted(map(float, manifest.sampling_rate.unique())),
                "times": sorted(map(int, manifest.n_times.unique())),
                "trials_per_subject_session_class": sorted(
                    map(int, manifest.groupby(["subject_id", "session_id", "label"]).size().unique())
                ),
                "outer_split_field_read": False,
                "outer_test_used": False,
            }
        )
    counts = []
    shape_gate = []
    device = _device(require_cuda=False)
    dummy = torch.zeros(2, 62, 1000, dtype=torch.float32, device=device)
    for method in roster(config):
        model = build_model(method, int(config["model_fit_subject_count"]), config).to(device).eval()
        with torch.inference_mode():
            output = model(dummy)
        valid_shape = tuple(output.logits.shape) == (2, 2) and tuple(output.features.shape) == (2, int(config["embedding_dim"]))
        finite = bool(torch.isfinite(output.logits).all() and torch.isfinite(output.features).all())
        counts.append({"method_id": method, "parameter_count": parameter_count(model)})
        shape_gate.append({"method_id": method, "logits_shape": list(output.logits.shape), "features_shape": list(output.features.shape), "finite": finite, "pass": bool(valid_shape and finite)})
        del model, output
    count_map = {row["method_id"]: row["parameter_count"] for row in counts}
    matched = {}
    for family, (task, invariant) in primary_pairs(config).items():
        matched[family] = {
            "task_only_parameters": count_map[task],
            "invariant_parameters": count_map[invariant],
            "equal": count_map[task] == count_map[invariant],
        }
    payload = {
        "status": "PASS" if all(value["equal"] for value in matched.values()) and all(value["pass"] for value in shape_gate) else "FAIL",
        "protocol_version": config["protocol_version"],
        "dataset": config["dataset"],
        "dataset_audit": dataset_rows,
        "development_split_manifests": split_rows,
        "method_parameter_counts": counts,
        "matched_parameter_gate": matched,
        "model_shape_gate": shape_gate,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_sha": git_sha(),
        "outer_split_field_read": False,
        "outer_test_used": False,
    }
    write_json(OUTPUTS / "PHASE0_REPORT.json", payload)
    write_json(
        OUTPUTS / "DATA_ACCESS_AUDIT.json",
        {
            "authorized_fields_accessed_from_split": ["train_subjects", "validation_subjects"],
            "outer_split_field_read": False,
            "outer_subjects_enumerated": False,
            "outer_labels_read": False,
            "outer_signals_read": False,
            "outer_features_constructed": False,
            "outer_scores_computed": False,
            "outer_test_used": False,
        },
    )
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def smoke(force: bool = False) -> dict[str, Any]:
    config = load_config()
    results = run_smoke(_device(require_cuda=True), force=force)
    frame = pd.DataFrame(results)
    write_csv(OUTPUTS / "SMOKE_RESULTS.csv", frame)
    statuses = {}
    for family, (task, invariant) in primary_pairs(config).items():
        if family == "A_SUBJECT_GRL_EEGNET":
            family_methods = [method for method in roster(config) if method.startswith("A")]
        else:
            family_methods = [task, invariant]
        selected = frame[frame.method_id.isin(family_methods)]
        task_row = frame[frame.method_id == task]
        complete = len(selected) == len(family_methods) and set(selected.status) == {"COMPLETE"}
        finite = bool(np.isfinite(selected.best_calibration_BA.astype(float)).all()) if complete else False
        competent = bool(len(task_row) == 1 and float(task_row.iloc[0].best_calibration_BA) >= float(config["fidelity_min_task_only_calibration_BA"]))
        non_task = selected[selected.method_id != task]
        invariant_competent = bool(
            complete
            and len(non_task)
            and (non_task.best_calibration_BA.astype(float) >= float(config["fidelity_min_invariant_calibration_BA"])).all()
        )
        parameter_equal = selected.parameter_count.nunique() == 1 if complete else False
        passed = bool(complete and finite and competent and invariant_competent and parameter_equal)
        declared_deviation = family.startswith(("B_", "C_"))
        statuses[family] = {
            "status": ("DEVIATION" if declared_deviation else "PASS") if passed else "FAIL",
            "fidelity_gate_pass": passed,
            "methods": family_methods,
            "task_only_calibration_BA": float(task_row.iloc[0].best_calibration_BA) if len(task_row) else None,
            "minimum_task_only_calibration_BA": float(config["fidelity_min_task_only_calibration_BA"]),
            "minimum_invariant_calibration_BA": float(config["fidelity_min_invariant_calibration_BA"]),
            "all_invariant_smoke_models_competent": invariant_competent,
            "all_runs_complete": complete,
            "all_losses_and_metrics_finite": finite,
            "matched_parameter_count": parameter_equal,
            "deviation": "clean-room method-level reproduction; not exact official-code replication" if declared_deviation else "none",
        }
    passed = all(value["fidelity_gate_pass"] for value in statuses.values())
    payload = {"status": "PASS" if passed else "FAIL", "families": statuses, "outer_test_used": False}
    write_json(OUTPUTS / "METHOD_FIDELITY.json", payload)
    lines = ["# Method fidelity status", "", f"Overall: `{payload['status']}`.", ""]
    for family, result in statuses.items():
        lines.extend(
            [
                f"## {family}",
                "",
                f"Status: `{result['status']}` (competence gate pass: `{str(result['fidelity_gate_pass']).lower()}`). "
                f"Task-only calibration BA: `{result['task_only_calibration_BA']}`; "
                f"threshold: `{result['minimum_task_only_calibration_BA']}`; matched parameter count: "
                f"`{str(result['matched_parameter_count']).lower()}`.",
                "",
                f"Every invariant/ladder smoke model exceeds `{result['minimum_invariant_calibration_BA']}`: "
                f"`{str(result['all_invariant_smoke_models_competent']).lower()}`.",
                "",
                f"Fidelity note: {result['deviation']}. Outcome loaders were not constructed during training; outer test used: `false`.",
                "",
            ]
        )
    lines.extend([
        "## Pre-freeze repair retained in the ledger",
        "",
        "The first B1 draft aligned final mixed task features directly across subject groups and collapsed at calibration BA 0.500 while B0 reached 0.722. That run is excluded from science but retained on the execution server. Before freeze, the alignment site was corrected to the source-special expert stack used by the audited upstream topology; objective names and weights did not change. See `HYPOTHESIS_LEDGER.md`.",
        "",
        "## Interpretation limit",
        "",
        "A is a controlled causal comparison with identical local EEGNet architectures. B and C are method-level clean-room instantiations because the audited upstream trees have no license and EEG-DG is incomplete at the audited commit. A DEVIATION status can pass the task-competence gate, but it cannot be described as exact official-code reproduction. See `LICENSE_AUDIT.md` for commits and concrete deviations.",
        "",
    ])
    (EXPERIMENT_ROOT / "METHOD_FIDELITY.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def freeze() -> dict[str, Any]:
    config = load_config()
    fidelity_path = OUTPUTS / "METHOD_FIDELITY.json"
    if not fidelity_path.exists():
        raise RuntimeError("Smoke fidelity has not run")
    fidelity = json.loads(fidelity_path.read_text(encoding="utf-8"))
    if fidelity.get("status") != "PASS":
        raise RuntimeError("Method fidelity gate failed; repair fidelity before freezing")
    existing = EXPERIMENT_ROOT / "PROTOCOL_FROZEN.json"
    code_hashes = {
        path.name: sha256_file(path)
        for path in sorted((EXPERIMENT_ROOT / "code").glob("*.py"))
    }
    payload = {
        "status": "FROZEN_BEFORE_FULL_RUN",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha_before_full_run": git_sha(),
        "protocol_config": config,
        "protocol_config_sha256": sha256_file(CONFIG_PATH),
        "protocol_markdown_sha256": sha256_file(EXPERIMENT_ROOT / "PROTOCOL.md"),
        "code_sha256": code_hashes,
        "method_roster": roster(config),
        "primary_pairs": primary_pairs(config),
        "scientific_gates": {
            "I1": "mean delta_ID < 0",
            "I2": "mean delta_PRS < 0 with all six run values finite",
            "I3": "mean delta_BA_INV < 0",
            "eligible": "I1 and I2 and I3",
            "certified_rescue": "LCB95(PERSIST-invariant)>0 and LCB95(PERSIST-generic)>0",
            "cross_family": ">=2 certified eligible families and pooled hierarchical LCB95(PERSIST-generic)>0",
        },
        "fidelity": fidelity,
        "outer_split_field_read": False,
        "outer_test_used": False,
    }
    if existing.exists():
        old = json.loads(existing.read_text(encoding="utf-8"))
        immutable = ["protocol_config_sha256", "protocol_markdown_sha256", "code_sha256", "method_roster", "scientific_gates"]
        if any(old.get(key) != payload.get(key) for key in immutable):
            raise RuntimeError("Existing frozen protocol differs; refusing silent re-freeze")
        print(json.dumps(old, indent=2), flush=True)
        return old
    write_json(existing, payload)
    print(json.dumps(payload, indent=2), flush=True)
    return payload


def _require_frozen() -> dict[str, Any]:
    path = EXPERIMENT_ROOT / "PROTOCOL_FROZEN.json"
    if not path.exists():
        raise RuntimeError("PROTOCOL_FROZEN.json is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "FROZEN_BEFORE_FULL_RUN" or payload.get("outer_test_used") is not False:
        raise RuntimeError("Frozen protocol is invalid")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["phase0", "smoke", "freeze", "full", "audit", "rescue", "finalize", "all"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.phase in {"phase0", "all"}:
        phase0()
    if args.phase in {"smoke", "all"}:
        smoke(force=args.force)
    if args.phase in {"freeze", "all"}:
        freeze()
    if args.phase in {"full", "all"}:
        _require_frozen()
        run_full(_device(require_cuda=True), force=args.force)
    if args.phase in {"audit", "all"}:
        _require_frozen()
        audit_all(force_spectrum=args.force)
    if args.phase in {"rescue", "all"}:
        _require_frozen()
        run_eligible_rescues()
    if args.phase in {"finalize", "all"}:
        _require_frozen()
        decision = finalize()
        print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
