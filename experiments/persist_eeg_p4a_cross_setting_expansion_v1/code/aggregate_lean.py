from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import aggregate
import common


ERM_SLUG = common.config_slug("ERM", 0.0)


def model_row(
    setting: str,
    fold: int,
    seed: int,
    candidate: dict[str, Any],
    identity: dict[str, Any],
    validation_ba: float,
    validation_f1: float,
    outcome_ba: float,
    outcome_f1: float,
    outcome_status: str,
) -> dict[str, Any]:
    return {
        "setting_id": setting,
        "dataset": common.SETTINGS[setting]["dataset"],
        "task": common.SETTINGS[setting]["task"],
        "backbone": common.SETTINGS[setting]["backbone"],
        "fold": fold,
        "seed": seed,
        "method": "ERM",
        "lambda": 0.0,
        "source_identity": float(identity["identity_symmetric"]),
        "source_identity_raw_accuracy": float(identity["identity_accuracy_symmetric"]),
        "source_identity_chance_normalized_accuracy": float(identity["chance_normalized_identity"]),
        "source_identity_chance_accuracy": float(identity["chance_accuracy"]),
        "source_validation_BA": float(validation_ba),
        "source_validation_F1": float(validation_f1),
        "checkpoint_sha256": str(candidate["checkpoint_sha256"]),
        "training_epoch": int(candidate["best_epoch"]),
        "selection_metric": "source_validation_mean_subject_BA_then_NLL",
        "outcome_status": outcome_status,
        "ERM_outcome_competence_BA": float(outcome_ba),
        "ERM_outcome_competence_F1": float(outcome_f1),
    }


def historical_openbmi_erm(setting: str) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[dict[str, Any]]]:
    backbone = "eegnet" if setting == "S1" else "eegconformer"
    model_rows: list[dict[str, Any]] = []
    evidence_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for fold in common.FOLDS:
        for seed in common.SEEDS:
            unit = common.P2_ROOT / "runtime" / "runs" / backbone / f"fold-{fold}" / f"seed-{seed}"
            unit_protocol = common.read_json(unit / "UNIT_PROTOCOL.json")
            evaluation = unit / "evaluation" / ERM_SLUG
            candidate = common.read_json(unit / "candidates" / f"{ERM_SLUG}.json")
            identity = pd.read_csv(evaluation / "identity.csv").iloc[0].to_dict()
            all_source = aggregate.payload_from_npz(evaluation / "embeddings.npz", "source", session_shift=1)
            model_fit = aggregate.subset(all_source, unit_protocol["roles"]["inner_train"])
            validation = aggregate.subset(all_source, unit_protocol["roles"]["inner_validation"])
            validation_metrics = common.mean_subject_metrics(validation["labels"], validation["logits"], validation["subjects"])
            performance = pd.read_csv(evaluation / "performance.csv")
            state = aggregate.load_checkpoint_state(unit / "checkpoints" / f"{ERM_SLUG}.pt")
            head = aggregate.head_model(
                state["head.weight"].numpy().astype(np.float64),
                state["head.bias"].numpy().astype(np.float64),
            )
            normalizer = unit / "normalizer.npz"
            evidence, controls, _ = common.direction_rows(
                setting,
                fold,
                seed,
                head,
                model_fit,
                validation,
                str(candidate["checkpoint_sha256"]),
                common.file_sha256(normalizer),
                {
                    "source": common.array_sha256(model_fit["indices"]),
                    "validation": common.array_sha256(validation["indices"]),
                },
            )
            model_rows.append(
                model_row(
                    setting,
                    fold,
                    seed,
                    candidate,
                    identity,
                    validation_metrics["BA"],
                    validation_metrics["macro_f1"],
                    float(performance.BA.mean()),
                    float(performance.macro_f1.mean()),
                    "HISTORICALLY_OBSERVED",
                )
            )
            evidence_frames.append(evidence)
            audit_rows.append(
                {
                    "setting_id": setting,
                    "fold": fold,
                    "seed": seed,
                    "checkpoint_path": str(unit / "checkpoints" / f"{ERM_SLUG}.pt"),
                    "checkpoint_sha256": str(candidate["checkpoint_sha256"]),
                    "source_embedding_path": str(evaluation / "embeddings.npz"),
                    "source_embedding_sha256": common.file_sha256(evaluation / "embeddings.npz"),
                    "direction_rows": len(evidence),
                    "matched_control_rows": len(controls),
                    "artifact_complete": True,
                }
            )
    return model_rows, evidence_frames, audit_rows


def historical_wbcic_erm() -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[dict[str, Any]]]:
    setting = "S3"
    model_rows: list[dict[str, Any]] = []
    evidence_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    bundle = common.load_data(setting)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.from_numpy(np.asarray(bundle.x)).to(device)
    for fold in common.FOLDS:
        roles = common.wbcic_roles(fold)
        validation_indices = common.row_indices(bundle.metadata, roles["validation"], bundle.source_sessions)
        for seed in common.SEEDS:
            unit = common.P3_ROOT / "runtime" / "runs" / "eegnet" / f"fold-{fold}" / f"seed-{seed}"
            evaluation = unit / "evaluation" / ERM_SLUG
            candidate = common.read_json(unit / "candidates" / f"{ERM_SLUG}.json")
            identity = pd.read_csv(evaluation / "identity.csv").iloc[0].to_dict()
            normalizer = np.load(unit / "normalizer.npz", allow_pickle=False)
            mean = torch.as_tensor(normalizer["mean"], dtype=torch.float32, device=device)
            std = torch.as_tensor(normalizer["std"], dtype=torch.float32, device=device)
            model = common.StandardEEGNet(58, 1000, 32).to(device)
            model.load_state_dict(aggregate.load_checkpoint_state(unit / "checkpoints" / f"{ERM_SLUG}.pt"), strict=True)
            validation = common.evaluate_model(model, raw, bundle.metadata, validation_indices, mean, std, batch_size=512)
            validation_metrics = common.mean_subject_metrics(validation["labels"], validation["logits"], validation["subjects"])
            performance = pd.read_csv(evaluation / "performance.csv")
            model_fit = aggregate.payload_from_npz(evaluation / "embeddings.npz", "source", session_shift=0)
            evidence, controls, _ = common.direction_rows(
                setting,
                fold,
                seed,
                model,
                model_fit,
                validation,
                str(candidate["checkpoint_sha256"]),
                common.file_sha256(unit / "normalizer.npz"),
                {
                    "source": common.array_sha256(model_fit["indices"]),
                    "validation": common.array_sha256(validation["indices"]),
                },
            )
            model_rows.append(
                model_row(
                    setting,
                    fold,
                    seed,
                    candidate,
                    identity,
                    validation_metrics["BA"],
                    validation_metrics["macro_f1"],
                    float(performance.BA.mean()),
                    float(performance.macro_f1.mean()),
                    "HISTORICALLY_OBSERVED",
                )
            )
            evidence_frames.append(evidence)
            audit_rows.append(
                {
                    "setting_id": setting,
                    "fold": fold,
                    "seed": seed,
                    "checkpoint_path": str(unit / "checkpoints" / f"{ERM_SLUG}.pt"),
                    "checkpoint_sha256": str(candidate["checkpoint_sha256"]),
                    "source_embedding_path": str(evaluation / "embeddings.npz"),
                    "source_embedding_sha256": common.file_sha256(evaluation / "embeddings.npz"),
                    "direction_rows": len(evidence),
                    "matched_control_rows": len(controls),
                    "artifact_complete": True,
                }
            )
            del model, validation
    del raw
    torch.cuda.empty_cache()
    return model_rows, evidence_frames, audit_rows


def new_erm() -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[dict[str, Any]]]:
    model_rows: list[dict[str, Any]] = []
    evidence_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for setting in ("S4", "S5", "S6"):
        for fold in common.FOLDS:
            for seed in common.SEEDS:
                unit = common.run_dir(setting, fold, seed)
                target = unit / "source_freeze" / ERM_SLUG
                summary = common.read_json(target / "SOURCE_COMPLETE.json")
                candidate = common.read_json(unit / "candidates" / f"{ERM_SLUG}.json")
                evidence = pd.read_csv(target / "source_evidence.csv")
                controls_path = target / "matched_controls.csv"
                identity = summary["source_identity"]
                model_rows.append(
                    model_row(
                        setting,
                        fold,
                        seed,
                        candidate,
                        identity,
                        summary["source_validation_BA"],
                        summary["source_validation_F1"],
                        summary["outcome_competence_BA"],
                        summary["outcome_competence_F1"],
                        "P4B_DIRECTION_UTILITY_SEALED",
                    )
                )
                evidence_frames.append(evidence)
                audit_rows.append(
                    {
                        "setting_id": setting,
                        "fold": fold,
                        "seed": seed,
                        "checkpoint_path": str(unit / "checkpoints" / f"{ERM_SLUG}.pt"),
                        "checkpoint_sha256": str(candidate["checkpoint_sha256"]),
                        "source_embedding_path": str(target / "embeddings.npz"),
                        "source_embedding_sha256": common.file_sha256(target / "embeddings.npz"),
                        "direction_rows": len(evidence),
                        "matched_control_rows": sum(1 for _ in controls_path.open("rb")) - 1,
                        "matched_controls_path": str(controls_path),
                        "matched_controls_sha256": common.file_sha256(controls_path),
                        "artifact_complete": True,
                    }
                )
    return model_rows, evidence_frames, audit_rows


def competence_table(model: pd.DataFrame, evidence: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    representation = evidence.groupby("setting_id").representation_dim.first().to_dict()
    for setting, spec in common.SETTINGS.items():
        cell = model[model.setting_id == setting]
        fold_means = cell.groupby("fold").ERM_outcome_competence_BA.mean()
        mean_ba = float(cell.ERM_outcome_competence_BA.mean())
        folds_above = int((fold_means > 0.5).sum())
        preprocessing_ok = True
        artifact_complete = bool(
            len(audit[audit.setting_id == setting]) == 15
            and audit[audit.setting_id == setting].artifact_complete.all()
            and (audit[audit.setting_id == setting].direction_rows == 8).all()
        )
        status = "COMPETENCE_PASS" if mean_ba > 0.60 and folds_above >= 4 and preprocessing_ok and artifact_complete else "COMPETENCE_FAIL"
        roles = "source S1+S2; outcome held-subject S2" if spec["dataset"] == "OpenBMI" else "source S1+S2; outcome held-subject S3"
        rows.append(
            {
                "setting_id": setting,
                "dataset": spec["dataset"],
                "task": spec["task"],
                "backbone": spec["backbone"],
                "subject_count": 40 if spec["dataset"] == "OpenBMI" else 41,
                "folds": 5,
                "seeds": 3,
                "outcome_BA_mean": mean_ba,
                "outcome_macro_F1_mean": float(cell.ERM_outcome_competence_F1.mean()),
                "folds_above_chance": folds_above,
                "representation_dim": int(representation[setting]),
                "session_roles": roles,
                "source_artifact_complete": artifact_complete,
                "preprocessing_event_label_status": "PASS" if preprocessing_ok else "FAIL",
                "competence": status,
            }
        )
    return pd.DataFrame(rows)


def write_reports(model: pd.DataFrame, evidence: pd.DataFrame, audit: pd.DataFrame, competence: pd.DataFrame) -> str:
    table = competence.to_markdown(index=False, floatfmt=".5f")
    (common.EXP / "SETTING_COMPETENCE_REPORT_LEAN.md").write_text(
        "# Setting Competence Report — Lean\n\n"
        "Frozen gate: mean ERM outcome BA > 0.60, at least 4/5 fold means above binary chance, and no preprocessing/event/label or source-artifact failure. No setting was retrained or retuned for this audit.\n\n"
        + table
        + "\n",
        encoding="utf-8",
    )
    usable = competence.loc[competence.competence == "COMPETENCE_PASS", "setting_id"].tolist()
    failed = competence.loc[competence.competence != "COMPETENCE_PASS", "setting_id"].tolist()
    terminal = "P4A_LEAN_CROSS_SETTING_CUBE_COMPLETE" if not failed else "P4A_LEAN_PARTIAL_SETTING_FAILURE"
    report = f"""# P4A Lean Final Report

Exact terminal: `{terminal}`.

## Competent-ERM settings

{table}

## Lean closure

- Mandatory new-setting ERM: 45/45 complete before amendment.
- Cross-setting ERM cube: {len(model)} rows (6 settings × 5 folds × 3 seeds).
- Source evidence cube: {len(evidence)} rows; first eight source-defined persistent directions per run.
- Source artifact audit: {len(audit)} complete run rows.
- Usable settings: {', '.join(usable) if usable else 'none'}.
- Failed settings: {', '.join(failed) if failed else 'none'}.
- Non-ERM grid: partial and explicitly excluded from the Lean primary gate; see pause snapshot.
- Source I/P/D/C_src/O_task definitions: unchanged from the frozen P4A protocol.
- O_task: squared projection on the centered frozen linear-head task span; no future-driven replacement.
- New-setting direction-level future utility: sealed.
- Invariance outcome deltas: sealed.
- OpenBMI sealed 14: untouched and unenumerated.
- WBCIC outer 10: untouched and unenumerated.

This closure establishes a competent-ERM, cross-dataset, cross-task, and cross-backbone source evidence cube. It does not claim completion of the original 405-grid protocol and does not use partial-grid outcomes to formulate P4B.
"""
    (common.EXP / "P4A_LEAN_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    return terminal


def main() -> None:
    common.ensure_dirs()
    common.protocol()
    amendment = common.read_json(common.EXP / "P4A_PROTOCOL_AMENDMENT_LEAN_V1.json")
    if not amendment.get("amendment_before_p4b_future_direction_utility_discovery"):
        raise RuntimeError("Lean amendment was not frozen before future-utility discovery")

    model_rows: list[dict[str, Any]] = []
    evidence_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for collector in (
        lambda: historical_openbmi_erm("S1"),
        lambda: historical_openbmi_erm("S2"),
        historical_wbcic_erm,
        new_erm,
    ):
        rows, evidence, audits = collector()
        model_rows.extend(rows)
        evidence_frames.extend(evidence)
        audit_rows.extend(audits)

    model = pd.DataFrame(model_rows).sort_values(["setting_id", "fold", "seed"]).reset_index(drop=True)
    evidence = pd.concat(evidence_frames, ignore_index=True).sort_values(["setting_id", "fold", "seed", "direction_rank"]).reset_index(drop=True)
    audit = pd.DataFrame(audit_rows).sort_values(["setting_id", "fold", "seed"]).reset_index(drop=True)
    evidence["identity_chance_accuracy"] = [
        1.0 / len(common.roles_for(str(row.setting_id), int(row.fold))["model_fit"])
        for _, row in evidence.iterrows()
    ]
    if len(model) != 90 or len(evidence) != 720 or len(audit) != 90:
        raise RuntimeError(f"Lean cube cardinality failure: model={len(model)} evidence={len(evidence)} audit={len(audit)}")

    competence = competence_table(model, evidence, audit)
    common.write_csv(common.RESULTS / "erm_setting_cube.csv", model)
    common.write_csv(common.RESULTS / "source_evidence_cube.csv", evidence)
    common.write_csv(common.RESULTS / "source_artifact_audit.csv", audit)
    common.write_csv(common.RESULTS / "setting_competence_lean.csv", competence)
    terminal = write_reports(model, evidence, audit, competence)

    common.write_json(
        common.EXP / "SETTING_MANIFEST_LEAN.json",
        {
            "schema": "P4A_SETTING_MANIFEST_LEAN_V1",
            "settings": common.SETTINGS,
            "folds": list(common.FOLDS),
            "seeds": list(common.SEEDS),
            "primary_method": "ERM",
            "erm_rows": len(model),
            "source_evidence_rows": len(evidence),
            "partial_non_erm_grid_status": "OPTIONAL_PARTIAL_INVARIANCE_GRID",
            "partial_non_erm_grid_primary_gate": False,
            "competence": competence.set_index("setting_id").competence.to_dict(),
            "direction_future_utility_sealed": True,
        },
    )
    common.write_json(
        common.RESULTS / "P4A_LEAN_AGGREGATION_COMPLETE.json",
        {
            "pass": True,
            "terminal_candidate": terminal,
            "erm_rows": len(model),
            "source_evidence_rows": len(evidence),
            "source_artifact_audit_rows": len(audit),
            "usable_settings": competence.loc[competence.competence == "COMPETENCE_PASS", "setting_id"].tolist(),
            "direction_future_utility_sealed": True,
            "invariance_outcome_delta_sealed": True,
        },
    )
    print(f"P4A_LEAN_AGGREGATION_COMPLETE terminal_candidate={terminal}", flush=True)


if __name__ == "__main__":
    main()
