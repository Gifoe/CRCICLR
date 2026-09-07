"""Global 16-candidate multiplicity and terminal closure synthesis."""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import torch

from common import (
    BACKBONES,
    BLOCKS,
    OUT,
    PROTOCOL,
    REFERENCE_OUT,
    REPO_ROOT,
    RESULTS,
    git_commit,
    holm,
    sha256_file,
    write_csv,
    write_json,
)


def records(path: Path) -> dict[str, dict[str, Any]]:
    frame = pd.read_csv(path)
    return {str(row["block"]): row.to_dict() for _, row in frame.iterrows()}


def new_backbone_rows(backbone: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frozen = json.loads((PROTOCOL / f"BACKBONE_{backbone.upper()}_FROZEN.json").read_text(encoding="utf-8"))
    task = dict(frozen["selection_metrics"])
    if not frozen["competence_gate_pass"]:
        rows = []
        for block, start, end in BLOCKS:
            rows.append(
                {
                    "Backbone": backbone,
                    "Task_BA": task["mean_subject_BA"],
                    "Task_BA_CI_L": task["subject_bootstrap_CI95_L"],
                    "Task_BA_CI_U": task["subject_bootstrap_CI95_U"],
                    "Representation_dimension": frozen["competence_checkpoint_set"][0]["representation_dim"],
                    "Competence": False,
                    "Block": block,
                    "Rank": end - start,
                    "H1": False,
                    "H2": False,
                    "H3": False,
                    "H4": False,
                    "H5": False,
                    "Protected": False,
                    "Preliminary_actionable": False,
                    "p_joint_raw": 1.0,
                    "Assignment": "NOT_AUDITED_INCOMPETENT",
                    "Action": "NO_OP",
                }
            )
        return frozen, rows
    prefix = f"BACKBONE_{backbone.upper()}"
    assign = records(RESULTS / f"{prefix}_BLOCK_ASSIGNMENTS.csv")
    persistence = records(RESULTS / f"{prefix}_PERSISTENCE_RESULTS.csv")
    utility = records(RESULTS / f"{prefix}_SIGNED_UTILITY_RESULTS.csv")
    decision = records(RESULTS / f"{prefix}_DECISION_DEPENDENCE_RESULTS.csv")
    actionability = records(RESULTS / f"{prefix}_ACTIONABILITY_RESULTS.csv")
    rows = []
    for block, start, end in BLOCKS:
        a, p, u, d, x = assign[block], persistence[block], utility[block], decision[block], actionability[block]
        rows.append(
            {
                "Backbone": backbone,
                "Task_BA": task["mean_subject_BA"],
                "Task_BA_CI_L": task["subject_bootstrap_CI95_L"],
                "Task_BA_CI_U": task["subject_bootstrap_CI95_U"],
                "Representation_dimension": frozen["competence_checkpoint_set"][0]["representation_dim"],
                "Competence": True,
                "Block": block,
                "Rank": end - start,
                "Persistence_effect": p["mean_specific_advantage"],
                "Persistence_CI_L": p["CI95_L"],
                "u_spec": u["u_spec_mean"],
                "u_spec_CI_L": u["u_spec_CI95_L"],
                "u_spec_CI_U": u["u_spec_CI95_U"],
                "Local_decision_ratio": d["local_ratio"],
                "Local_decision_ratio_CI_L": d["local_ratio_CI95_L"],
                "Finite_decision_ratio": d["finite_ratio_mean"],
                "Finite_decision_ratio_CI_L": d["finite_ratio_CI95_L"],
                "delta_BA_specific": x["delta_BA_specific_mean"],
                "delta_BA_specific_CI_L": x["delta_BA_specific_CI95_L"],
                "H1": bool(a["H1"]),
                "H2": bool(a["H2"]),
                "H3": bool(a["H3"]),
                "H4": bool(a["H4"]),
                "H5": bool(a["H5"]),
                "Protected": bool(a["protected_utility_gate"]),
                "Preliminary_actionable": bool(a["preliminary_all_H1_H5"]),
                "p_H1_raw": a["p_H1_raw"],
                "p_H2_raw": a["p_H2_raw"],
                "p_H3_local_raw": a["p_H3_local_raw"],
                "p_H3_finite_raw": a["p_H3_finite_raw"],
                "p_H4_raw": a["p_H4_raw"],
                "p_joint_raw": a["p_joint_raw"],
                "Assignment": a["assignment"],
                "Action": a["action"],
            }
        )
    return frozen, rows


def eegnet_rows() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frozen = json.loads((REFERENCE_OUT / "protocol" / "REPRESENTATION_FROZEN.json").read_text(encoding="utf-8"))
    task = frozen["selection_metrics"]
    assign = records(REFERENCE_OUT / "results" / "BLOCK_ASSIGNMENTS.csv")
    persistence = records(REFERENCE_OUT / "results" / "PERSISTENCE_RESULTS.csv")
    utility = records(REFERENCE_OUT / "results" / "SIGNED_UTILITY_RESULTS.csv")
    decision = records(REFERENCE_OUT / "results" / "DECISION_DEPENDENCE_RESULTS.csv")
    actionability = records(REFERENCE_OUT / "results" / "ACTIONABILITY_RESULTS.csv")
    rows = []
    for block, start, end in BLOCKS:
        a, p, u, d, x = assign[block], persistence[block], utility[block], decision[block], actionability[block]
        rows.append(
            {
                "Backbone": "EEGNet",
                "Task_BA": task["mean_subject_BA"],
                "Task_BA_CI_L": task["subject_bootstrap_CI95_L"],
                "Task_BA_CI_U": task["subject_bootstrap_CI95_U"],
                "Representation_dimension": 32,
                "Competence": True,
                "Block": block,
                "Rank": end - start,
                "Persistence_effect": p["mean_specific_advantage"],
                "Persistence_CI_L": p["CI95_L"],
                "u_spec": u["u_spec_mean"],
                "u_spec_CI_L": u["u_spec_CI95_L"],
                "u_spec_CI_U": u["u_spec_CI95_U"],
                "Local_decision_ratio": d["local_ratio"],
                "Local_decision_ratio_CI_L": d["local_ratio_CI95_L"],
                "Finite_decision_ratio": d["finite_ratio_mean"],
                "Finite_decision_ratio_CI_L": d["finite_ratio_CI95_L"],
                "delta_BA_specific": x["delta_BA_specific_mean"],
                "delta_BA_specific_CI_L": x["delta_BA_specific_CI95_L"],
                "H1": bool(a["H1"]),
                "H2": bool(a["H2"]),
                "H3": bool(a["H3"]),
                "H4": bool(a["H4"]),
                "H5": bool(a["H5"]),
                "Protected": bool(a["protected_utility_gate"]),
                "Preliminary_actionable": False,
                "p_joint_raw": np.nan,
                "p_joint_global_holm": np.nan,
                "Global_multiplicity_pass": False,
                "Globally_qualified_actionable": False,
                "Assignment": a["assignment"],
                "Action": a["action"],
            }
        )
    return frozen, rows


def main() -> None:
    reference_frozen, eeg_rows = eegnet_rows()
    new_rows, frozen_by_backbone = [], {}
    for backbone in BACKBONES:
        path = RESULTS / f"BACKBONE_{backbone.upper()}_COMPETENCE_RESULT.json"
        if not path.is_file():
            raise RuntimeError(f"Incomplete multi-backbone run: {backbone} competence missing")
        frozen, rows = new_backbone_rows(backbone)
        frozen_by_backbone[backbone] = frozen
        new_rows.extend(rows)
    if len(new_rows) != 16:
        raise RuntimeError("Global family must contain exactly 16 new backbone/block slots")
    pvalues = {f"{row['Backbone']}::{row['Block']}": float(row["p_joint_raw"]) for row in new_rows}
    adjusted = holm(pvalues)
    qualified = []
    for row in new_rows:
        key = f"{row['Backbone']}::{row['Block']}"
        row["p_joint_global_holm"] = adjusted[key]
        row["Global_multiplicity_pass"] = bool(adjusted[key] < 0.05)
        row["Globally_qualified_actionable"] = bool(
            row["Preliminary_actionable"] and row["Global_multiplicity_pass"]
        )
        if row["Globally_qualified_actionable"]:
            row["Assignment"] = "GLOBALLY_QUALIFIED_ACTIONABLE-HARMFUL"
            row["Action"] = "REPLICATION_REQUIRED"
            qualified.append(row)
        elif row["Preliminary_actionable"]:
            row["Assignment"] = "MULTIPLICITY-FAILED PRELIMINARY TARGET"
            row["Action"] = "NO_OP"
    master_rows = eeg_rows + new_rows
    write_csv(RESULTS / "MASTER_BLOCK_RESULTS.csv", master_rows)
    write_csv(
        RESULTS / "MASTER_ACTION_MATRIX.csv",
        [
            {
                "Backbone": row["Backbone"],
                "Block": row["Block"],
                "Competence": row["Competence"],
                "Protected": row["Protected"],
                "Actionable": row["Globally_qualified_actionable"],
                "Assignment": row["Assignment"],
                "Action": row["Action"],
            }
            for row in master_rows
        ],
    )
    backbone_rows = []
    for backbone in ("EEGNet", *BACKBONES):
        rows = [row for row in master_rows if row["Backbone"] == backbone]
        competence = bool(rows[0]["Competence"])
        if backbone == "EEGNet":
            audit_payload = json.loads((REFERENCE_OUT / "FINAL_DECISION.json").read_text(encoding="utf-8"))
        else:
            audit_path = RESULTS / f"BACKBONE_{backbone.upper()}_AUDIT_RESULT.json"
            audit_payload = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else {}
        backbone_rows.append(
            {
                "Backbone": backbone,
                "Competence": competence,
                "Best_task_BA": rows[0]["Task_BA"],
                "Task_BA_CI_L": rows[0]["Task_BA_CI_L"],
                "Task_BA_CI_U": rows[0]["Task_BA_CI_U"],
                "Representation_dimension": rows[0]["Representation_dimension"],
                "Audit_baseline_BA": audit_payload.get("baseline_mean_subject_BA"),
                "Persistent_blocks": sum(bool(row.get("H1", False)) for row in rows),
                "Protected_blocks": sum(bool(row.get("Protected", False)) for row in rows),
                "Harmful_utility_blocks": sum(bool(row.get("H1", False) and row.get("H2", False)) for row in rows),
                "Decision_active_harmful_blocks": sum(bool(row.get("H1", False) and row.get("H2", False) and row.get("H3", False)) for row in rows),
                "Actionable_harmful_blocks": sum(bool(row.get("Globally_qualified_actionable", False)) for row in rows),
                "Final_recommended_action": "REPLICATION_REQUIRED" if any(row.get("Globally_qualified_actionable", False) for row in rows) else ("PRESERVE" if any(row.get("Protected", False) for row in rows) else "NO_OP"),
                "AGDI_authorized": False,
                "AGDI_dev_result": "NOT_AUTHORIZED",
                "Outer_result": "SEALED_UNUSED",
            }
        )
    write_csv(RESULTS / "MASTER_BACKBONE_RESULTS.csv", backbone_rows)

    fixed_order = {name: index for index, name in enumerate(BACKBONES)}
    qualified.sort(
        key=lambda row: (
            -float(row["delta_BA_specific_CI_L"]),
            int(row["Rank"]),
            -float(row["Task_BA"]),
            fixed_order[row["Backbone"]],
        )
    )
    selection = {
        "status": "QUALIFIED_TARGET_REQUIRES_SEED_REPLICATION" if qualified else "NO_GLOBALLY_QUALIFIED_TARGET",
        "eligible_family_size": 16,
        "qualified_targets": [f"{row['Backbone']}::{row['Block']}" for row in qualified],
        "selected_target": (f"{qualified[0]['Backbone']}::{qualified[0]['Block']}" if qualified else None),
        "selection_rule": "highest H4 LCB; lower rank; higher task BA; frozen backbone order",
        "outer_test_used": False,
    }
    write_json(RESULTS / "ACTIONABLE_TARGET_SELECTION.json", selection)
    if qualified:
        terminal_state = "ACTIONABLE_TARGET_SEED_REPLICATION_REQUIRED"
        next_action = "RUN_TWO_FROZEN_SEED_REPLICATIONS"
        conclusion = "At least one target survived H1-H5 and the global 16-candidate Holm correction; AGDI remains unauthorized pending fixed-seed replication."
    else:
        terminal_state = "FINAL_MULTIBACKBONE_FALSIFICATION_CLOSURE"
        next_action = "STOP_CONSTRUCTIVE_SEARCH"
        conclusion = "No new backbone/block target jointly survived H1-H5 and prospective global multiplicity. The frozen five-backbone search is closed; AGDI is not authorized."
        write_json(
            RESULTS / "ACTIONABLE_TARGET_SEED_REPLICATION.json",
            {"status": "NOT_APPLICABLE_NO_GLOBALLY_QUALIFIED_TARGET", "seeds_run": [], "AGDI_authorized": False},
        )
        write_json(
            PROTOCOL / "AGDI_PROTOCOL_LOCK.json",
            {"status": "NOT_AUTHORIZED_NO_REPLICATED_TARGET", "alpha_grid": [0, 0.25, 0.5, 0.75, 1.0], "outer_test_used": False},
        )
        write_json(
            PROTOCOL / "FINAL_WBCIC_OUTER_EVALUATION_LOCK.json",
            {"status": "NOT_APPLICABLE_AGDI_UNAUTHORIZED", "outer_test_state": "SEALED_UNUSED", "outer_test_used": False},
        )
    final = {
        "terminal_state": terminal_state,
        "scientific_conclusion": conclusion,
        "next_action": next_action,
        "MULTIBACKBONE_NO_REPLICATED_ACTIONABLE_TARGET": not bool(qualified),
        "STOP_CONSTRUCTIVE_SEARCH": not bool(qualified),
        "AGDI_AUTHORIZED": False,
        "qualified_targets_before_replication": selection["qualified_targets"],
        "outer_test_state": "OUTER_TEST_LOCKED_AND_UNUSED",
        "outer_test_used": False,
        "frozen_EEGNet_conclusion_preserved": True,
        "global_family_size": 16,
        "global_multiplicity": "Holm FWER on p_joint=max(required component p-values)",
    }
    write_json(OUT / "FINAL_DECISION.json", final)
    write_reports(final, backbone_rows, master_rows)
    reproducibility(final)
    print(json.dumps(final, indent=2))


def write_reports(final: dict[str, Any], backbone_rows: list[dict[str, Any]], block_rows: list[dict[str, Any]]) -> None:
    table = pd.DataFrame(backbone_rows).to_markdown(index=False)
    competent = [row["Backbone"] for row in backbone_rows if row["Competence"]]
    incompetent = [row["Backbone"] for row in backbone_rows if not row["Competence"]]
    weak_audit = [
        row["Backbone"]
        for row in backbone_rows
        if row["Competence"] and row.get("Audit_baseline_BA") is not None and float(row["Audit_baseline_BA"]) < 0.60
    ]
    compact = pd.DataFrame(block_rows)[
        ["Backbone", "Block", "H1", "H2", "H3", "H4", "H5", "Protected", "Globally_qualified_actionable", "Action"]
    ].to_markdown(index=False)
    text = f"""# PERSIST-EEG final multi-backbone closure

## Outcome

`{final['terminal_state']}`

{final['scientific_conclusion']}

The result is prospective over exactly four new representation families and
four fixed rank blocks. The family-wise test uses all 16 slots, including
incompetent-backbone slots as p=1. The ten outer WBCIC subjects remain sealed
and unused. EEGNet was not rerun or reinterpreted.

Competent representations: `{', '.join(competent)}`. Competence failures:
`{', '.join(incompetent) if incompetent else 'none'}`. A failed backbone does
not contribute H1--H5 evidence.

## Backbone summary

{table}

## Gate/action matrix

{compact}

## Interpretation limits

- A competence failure is not evidence that persistent nuisance is absent; it
  means that representation cannot support an interpretable PERSIST audit.
- The spectral FBCNet family failed near chance, so the closure contains no
  competent filter-bank audit. This weakens inductive-bias coverage and must
  not be described as five competent backbones.
- `{', '.join(weak_audit) if weak_audit else 'No competent backbone'}` had a
  cross-fitted audit baseline below 0.60 after restricting training to the
  three model-fit folds; its block-level audit is correspondingly weaker than
  its task-search competence result.
- A negative H4/H5 result is evidence against safe removability under this
  frozen intervention, not proof that the block contains no subject signal.
- The study tests four pre-registered blocks in five representation families;
  it does not support claims about every possible architecture or latent basis.
- No theorem guarantees that removing a harmful-utility direction improves
  generalization. That empirical claim is exactly what H4/H5 test.
"""
    (OUT / "scientific_report.md").write_text(text, encoding="utf-8")
    paper = f"""# Paper-ready results

**Terminal state:** `{final['terminal_state']}`

We prospectively evaluated four new EEG representation families—FBCNet,
EEGConformer, DeepConvNet and TeCh—against the frozen EEGNet reference using
the identical 41-subject development cohort, S1+S2→S3 protocol and four
persistence-rank blocks. Candidate actionability required persistence, harmful
signed utility, local and finite decision dependence, ≥0.5 percentage-point
specific balanced-accuracy gain, subject/fold stability, and Holm control over
the full 4×4 candidate family. {final['scientific_conclusion']} The ten held-out
subjects were not opened.

{table}
"""
    (OUT / "PAPER_READY_RESULTS.md").write_text(paper, encoding="utf-8")


def reproducibility(final: dict[str, Any]) -> None:
    files = [
        PROTOCOL / "BACKBONE_ROSTER_LOCK.json",
        PROTOCOL / "MULTIBACKBONE_PROTOCOL_LOCK.json",
        PROTOCOL / "MULTIBACKBONE_MULTIPLICITY_LOCK.json",
        RESULTS / "MASTER_BACKBONE_RESULTS.csv",
        RESULTS / "MASTER_BLOCK_RESULTS.csv",
        RESULTS / "MASTER_ACTION_MATRIX.csv",
    ]
    reference_files = {
        "development_scope": REFERENCE_OUT / "protocol" / "DEVELOPMENT_SCOPE_LOCK.json",
        "cache_scope": REFERENCE_OUT / "protocol" / "CACHE_SCOPE_AUDIT.json",
        "cache_inventory": REFERENCE_OUT / "protocol" / "CACHE_INVENTORY.csv",
        "preprocessing": REFERENCE_OUT / "protocol" / "PREPROCESSING_PROTOCOL_LOCK.json",
        "raw_data": REFERENCE_OUT / "protocol" / "WBCIC_RAW_DATA_LOCK.json",
    }
    checkpoint_hashes: dict[str, str] = {}
    elapsed: dict[str, dict[str, float | None]] = {}
    for backbone in BACKBONES:
        competence_path = RESULTS / f"BACKBONE_{backbone.upper()}_COMPETENCE_RESULT.json"
        audit_path = RESULTS / f"BACKBONE_{backbone.upper()}_AUDIT_RESULT.json"
        competence_payload = json.loads(competence_path.read_text(encoding="utf-8"))
        audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
        for item in competence_payload.get("checkpoints", []):
            checkpoint_hashes[str(item["checkpoint"])] = str(item["checkpoint_sha256"])
        for item in audit_payload.get("checkpoints", []):
            checkpoint_hashes[str(item["checkpoint"])] = str(item["checkpoint_sha256"])
        elapsed[backbone] = {
            "task_search_seconds": competence_payload.get("elapsed_seconds"),
            "audit_seconds": audit_payload.get("elapsed_seconds"),
        }
    payload = {
        "git_sha": git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available_at_finalize": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "reference_scope_hash": "dae8e7ec00cbcf6dcc8c5b25829f2148fd0b5fdf162f75a0cddc18b096af7db4",
        "reference_artifact_sha256": {name: sha256_file(path) for name, path in reference_files.items()},
        "reference_EEGNet_commit": "61e4157817bc9c04f50471fb9dd6b865d74e21e4",
        "checkpoint_sha256": checkpoint_hashes,
        "elapsed_seconds_by_backbone": elapsed,
        "artifact_sha256": {str(path.relative_to(REPO_ROOT)).replace("\\", "/"): sha256_file(path) for path in files},
        "commands": [
            "python code/freeze_protocol.py",
            "python code/pipeline.py all --device cuda --workers 0",
            "python code/finalize.py",
            "python code/figures.py",
        ],
        "final_terminal_state": final["terminal_state"],
        "outer_test_used": False,
    }
    write_json(OUT / "REPRODUCIBILITY.json", payload)


if __name__ == "__main__":
    main()
