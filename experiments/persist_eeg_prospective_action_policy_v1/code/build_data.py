from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import (
    DATA,
    PROTOCOL,
    REPO_ROOT,
    assert_no_outer_columns,
    recursive_outer_true,
    require_files,
    router_pilot_root,
    sha256_file,
    write_csv,
    write_json,
)


ID_COLUMNS = ["fold", "seed", "router_fold", "manifest_index", "subject", "session", "label"]


PILOTS: list[dict[str, Any]] = [
    {
        "id": "signed_utility_v3_1",
        "artifact": "experiments/persist_eeg_p4_signed_v3_1/results_v3_1/SIGNED_V3_1_FINAL_REPORT.json",
        "decision_unit": "run x persistence block",
        "actions": ["KEEP", "ERASE_BLOCK"],
        "outcome": "signed CE utility",
        "pre_action_features": ["persistence rank/effect", "matched random control"],
        "future_outcome_quantities": ["signed_u_abs", "signed_u_spec"],
        "model_use": "audit only; direct U is not reused on its own outcome group",
    },
    {
        "id": "shared_geometry_v1_2",
        "artifact": "experiments/persist_eeg_p4_shared_geometry/results_v1_2/SHARED_GEOMETRY_FINAL_REPORT.json",
        "decision_unit": "run x paradigm x block",
        "actions": ["diagnostic only"],
        "outcome": "cross-paradigm geometry/utility",
        "pre_action_features": ["geometry strength", "principal/contrast geometry"],
        "future_outcome_quantities": ["utility measured on validation roles"],
        "model_use": "provenance and feature semantics only",
    },
    {
        "id": "p5_p6",
        "artifact": "experiments/persist_eeg_p5_1_p6/outputs/P5_1_P6_FINAL_REPORT.json",
        "decision_unit": "run and subject for strict-inductive fusion",
        "actions": ["BASE", "PROTECTED_FUSION", "ALL_PERSISTENCE", "RANDOM"],
        "outcome": "subject-balanced BA",
        "pre_action_features": ["frozen signed assignments", "task-only configuration"],
        "future_outcome_quantities": ["subject delta BA"],
        "model_use": "oracle/action history audit; incompatible with sample router",
    },
    {
        "id": "historical_router",
        "artifact": "experiments/persist_eeg_router/outputs/final/PERSIST_ROUTER_FINAL_REPORT.json",
        "decision_unit": "individual OpenBMI trial",
        "actions": ["KEEP", "ERASE", "AMPLIFY", "GEOMETRY"],
        "outcome": "subject-balanced BA from trial decisions",
        "pre_action_features": ["label-free confidence", "counterfactual logit response", "protected geometry"],
        "future_outcome_quantities": ["rescue/harm", "realised correctness effect"],
        "model_use": "sample-family modelling with subject-grouped validation",
    },
    {
        "id": "persist_cf",
        "artifact": "experiments/persist_eeg_cf/outputs/final/PERSIST_CF_FINAL_REPORT.json",
        "decision_unit": "run/configuration; nested subject folds",
        "actions": ["BASE", "CF", "DUPLICATE", "FULL", "HISTORICAL", "RANDOM"],
        "outcome": "BA/NLL by configuration",
        "pre_action_features": ["frozen configuration and TRAIN-only estimates"],
        "future_outcome_quantities": ["configuration result"],
        "model_use": "audit only; configuration-level unit incompatible with trial/block units",
    },
    {
        "id": "dda_v1",
        "artifact": "experiments/persist_eeg_dda_v1/outputs/DDA_FINAL_REPORT.json",
        "decision_unit": "run x audit fold x persistence block",
        "actions": ["NO_OP", "SUPPRESS_BLOCK"],
        "outcome": "held-out outcome-role CE and BA change",
        "pre_action_features": ["persistence", "decision dependence", "geometry"],
        "future_outcome_quantities": ["outcome_ce_effect", "outcome_ba_change", "same-cell signed U"],
        "model_use": "block-family modelling; same-cell U excluded and cross-fitted U constructed",
    },
    {
        "id": "wbcic_eegnet",
        "artifact": "experiments/persist_eeg_wbcic_actionability_v2/outputs/FINAL_DECISION.json",
        "decision_unit": "backbone x cross-fit fold x persistence block",
        "actions": ["NO_OP", "SUPPRESS_BLOCK"],
        "outcome": "development-only S3 subject-balanced delta BA",
        "pre_action_features": ["S1/S2 persistence", "label-free decision response", "basis geometry"],
        "future_outcome_quantities": ["S3 signed utility", "S3 delta BA"],
        "model_use": "WBCIC development block family; cross-fitted U only",
    },
    {
        "id": "wbcic_multibackbone",
        "artifact": "experiments/persist_eeg_multibackbone_final_closure/outputs/FINAL_DECISION.json",
        "decision_unit": "backbone x cross-fit fold x persistence block",
        "actions": ["NO_OP", "SUPPRESS_BLOCK"],
        "outcome": "development-only S3 subject-balanced delta BA",
        "pre_action_features": ["S1/S2 persistence", "label-free decision response", "basis geometry"],
        "future_outcome_quantities": ["S3 signed utility", "S3 delta BA"],
        "model_use": "competent WBCIC backbones only; FBCNet excluded after competence failure",
    },
]


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def provenance_audit() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for spec in PILOTS:
        path = REPO_ROOT / spec["artifact"]
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        outer_true = recursive_outer_true(payload)
        row = dict(spec)
        row.update(
            {
                "artifact_sha256": sha256_file(path),
                "outer_true_paths": outer_true,
                "eligible": not outer_true,
            }
        )
        rows.append(row)
    if any(not row["eligible"] for row in rows):
        offenders = {row["id"]: row["outer_true_paths"] for row in rows if not row["eligible"]}
        raise RuntimeError(f"Selected historical artifact depends on outer data: {offenders}")
    payload = {
        "status": "PILOT_PROVENANCE_AUDIT_PASS",
        "OUTER_TEST_USED": False,
        "outer_subject_ids_loaded": False,
        "outer_samples_materialized": False,
        "pooling_rule": "Incompatible sample, block, subject, fold, and configuration units remain separate families.",
        "pilots": rows,
    }
    write_json(PROTOCOL / "PILOT_PROVENANCE_AUDIT.json", payload)
    table = pd.DataFrame(rows)[
        ["id", "decision_unit", "actions", "outcome", "model_use", "eligible"]
    ].copy()
    table["actions"] = table.actions.map(lambda value: ", ".join(value))
    md = f"""# Pilot provenance audit

`OUTER_TEST_USED = false`

No sealed WBCIC outer subject identifier, sample, embedding, label, or outcome
was loaded. All selected artifacts explicitly report no outer-test use.

{_markdown_table(table)}

## Pooling decision

The historical router acts per trial, DDA acts per run/fold/block, P5/P6 acts
per run or subject, and the WBCIC audit acts per backbone/fold/block. Treating
these rows as exchangeable would be pseudo-replication. This experiment fits
and evaluates a separate policy inside each compatible decision family.

Same-cell realised signed utility is an outcome, not a legal predictor. DDA
and WBCIC therefore receive only leave-group-out utility estimates constructed
without the target outcome cell.
"""
    (PROTOCOL / "PILOT_PROVENANCE_AUDIT.md").write_text(md, encoding="utf-8")
    return payload


def _merge_router_frames(root: Path) -> pd.DataFrame:
    cache = root / "experiments" / "persist_eeg_router" / "outputs" / "cache"
    paths = {
        "features": cache / "OOF_ROUTER_FEATURES.parquet",
        "base": cache / "OOF_BASE_LOGITS.parquet",
        "counter": cache / "OOF_COUNTERFACTUAL_LOGITS.parquet",
        "geometry": cache / "OOF_GEOMETRY_FEATURES.parquet",
    }
    require_files(paths.values())
    frames = {name: pd.read_parquet(path).sort_values(ID_COLUMNS).reset_index(drop=True) for name, path in paths.items()}
    reference = frames["features"][ID_COLUMNS]
    for name, frame in frames.items():
        assert_no_outer_columns(frame, f"router:{name}")
        if not reference.equals(frame[ID_COLUMNS]):
            raise RuntimeError(f"Router cache identity mismatch: {name}")
    merged = frames["features"].copy()
    for name in ("base", "counter", "geometry"):
        extra = [column for column in frames[name].columns if column not in ID_COLUMNS]
        merged = pd.concat([merged, frames[name][extra]], axis=1)
    if len(merged) != 40800 or merged.subject.astype(str).nunique() != 54:
        raise RuntimeError(f"Unexpected Router cache scope: rows={len(merged)}, subjects={merged.subject.nunique()}")
    return merged


def router_dataset() -> tuple[pd.DataFrame, dict[str, Any]]:
    pilot_root = router_pilot_root()
    raw = _merge_router_frames(pilot_root)
    label = raw.label.to_numpy(dtype=int)
    logits = {
        "noop": raw[["keep_logit_0", "keep_logit_1"]].to_numpy(),
        "erase": raw[["erase_logit_0", "erase_logit_1"]].to_numpy(),
        "amplify": raw[["amplify_logit_0", "amplify_logit_1"]].to_numpy(),
        "geometry": raw[["geometry_logit_0", "geometry_logit_1"]].to_numpy(),
    }
    pred = {name: value.argmax(axis=1) for name, value in logits.items()}
    correct = {name: (value == label).astype(np.int8) for name, value in pred.items()}
    frame = pd.DataFrame(
        {
            "family_id": "openbmi_sample_router",
            "decision_unit_type": "trial",
            "dataset_id": "OpenBMI",
            "backbone_id": "EEGNet",
            "fold_id": raw.fold.astype(int),
            "seed_id": raw.seed.astype(int),
            "audit_fold_id": raw.router_fold.astype(int),
            "subject_id": raw.subject.astype(str),
            "session_id": raw.session.astype(str),
            "block_id": "PROTECTED_UNION",
            "manifest_index": raw.manifest_index.astype(int),
            "outcome_label": label,
            "pred_noop": pred["noop"],
            "pred_erase": pred["erase"],
            "pred_amplify": pred["amplify"],
            "pred_geometry": pred["geometry"],
            "effect_erase": correct["erase"] - correct["noop"],
            "effect_amplify": correct["amplify"] - correct["noop"],
            "effect_geometry": correct["geometry"] - correct["noop"],
            "outer_test_used": False,
        }
    )
    feature_columns = [column for column in raw.columns if column not in ID_COLUMNS and "logit_" not in column]
    for column in feature_columns:
        frame[f"f_router_{column}"] = raw[column].astype(float)
    repeats = frame.groupby("manifest_index").manifest_index.transform("size")
    frame["unit_weight"] = 1.0 / repeats
    meta = {
        "source_root": str(pilot_root),
        "rows": len(frame),
        "subjects": int(frame.subject_id.nunique()),
        "runs": int(frame[["fold_id", "seed_id"]].drop_duplicates().shape[0]),
        "actions": ["NO_OP", "ERASE", "AMPLIFY", "GEOMETRY"],
        "decision_unit": "individual trial; validation grouped by subject across all repeated runs",
        "feature_columns": [f"f_router_{column}" for column in feature_columns],
    }
    return frame, meta


def _leave_cell_out_mean(frame: pd.DataFrame, group: list[str], value: str) -> pd.Series:
    total = frame.groupby(group)[value].transform("sum")
    count = frame.groupby(group)[value].transform("count")
    result = (total - frame[value]) / np.maximum(count - 1, 1)
    return result.where(count > 1)


def _leave_cell_out_std(frame: pd.DataFrame, group: list[str], value: str) -> pd.Series:
    result = np.full(len(frame), np.nan, dtype=float)
    for _, idx in frame.groupby(group).groups.items():
        positions = np.asarray(list(idx), dtype=int)
        values = frame.loc[positions, value].to_numpy(dtype=float)
        for offset, position in enumerate(positions):
            others = np.delete(values, offset)
            result[position] = float(np.std(others, ddof=1)) if len(others) > 1 else 0.0
    return pd.Series(result, index=frame.index)


def dda_dataset() -> tuple[pd.DataFrame, dict[str, Any]]:
    path = REPO_ROOT / "experiments" / "persist_eeg_dda_v1" / "outputs" / "results" / "DDA_BLOCK_CROSSFIT.csv"
    raw = pd.read_csv(path)
    assert_no_outer_columns(raw, "DDA_BLOCK_CROSSFIT")
    if len(raw) != 215 or raw.run.nunique() != 6:
        raise RuntimeError("DDA block table does not match the frozen pilot")
    raw = raw.reset_index(drop=True)
    # signed_u_spec is constant across the five audit folds of a run/block.
    # Leaving out one audit cell would therefore copy the same outcome-derived
    # value back into that cell. Construct priors from entirely different runs
    # or outer folds instead.
    run_u = raw.groupby(["run", "fold", "block"], as_index=False).signed_u_spec.first()
    crossrun: dict[tuple[str, int], tuple[float, float]] = {}
    crossfold: dict[tuple[int, int], tuple[float, float]] = {}
    def prior_stats(values: np.ndarray) -> tuple[float, float]:
        if len(values) == 0:
            return float("nan"), float("nan")
        return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0

    for item in run_u.itertuples(index=False):
        other_runs = run_u[(run_u.block == item.block) & (run_u.run != item.run)].signed_u_spec.to_numpy(dtype=float)
        other_folds = run_u[(run_u.block == item.block) & (run_u.fold != item.fold)].signed_u_spec.to_numpy(dtype=float)
        crossrun[(str(item.run), int(item.block))] = prior_stats(other_runs)
        crossfold[(int(item.fold), int(item.block))] = prior_stats(other_folds)
    raw["u_crossrun"] = [crossrun[(str(run), int(block))][0] for run, block in zip(raw.run, raw.block)]
    raw["u_crossrun_sd"] = [crossrun[(str(run), int(block))][1] for run, block in zip(raw.run, raw.block)]
    raw["u_crossouterfold"] = [crossfold[(int(fold), int(block))][0] for fold, block in zip(raw.fold, raw.block)]
    raw["u_crossouterfold_sd"] = [crossfold[(int(fold), int(block))][1] for fold, block in zip(raw.fold, raw.block)]
    frame = pd.DataFrame(
        {
            "family_id": "openbmi_dda_block",
            "decision_unit_type": "run_audit_fold_block",
            "dataset_id": "OpenBMI",
            "backbone_id": "EEGNet",
            "fold_id": raw.fold.astype(int),
            "seed_id": raw.seed.astype(int),
            "audit_fold_id": raw.audit_fold.astype(int),
            "subject_id": "GROUPED_OUTCOME_SUBJECTS",
            "session_id": "future_session",
            "block_id": raw.block.map(lambda value: f"B{int(value):02d}"),
            "manifest_index": -1,
            "effect_suppress": raw.outcome_ba_change.astype(float),
            "effect_suppress_ce": raw.outcome_ce_effect.astype(float),
            "outer_test_used": False,
            "unit_weight": 1.0,
            "f_persistence_strength": raw.persistence_strength,
            "f_geometry_strength": raw.geometry_strength,
            "f_rank": raw["rank"],
            "f_jacobian_ratio": raw.jacobian_ratio,
            "f_decision_logit_ratio": raw.decision_logit_ratio,
            "f_decision_margin": raw.decision_margin_displacement,
            "f_decision_flip": raw.decision_flip_rate,
            "f_decision_tv": raw.decision_total_variation,
            "f_u_crossrun": raw.u_crossrun,
            "f_u_crossrun_sd": raw.u_crossrun_sd,
            "f_u_crossouterfold": raw.u_crossouterfold,
            "f_u_crossouterfold_sd": raw.u_crossouterfold_sd,
        }
    )
    meta = {
        "rows": len(frame),
        "runs": int(raw.run.nunique()),
        "actions": ["NO_OP", "SUPPRESS_BLOCK"],
        "decision_unit": "run x audit fold x block; outcome subjects are disjoint from feature/decision subjects",
        "same_cell_u_excluded": True,
        "same_run_u_excluded_for_leave_run_out": True,
        "same_outer_fold_u_excluded_for_leave_outer_fold_out": True,
        "validation": ["leave-one-run-out", "leave-one-outer-fold-out"],
    }
    return frame, meta


def _read_wbcic_backbone(backbone: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if backbone == "EEGNet":
        root = REPO_ROOT / "experiments" / "persist_eeg_wbcic_actionability_v2" / "outputs" / "results"
        audit = pd.read_csv(root / "WBCIC_AUDIT_SUBJECT_RESULTS.csv").assign(backbone="EEGNet")
        random = pd.read_csv(root / "WBCIC_AUDIT_RANDOM_SUBJECT_RESULTS.csv").assign(backbone="EEGNet")
        persistence = pd.read_csv(root / "WBCIC_PERSISTENCE_SUBJECT_RESULTS.csv").assign(backbone="EEGNet")
        basis = pd.read_csv(root / "PERSISTENCE_BASIS_RESULTS.csv").assign(backbone="EEGNet")
    else:
        root = REPO_ROOT / "experiments" / "persist_eeg_multibackbone_final_closure" / "outputs" / "results"
        prefix = f"BACKBONE_{backbone.upper()}"
        audit = pd.read_csv(root / f"{prefix}_AUDIT_SUBJECT_RESULTS.csv")
        random = pd.read_csv(root / f"{prefix}_AUDIT_RANDOM_SUBJECT_RESULTS.csv")
        persistence = pd.read_csv(root / f"{prefix}_PERSISTENCE_SUBJECT_RESULTS.csv")
        basis = pd.read_csv(root / f"{prefix}_BASIS_RESULTS.csv")
    for name, item in (("audit", audit), ("random", random), ("persistence", persistence), ("basis", basis)):
        assert_no_outer_columns(item, f"WBCIC:{backbone}:{name}")
    return audit, random, persistence, basis


def wbcic_dataset() -> tuple[pd.DataFrame, dict[str, Any]]:
    master = pd.read_csv(
        REPO_ROOT
        / "experiments"
        / "persist_eeg_multibackbone_final_closure"
        / "outputs"
        / "results"
        / "MASTER_BACKBONE_RESULTS.csv"
    )
    competent = [str(value) for value in master.loc[master.Competence.astype(bool), "Backbone"]]
    if set(competent) != {"EEGNet", "EEGConformer", "DeepConvNet", "TeCh"}:
        raise RuntimeError(f"Unexpected competent backbone set: {competent}")
    task_ba = dict(zip(master.Backbone.astype(str), master.Best_task_BA.astype(float)))
    rows: list[dict[str, Any]] = []
    for backbone in competent:
        audit, random, persistence, basis = _read_wbcic_backbone(backbone)
        finite_random = random.groupby(["fold", "block"], as_index=False).finite_logit_RMS.mean().rename(
            columns={"finite_logit_RMS": "random_logit_rms"}
        )
        p_summary = persistence.groupby(["fold", "block"], as_index=False).agg(
            persistence_specific=("persistence_specific", "mean"),
            persistence_subject_sd=("persistence_specific", "std"),
        )
        b_columns = [
            "fold",
            "block",
            "rank",
            "eigenvalue_sum",
            "minimum_eigenvalue",
            "effective_positive_rank",
            "cross_covariance_condition_positive",
            "s1_s2_centroid_correlation",
            "s1_s2_principal_angle_mean_deg",
            "s1_s2_principal_angle_max_deg",
        ]
        b_columns = [column for column in b_columns if column in basis.columns]
        cell = audit.groupby(["fold", "block"], as_index=False).agg(
            base_BA=("base_BA", "mean"),
            effect_suppress=("delta_BA_specific", "mean"),
            effect_suppress_raw=("delta_BA", "mean"),
            u_fold=("u_spec", "mean"),
            u_subject_sd=("u_spec", "std"),
            candidate_logit_rms=("logit_RMS", "mean"),
            decision_margin=("margin_displacement", "mean"),
            decision_flip=("prediction_flip_rate", "mean"),
            decision_tv=("total_variation", "mean"),
        )
        cell = cell.merge(finite_random, on=["fold", "block"], validate="one_to_one")
        cell = cell.merge(p_summary, on=["fold", "block"], validate="one_to_one")
        cell = cell.merge(basis[b_columns], on=["fold", "block"], validate="one_to_one")
        cell["u_crossfold"] = _leave_cell_out_mean(cell, ["block"], "u_fold")
        cell["u_crossfold_sd"] = _leave_cell_out_std(cell, ["block"], "u_fold")
        cell["backbone"] = backbone
        for row in cell.to_dict(orient="records"):
            rows.append(row)
    raw = pd.DataFrame(rows)
    # For leave-one-backbone-out, utility priors must not use any outcome from
    # the held-out representation. Build an explicit cross-backbone prior.
    raw["u_crossbackbone"] = _leave_cell_out_mean(raw, ["fold", "block"], "u_fold")
    raw["u_crossbackbone_sd"] = _leave_cell_out_std(raw, ["fold", "block"], "u_fold")
    frame = pd.DataFrame(
        {
            "family_id": "wbcic_development_block",
            "decision_unit_type": "backbone_fold_block",
            "dataset_id": "WBCIC",
            "backbone_id": raw.backbone,
            "fold_id": raw.fold.astype(int),
            "seed_id": 20260817,
            "audit_fold_id": raw.fold.astype(int),
            "subject_id": "GROUPED_DEVELOPMENT_SUBJECTS",
            "session_id": "S3",
            "block_id": raw.block,
            "manifest_index": -1,
            "effect_suppress": raw.effect_suppress,
            "effect_suppress_raw": raw.effect_suppress_raw,
            "outer_test_used": False,
            "unit_weight": 1.0,
            "f_task_BA": raw.backbone.map(task_ba),
            "f_baseline_BA": raw.base_BA,
            "f_persistence_strength": raw.persistence_specific,
            "f_persistence_sd": raw.persistence_subject_sd,
            "f_rank": raw["rank"],
            "f_decision_logit_ratio": raw.candidate_logit_rms / raw.random_logit_rms.clip(lower=1e-12),
            "f_decision_margin": raw.decision_margin,
            "f_decision_flip": raw.decision_flip,
            "f_decision_tv": raw.decision_tv,
            "f_u_crossfold": raw.u_crossfold,
            "f_u_crossfold_sd": raw.u_crossfold_sd,
            "f_u_crossbackbone": raw.u_crossbackbone,
            "f_u_crossbackbone_sd": raw.u_crossbackbone_sd,
            "f_eigenvalue_sum": raw.eigenvalue_sum,
            "f_minimum_eigenvalue": raw.minimum_eigenvalue,
            "f_effective_positive_rank": raw.effective_positive_rank,
            "f_cross_covariance_condition": raw.cross_covariance_condition_positive,
            "f_centroid_correlation": raw.s1_s2_centroid_correlation,
            "f_principal_angle_mean": raw.s1_s2_principal_angle_mean_deg,
            "f_principal_angle_max": raw.s1_s2_principal_angle_max_deg,
        }
    )
    meta = {
        "rows": len(frame),
        "backbones": competent,
        "excluded_backbones": ["FBCNet: REPRESENTATION_COMPETENCE_FAIL"],
        "actions": ["NO_OP", "SUPPRESS_BLOCK"],
        "decision_unit": "competent backbone x development subject fold x fixed persistence block",
        "same_fold_u_excluded": True,
        "validation": ["leave-one-fold-out with cross-fold U", "leave-one-backbone-out with cross-backbone U"],
        "OUTER_TEST_USED": False,
    }
    return frame, meta


def feature_dictionary(frame: pd.DataFrame, family_meta: dict[str, Any]) -> None:
    features = [column for column in frame.columns if column.startswith("f_")]
    target = [column for column in frame.columns if column.startswith("effect_")]
    schema = {
        "version": "prospective_action_policy_v1",
        "row_layout": "wide action outcomes; one row per real decision unit, never one pseudo-independent row per action",
        "families": family_meta,
        "feature_columns": features,
        "target_columns": target,
        "forbidden_model_columns": [
            "outcome_label",
            "pred_noop",
            "pred_erase",
            "pred_amplify",
            "pred_geometry",
            *target,
        ],
        "OUTER_TEST_USED": False,
    }
    write_json(DATA / "META_DATASET_SCHEMA.json", schema)
    md = """# Action feature dictionary

The CSV is wide by action so repeated outcomes are not presented as
independent meta-samples. `family_id` determines the legal decision unit and
the applicable action menu.

## Legal feature families

- `f_router_*`: label-free task confidence, counterfactual logit response,
  and protected-geometry features computed by subject-cross-fitting.
- `f_persistence_*`, `f_rank`: persistence evidence constructed outside the
  held-out consequence role.
- `f_u_crossrun*` / `f_u_crossouterfold*`: DDA utility priors from entirely
  different runs or outer folds; repeated audit cells from the target run are
  excluded.
- `f_u_crossfold*`: WBCIC signed-utility estimates from other outcome folds only.
- `f_u_crossbackbone*`: utility prior from other backbones only, used solely
  for leave-one-backbone-out validation.
- `f_decision_*`, `f_jacobian_*`: label-free local/finite decision response.
- `f_geometry_*`, eigenvalue/angle/condition features: pre-outcome geometry.
- `f_task_BA`, `f_baseline_BA`: development-only representation competence.

## Targets and prohibited inputs

All `effect_*` columns, target labels, action predictions, realised oracle
actions, and same-cell signed utility are outcomes. They are never supplied to
a model. The sealed WBCIC outer test is absent. `OUTER_TEST_USED = false`.

## Grouping

OpenBMI router rows are grouped by subject across every fold and seed. DDA
rows are grouped by complete run. WBCIC rows are evaluated with complete fold
or backbone holdout. Families are never pooled in a single fitted model.
"""
    (DATA / "ACTION_FEATURE_DICTIONARY.md").write_text(md, encoding="utf-8")


def build_all() -> tuple[pd.DataFrame, dict[str, Any]]:
    provenance_audit()
    router, router_meta = router_dataset()
    dda, dda_meta = dda_dataset()
    wbcic, wbcic_meta = wbcic_dataset()
    combined = pd.concat([router, dda, wbcic], ignore_index=True, sort=False)
    if bool(combined.outer_test_used.any()):
        raise RuntimeError("Outer-test row entered the legal action-outcome dataset")
    if combined.family_id.value_counts().to_dict() != {
        "openbmi_sample_router": 40800,
        "openbmi_dda_block": 215,
        "wbcic_development_block": 80,
    }:
        raise RuntimeError(f"Unexpected family counts: {combined.family_id.value_counts().to_dict()}")
    write_csv(DATA / "ACTION_OUTCOME_DATASET.csv", combined)
    meta = {
        "openbmi_sample_router": router_meta,
        "openbmi_dda_block": dda_meta,
        "wbcic_development_block": wbcic_meta,
    }
    feature_dictionary(combined, meta)
    return combined, meta


if __name__ == "__main__":
    data, metadata = build_all()
    print(json.dumps({"rows": len(data), "families": data.family_id.value_counts().to_dict()}, indent=2))
