from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
RESULTS = EXP / "results"
FIGURES = EXP / "figures"
P4A = REPO / "experiments" / "persist_eeg_p4a_cross_setting_expansion_v1"
P4B = REPO / "experiments" / "persist_eeg_p4b_identity_reliability_discovery_v1"
BRANCH = "codex/persist-eeg-p4c-suppression-safety-validation-v1"
P4B_TIP = "8b26c073cea98743f73734cff4f60b58c8e3fe71"
RESERVED = ["S4", "S6"]
DISCOVERY = ["S1", "S2", "S3", "S5"]
RANKS = list(range(1, 9))
CONTROL_COUNT = 100

sys.path.insert(0, str(HERE))
from p4c_safety_common import dataframe_markdown, now_utc, read_json, sha256, write_json  # noqa: E402

sys.path.insert(0, str(P4A / "code"))
import common as p4a_common  # noqa: E402


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def source_payload(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=True)
    return {
        "features": payload["source_features"].astype(np.float64),
        "subjects": payload["source_subjects"].astype(str),
        "sessions": payload["source_sessions"].astype(np.int64),
    }


def source_direction_equivalent(unit: Path, center: np.ndarray, direction: np.ndarray, expected: pd.Series) -> tuple[bool, dict[str, float]]:
    source = source_payload(unit / "source_freeze" / "erm__lambda-0.00" / "embeddings.npz")
    state = torch.load(unit / "checkpoints" / "erm__lambda-0.00.pt", map_location="cpu", weights_only=True)["state_dict"]
    weight = state["head.weight"].numpy().astype(np.float64)
    bias = state["head.bias"].numpy().astype(np.float64)
    features = source["features"]
    clean_logits = features @ weight.T + bias
    erased = p4a_common.erase_direction(features, center, direction)
    erased_logits = erased @ weight.T + bias
    observed_d = p4a_common.exact_d_finite(clean_logits, erased_logits)
    observed_o = p4a_common.task_subspace_overlap(weight, direction)
    observed_geometry = float(np.sqrt(np.mean(np.square((features - center) @ direction))))
    ordered = p4a_common.subject_sort(np.unique(source["subjects"]))
    means1 = np.stack([features[(source["subjects"] == name) & (source["sessions"] == 0)].mean(0) for name in ordered])
    means2 = np.stack([features[(source["subjects"] == name) & (source["sessions"] == 1)].mean(0) for name in ordered])
    p1 = (means1 - center) @ direction
    p2 = (means2 - center) @ direction
    observed_p = 0.0 if min(np.std(p1), np.std(p2)) < 1e-12 else float(np.corrcoef(p1, p2)[0, 1])
    values = {"D_finite": observed_d, "O_task": observed_o, "geometry_strength": observed_geometry, "persistence": observed_p}
    valid = all(np.isclose(value, float(expected[key]), rtol=1e-7, atol=1e-9) for key, value in values.items())
    return bool(valid), values


def p4b_audit() -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    required = [
        "P4B_PROTOCOL_FROZEN.json",
        "SOURCE_NORMALIZATION_FROZEN.json",
        "DISCOVERY_SETTING_ASSIGNMENT.json",
        "PRE_OUTCOME_FREEZE_COMPLETE.json",
        "P4B_FINAL_REPORT.md",
        "P4C_FINAL_REPORT.md",
        "results/P4B_FINAL_VALIDATION.json",
        "results/P4C_FINAL_VALIDATION.json",
        "results/source_evidence_normalized.csv",
        "results/discovery_future_utility_direction.csv",
        "results/regime_summary.csv",
    ]
    missing = [name for name in required if not (P4B / name).is_file()]
    if missing:
        raise RuntimeError(f"P4B required artifacts missing: {missing}")
    validation = read_json(P4B / "results" / "P4B_FINAL_VALIDATION.json")
    closure = read_json(P4B / "results" / "P4C_FINAL_VALIDATION.json")
    protocol = read_json(P4B / "P4B_PROTOCOL_FROZEN.json")
    normalization = read_json(P4B / "SOURCE_NORMALIZATION_FROZEN.json")
    source = pd.read_csv(P4B / "results" / "source_evidence_normalized.csv")
    source_cube_hash = sha256(P4A / "results" / "source_evidence_cube.csv")
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError("wrong P4C-Safety branch")
    if git("merge-base", "HEAD", P4B_TIP) != P4B_TIP:
        raise RuntimeError("P4C-Safety branch is not descended from validated P4B tip")
    checks = {
        "p4b_validator_pass": validation.get("pass") is True,
        "p4b_negative_terminal_preserved": validation.get("terminal") == "P4B_IDENTITY_RELIABILITY_CONDITION_NOT_SUPPORTED",
        "old_p4c_block_preserved": closure.get("terminal") == "P4C_NOT_AUTHORIZED_BY_P4B" and closure.get("FINAL_MODEL_AUTHORIZATION") == "NOT_AUTHORIZED",
        "reserved_exact": protocol.get("p4c_reserved_settings") == RESERVED,
        "reserved_untouched": validation.get("p4c_reserved_future_utility_accessed") is False,
        "openbmi_holdout_untouched": validation.get("OpenBMI_sealed_internal_holdout") == "UNTOUCHED",
        "wbcic_outer_untouched": validation.get("WBCIC_outer_10") == "UNTOUCHED_NOT_ENUMERATED",
        "source_cube_hash": normalization.get("source_cube_sha256") == source_cube_hash,
        "normalization_formula": normalization.get("E_task_formula") == "(z_D + z_C + z_O)/3",
        "normalization_primitives": normalization.get("active_E_task_primitives") == ["D", "C", "O"],
        "source_cardinality": len(source) == 720 and set(source.setting_id) == set(DISCOVERY + RESERVED),
    }
    if not all(checks.values()):
        raise RuntimeError(f"P4B/source audit failed: {checks}")
    audit = {
        "schema": "P4C_SAFETY_INPUT_AUDIT_V1",
        "timestamp_utc": now_utc(),
        "pass": True,
        "checks": checks,
        "p4b_parent_tip": P4B_TIP,
        "p4b_terminal": validation["terminal"],
        "p4b_validator_sha256": sha256(P4B / "results" / "P4B_FINAL_VALIDATION.json"),
        "old_p4c_validation_sha256": sha256(P4B / "results" / "P4C_FINAL_VALIDATION.json"),
        "source_cube_sha256": source_cube_hash,
        "source_evidence_normalized_sha256": sha256(P4B / "results" / "source_evidence_normalized.csv"),
        "normalization_sha256": sha256(P4B / "SOURCE_NORMALIZATION_FROZEN.json"),
        "OpenBMI_sealed_internal_holdout": "UNTOUCHED",
        "WBCIC_outer_10": "UNTOUCHED_NOT_ENUMERATED",
    }
    write_json(EXP / "P4C_SAFETY_INPUT_AUDIT.json", audit)
    (EXP / "P4C_SAFETY_INPUT_AUDIT.md").write_text(
        "# P4C-Safety Input Audit\n\n"
        f"PASS. The branch descends from P4B tip `{P4B_TIP}`. P4B remains `{validation['terminal']}` and its blocked original P4C closure is unchanged. "
        f"The frozen source cube is `{source_cube_hash}`; normalization is `{audit['normalization_sha256']}`. S4/S6 future outcomes had not been accessed.\n",
        encoding="utf-8",
    )
    (EXP / "P4B_NEGATIVE_RESULT_PRESERVATION.md").write_text(
        "# P4B Negative Result Preservation\n\n"
        "P4B remains `P4B_IDENTITY_RELIABILITY_CONDITION_NOT_SUPPORTED`. Its continuous MINT transport failed and is not rerun or reinterpreted here. "
        "P4C-Safety tests only the separately pre-specified coarse asymmetric safety regime. The original `P4C_NOT_AUTHORIZED_BY_P4B` closure remains intact in the P4B experiment.\n",
        encoding="utf-8",
    )
    return audit, protocol, source


def reproduce_discovery(protocol: dict[str, Any], source: pd.DataFrame) -> pd.DataFrame:
    direction = pd.read_csv(P4B / "results" / "discovery_future_utility_direction.csv")
    frozen_summary = pd.read_csv(P4B / "results" / "regime_summary.csv")
    rows: list[dict[str, Any]] = []
    for setting in DISCOVERY:
        threshold = protocol["regime"]["thresholds"][setting]
        cell = source[source.setting_id == setting].copy()
        cell["low"] = (cell.z_I >= threshold["high_I_lower"]) & (cell.E_task <= threshold["low_E_upper"])
        cell["high"] = (cell.z_I >= threshold["high_I_lower"]) & (cell.E_task >= threshold["high_E_lower"])
        observed = direction[direction.setting_id == setting].copy()
        merged = cell.merge(observed[["fold", "seed", "direction_rank", "highI_lowE", "highI_highE", "U_BA"]], on=["fold", "seed", "direction_rank"], validate="one_to_one")
        membership_exact = bool((merged.low == merged.highI_lowE).all() and (merged.high == merged.highI_highE).all())
        low_count = int(merged.low.sum())
        high_count = int(merged.high.sum())
        delta = float(merged.loc[merged.low, "U_BA"].mean() - merged.loc[merged.high, "U_BA"].mean())
        expected = frozen_summary[frozen_summary.setting_id == setting].iloc[0]
        exact = membership_exact and low_count == int(expected.low_count) and high_count == int(expected.high_count) and np.isclose(delta, float(expected.DeltaRegime), rtol=0, atol=1e-15)
        rows.append({"setting_id": setting, "low_count": low_count, "high_count": high_count, "DeltaRegime": delta, "membership_exact": membership_exact, "p4b_exact": bool(exact)})
    result = pd.DataFrame(rows)
    if not result.p4b_exact.all():
        raise RuntimeError("P4B discovery regime membership reproduction failed")
    result.to_csv(RESULTS / "discovery_regime_reproduction.csv", index=False)
    (EXP / "REGIME_REPRODUCTION_AUDIT.md").write_text(
        "# Regime Reproduction Audit\n\nPASS. The exact P4B quantile thresholds and inclusive comparisons reproduced all S1/S2/S3/S5 memberships, counts and DeltaRegime values.\n\n"
        + dataframe_markdown(result) + "\n",
        encoding="utf-8",
    )
    return result


def preflight_and_assign(audit: dict[str, Any], source: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float]]]:
    reserved = source[source.setting_id.isin(RESERVED)].copy()
    thresholds: dict[str, dict[str, float]] = {}
    run_rows: list[dict[str, Any]] = []
    reserved["source_direction_sha256"] = reserved.direction_sha256.astype(str)
    reserved["direction_equivalence_status"] = "BYTE_EXACT"
    for setting in RESERVED:
        cell = reserved[reserved.setting_id == setting]
        thresholds[setting] = {
            "high_I_lower": float(cell.z_I.quantile(2.0 / 3.0)),
            "low_I_upper": float(cell.z_I.quantile(1.0 / 3.0)),
            "low_E_upper": float(cell.E_task.quantile(1.0 / 3.0)),
            "high_E_lower": float(cell.E_task.quantile(2.0 / 3.0)),
        }
        for fold in p4a_common.FOLDS:
            for seed in p4a_common.SEEDS:
                run = p4a_common.run_dir(setting, fold, seed)
                unit = cell[(cell.fold == fold) & (cell.seed == seed)].sort_values("direction_rank")
                checkpoint = run / "checkpoints" / "erm__lambda-0.00.pt"
                normalizer = run / "normalizer.npz"
                basis_path = run / "source_freeze" / "erm__lambda-0.00" / "persistence_basis.npz"
                embedding = run / "source_freeze" / "erm__lambda-0.00" / "embeddings.npz"
                if len(unit) != 8 or not all(path.is_file() for path in [checkpoint, normalizer, basis_path, embedding]):
                    raise RuntimeError(f"runtime source artifact incomplete {setting}/{fold}/{seed}")
                if set(unit.checkpoint_sha256.astype(str)) != {sha256(checkpoint)} or set(unit.normalizer_sha256.astype(str)) != {sha256(normalizer)}:
                    raise RuntimeError(f"checkpoint/normalizer hash mismatch {setting}/{fold}/{seed}")
                artifact = np.load(basis_path, allow_pickle=False)
                center = artifact["center"].astype(np.float64)
                basis = artifact["basis"].astype(np.float64)
                runtime_basis_hash = p4a_common.array_sha256(basis)
                mismatch_count = 0
                for rank in RANKS:
                    row = unit[unit.direction_rank == rank].iloc[0]
                    runtime_hash = p4a_common.array_sha256(basis[:, rank - 1])
                    status = "BYTE_EXACT"
                    if runtime_hash != str(row.direction_sha256):
                        valid, _metrics = source_direction_equivalent(run, center, basis[:, rank - 1], row)
                        if not valid:
                            raise RuntimeError(f"direction source equivalence failure {setting}/{fold}/{seed}/{rank}")
                        status = "NUMERIC_EQUIVALENT_SOURCE_METRICS"
                        mismatch_count += 1
                    mask = (reserved.setting_id == setting) & (reserved.fold == fold) & (reserved.seed == seed) & (reserved.direction_rank == rank)
                    reserved.loc[mask, "direction_sha256"] = runtime_hash
                    reserved.loc[mask, "direction_equivalence_status"] = status
                run_rows.append({
                    "setting_id": setting,
                    "fold": fold,
                    "seed": seed,
                    "checkpoint_sha256": sha256(checkpoint),
                    "normalizer_sha256": sha256(normalizer),
                    "source_basis_sha256": str(unit.persistence_basis_sha256.iloc[0]),
                    "runtime_basis_sha256": runtime_basis_hash,
                    "direction_hash_mismatch_count": mismatch_count,
                    "source_numeric_equivalence": True,
                })
    for setting in RESERVED:
        threshold = thresholds[setting]
        mask = reserved.setting_id == setting
        reserved.loc[mask, "identity_tertile"] = np.where(reserved.loc[mask, "z_I"] >= threshold["high_I_lower"], "HIGH", np.where(reserved.loc[mask, "z_I"] <= threshold["low_I_upper"], "LOW", "MID"))
        reserved.loc[mask, "entanglement_tertile"] = np.where(reserved.loc[mask, "E_task"] >= threshold["high_E_lower"], "HIGH", np.where(reserved.loc[mask, "E_task"] <= threshold["low_E_upper"], "LOW", "MID"))
    reserved["regime_label"] = "UNCLASSIFIED"
    reserved.loc[(reserved.identity_tertile == "HIGH") & (reserved.entanglement_tertile == "LOW"), "regime_label"] = "REGIME_LOW"
    reserved.loc[(reserved.identity_tertile == "HIGH") & (reserved.entanglement_tertile == "HIGH"), "regime_label"] = "REGIME_HIGH"
    reserved["highest_identity"] = False
    for _key, cell in reserved.groupby(["setting_id", "fold", "seed"], sort=True):
        chosen = cell.sort_values(["z_I", "direction_rank"], ascending=[False, True]).index[0]
        reserved.loc[chosen, "highest_identity"] = True
    reserved["source_cube_hash"] = audit["source_cube_sha256"]
    reserved["normalization_hash"] = audit["normalization_sha256"]
    reserved["source_cube_hash"] = audit["source_cube_sha256"]
    reserved["E_task_definition"] = "(z_D + z_C + z_O)/3"
    columns = [
        "setting_id", "dataset", "task", "backbone", "fold", "seed", "direction_rank",
        "z_I", "z_D", "z_C", "z_O", "E_task", "identity_tertile", "entanglement_tertile",
        "regime_label", "highest_identity", "direction_sha256", "source_direction_sha256",
        "direction_equivalence_status", "checkpoint_sha256", "normalizer_sha256", "normalization_hash",
        "source_cube_hash", "E_task_definition",
    ]
    assignments = reserved[columns].sort_values(["setting_id", "fold", "seed", "direction_rank"]).reset_index(drop=True)
    assignments.to_csv(RESULTS / "P4C_SAFETY_PREOUTCOME_REGIME_ASSIGNMENTS.csv", index=False)
    runs = pd.DataFrame(run_rows)
    runs.to_csv(RESULTS / "p4c_safety_source_preflight_runs.csv", index=False)
    return assignments, runs, thresholds


def coverage(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for setting in RESERVED:
        for regime in ["REGIME_LOW", "REGIME_HIGH"]:
            cell = assignments[(assignments.setting_id == setting) & (assignments.regime_label == regime)]
            rows.append({
                "setting_id": setting,
                "regime_label": regime,
                "run_direction_cells": len(cell),
                "folds": cell.fold.nunique(),
                "seeds": cell.seed.nunique(),
                "coverage_pass": len(cell) >= 6 and cell.fold.nunique() >= 3 and cell.seed.nunique() >= 2,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "regime_coverage.csv", index=False)
    (EXP / "REGIME_COVERAGE_AUDIT.md").write_text(
        "# Regime Coverage Audit\n\nThe gate requires at least six run-direction cells, three folds and two seeds in every reserved-setting regime.\n\n"
        + dataframe_markdown(frame) + "\n",
        encoding="utf-8",
    )
    return frame


def figure_source_map(assignments: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = {"REGIME_LOW": "#4C78A8", "REGIME_HIGH": "#E45756", "UNCLASSIFIED": "#B8B8B8"}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharex=True, sharey=True)
    for ax, setting in zip(axes, RESERVED):
        cell = assignments[assignments.setting_id == setting]
        for regime in ["UNCLASSIFIED", "REGIME_LOW", "REGIME_HIGH"]:
            subset = cell[cell.regime_label == regime]
            ax.scatter(subset.z_I, subset.E_task, s=28, alpha=.75, color=colors[regime], label=regime)
        ax.axvline(cell[cell.identity_tertile == "HIGH"].z_I.min(), color="black", ls="--", lw=.8)
        ax.set(title=f"{setting} source-only regime map", xlabel="z_I", ylabel="E_task")
        ax.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "figure1_preoutcome_regime_map.png", dpi=220)
    plt.close(fig)


def freeze(audit: dict[str, Any], thresholds: dict[str, dict[str, float]], assignments: pd.DataFrame, runs: pd.DataFrame, coverage_frame: pd.DataFrame) -> None:
    if not coverage_frame.coverage_pass.all():
        raise RuntimeError("P4C_SAFETY_INSUFFICIENT_REGIME_COVERAGE")
    protocol = {
        "schema": "P4C_SUPPRESSION_SAFETY_PROTOCOL_V1",
        "timestamp_utc": now_utc(),
        "parent_p4b_tip": P4B_TIP,
        "p4b_terminal_preserved": "P4B_IDENTITY_RELIABILITY_CONDITION_NOT_SUPPORTED",
        "reserved_settings": RESERVED,
        "direction_ranks": RANKS,
        "source_definitions": {"I": "identity_direction_effect", "P": "persistence", "D": "D_finite", "C": "C_src_CE", "O": "O_task"},
        "normalization": "P4B within-setting median/MAD with frozen SD fallback",
        "normalization_sha256": audit["normalization_sha256"],
        "source_cube_sha256": audit["source_cube_sha256"],
        "E_task": "(z_D + z_C + z_O)/3",
        "regime_thresholds": thresholds,
        "regime_rules": {"REGIME_LOW": "z_I >= high_I_lower and E_task <= low_E_upper", "REGIME_HIGH": "z_I >= high_I_lower and E_task >= high_E_lower", "comparison": "inclusive pandas quantiles at 1/3 and 2/3"},
        "intervention": "h_erased = h - ((h-mu)^T v)v; frozen representation, source center, direction and classifier",
        "future_utility": {"primary": "U_BA=BA_erased-BA_intact", "secondary_F1": "U_F1=F1_erased-F1_intact", "secondary_CE": "U_CE=CE_intact-CE_erased", "positive": "beneficial suppression", "subject_first": True},
        "matched_random": {"count": CONTROL_COUNT, "seed": "stable_seed('P4A-control', setting, fold, seed, rank, control_id)", "space": "full representation", "matching": "per-trial displacement norm", "controls_not_independent_N": True},
        "highest_identity": "source-only maximum z_I per run; ties lowest direction rank",
        "noop": "U=0",
        "bootstrap": {"draws": 10000, "seed": 441027, "hierarchy_pooled": ["setting", "fold", "seed/run", "direction", "outcome_subject"], "hierarchy_setting": ["fold", "seed/run", "direction", "outcome_subject"]},
        "coverage_gate": {"run_direction_cells": 6, "folds": 3, "seeds": 2},
        "strong_gates": {"G1": "pooled DeltaRegime>0 and CI lower>0", "G2": "pooled U_high<0 and CI upper<0", "G3": "S4 and S6 DeltaRegime>0", "G4": "S4 and S6 U_high<0", "G5": "purity pass"},
        "partial_rule": "pooled DeltaRegime>0 and pooled U_high<0; at least 3/4 per-setting direction indicators correct; neither setting reverses both",
        "low_E_actionability": {"BENEFICIAL": "pooled U_low>0, CI lower>0, S4/S6 points>=0", "NOT_BENEFICIAL": "pooled U_low<=0 or CI upper<0 or S4/S6 points both<0", "otherwise": "INCONCLUSIVE"},
        "forbidden": ["continuous MINT validation", "MINT refit", "E_task change", "threshold search", "setting rescue", "direction reselection", "retraining", "sealed holdout access", "P4A grid restart"],
        "next_stage": {"strong": "METHOD_LEVEL_BRIDGE_AUTHORIZATION=AUTHORIZED", "partial": "CONDITIONAL", "not_supported": "NOT_AUTHORIZED", "FINAL_NEW_MODEL_AUTHORIZATION": "NOT_AUTHORIZED_AT_THIS_STAGE"},
    }
    write_json(EXP / "P4C_SAFETY_PROTOCOL_FROZEN.json", protocol)
    hashes = {
        "P4C_SAFETY_INPUT_AUDIT.json": sha256(EXP / "P4C_SAFETY_INPUT_AUDIT.json"),
        "P4B_FINAL_VALIDATION.json": sha256(P4B / "results" / "P4B_FINAL_VALIDATION.json"),
        "P4B_PROTOCOL_FROZEN.json": sha256(P4B / "P4B_PROTOCOL_FROZEN.json"),
        "SOURCE_NORMALIZATION_FROZEN.json": sha256(P4B / "SOURCE_NORMALIZATION_FROZEN.json"),
        "source_evidence_normalized.csv": sha256(P4B / "results" / "source_evidence_normalized.csv"),
        "P4C_SAFETY_PROTOCOL_FROZEN.json": sha256(EXP / "P4C_SAFETY_PROTOCOL_FROZEN.json"),
        "P4C_SAFETY_PREOUTCOME_REGIME_ASSIGNMENTS.csv": sha256(RESULTS / "P4C_SAFETY_PREOUTCOME_REGIME_ASSIGNMENTS.csv"),
        "p4c_safety_source_preflight_runs.csv": sha256(RESULTS / "p4c_safety_source_preflight_runs.csv"),
        "discovery_regime_reproduction.csv": sha256(RESULTS / "discovery_regime_reproduction.csv"),
        "regime_coverage.csv": sha256(RESULTS / "regime_coverage.csv"),
    }
    pre = {
        "schema": "P4C_SAFETY_PREOUTCOME_FREEZE_V1",
        "timestamp_utc": now_utc(),
        "pass": True,
        "reserved_settings": RESERVED,
        "future_outcome_access_count_before_freeze": 0,
        "assignments_rows": len(assignments),
        "source_preflight_runs": len(runs),
        "coverage_pass": True,
        "hashes": hashes,
        "OpenBMI_sealed_internal_holdout": "UNTOUCHED",
        "WBCIC_outer_10": "UNTOUCHED_NOT_ENUMERATED",
        "post_outcome_protocol_modification": False,
    }
    write_json(EXP / "P4C_SAFETY_PREOUTCOME_FREEZE.json", pre)
    (EXP / "README.md").write_text(
        "# PERSIST-EEG Phase 4C-Safety\n\nProspective suppression-safety validation on sealed S4/S6. This confirmatory experiment preserves P4B's negative continuous-prediction result and tests only the frozen High-I × task-entanglement safety boundary.\n",
        encoding="utf-8",
    )


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    audit, protocol, source = p4b_audit()
    reproduce_discovery(protocol, source)
    assignments, runs, thresholds = preflight_and_assign(audit, source)
    coverage_frame = coverage(assignments)
    figure_source_map(assignments)
    freeze(audit, thresholds, assignments, runs, coverage_frame)
    print(f"P4C_SAFETY_PREOUTCOME_FROZEN rows={len(assignments)} hash={sha256(RESULTS / 'P4C_SAFETY_PREOUTCOME_REGIME_ASSIGNMENTS.csv')}", flush=True)


if __name__ == "__main__":
    main()
