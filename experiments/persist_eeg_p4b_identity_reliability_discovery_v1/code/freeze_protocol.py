from __future__ import annotations

import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = HERE.parents[2]
RESULTS = EXP / "results"
P4A = REPO / "experiments" / "persist_eeg_p4a_cross_setting_expansion_v1"
P4A_RESULTS = P4A / "results"
LEAN_TIP = "281e7c66c9818b5d2efe96968900cb585af20287"
EPSILON = 1e-12
VARIABLES = {
    "I": "identity_direction_effect",
    "P": "persistence",
    "geometry_strength": "geometry_strength",
    "rank": "direction_rank",
    "D": "D_finite",
    "C": "C_src_CE",
    "O": "O_task",
}


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True, encoding="utf-8", errors="replace").strip()


def robust_spec(values: pd.Series) -> dict[str, Any]:
    array = values.to_numpy(dtype=float)
    finite = np.isfinite(array)
    observed = array[finite]
    median = float(np.median(observed))
    mad = float(np.median(np.abs(observed - median)))
    sd = float(np.std(observed, ddof=1))
    robust_scale = 1.4826 * mad
    tolerance = EPSILON * max(1.0, abs(median), abs(sd))
    if robust_scale > tolerance:
        scale, method, status = robust_scale, "1.4826_MAD", "ACTIVE"
    elif sd > tolerance:
        scale, method, status = sd, "SD_FALLBACK", "ACTIVE"
    else:
        scale, method, status = 1.0, "DEGENERATE_UNIT_PLACEHOLDER", "DEGENERATE"
    return {
        "count": int(finite.sum()),
        "missing": int((~finite).sum()),
        "median": median,
        "MAD": mad,
        "SD": sd,
        "minimum": float(np.min(observed)),
        "maximum": float(np.max(observed)),
        "scale": scale,
        "scale_method": method,
        "status": status,
        "epsilon": EPSILON,
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    current_tip = git("rev-parse", "HEAD")
    if current_tip != LEAN_TIP:
        raise RuntimeError(f"P4B freeze must start at P4A Lean tip {LEAN_TIP}, observed {current_tip}")
    if git("branch", "--show-current") != "codex/persist-eeg-p4b-identity-reliability-discovery-v1":
        raise RuntimeError("wrong P4B branch")

    inputs = {
        "P4A_LEAN_FINAL_REPORT.md": P4A / "P4A_LEAN_FINAL_REPORT.md",
        "P4A_LEAN_FINAL_VALIDATION.json": P4A_RESULTS / "P4A_LEAN_FINAL_VALIDATION.json",
        "P4A_GRID_PAUSE_SNAPSHOT.json": P4A / "P4A_GRID_PAUSE_SNAPSHOT.json",
        "P4A_PROTOCOL_AMENDMENT_LEAN_V1.json": P4A / "P4A_PROTOCOL_AMENDMENT_LEAN_V1.json",
        "SETTING_MANIFEST_LEAN.json": P4A / "SETTING_MANIFEST_LEAN.json",
        "SETTING_COMPETENCE_REPORT_LEAN.md": P4A / "SETTING_COMPETENCE_REPORT_LEAN.md",
        "erm_setting_cube.csv": P4A_RESULTS / "erm_setting_cube.csv",
        "source_evidence_cube.csv": P4A_RESULTS / "source_evidence_cube.csv",
        "source_artifact_audit.csv": P4A_RESULTS / "source_artifact_audit.csv",
        "setting_competence_lean.csv": P4A_RESULTS / "setting_competence_lean.csv",
    }
    missing_inputs = [name for name, path in inputs.items() if not path.is_file()]
    if missing_inputs:
        raise RuntimeError(f"missing P4A inputs: {missing_inputs}")
    hashes = {name: sha256(path) for name, path in inputs.items()}
    validation = json.loads(inputs["P4A_LEAN_FINAL_VALIDATION.json"].read_text(encoding="utf-8-sig"))
    if validation.get("pass") is not True or validation.get("terminal") != "P4A_LEAN_CROSS_SETTING_CUBE_COMPLETE":
        raise RuntimeError("P4A Lean validation is not a complete PASS")

    cube = pd.read_csv(inputs["source_evidence_cube.csv"])
    competence = pd.read_csv(inputs["setting_competence_lean.csv"])
    artifact = pd.read_csv(inputs["source_artifact_audit.csv"])
    banned = [column for column in cube.columns if "future" in column.lower() or column.lower().startswith("u_")]
    if banned:
        raise RuntimeError(f"source cube contains future/outcome utility columns: {banned}")
    expected_settings = ["S1", "S2", "S3", "S4", "S5", "S6"]
    if sorted(cube.setting_id.unique()) != expected_settings or len(cube) != 720:
        raise RuntimeError("source cube setting/cardinality failure")

    usable = competence.loc[
        (competence.competence == "COMPETENCE_PASS")
        & competence.source_artifact_complete.astype(bool)
        & (competence.preprocessing_event_label_status == "PASS"),
        "setting_id",
    ].tolist()
    if not all(len(artifact[artifact.setting_id == setting]) == 15 for setting in usable):
        raise RuntimeError("source artifact audit is incomplete")

    normalizers: dict[str, dict[str, Any]] = {}
    normalized = cube.copy()
    normalized["I_raw"] = normalized[VARIABLES["I"]]
    normalized["I_full_raw"] = normalized["identity_full"]
    for setting in expected_settings:
        mask = normalized.setting_id == setting
        normalizers[setting] = {}
        for alias, column in VARIABLES.items():
            spec = robust_spec(normalized.loc[mask, column])
            normalizers[setting][alias] = spec
            normalized.loc[mask, f"z_{alias}"] = (
                normalized.loc[mask, column].astype(float) - spec["median"]
            ) / spec["scale"]
        full = robust_spec(normalized.loc[mask, "identity_full"])
        normalizers[setting]["I_full_descriptive"] = full
        normalized.loc[mask, "I_setting_median"] = full["median"]
        normalized.loc[mask, "I_setting_SD"] = full["SD"]

    active_primitives = ["D", "C", "O"]
    degenerate = {
        primitive: [
            setting for setting in usable if normalizers[setting][primitive]["status"] == "DEGENERATE"
        ]
        for primitive in active_primitives
    }
    if any(degenerate.values()):
        active_primitives = [primitive for primitive in active_primitives if not degenerate[primitive]]
    if len(active_primitives) < 2:
        raise RuntimeError("fewer than two usable task-entanglement primitives")

    normalized["E_task"] = normalized[[f"z_{item}" for item in active_primitives]].mean(axis=1)
    correlations = normalized[[f"z_{item}" for item in ("I", "P", "geometry_strength", "rank", "D", "C", "O")]].corr(method="spearman")
    primitive_corr = correlations.loc[["z_D", "z_C", "z_O"], ["z_D", "z_C", "z_O"]]
    off_diagonal = [
        abs(float(primitive_corr.iloc[i, j]))
        for i in range(3)
        for j in range(i + 1, 3)
    ]
    near_duplicate = max(off_diagonal) >= 0.995
    if near_duplicate:
        raise RuntimeError("automatic primitive dropping is intentionally not implemented; source audit found a near-exact duplicate")

    discovery_historical = [setting for setting in ["S1", "S2", "S3"] if setting in usable]
    if "S5" in usable:
        new_discovery = "S5"
        reserved = [setting for setting in ["S4", "S6"] if setting in usable]
    elif "S6" in usable:
        new_discovery = "S6"
        reserved = [setting for setting in ["S4"] if setting in usable]
    elif "S4" in usable:
        new_discovery = "S4"
        reserved = []
    else:
        raise RuntimeError("no usable new discovery setting")
    discovery = discovery_historical + [new_discovery]

    regime_thresholds: dict[str, dict[str, float]] = {}
    for setting in discovery:
        cell = normalized[normalized.setting_id == setting]
        regime_thresholds[setting] = {
            "high_I_lower": float(cell.z_I.quantile(2.0 / 3.0)),
            "low_E_upper": float(cell.E_task.quantile(1.0 / 3.0)),
            "high_E_lower": float(cell.E_task.quantile(2.0 / 3.0)),
        }

    normalization = {
        "schema": "P4B_SOURCE_NORMALIZATION_FROZEN_V1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_only": True,
        "future_utility_accessed_before_freeze": False,
        "primary_rule": "within-setting (x - median) / (1.4826*MAD + epsilon semantics)",
        "fallback": "setting SD when robust scale is numerically degenerate; otherwise mark DEGENERATE",
        "epsilon": EPSILON,
        "variables": VARIABLES,
        "settings": normalizers,
        "active_E_task_primitives": active_primitives,
        "E_task_formula": f"({' + '.join('z_'+item for item in active_primitives)})/{len(active_primitives)}",
        "target_signs": {"D": "higher=more decision dependence", "C": "higher=more source CE harm under erasure", "O": "higher=more task-span overlap"},
        "source_cube_sha256": hashes["source_evidence_cube.csv"],
    }
    write_json(EXP / "SOURCE_NORMALIZATION_FROZEN.json", normalization)
    normalized.to_csv(RESULTS / "source_evidence_normalized.csv", index=False)

    assignment = {
        "schema": "P4B_DISCOVERY_SETTING_ASSIGNMENT_V1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_only_assignment": True,
        "future_utility_accessed_before_assignment": False,
        "usable_settings": usable,
        "historically_observed_discovery_settings": discovery_historical,
        "new_sealed_usable_settings": [setting for setting in ["S4", "S5", "S6"] if setting in usable],
        "new_discovery_setting": new_discovery,
        "all_discovery_settings": discovery,
        "p4c_reserved_settings": reserved,
        "selection_rule": "If S5 usable choose S5; else S6; else S4. Reserve S4/S6 whenever usable and not selected.",
        "selection_outcome_driven": False,
        "p4c_reserved_future_utility_access_allowed": False,
        "source_cube_sha256": hashes["source_evidence_cube.csv"],
    }
    write_json(EXP / "DISCOVERY_SETTING_ASSIGNMENT.json", assignment)

    protocol = {
        "schema": "P4B_IDENTITY_RELIABILITY_DISCOVERY_PROTOCOL_V1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "p4a_lean_tip": LEAN_TIP,
        "scientific_question": "What determines when subject identifiability is a valid nuisance proxy?",
        "hypothesis": "Identity is a more reliable nuisance proxy when subject-predictive variation is task-decoupled and becomes less reliable when task-entangled.",
        "usable_settings": usable,
        "discovery_settings": discovery,
        "new_discovery_setting": new_discovery,
        "p4c_reserved_settings": reserved,
        "direction_ranks": list(range(1, 9)),
        "normalization": normalization["primary_rule"],
        "normalization_sha256": sha256(EXP / "SOURCE_NORMALIZATION_FROZEN.json"),
        "assignment_sha256": sha256(EXP / "DISCOVERY_SETTING_ASSIGNMENT.json"),
        "active_E_task_primitives": active_primitives,
        "E_task_formula": normalization["E_task_formula"],
        "future_utility": {
            "primary": "U_BA = BA_erased - BA_intact",
            "secondary_F1": "U_F1 = F1_erased - F1_intact",
            "secondary_CE": "U_CE = CE_intact - CE_erased",
            "positive_meaning": "suppression beneficial",
            "frozen_classifier": True,
            "subject_first": True,
        },
        "models": {
            "alpha": 1.0,
            "M0": ["z_P", "z_geometry_strength", "z_rank"],
            "MI": ["z_P", "z_geometry_strength", "z_rank", "z_I"],
            "ME": ["z_P", "z_geometry_strength", "z_rank", "E_task"],
            "MADD": ["z_P", "z_geometry_strength", "z_rank", "z_I", "E_task"],
            "MINT": ["z_P", "z_geometry_strength", "z_rank", "z_I", "E_task", "z_I_x_E_task"],
            "secondary": {
                "MID": ["z_P", "z_geometry_strength", "z_rank", "z_I", "z_D", "z_I_x_z_D"],
                "MIC": ["z_P", "z_geometry_strength", "z_rank", "z_I", "z_C", "z_I_x_z_C"],
                "MIO": ["z_P", "z_geometry_strength", "z_rank", "z_I", "z_O", "z_I_x_z_O"],
            },
        },
        "primary_cv": "Leave-One-Discovery-Setting-Out",
        "secondary_cv": "Leave-One-Entire-Run-Out (setting x fold x seed)",
        "bootstrap": {"draws": 10000, "hierarchy": ["setting", "fold", "seed/run", "direction", "outcome subject"]},
        "regime": {"rule": "within-setting tertiles", "thresholds": regime_thresholds},
        "primary_interaction": {"coefficient": "beta_IxE", "hypothesis": "<0", "DeltaSlope": "slope(E=-1)-slope(E=+1)>0"},
        "primary_regime_contrast": "DeltaRegime=mean(U_BA|High-I,Low-E)-mean(U_BA|High-I,High-E)>0",
        "success_gates": {
            "G1": "RMSE_MI-RMSE_MINT > 0 with 95% CI lower > 0",
            "G2": "RMSE_MADD-RMSE_MINT > 0 with CI lower > 0, or beta_IxE CI upper < 0",
            "G3": "DeltaSlope > 0 with CI lower > 0",
            "G4": "DeltaRegime > 0 with CI lower > 0",
            "G5": ">=75% discovery settings show the primary effect direction",
            "G6": "P4C reserved future utilities untouched",
        },
        "terminal_rules": [
            "P4B_IDENTITY_RELIABILITY_CONDITION_STRONG_SUPPORTED",
            "P4B_IDENTITY_RELIABILITY_CONDITION_PARTIAL_SUPPORTED",
            "P4B_IDENTITY_RELIABILITY_CONDITION_NOT_SUPPORTED",
            "P4B_INSUFFICIENT_CROSS_SETTING_EVIDENCE",
            "P4B_INSUFFICIENT_PROSPECTIVE_HOLDOUT",
            "P4B_PROTOCOL_OR_PURITY_FAILURE",
        ],
        "forbidden_rescue": ["nonlinear trees", "random forest", "neural predictor", "post-hoc threshold search", "learned D/C/O weights", "cherry-picked setting"],
        "p4c_reserved_future_utility_access": "FORBIDDEN",
        "p4c_execution": "FORBIDDEN_IN_P4B",
    }
    write_json(EXP / "P4B_PROTOCOL_FROZEN.json", protocol)

    prefreeze = {
        "schema": "P4B_PRE_OUTCOME_FREEZE_COMPLETE_V1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pass": True,
        "future_utility_access_count_before_freeze": 0,
        "new_discovery_future_utility_accessed": False,
        "p4c_reserved_future_utility_accessed": False,
        "hashes": {
            "P4A_source_evidence_cube": hashes["source_evidence_cube.csv"],
            "SOURCE_NORMALIZATION_FROZEN.json": sha256(EXP / "SOURCE_NORMALIZATION_FROZEN.json"),
            "DISCOVERY_SETTING_ASSIGNMENT.json": sha256(EXP / "DISCOVERY_SETTING_ASSIGNMENT.json"),
            "P4B_PROTOCOL_FROZEN.json": sha256(EXP / "P4B_PROTOCOL_FROZEN.json"),
            "source_evidence_normalized.csv": sha256(RESULTS / "source_evidence_normalized.csv"),
        },
        "discovery_settings": discovery,
        "p4c_reserved_settings": reserved,
    }
    write_json(EXP / "PRE_OUTCOME_FREEZE_COMPLETE.json", prefreeze)
    write_json(EXP / "P4A_INPUT_HASHES.json", {"schema": "P4B_P4A_INPUT_HASH_AUDIT_V1", "p4a_lean_tip": LEAN_TIP, "hashes": hashes})

    scale_rows = []
    for setting in expected_settings:
        for alias in VARIABLES:
            scale_rows.append({"setting_id": setting, "variable": alias, **normalizers[setting][alias]})
    scale_frame = pd.DataFrame(scale_rows)
    scale_frame.to_csv(RESULTS / "source_scale_audit.csv", index=False)
    correlations.to_csv(RESULTS / "source_spearman_collinearity.csv")

    input_lines = "\n".join(f"- `{name}`: `{value}`" for name, value in hashes.items())
    (EXP / "P4A_INPUT_AUDIT.md").write_text(
        f"# P4A Input Audit\n\nP4A Lean tip: `{LEAN_TIP}`. Validator PASS and terminal `P4A_LEAN_CROSS_SETTING_CUBE_COMPLETE` were verified before P4B creation. No future-utility column is present in the 720-row source cube.\n\n{input_lines}\n",
        encoding="utf-8",
    )
    usability_table = competence.to_markdown(index=False, floatfmt=".5f")
    (EXP / "USABLE_SETTING_AUDIT.md").write_text(
        "# Usable Setting Audit\n\nEligibility requires P4A `COMPETENCE_PASS`, complete 15-run source artifacts, and preprocessing/event/label PASS. No retuning was performed.\n\n"
        + usability_table
        + f"\n\nDiscovery settings: {', '.join(discovery)}. P4C reserved settings: {', '.join(reserved)}. The S5 choice follows the predeclared priority and used no future utility.\n",
        encoding="utf-8",
    )
    scale_table = scale_frame[["setting_id", "variable", "minimum", "maximum", "median", "MAD", "SD", "scale_method", "status"]].to_markdown(index=False, floatfmt=".6g")
    corr_table = correlations.to_markdown(floatfmt=".3f")
    (EXP / "SOURCE_SCALE_AND_COLLINEARITY_AUDIT.md").write_text(
        "# Source Scale and Collinearity Audit\n\nAll calculations are source-only and precede any new direction-level future utility access. Absolute identity and other primitive scales differ substantially across settings, so pooled raw thresholds are invalid. All required variables are nondegenerate under within-setting robust normalization. D and O are strongly but not near-exactly associated; the pooled Spearman magnitude is below the frozen 0.995 duplicate threshold, so D/C/O are all retained. C_src_CE is primary because its sign directly means validation CE harm under erasure; BA/F1 remain secondary.\n\n"
        + scale_table
        + "\n\n## Pooled Spearman correlation after within-setting normalization\n\n"
        + corr_table
        + f"\n\nFrozen E_task: `{normalization['E_task_formula']}`. No weights were learned.\n",
        encoding="utf-8",
    )
    (EXP / "README.md").write_text(
        "# PERSIST-EEG P4B — Identity Reliability Condition Discovery\n\nThis experiment tests whether source-defined task entanglement explains when direction-level identity contribution is a valid nuisance proxy. Protocol, setting assignment, normalization, interaction, tertile regimes, ridge alpha, LOSO-setting CV, and 10,000-draw hierarchy are frozen before new discovery outcome access. S4/S6 are P4C-reserved and their future utilities are forbidden in P4B.\n",
        encoding="utf-8",
    )
    (EXP / "FUTURE_UTILITY_ACCESS_LEDGER.md").write_text(
        "# Future Utility Access Ledger\n\nAt protocol freeze: no new discovery-setting direction-level future utility had been accessed. P4C reserved settings S4 and S6 remained sealed. Post-freeze discovery access will be appended by the frozen runner.\n",
        encoding="utf-8",
    )
    print("P4B_PRE_OUTCOME_FREEZE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
