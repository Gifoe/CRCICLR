from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import p4d_common as c


ASSIGNMENTS = c.P4C / "results" / "P4C_SAFETY_PREOUTCOME_REGIME_ASSIGNMENTS.csv"
P4C_VALIDATION = c.P4C / "results" / "P4C_SAFETY_FINAL_VALIDATION.json"
P4C_PROTOCOL = c.P4C / "P4C_SAFETY_PROTOCOL_FROZEN.json"
P4C_SUMMARY = c.P4C / "results" / "p4c_safety_regime_summary.csv"


def audit_p4c() -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    required = [
        c.P4C / "P4B_NEGATIVE_RESULT_PRESERVATION.md",
        c.P4C / "P4C_SAFETY_INPUT_AUDIT.md",
        P4C_PROTOCOL,
        c.P4C / "P4C_SAFETY_PREOUTCOME_FREEZE.json",
        ASSIGNMENTS,
        c.P4C / "P4C_SAFETY_OUTCOME_ACCESS_LEDGER.md",
        c.P4C / "REGIME_REPRODUCTION_AUDIT.md",
        c.P4C / "REGIME_COVERAGE_AUDIT.md",
        c.P4C / "REGIME_SEPARATION_PROSPECTIVE_AUDIT.md",
        c.P4C / "SUPPRESSION_VETO_AUDIT.md",
        c.P4C / "MATCHED_RANDOM_SPECIFICITY_AUDIT.md",
        c.P4C / "LOW_ENTANGLEMENT_ACTIONABILITY_AUDIT.md",
        c.P4C / "CROSS_SETTING_SAFETY_STABILITY.md",
        c.P4C / "THEORY_ADMISSIBILITY_NOTE.md",
        c.P4C / "HOLDOUT_PURITY_AUDIT.md",
        c.P4C / "P4C_SAFETY_FINAL_REPORT.md",
        P4C_VALIDATION,
        P4C_SUMMARY,
    ]
    for path in required:
        c.require_file(path)
    validation = c.read_json(P4C_VALIDATION)
    protocol = c.read_json(P4C_PROTOCOL)
    summary = pd.read_csv(P4C_SUMMARY)
    if validation.get("pass") is not True:
        raise RuntimeError("P4C validator is not PASS")
    if validation.get("preoutcome_assignment_sha256") != c.sha256(ASSIGNMENTS):
        raise RuntimeError("P4C assignment hash mismatch")
    if protocol.get("source_cube_sha256") != "41c5373bd73f327a652c3d155ffcf90642589f35e48ce0b2a47ee30307443ec0":
        raise RuntimeError("P4C source cube hash changed")
    if protocol.get("normalization_sha256") != "dfcbcfcde0536e5c673637ab6b300377b4162e5205ba555c90f73274b1c6720f":
        raise RuntimeError("P4C normalization hash changed")
    return validation, protocol, summary


def authorization(validation: dict[str, Any], summary: pd.DataFrame) -> dict[str, Any]:
    status = str(validation["SAFETY_STATUS"])
    pooled = summary.loc[summary["setting_id"].astype(str) == "POOLED"].iloc[0]
    setting_rows = summary.loc[summary["setting_id"].isin(["S4", "S6"])].set_index("setting_id")
    directional = {
        setting: {
            "DeltaRegime_positive": bool(setting_rows.loc[setting, "DeltaRegime"] > 0),
            "U_high_negative": bool(setting_rows.loc[setting, "U_high"] < 0),
        }
        for setting in ("S4", "S6")
    }
    purity = validation.get("OpenBMI_sealed_internal_holdout") == "UNTOUCHED" and validation.get("WBCIC_outer_10") == "UNTOUCHED_NOT_ENUMERATED"
    partial_conditions = (
        float(pooled["DeltaRegime"]) > 0
        and float(pooled["U_high"]) < 0
        and all(value["DeltaRegime_positive"] and value["U_high_negative"] for value in directional.values())
        and purity
    )
    if status == "P4C_SAFETY_BOUNDARY_STRONG_SUPPORTED" and purity:
        decision = "AUTHORIZED"
    elif status == "P4C_SAFETY_BOUNDARY_PARTIAL_SUPPORTED" and partial_conditions:
        decision = "CONDITIONAL"
    else:
        decision = "NOT_AUTHORIZED"
    payload: dict[str, Any] = {
        "schema": "PERSIST_EEG_P4D_AUTHORIZATION_V1",
        "timestamp_utc": c.now_utc(),
        "P4C_SAFETY_STATUS": status,
        "P4C_validator_pass": True,
        "P4D_AUTHORIZATION": decision,
        "pooled_DeltaRegime_BA": float(pooled["DeltaRegime"]),
        "pooled_U_high_BA": float(pooled["U_high"]),
        "setting_direction_checks": directional,
        "purity_pass": purity,
        "P4E_constraint": "P4C is not strong; P4E remains NOT_AUTHORIZED regardless of P4D outcome" if status != "P4C_SAFETY_BOUNDARY_STRONG_SUPPORTED" else "subject to P4D and headroom gates",
        "p4c_validation_sha256": c.sha256(P4C_VALIDATION),
        "p4c_assignment_sha256": c.sha256(ASSIGNMENTS),
        "p4c_protocol_sha256": c.sha256(P4C_PROTOCOL),
    }
    payload["content_sha256"] = c.canonical_sha256(payload)
    c.write_json(c.EXP / "P4D_AUTHORIZATION.json", payload)
    return payload


def source_burden(protocol: dict[str, Any]) -> pd.DataFrame:
    frame = pd.read_csv(ASSIGNMENTS)
    if len(frame) != 240 or set(frame.setting_id) != {"S4", "S6"}:
        raise RuntimeError("P4C assignment cardinality changed")
    rows: list[dict[str, Any]] = []
    assignment_hash = c.sha256(ASSIGNMENTS)
    for keys, group in frame.groupby(["setting_id", "fold", "seed"], sort=True):
        setting, fold, seed = keys
        high_i = group.identity_tertile.astype(str).eq("HIGH")
        high_e = group.entanglement_tertile.astype(str).eq("HIGH")
        low_e = group.entanglement_tertile.astype(str).eq("LOW")
        n_identity_high = int(high_i.sum())
        n_high_high = int((high_i & high_e).sum())
        n_high_low = int((high_i & low_e).sum())
        if n_identity_high <= 0:
            raise RuntimeError(f"no High-I directions for {keys}")
        identity_mass = np.maximum(group.z_I.to_numpy(float), 0.0)
        denominator_mass = float(identity_mass[high_i.to_numpy()].sum())
        unsafe_mass = float(identity_mass[(high_i & high_e).to_numpy()].sum()) / max(denominator_mass, c.EPS)
        admissible_mass = float(identity_mass[(high_i & low_e).to_numpy()].sum()) / max(denominator_mass, c.EPS)
        checkpoint_hashes = group.checkpoint_sha256.astype(str).unique().tolist()
        if len(checkpoint_hashes) != 1:
            raise RuntimeError(f"multiple ERM checkpoints in burden unit {keys}")
        spec = c.SETTINGS[str(setting)]
        rows.append(
            {
                "setting_id": setting,
                "dataset": spec["dataset"],
                "task": spec["task"],
                "backbone": spec["backbone"],
                "fold": int(fold),
                "seed": int(seed),
                "n_identity_high": n_identity_high,
                "n_highI_highE": n_high_high,
                "n_highI_lowE": n_high_low,
                "R_unsafe": n_high_high / n_identity_high,
                "R_admissible": n_high_low / n_identity_high,
                "R_unsafe_mass": unsafe_mass,
                "R_admissible_mass": admissible_mass,
                "positive_highI_identity_mass": denominator_mass,
                "source_cube_hash": protocol["source_cube_sha256"],
                "p4c_regime_hash": assignment_hash,
                "checkpoint_hash": checkpoint_hashes[0],
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 30 or result[["setting_id", "fold", "seed"]].duplicated().any():
        raise RuntimeError("burden run matrix is not 2 settings x 5 folds x 3 seeds")
    c.write_csv(c.RESULTS / "P4D_SOURCE_UNSAFE_BURDEN.csv", result)
    freeze: dict[str, Any] = {
        "schema": "PERSIST_EEG_P4D_SOURCE_BURDEN_FREEZE_V1",
        "timestamp_utc": c.now_utc(),
        "primary": "R_unsafe=count(High-I AND High-E)/count(High-I)",
        "secondary_admissible": "R_admissible=count(High-I AND Low-E)/count(High-I)",
        "secondary_mass": "positive z_I mass in target High-I regime / positive z_I mass in all High-I directions",
        "source_only": True,
        "method_future_task_outcomes_accessed": False,
        "rows": len(result),
        "settings": ["S4", "S6"],
        "p4c_assignment_sha256": assignment_hash,
        "burden_csv_sha256": c.sha256(c.RESULTS / "P4D_SOURCE_UNSAFE_BURDEN.csv"),
    }
    freeze["content_sha256"] = c.canonical_sha256(freeze)
    c.write_json(c.EXP / "P4D_SOURCE_BURDEN_FREEZE.json", freeze)
    return result


def grid_inventory() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    configurations = (("ERM", 0.0),) + tuple((method, lam) for method in c.METHODS for lam in c.LAMBDAS)
    for setting, spec in c.SETTINGS.items():
        for fold in c.FOLDS:
            for seed in c.SEEDS:
                for method, lam in configurations:
                    complete = c.source_complete(setting, fold, seed, method, lam)
                    checkpoint = c.checkpoint_path(setting, fold, seed, method, lam)
                    payload = c.read_json(complete) if complete.is_file() else {}
                    trained = bool(payload.get("pass") is True and checkpoint.is_file())
                    if setting in {"S1", "S2", "S3"}:
                        status = "HISTORICALLY_OBSERVED_TASK_OUTCOME"
                        evidence = "P4A historical reconstruction backed by Phase2/Phase3 compact results"
                    elif trained and method == "ERM":
                        status = "HISTORICALLY_OBSERVED_TASK_OUTCOME"
                        evidence = "P4A ERM outcome competence was already accessed under frozen protocol"
                    elif trained and payload.get("invariance_outcome_accessed") is False and payload.get("direction_future_utility_accessed") is False:
                        status = "TRAINED_BUT_TASK_OUTCOME_SEALED"
                        evidence = "P4A SOURCE_COMPLETE explicit sealed flags"
                    elif trained:
                        status = "HISTORICALLY_OBSERVED_TASK_OUTCOME"
                        evidence = "trained artifact without prospective sealed flags"
                    else:
                        status = "UNTRAINED"
                        evidence = "no complete P4A source artifact/checkpoint"
                    rows.append(
                        {
                            "setting_id": setting,
                            "dataset": spec["dataset"],
                            "task": spec["task"],
                            "backbone": spec["backbone"],
                            "method": method,
                            "lambda": lam,
                            "fold": fold,
                            "seed": seed,
                            "status": status,
                            "trained_artifact_present": trained,
                            "source_complete_path": str(complete) if complete.is_file() else "",
                            "checkpoint_sha256": payload.get("checkpoint_sha256", ""),
                            "outcome_access_evidence": evidence,
                        }
                    )
    result = pd.DataFrame(rows)
    if len(result) != 900:
        raise RuntimeError("grid inventory must contain 6x5x3x10 rows")
    c.write_csv(c.RESULTS / "invariance_grid_inventory.csv", result)
    return result


def identity_selection() -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in c.FOLDS:
        for seed in c.SEEDS:
            erm_path = c.source_complete("S4", fold, seed, "ERM", 0.0)
            erm = c.read_json(c.require_file(erm_path))
            i_erm = float(erm["source_identity"]["identity_symmetric"])
            for method in c.METHODS:
                for lam in c.LAMBDAS:
                    path = c.require_file(c.source_complete("S4", fold, seed, method, lam))
                    payload = c.read_json(path)
                    if payload.get("pass") is not True or payload.get("invariance_outcome_accessed") is not False:
                        raise RuntimeError(f"S4 candidate is not sealed: {path}")
                    i_method = float(payload["source_identity"]["identity_symmetric"])
                    suppression = i_erm - i_method
                    rows.append(
                        {
                            "setting_id": "S4",
                            "fold": fold,
                            "seed": seed,
                            "method": method,
                            "lambda": lam,
                            "I_ERM": i_erm,
                            "I_method": i_method,
                            "S_I_abs": suppression,
                            "S_I_rel": suppression / (abs(i_erm) + c.EPS),
                            "checkpoint_sha256": payload["checkpoint_sha256"],
                            "task_outcome_accessed_for_selection": False,
                        }
                    )
    frame = pd.DataFrame(rows)
    if len(frame) != 135:
        raise RuntimeError("canonical selection requires the complete 135-cell S4 grid")
    selections: list[dict[str, Any]] = []
    for method in c.METHODS:
        candidates = []
        for lam in c.LAMBDAS:
            part = frame[(frame.method == method) & np.isclose(frame["lambda"], lam)]
            median = float(part.S_I_abs.median())
            fraction = float((part.S_I_abs > 0).mean())
            competent = median > 0 and fraction >= 0.60 and len(part) == 15
            candidates.append(
                {
                    "lambda": lam,
                    "completed_runs": len(part),
                    "median_S_I_abs": median,
                    "fraction_S_I_abs_positive": fraction,
                    "competent": competent,
                }
            )
        eligible = [item for item in candidates if item["competent"]]
        chosen = sorted(eligible, key=lambda item: (-item["median_S_I_abs"], item["lambda"]))[0] if eligible else None
        selections.append(
            {
                "method": method,
                "status": "IDENTITY_MANIPULATION_COMPETENT" if chosen else "IDENTITY_MANIPULATION_INCOMPETENT",
                "lambda_star": None if chosen is None else chosen["lambda"],
                "median_S_I_abs": None if chosen is None else chosen["median_S_I_abs"],
                "fraction_S_I_abs_positive": None if chosen is None else chosen["fraction_S_I_abs_positive"],
                "lambda_candidates": candidates,
            }
        )
    competent = [row for row in selections if row["status"] == "IDENTITY_MANIPULATION_COMPETENT"]
    tie_order = {"DANN": 0, "MMD": 1, "CORAL": 2}
    headroom = sorted(competent, key=lambda row: (-float(row["median_S_I_abs"]), tie_order[row["method"]]))[0]["method"] if competent else None
    payload: dict[str, Any] = {
        "schema": "PERSIST_EEG_P4D_CANONICAL_INVARIANCE_CONFIGS_V1",
        "timestamp_utc": c.now_utc(),
        "selection_setting": "S4",
        "selection_scope": "15 complete prospective source-only runs per lambda",
        "selection_rule": "largest median S_I_abs among lambdas with median>0 and >=60% positive; ties smaller lambda",
        "S_I_abs": "identity_symmetric_ERM - identity_symmetric_method",
        "methods": selections,
        "competent_method_count": len(competent),
        "CANONICAL_HEADROOM_METHOD": headroom,
        "headroom_tie_order": ["DANN", "MMD", "CORAL"],
        "future_BA_F1_CE_accessed_for_selection": False,
        "identity_manipulation_csv_sha256": "PENDING",
    }
    c.write_csv(c.RESULTS / "identity_manipulation.csv", frame)
    payload["identity_manipulation_csv_sha256"] = c.sha256(c.RESULTS / "identity_manipulation.csv")
    payload["content_sha256"] = c.canonical_sha256(payload)
    c.write_json(c.EXP / "CANONICAL_INVARIANCE_CONFIGS.json", payload)
    return frame, payload


def freeze_protocol(
    authorization_payload: dict[str, Any],
    burden: pd.DataFrame,
    canonical: dict[str, Any],
    inventory: pd.DataFrame,
) -> dict[str, Any]:
    r_values = burden.R_unsafe.to_numpy(float)
    payload: dict[str, Any] = {
        "schema": "PERSIST_EEG_P4D_PROTOCOL_FROZEN_V1",
        "timestamp_utc": c.now_utc(),
        "P4D_AUTHORIZATION": authorization_payload["P4D_AUTHORIZATION"],
        "parent_p4c_tip": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=c.REPO, text=True).strip(),
        "prospective_settings": ["S4", "S6"],
        "supplementary_partial_setting": "S5",
        "primary_burden": "R_unsafe=count(High-I AND High-E)/count(High-I)",
        "frozen_R_quantiles": {
            "R_low_q25": float(np.quantile(r_values, 0.25)),
            "R_high_q75": float(np.quantile(r_values, 0.75)),
            "R_split_median": float(np.quantile(r_values, 0.50)),
            "quantile_method": "numpy linear",
        },
        "canonical_configs_sha256": c.sha256(c.EXP / "CANONICAL_INVARIANCE_CONFIGS.json"),
        "canonical_headroom_method": canonical["CANONICAL_HEADROOM_METHOD"],
        "identity_normalization": "within-setting (S_I_abs-median)/(1.4826*MAD+1e-12); fallback setting SD when MAD degenerate",
        "primary_model": "DeltaG_BA ~ z_SI + R_unsafe + z_SI*R_unsafe + method fixed effects",
        "primary_interaction_hypothesis": "beta_zSI_x_Runsafe < 0",
        "simple_slope_contrast": "DeltaSlope_bridge=slope(R_low)-slope(R_high)>0",
        "risk_split": "LOW when R_unsafe <= frozen pooled median; HIGH otherwise",
        "bootstrap": {
            "draws": c.BOOTSTRAP_DRAWS,
            "seed": c.BOOTSTRAP_SEED,
            "hierarchy": ["setting", "fold", "seed/run"],
            "method_configs_nested_within_run": True,
            "interval": "percentile 95%",
        },
        "terminal_gates": {
            "strong": ["G1>=2 competent methods", "G2 interaction<0 CI upper<0", "G3 DeltaSlope>0 CI lower>0", "G4 S4/S6 direction correct", "G5 >=2 methods direction correct", "G6 purity PASS"],
            "partial": "interaction and DeltaSlope directions correct with broad S4/S6 consistency but incomplete CI/method replication gates",
            "otherwise": "P4D_METHOD_LEVEL_BRIDGE_NOT_SUPPORTED",
        },
        "P4E_gate": "requires P4C strong, P4D strong, canonical low-risk mean>=0.005 with CI lower>0, headroom contrast CI lower>0, purity",
        "P4E_pre_authorization": "NOT_AUTHORIZED",
        "forbidden": ["outcome-driven lambda selection", "outcome-driven method selection", "405-grid restart", "S5 completion", "threshold tuning", "nonlinear rescue", "P4C rule changes", "P4E training", "sealed outer holdout access"],
        "method_future_task_outcomes_accessed_before_freeze": False,
        "hashes": {
            "P4D_AUTHORIZATION.json": c.sha256(c.EXP / "P4D_AUTHORIZATION.json"),
            "P4D_SOURCE_BURDEN_FREEZE.json": c.sha256(c.EXP / "P4D_SOURCE_BURDEN_FREEZE.json"),
            "P4D_SOURCE_UNSAFE_BURDEN.csv": c.sha256(c.RESULTS / "P4D_SOURCE_UNSAFE_BURDEN.csv"),
            "invariance_grid_inventory.csv": c.sha256(c.RESULTS / "invariance_grid_inventory.csv"),
            "identity_manipulation.csv": c.sha256(c.RESULTS / "identity_manipulation.csv"),
        },
        "inventory_status_counts": inventory.status.value_counts().to_dict(),
    }
    payload["content_sha256"] = c.canonical_sha256(payload)
    c.write_json(c.EXP / "P4D_PROTOCOL_FROZEN.json", payload)
    return payload


def write_reports(
    validation: dict[str, Any],
    authorization_payload: dict[str, Any],
    burden: pd.DataFrame,
    inventory: pd.DataFrame,
    canonical: dict[str, Any],
) -> None:
    burden_summary = burden.groupby("setting_id").R_unsafe.agg(["count", "min", "median", "mean", "max"]).reset_index()
    inventory_summary = inventory.groupby(["setting_id", "status"]).size().rename("cells").reset_index()
    method_rows = pd.DataFrame(
        [
            {
                "method": row["method"],
                "status": row["status"],
                "lambda_star": row["lambda_star"],
                "median_S_I_abs": row["median_S_I_abs"],
                "fraction_positive": row["fraction_S_I_abs_positive"],
            }
            for row in canonical["methods"]
        ]
    )
    c.write_text(
        c.EXP / "P4C_INPUT_AUDIT.md",
        f"""# P4C Input Audit

- Exact terminal: `{validation['SAFETY_STATUS']}`.
- Validator: `PASS`.
- Low-E actionability: `{validation['LOW_E_ACTIONABILITY_STATUS']}`.
- P4D authorization: `{authorization_payload['P4D_AUTHORIZATION']}`.
- P4C assignment SHA-256: `{authorization_payload['p4c_assignment_sha256']}`.
- P4C source cube SHA-256: `41c5373bd73f327a652c3d155ffcf90642589f35e48ce0b2a47ee30307443ec0`.
- P4B normalization SHA-256: `dfcbcfcde0536e5c673637ab6b300377b4162e5205ba555c90f73274b1c6720f`.
- OpenBMI sealed internal holdout: untouched.
- WBCIC outer 10: untouched and not enumerated.

The partial P4C result is not upgraded. Conditional P4D authorization follows only because pooled DeltaRegime is positive, pooled U_high is negative, both S4 and S6 share those directions, and purity passes. P4E remains unauthorized because P4C is not strong.
""",
    )
    c.write_text(
        c.EXP / "P4D_SOURCE_BURDEN_DEFINITION.md",
        """# P4D Source Burden Definition

Primary `R_unsafe` is the count of frozen P4C `High-I AND High-E` directions divided by all frozen `High-I` directions within each ERM setting/fold/seed run. `R_admissible` replaces High-E with Low-E. Secondary mass ratios use positive `z_I` mass and were fixed before method outcomes. No task outcome, learned weight, or alternative threshold enters this definition.
""" + "\n" + c.markdown_table(burden_summary),
    )
    c.write_text(
        c.EXP / "INVARIANCE_GRID_INVENTORY.md",
        """# Invariance Grid Inventory

The inventory distinguishes historical/observed outcomes, trained but task-outcome-sealed P4A artifacts, and untrained cells. S4 is complete and sealed for non-ERM methods. S5 is partial and excluded from the primary matrix. S6 is ERM-only at this freeze and will receive only frozen canonical competent configurations. The paused 405-grid is not resumed.

""" + c.markdown_table(inventory_summary),
    )
    c.write_text(
        c.EXP / "IDENTITY_MANIPULATION_AUDIT.md",
        """# Identity Manipulation Audit

Canonical lambdas were selected from the 15 complete S4 source-only runs per lambda using `S_I_abs = identity_symmetric_ERM - identity_symmetric_method`. Competence requires median suppression above zero and at least 60% positive runs. Ties select the smaller lambda. Future BA/F1/CE was not accessed.

""" + c.markdown_table(method_rows),
    )
    c.write_text(
        c.EXP / "README.md",
        """# PERSIST-EEG Phase 4D — Mechanism-to-Method Bridge

This experiment tests whether frozen source-side unsafe identity burden moderates the effect of canonical global subject-invariance methods on future task generalization. It uses P4C assignments unchanged, identity-only lambda selection, prospective S4/S6 evaluation, and cluster-aware 10,000-draw bootstrap inference. It does not restart the 405-grid and cannot authorize P4E because its P4C parent is only partial.
""",
    )


def main() -> None:
    c.RESULTS.mkdir(parents=True, exist_ok=True)
    c.FIGURES.mkdir(parents=True, exist_ok=True)
    validation, p4c_protocol, p4c_summary = audit_p4c()
    auth = authorization(validation, p4c_summary)
    if auth["P4D_AUTHORIZATION"] == "NOT_AUTHORIZED":
        c.write_text(c.EXP / "P4D_FINAL_REPORT.md", "# P4D Final Report\n\n`P4D_NOT_AUTHORIZED_BY_P4C`. No method outcome was accessed.")
        raise SystemExit("P4D_NOT_AUTHORIZED_BY_P4C")
    burden = source_burden(p4c_protocol)
    inventory = grid_inventory()
    _identity, canonical = identity_selection()
    freeze_protocol(auth, burden, canonical, inventory)
    write_reports(validation, auth, burden, inventory, canonical)
    print(json.dumps({"P4D_AUTHORIZATION": auth["P4D_AUTHORIZATION"], "canonical": canonical["methods"], "headroom": canonical["CANONICAL_HEADROOM_METHOD"]}, indent=2))
    print("P4D_PREPARATION_COMPLETE_NO_METHOD_FUTURE_OUTCOME_ACCESSED")


if __name__ == "__main__":
    main()
