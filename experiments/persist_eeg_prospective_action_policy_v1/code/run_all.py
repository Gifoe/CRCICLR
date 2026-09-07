from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_data import build_all
from common import (
    DATA,
    DIAGNOSTICS,
    EXP_ROOT,
    FIGURES,
    NEXT_STAGE,
    OUT,
    PROTOCOL,
    REPO_ROOT,
    RESULTS,
    git_sha,
    package_versions,
    router_pilot_root,
    sha256_file,
    write_json,
)
from modeling import run_modeling
from oracle_analysis import run_oracle_analysis


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _top_associations(frame: pd.DataFrame, family: str, limit: int = 5) -> list[dict[str, Any]]:
    block = frame[frame.family_id == family].copy()
    if block.empty:
        return []
    block["abs_spearman"] = block.spearman_r.abs()
    return block.sort_values(["grouped_permutation_p", "abs_spearman"], ascending=[True, False]).head(limit).to_dict(
        orient="records"
    )


def _feature_increment(learn: pd.DataFrame, family: str, scheme: str, left: str, right: str) -> dict[str, Any]:
    subset = learn[
        (learn.family_id == family)
        & (learn.validation_scheme == scheme)
        & (learn.feature_family.isin([left, right]))
        & (learn.status == "EVALUATED")
    ].set_index("feature_family")
    if left not in subset.index or right not in subset.index:
        return {"status": "NOT_ESTIMABLE_WITH_LEGAL_FEATURES"}
    return {
        "status": "ESTIMATED",
        "baseline": left,
        "augmented": right,
        "R2_baseline": float(subset.loc[left, "R2"]),
        "R2_augmented": float(subset.loc[right, "R2"]),
        "delta_R2": float(subset.loc[right, "R2"] - subset.loc[left, "R2"]),
        "RMSE_baseline": float(subset.loc[left, "RMSE"]),
        "RMSE_augmented": float(subset.loc[right, "RMSE"]),
    }


def _policy_comparison(summary: pd.DataFrame, family: str, scheme: str) -> dict[str, Any]:
    rows = summary[(summary.family_id == family) & (summary.validation_scheme == scheme)]
    return {
        str(row.method): {
            "mean_delta_BA": float(row.mean_delta_BA),
            "LCB95": float(row.delta_BA_group_bootstrap_CI_L),
            "positive_group_fraction": float(row.positive_group_fraction),
            "unsafe_intervention_rate": float(row.unsafe_intervention_rate),
            "recovered_headroom": float(row.mean_recovered_headroom) if pd.notna(row.mean_recovered_headroom) else None,
        }
        for row in rows.itertuples(index=False)
    }


def decide(
    oracle: dict[str, Any],
    model: dict[str, Any],
    learnability: pd.DataFrame,
    associations: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, Any]:
    candidate = model["candidate"]
    router_concentration = oracle["concentration"]["openbmi_sample_router"]
    router_summary = oracle["summaries"]["openbmi_sample_router"]
    structural = bool(
        router_summary["selection_value"] >= 0.005
        and router_concentration["largest_subject_share"] < 0.50
        and router_concentration["largest_run_share"] < 0.50
        and router_concentration["subjects_with_rescue_fraction"] >= 0.60
    )
    if candidate["status"] == "READY_FOR_PROSPECTIVE_POLICY_FREEZE":
        terminal = "READY_FOR_PROSPECTIVE_POLICY_FREEZE"
    elif candidate["status"] == "PROMISING_PROSPECTIVE_POLICY":
        terminal = "PROMISING_PROSPECTIVE_POLICY"
    elif structural:
        terminal = "STOP_ACTIONABILITY_NOT_PREDICTABLE"
    else:
        terminal = "STOP_NO_STRUCTURAL_HEADROOM"
    decision_increment = {
        "dda_leave_run": _feature_increment(
            learnability, "openbmi_dda_block", "leave_one_run_out", "P+U", "P+U+D"
        ),
        "wbcic_leave_fold": _feature_increment(
            learnability, "wbcic_development_block", "leave_one_fold_out", "P+U", "P+U+D"
        ),
    }
    geometry_increment = {
        "dda_leave_run": _feature_increment(
            learnability, "openbmi_dda_block", "leave_one_run_out", "P+U+D", "P+U+D+geometry"
        ),
        "wbcic_leave_fold": _feature_increment(
            learnability,
            "wbcic_development_block",
            "leave_one_fold_out",
            "P+U+D",
            "P+U+D+geometry",
        ),
    }
    payload = {
        "terminal_state": terminal,
        "experiment_type": "exploratory development-only policy study",
        "structural_oracle_headroom": structural,
        "oracle": oracle["summaries"],
        "concentration": router_concentration,
        "policy_candidate": candidate,
        "decision_dependence_increment": decision_increment,
        "geometry_increment": geometry_increment,
        "top_legal_feature_associations": {
            family: _top_associations(associations, family)
            for family in ("openbmi_sample_router", "openbmi_dda_block", "wbcic_development_block")
        },
        "primary_policy_comparisons": {
            "openbmi_sample_router": _policy_comparison(
                summary, "openbmi_sample_router", "leave_one_subject_group_out"
            ),
            "openbmi_dda_block": _policy_comparison(summary, "openbmi_dda_block", "leave_one_run_out"),
            "wbcic_development_block": _policy_comparison(
                summary, "wbcic_development_block", "leave_one_fold_out"
            ),
        },
        "leave_one_dataset_out": {
            "status": "NOT_ESTIMABLE",
            "reason": "The compatible OpenBMI sample, OpenBMI block, and WBCIC block policies have different decision units and action semantics; pooling them would be invalid.",
        },
        "outer_test_authorized": False,
        "OUTER_TEST_USED": False,
    }
    write_json(OUT / "FINAL_DECISION.json", payload)
    return payload


def next_stage(decision: dict[str, Any]) -> None:
    NEXT_STAGE.mkdir(parents=True, exist_ok=True)
    authorized = decision["terminal_state"] == "READY_FOR_PROSPECTIVE_POLICY_FREEZE"
    if authorized:
        candidate = decision["policy_candidate"]["best_risk_policy"]
        spec = {
            "status": "DRAFT_ONLY_NOT_EXECUTED",
            "policy_family": candidate["family_id"],
            "validation_scheme": candidate["validation_scheme"],
            "uncertainty": decision["policy_candidate"]["fixed_policy"],
            "default_action": "NO_OP",
            "protected_priority": True,
            "outer_evaluation_authorized": False,
            "OUTER_TEST_USED": False,
        }
        body = """# Prospective policy lock draft

This is a draft for a new experiment, not an outer-test authorization.
Discovery, policy-development, and evaluation groups must remain independent.
The frozen policy must predict `A_hat` before realised `A` is observed.
"""
        experiment = """# Next prospective experiment

1. Freeze discovery subjects and estimate persistence.
2. Freeze independent policy-development subjects and legal P/U/D features.
3. Lock the risk-aware policy and all thresholds.
4. Apply it once to an unseen evaluation group.
5. Observe realised actionability only after the action is fixed.

The current sealed WBCIC outer set is not opened by this draft.
"""
    else:
        spec = {
            "status": "NOT_AUTHORIZED",
            "reason": decision["terminal_state"],
            "outer_evaluation_authorized": False,
            "OUTER_TEST_USED": False,
        }
        body = f"""# Prospective policy lock draft

Not authorized. The development experiment terminated as
`{decision['terminal_state']}`. No policy is stable enough to freeze.
"""
        experiment = """# Next prospective experiment

No next prospective action-policy experiment is authorized. Opening the
sealed outer test or tuning another policy would be invalid.
"""
    write_json(NEXT_STAGE / "PROSPECTIVE_POLICY_SPEC.json", spec)
    (NEXT_STAGE / "PROSPECTIVE_POLICY_LOCK_DRAFT.md").write_text(body, encoding="utf-8")
    (NEXT_STAGE / "NEXT_PROSPECTIVE_EXPERIMENT.md").write_text(experiment, encoding="utf-8")


def scientific_report(decision: dict[str, Any], summary: pd.DataFrame, associations: pd.DataFrame) -> None:
    router = decision["oracle"]["openbmi_sample_router"]
    concentration = decision["concentration"]
    actions = pd.read_csv(DIAGNOSTICS / "ORACLE_HEADROOM_BY_ACTION.csv")
    router_actions = actions[
        (actions.family_id == "openbmi_sample_router")
        & actions.action.isin(["ERASE", "AMPLIFY", "GEOMETRY"])
    ]
    risk = summary[summary.method == "RiskAwarePERSIST"].sort_values("mean_delta_BA", ascending=False).iloc[0]
    best_associations = _top_associations(associations, str(risk.family_id), 5)
    assoc_text = "\n".join(
        f"- `{row['feature']}`: Spearman={row['spearman_r']:.3f}, grouped permutation p={row['grouped_permutation_p']:.4f}."
        for row in best_associations
    ) or "- No stable legal univariate association."
    action_text = "\n".join(
        f"- `{row.action}`: always-action ΔBA={row.mean_delta_BA:.5f}; pair-oracle rescue gain={row.mean_pair_oracle_gain:.5f}; harm fraction={row.harm_fraction:.3f}."
        for row in router_actions.itertuples(index=False)
    )
    comparisons = decision["primary_policy_comparisons"]
    comp_lines: list[str] = []
    for family, methods in comparisons.items():
        comp_lines.append(f"### {family}")
        for method, values in methods.items():
            comp_lines.append(
                f"- `{method}`: ΔBA={values['mean_delta_BA']:.6f}, LCB95={values['LCB95']:.6f}, "
                f"unsafe={values['unsafe_intervention_rate']:.3f}."
            )
    report = f"""# PERSIST-EEG prospective action policy V1

## Terminal interpretation

`{decision['terminal_state']}`

This is exploratory development evidence, not a confirmatory result. The
sealed WBCIC outer subjects were not read or used.

## Direct answers

1. **Original oracle headroom.** Exact subject-balanced OpenBMI reconstruction
   gives `{router['oracle_action_gain']:.6f}` BA.
2. **Distribution/concentration.** `{concentration['subjects_with_rescue_fraction']:.3f}`
   of subjects have at least one rescue; the largest subject contributes
   `{concentration['largest_subject_share']:.3f}` and the largest run
   `{concentration['largest_run_share']:.3f}` of oracle gain. At the trial
   level rescue remains sparse (`{concentration['rescue_unit_fraction']:.3f}`).
3. **Action-selection value.** Best fixed action including NO_OP gains
   `{router['best_fixed_including_noop_gain']:.6f}`; selection value is
   `{router['selection_value']:.6f}` BA.
4. **Rescue actions.** All three actions rescue some errors under oracle
   selection:

{action_text}

5. **Harmful actions.** Every non-trivial fixed OpenBMI intervention is net
   harmful despite its rescue cases.
6. **Predictive features.** Strongest legal associations for the best risk
   family are:

{assoc_text}

7. **Does D add beyond P/U?**
   `{json.dumps(decision['decision_dependence_increment'], ensure_ascii=False)}`
8. **Does geometry add?**
   `{json.dumps(decision['geometry_increment'], ensure_ascii=False)}`
9. **Does uncertainty reduce harm?** The best risk policy has unsafe rate
   `{risk.unsafe_intervention_rate:.3f}`. Its exact comparison to the
   non-conservative ridge is stored in `FINAL_POLICY_CANDIDATE.json`.
10. **Baseline comparisons.**

{chr(10).join(comp_lines)}

11. **Recovered headroom.** Best risk policy recovers
    `{risk.mean_recovered_headroom:.3f}` of its grouped oracle headroom.
12. **Replication across groups.** Positive-group fraction is
    `{risk.positive_group_fraction:.3f}`; largest positive-group share is
    `{risk.largest_positive_group_share if pd.notna(risk.largest_positive_group_share) else 'N/A'}`.
13. **Unsafe intervention rate.** `{risk.unsafe_intervention_rate:.3f}` for the
    best risk-aware candidate.
14. **Freeze decision.** `{decision['policy_candidate']['status']}`. This never
    authorizes opening outer test.

## Limits

- Leave-one-dataset-out is not estimable without pooling incompatible trial
  and block decision units.
- FBCNet is excluded from the WBCIC policy meta-data because its representation
  competence failed near chance.
- OpenBMI router U is unavailable independently of the target trial outcome;
  reporting a P+U router by inserting rescue/harm would be leakage.
- DDA and WBCIC use cross-fitted U only; same-cell realised U is excluded.
- A positive oracle is not a deployable policy and does not justify outer use.
"""
    (OUT / "SCIENTIFIC_REPORT.md").write_text(report, encoding="utf-8")


def validate_outputs(data: pd.DataFrame, decision: dict[str, Any]) -> None:
    expected = {
        "openbmi_sample_router": 40800,
        "openbmi_dda_block": 215,
        "wbcic_development_block": 80,
    }
    assert data.family_id.value_counts().to_dict() == expected
    assert not bool(data.outer_test_used.astype(bool).any())
    assert decision["OUTER_TEST_USED"] is False
    required = [
        PROTOCOL / "PILOT_PROVENANCE_AUDIT.md",
        PROTOCOL / "PILOT_PROVENANCE_AUDIT.json",
        DATA / "ACTION_OUTCOME_DATASET.csv",
        DATA / "ACTION_FEATURE_DICTIONARY.md",
        DATA / "META_DATASET_SCHEMA.json",
        DIAGNOSTICS / "ORACLE_HEADROOM_DECOMPOSITION.csv",
        DIAGNOSTICS / "ORACLE_HEADROOM_BY_GROUP.csv",
        DIAGNOSTICS / "ORACLE_HEADROOM_BY_ACTION.csv",
        DIAGNOSTICS / "ORACLE_CONCENTRATION_ANALYSIS.json",
        DIAGNOSTICS / "ORACLE_HEADROOM_REPORT.md",
        DIAGNOSTICS / "ACTIONABILITY_LEARNABILITY.csv",
        DIAGNOSTICS / "FEATURE_ASSOCIATION.csv",
        DIAGNOSTICS / "LEARNABILITY_REPORT.md",
        RESULTS / "MODEL_LADDER_RESULTS.csv",
        RESULTS / "POLICY_GROUP_RESULTS.csv",
        RESULTS / "POLICY_ACTION_RESULTS.csv",
        RESULTS / "RISK_POLICY_RESULTS.csv",
        RESULTS / "ORACLE_RECOVERY_RESULTS.csv",
        RESULTS / "FINAL_POLICY_CANDIDATE.json",
        NEXT_STAGE / "PROSPECTIVE_POLICY_SPEC.json",
        NEXT_STAGE / "PROSPECTIVE_POLICY_LOCK_DRAFT.md",
        NEXT_STAGE / "NEXT_PROSPECTIVE_EXPERIMENT.md",
        OUT / "FINAL_DECISION.json",
        OUT / "SCIENTIFIC_REPORT.md",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"Required outputs missing: {missing}")
    if len(list(FIGURES.glob("*.png"))) < 7 or len(list(FIGURES.glob("*.pdf"))) < 7:
        raise RuntimeError("Publication figure set is incomplete")
    schema = _json(DATA / "META_DATASET_SCHEMA.json")
    forbidden = set(schema["forbidden_model_columns"])
    if forbidden & set(schema["feature_columns"]):
        raise RuntimeError("Outcome column entered the legal feature inventory")


def reproducibility(started: float) -> None:
    source_files = [
        REPO_ROOT / "experiments" / "persist_eeg_dda_v1" / "outputs" / "results" / "DDA_BLOCK_CROSSFIT.csv",
        REPO_ROOT
        / "experiments"
        / "persist_eeg_multibackbone_final_closure"
        / "outputs"
        / "results"
        / "MASTER_BLOCK_RESULTS.csv",
        REPO_ROOT / "experiments" / "persist_eeg_router" / "outputs" / "final" / "PERSIST_ROUTER_FINAL_REPORT.json",
    ]
    router_root = router_pilot_root() / "experiments" / "persist_eeg_router" / "outputs" / "cache"
    source_files.extend(
        router_root / name
        for name in (
            "OOF_ROUTER_FEATURES.parquet",
            "OOF_BASE_LOGITS.parquet",
            "OOF_COUNTERFACTUAL_LOGITS.parquet",
            "OOF_GEOMETRY_FEATURES.parquet",
        )
    )
    output_files = sorted(
        path
        for path in OUT.rglob("*")
        if path.is_file() and path.name != "REPRODUCIBILITY.json" and "__pycache__" not in path.parts
    )
    payload = {
        "git_sha": git_sha(),
        "seed": 20260819,
        "packages": package_versions(),
        "commands": [
            "python experiments/persist_eeg_prospective_action_policy_v1/code/run_all.py"
        ],
        "source_sha256": {str(path): sha256_file(path) for path in source_files},
        "output_sha256": {
            str(path.relative_to(REPO_ROOT)).replace("\\", "/"): sha256_file(path) for path in output_files
        },
        "elapsed_seconds": time.time() - started,
        "outer_test_subject_ids_loaded": False,
        "outer_test_samples_materialized": False,
        "OUTER_TEST_USED": False,
    }
    write_json(OUT / "REPRODUCIBILITY.json", payload)


def main() -> None:
    started = time.time()
    data, metadata = build_all()
    oracle = run_oracle_analysis(data)
    model = run_modeling(data)
    learnability = pd.read_csv(DIAGNOSTICS / "ACTIONABILITY_LEARNABILITY.csv")
    associations = pd.read_csv(DIAGNOSTICS / "FEATURE_ASSOCIATION.csv")
    summary = pd.read_csv(RESULTS / "MODEL_LADDER_RESULTS.csv")
    decision = decide(oracle, model, learnability, associations, summary)
    next_stage(decision)
    scientific_report(decision, summary, associations)
    validate_outputs(data, decision)
    reproducibility(started)
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
