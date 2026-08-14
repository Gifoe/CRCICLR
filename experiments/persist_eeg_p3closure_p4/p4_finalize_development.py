from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "outputs" / "persist_eeg_p3closure_p4"
P3 = BASE / "p3_closure"
P4 = BASE / "p4"
VERSIONS = ("V0", "V1", "V2", "V3")
TASKS = ("mi", "erp", "ssvep")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    p3 = json.loads((P3 / "P3_FINAL_REPORT_V2.json").read_text(encoding="utf-8"))
    if p3["status"] != "P3_CLOSED_AND_FROZEN":
        raise RuntimeError("P3 is not frozen")
    adaptation = json.loads((P4 / "P4_ADAPTATION_LOG.json").read_text(encoding="utf-8"))
    if adaptation["held_out_test_used"] or len(adaptation["entries"]) != 3:
        raise RuntimeError("Invalid adaptation log")
    results: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for version in VERSIONS:
        path = P4 / "development" / version / "fold-0" / "seed-0" / "DEVELOPMENT_RESULT.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["held_out_test_used"]:
            raise RuntimeError(f"{version} used held-out test")
        results[version] = result
        row: dict[str, Any] = {
            "version": version,
            "status": result["status"],
            "rank": result["method_config"]["rank"],
            "readout": result["method_config"]["readout"],
            "best_epoch": result["best_epoch"],
            "validation_macro_BA": result["validation"]["macro_BA"],
            "semantic_zp_AUROC": result["semantic"]["zp_macro_AUROC"],
            "semantic_hf_AUROC": result["semantic"]["hf_macro_AUROC"],
            "semantic_gap": result["semantic"]["macro_gap_zp_minus_hf"],
            "held_out_test_used": False,
        }
        for task in TASKS:
            row[f"{task}_BA"] = result["validation"]["task_BA"][task]
            row[f"{task}_minus_historical_reference"] = result["validation"]["PERSIST_minus_historical_reference"][task]
            row[f"{task}_budget_mean"] = result["gates"][task]["mean"]
            row[f"{task}_active_dimensions"] = result["gates"][task]["effective_active_dimensions_gt_0_5"]
            row[f"{task}_learned_minus_zero_BA"] = result["gates"][task]["learned_minus_zero_BA"]
        row.update({f"check_{key}": value for key, value in result["checks"].items()})
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(P4 / "P4_DEVELOPMENT_SUMMARY.csv", index=False)
    if (frame.status == "DEVELOPMENT_GATES_PASS").any():
        raise RuntimeError("A passing candidate exists; refusing failure finalization")
    test_markers = list(P4.glob("**/TEST_ACCESS_STARTED.json")) + list(P4.glob("**/TEST_COMPLETE.json"))
    if test_markers:
        raise RuntimeError(f"Unexpected outer-test access markers: {test_markers}")
    if (P4 / "P4_LOCKED_METHOD.json").exists():
        raise RuntimeError("A locked method exists despite no passing candidate")

    refusal = {
        "status": "P4_LOCK_REFUSED",
        "reason": "No V0-V3 candidate passed all development gates.",
        "candidate_versions": list(VERSIONS),
        "held_out_test_used": False,
        "formal_openbmi_evaluation_started": False,
        "risk_curve_started": False,
        "external_dataset_started": False,
        "next_step_requires_new_protocol_authorization": True,
    }
    write_json(P4 / "P4_LOCK_REFUSED.json", refusal)
    formal = {
        "status": "FORMAL_EVALUATION_NOT_RUN",
        "reason": "The method did not pass TRAIN/VALIDATION development gates, so no method was locked and outer-test remained unopened.",
        "success_level": "P4_MAIN_METHOD_NOT_SUPPORTED",
        "development_decision": "P4_MAIN_METHOD_NOT_YET_SUPPORTED",
        "formal_baseline_comparison": "NOT_RUN",
        "openbmi_5_seed_outer_test": "NOT_RUN",
        "persistence_risk_curve": "NOT_RUN",
        "other_current_dataset": "NOT_RUN",
        "held_out_test_used": False,
    }
    write_json(P4 / "P4_MAIN_RESULTS.json", formal)
    pd.DataFrame(
        [
            {
                "dataset": "OpenBMI",
                "split": "outer_test",
                "status": "NOT_RUN",
                "reason": "no development candidate passed; no locked method",
                "held_out_test_used": False,
            }
        ]
    ).to_csv(P4 / "P4_MAIN_RESULTS.csv", index=False)
    write_json(
        P4 / "PERSISTENCE_RISK_CURVE_NOT_RUN.json",
        {
            "status": "NOT_RUN",
            "reason": "Risk curves are permitted only after a method is locked; no candidate was lockable.",
            "held_out_test_used": False,
        },
    )

    checks_by_version = {version: result["checks"] for version, result in results.items()}
    final = {
        "status": "P4_MAIN_METHOD_NOT_SUPPORTED",
        "development_decision": "P4_MAIN_METHOD_NOT_YET_SUPPORTED",
        "p3_status": p3["status"],
        "p4_core_method_abandoned": False,
        "method_lock": "REFUSED",
        "formal_outer_test": "NOT_RUN",
        "test_driven_tuning_used": False,
        "versions": {
            version: {
                "status": result["status"],
                "method_config": result["method_config"],
                "validation": result["validation"],
                "semantic": result["semantic"],
                "gates": result["gates"],
                "geometry": result["geometry"],
                "checks": result["checks"],
            }
            for version, result in results.items()
        },
        "required_answers": {
            "A_real_persistence_subspace": {
                "answer": "NOT_STABLY_SUPPORTED",
                "evidence": {
                    version: {
                        "Persist_zP": result["semantic"]["zp_macro_AUROC"],
                        "Persist_hF": result["semantic"]["hf_macro_AUROC"],
                        "gap": result["semantic"]["macro_gap_zp_minus_hf"],
                    }
                    for version, result in results.items()
                },
                "interpretation": "Only V0 exceeded the 0.05 macro heuristic; later versions that improved branch use did not preserve the semantic ordering.",
            },
            "B_orthogonal_decomposition": {
                "answer": "SUPPORTED_AS_ENGINEERING_INVARIANT",
                "evidence": {
                    version: result["geometry"] for version, result in results.items()
                },
            },
            "C_task_specific_budgets": {
                "answer": "NUMERIC_DIFFERENCES_EXIST_BUT_SCIENTIFIC_INTERPRETATION_NOT_SUPPORTED",
                "evidence": {
                    version: {
                        task: {
                            "mean": result["gates"][task]["mean"],
                            "active_dimensions": result["gates"][task]["effective_active_dimensions_gt_0_5"],
                        }
                        for task in TASKS
                    }
                    for version, result in results.items()
                },
                "warning": "SSVEP had the largest mean gate in every version while MI had stronger P2 erasure utility. Gate magnitude did not recover the expected utility structure and was never supervised to do so.",
            },
            "D_task_decoding": {
                "answer": "NO_CANDIDATE_MET_THE_NO_GREATER_THAN_1PP_PER_TASK_WARNING",
                "evidence": {
                    version: result["validation"]["PERSIST_minus_historical_reference"]
                    for version, result in results.items()
                },
                "warning": "These are development validation comparisons to a historical EEGNet reference, NOT A FORMAL BASELINE COMPARISON.",
            },
            "E_modifications": adaptation["entries"],
            "F_test_driven_tuning": "NO; OUTER_TEST_WAS_NEVER_ACCESSED",
        },
        "failure_analysis": [
            "V0 organized persistence but the MI persistent branch was bypassed and MI BA fell.",
            "V1 curriculum improved optimization but still failed semantic, MI-use, and per-task performance gates.",
            "V2 residual readout made MI use persistence, but semantic ordering weakened and MI/ERP performance fell.",
            "V3 rank reduction improved MI/ERP, but SSVEP fell by 3.5pp and SSVEP persistence ordering reversed.",
            "Simple L1 gate magnitude is scale-sensitive and did not behave as a validated minimum-sufficient persistence measure.",
        ],
        "development_checks": checks_by_version,
        "external_dataset": {
            "candidate": "EEGMMIDB run-level persistence",
            "status": "NOT_RUN",
            "reason": "The OpenBMI method did not meet the prerequisite viable development state.",
        },
        "next": "STOP. A new protocol is required before more method search, formal baselines, ablations, or outer-test evaluation.",
    }
    write_json(P4 / "P4_FINAL_REPORT.json", final)
    report = f"""# PERSIST-EEG P4 Final Report

Decision: `P4_MAIN_METHOD_NOT_SUPPORTED`

Development decision: `P4_MAIN_METHOD_NOT_YET_SUPPORTED`

No method was locked. OpenBMI outer-test, the five-seed formal evaluation, persistence-risk curves, and EEGMMIDB were **not run**.

## A. Did PERSIST learn a real persistence subspace?

Not stably. V0 had zP/hF macro AUROC {results['V0']['semantic']['zp_macro_AUROC']:.4f}/{results['V0']['semantic']['hf_macro_AUROC']:.4f} (gap {results['V0']['semantic']['macro_gap_zp_minus_hf']:.4f}), but failed task performance and MI-use gates. Gaps for V1/V2/V3 were {results['V1']['semantic']['macro_gap_zp_minus_hf']:.4f}, {results['V2']['semantic']['macro_gap_zp_minus_hf']:.4f}, and {results['V3']['semantic']['macro_gap_zp_minus_hf']:.4f}. The semantic ordering did not survive the changes that forced task use of the persistent branch.

## B. Did the orthogonal decomposition behave correctly?

Yes as an engineering invariant. Every version passed orthogonality, reconstruction, finiteness, and rank checks. This is insufficient for scientific success because exact geometry alone does not establish persistence semantics or utility.

## C. Did tasks learn different persistence budgets?

Numerically yes, scientifically no. Mean gates (MI/ERP/SSVEP) were:

- V0: {results['V0']['gates']['mi']['mean']:.3f}/{results['V0']['gates']['erp']['mean']:.3f}/{results['V0']['gates']['ssvep']['mean']:.3f}
- V1: {results['V1']['gates']['mi']['mean']:.3f}/{results['V1']['gates']['erp']['mean']:.3f}/{results['V1']['gates']['ssvep']['mean']:.3f}
- V2: {results['V2']['gates']['mi']['mean']:.3f}/{results['V2']['gates']['erp']['mean']:.3f}/{results['V2']['gates']['ssvep']['mean']:.3f}
- V3: {results['V3']['gates']['mi']['mean']:.3f}/{results['V3']['gates']['erp']['mean']:.3f}/{results['V3']['gates']['ssvep']['mean']:.3f}

SSVEP was largest in every version although P2 found MI had the strongest erasure utility. No ordering was hard-coded. The discrepancy must be reported, not hidden.

## D. Did PERSIST preserve or improve decoding?

No version met the development warning that no task be more than 1pp below the historical EEGNet validation reference. V3 improved MI by {100*results['V3']['validation']['PERSIST_minus_historical_reference']['mi']:.2f}pp and ERP by {100*results['V3']['validation']['PERSIST_minus_historical_reference']['erp']:.2f}pp, but reduced SSVEP by {100*results['V3']['validation']['PERSIST_minus_historical_reference']['ssvep']:.2f}pp.

These are development-only historical-reference comparisons, **NOT A FORMAL BASELINE COMPARISON**.

## E. What modifications were necessary?

- V0→V1: task warmup, auxiliary ramps, delayed/weaker budget.
- V1→V2: residual readout `C_F(hF)+C_P(g⊙zP)` to prevent persistent-branch bypass.
- V2→V3: rank 8→4, stronger/longer persistence ordering, weaker budget.

All modifications used TRAIN/VALIDATION only and are recorded in `P4_ADAPTATION_LOG.json`.

## F. Was any test-driven tuning used?

`NO`. Outer-test was never accessed. Because no V0–V3 candidate passed all gates, creating `P4_LOCKED_METHOD.json` or running the formal test would violate the protocol.

## Failure interpretation

The exact projector is numerically sound, but this implementation does not jointly deliver stable persistence semantics, interpretable minimum-sufficient budgets, and non-catastrophic task performance. The current method is not ready for formal baselines, ablations, external replication, or a paper claim.
"""
    (P4 / "P4_FINAL_REPORT.md").write_text(report, encoding="utf-8")
    decision = """# PERSIST-EEG P3/P4 Final Decision

- P3: `P3_CLOSED_AND_FROZEN`
- P2 persistence utility remains supported.
- Selective compression: `NOT_SUPPORTED`.
- Medium as an independent core scale: `NOT_SUPPORTED`.
- P4 development: `P4_MAIN_METHOD_NOT_YET_SUPPORTED`.
- P4 success level: `P4_MAIN_METHOD_NOT_SUPPORTED`.
- Method lock: refused.
- Outer-test tuning/access: no.
- Formal 5-seed test, risk curves, baselines, ablations, and external dataset: not run.

STOP. Further work requires a new protocol rather than another validation-driven version.
"""
    (BASE / "P3P4_FINAL_DECISION.md").write_text(decision, encoding="utf-8")

    files = [path for path in BASE.rglob("*") if path.is_file() and path.name != "INTEGRITY.json"]
    integrity = {
        "status": "COMPLETE_WITH_P4_DEVELOPMENT_FAILURE",
        "p3_frozen": True,
        "p4_versions_complete": list(VERSIONS),
        "p4_method_locked": False,
        "outer_test_accessed": False,
        "formal_evaluation_run": False,
        "files_sha256": {str(path.relative_to(BASE)).replace("\\", "/"): sha256(path) for path in sorted(files)},
    }
    write_json(BASE / "INTEGRITY.json", integrity)
    write_json(
        BASE / "COMPLETE.json",
        {
            "status": "COMPLETE",
            "p3": p3["status"],
            "p4": final["status"],
            "outer_test_accessed": False,
            "next": "STOP",
        },
    )
    print(json.dumps({"p3": p3["status"], "p4": final["status"], "outer_test_accessed": False}, indent=2))


if __name__ == "__main__":
    main()
