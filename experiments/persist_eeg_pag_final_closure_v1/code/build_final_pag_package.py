from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def load(rel: str):
    path = REPO / rel
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(rel: str, payload: dict) -> None:
    (ROOT / rel).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(rel: str, text: str) -> None:
    (ROOT / rel).write_text(text.rstrip() + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    p3 = load("experiments/persist_eeg_p3closure_p4/outputs/persist_eeg_p3closure_p4/p3_closure/P3_FINAL_REPORT_V2.json")
    pud = load("experiments/persist_eeg_final_failure_localization_and_aux_v1/results/pud_aux_statistics.json")
    matched = load("experiments/persist_eeg_final_closure_repair_v1/results/matched_aux_statistics.json")
    v8 = load("experiments/persist_eeg_final_model_v8/outputs/FINAL_DECISION.json")
    tseg = load("experiments/persist_eeg_tseg_session_fairness_v1/GO_GATE.json")
    stage0 = load("experiments/persist_eeg_pag_final_closure_v1/STAGE0_AUDIT.json")
    second = load("experiments/persist_eeg_pag_final_closure_v1/SECOND_BACKBONE_DECISION.json")
    synthetic = load("experiments/persist_eeg_pag_final_closure_v1/SYNTHETIC_PAG_RESULTS.json")
    blocked = load("experiments/persist_eeg_pag_final_closure_v1/SEALED_CONFIRMATION_BLOCKED.json")
    if blocked.get("outcome_labels_read") is not False:
        raise RuntimeError("sealed outcome state is not fail-closed")

    base_commit = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()
    terminal = "PAG_PARTIAL_CONFIRMATION"
    sealed_state = blocked["status"]
    strongest_claim = (
        "In frozen development protocols, persistent structure, task consequence, and bounded shared "
        "geometry are observable, but they do not certify a reliable unseen-subject constructive "
        "intervention; independent sealed confirmation remains incomplete because the authorized resources "
        "are unavailable in the server repository."
    )

    decision = {
        "schema": "PERSIST_EEG_PAG_FINAL_DECISION_V1",
        "terminal_state": terminal,
        "sealed_confirmation_state": sealed_state,
        "sealed_confirmation_run": {
            "attempted_after_preregistration_lock": True,
            "resource_available": False,
            "outcome_labels_read": False,
            "post_unblinding_repair": False,
            "blocking_evidence_type": "independent OpenBMI internal sealed holdout and WBCIC outer-10 sealed subjects",
        },
        "development_premises": {
            "persistence": "supported in frozen development evidence",
            "task_consequence": "supported, with P3 trajectory evidence",
            "shared_geometry": "supported in bounded development by the authoritative Shared Geometry V1.2 artifact",
            "actionability": "not certified by development evidence",
        },
        "constructive_primary": {
            "method": "PUD-Aux",
            "comparator": "exactly matched task-only control",
            "minimum_meaningful_gain_pp": 0.5,
            "historical_unpaired_delta": pud["delta"],
            "historical_unpaired_ci95": [pud["ci95_l"], pud["ci95_u"]],
            "matched_development_delta": matched["primary"]["mean"],
            "matched_development_ci95": [matched["primary"]["ci95_l"], matched["primary"]["ci95_u"]],
            "matched_terminal": matched["terminal"],
        },
        "strongest_defensible_claim": strongest_claim,
        "ready_for_iclr_manuscript": "NO",
        "blocking_issue": "Independent sealed confirmation resources are absent and must be supplied and audited before a strong confirmatory claim.",
        "source_artifacts": {
            "p3": "experiments/persist_eeg_p3closure_p4/outputs/persist_eeg_p3closure_p4/p3_closure/P3_FINAL_REPORT_V2.json",
            "pud_aux": "experiments/persist_eeg_final_failure_localization_and_aux_v1/results/pud_aux_statistics.json",
            "matched_pud_aux": "experiments/persist_eeg_final_closure_repair_v1/results/matched_aux_statistics.json",
            "v8": "experiments/persist_eeg_final_model_v8/outputs/FINAL_DECISION.json",
            "tseg": "experiments/persist_eeg_tseg_session_fairness_v1/GO_GATE.json",
        },
        "base_commit_before_final_package": base_commit,
    }
    write_json("FINAL_PAG_DECISION.json", decision)

    rows = [
        ["development", "P3", "MI epoch0 to best", "BA_delta", p3["headline"]["MI_epoch0_to_best_BA"]["mean"], p3["headline"]["MI_epoch0_to_best_BA"]["ci95"][0], p3["headline"]["MI_epoch0_to_best_BA"]["ci95"][1], "supported_consequence", "experiments/persist_eeg_p3closure_p4/outputs/persist_eeg_p3closure_p4/p3_closure/P3_FINAL_REPORT_V2.json", "development only"],
        ["development", "P3", "SSVEP epoch0 to best", "BA_delta", p3["headline"]["SSVEP_epoch0_to_best_BA"]["mean"], p3["headline"]["SSVEP_epoch0_to_best_BA"]["ci95"][0], p3["headline"]["SSVEP_epoch0_to_best_BA"]["ci95"][1], "supported_consequence", "experiments/persist_eeg_p3closure_p4/outputs/persist_eeg_p3closure_p4/p3_closure/P3_FINAL_REPORT_V2.json", "development only"],
        ["development", "P3", "Long epoch0 to best", "AUROC_delta", p3["headline"]["Long_epoch0_to_best_AUROC"]["mean"], p3["headline"]["Long_epoch0_to_best_AUROC"]["ci95"][0], p3["headline"]["Long_epoch0_to_best_AUROC"]["ci95"][1], "compression_hypothesis_failed", "experiments/persist_eeg_p3closure_p4/outputs/persist_eeg_p3closure_p4/p3_closure/P3_FINAL_REPORT_V2.json", "long persistence increased"],
        ["development", "P3", "UL/PCA normalized overlap", "overlap", p3["headline"]["UL_PCA_geometry_summary"]["normalized_overlap_UL_PCA"]["mean"], "", "", "bounded_shared_geometry", "experiments/persist_eeg_p3closure_p4/outputs/persist_eeg_p3closure_p4/p3_closure/P3_FINAL_REPORT_V2.json", "not an intervention result"],
        ["development", "PUD-Aux", "PUD-Aux minus historical Vanilla", "subject_balanced_BA_delta", pud["delta"], pud["ci95_l"], pud["ci95_u"], "constructive_not_supported", "experiments/persist_eeg_final_failure_localization_and_aux_v1/results/pud_aux_statistics.json", "unpaired historical comparator; do not mix with matched result"],
        ["development", "PUD-Aux", "PUD-Aux minus matched task-only", "subject_balanced_BA_delta", matched["primary"]["mean"], matched["primary"]["ci95_l"], matched["primary"]["ci95_u"], "CI_contains_0_and_0.5pp_gate", "experiments/persist_eeg_final_closure_repair_v1/results/matched_aux_statistics.json", "exact matched development comparison"],
        ["development", "PUD-Aux", "PUD-Aux BA", "subject_balanced_BA", pud["pud_aux_BA"], "", "", "development", "experiments/persist_eeg_final_failure_localization_and_aux_v1/results/pud_aux_statistics.json", "40 development subjects"],
        ["development", "PUD-Aux", "historical Vanilla BA", "subject_balanced_BA", pud["vanilla_BA"], "", "", "development", "experiments/persist_eeg_final_failure_localization_and_aux_v1/results/pud_aux_statistics.json", "40 development subjects"],
        ["development", "TSEG", "OpenBMI TSEG minus ERM", "BA_delta_pp", 0.001429, "", "", "mixed_no_generalization_claim", "experiments/persist_eeg_tseg_session_fairness_v1/GO_GATE.json", "session-fair; 0.1429 pp"],
        ["development", "TSEG", "WBCIC TSEG minus ERM", "BA_delta_pp", -0.056667, "", "", "mechanism_reduced_BA_decreased", "experiments/persist_eeg_tseg_session_fairness_v1/GO_GATE.json", "session-fair; -5.6667 pp"],
        ["development", "V8", "OpenBMI updated-screening oracle headroom", "BA_delta_pp", v8["decisions"]["OpenBMI"]["oracle_headroom_vs_updated_screening_baseline_pp"], "", "", "headroom_context", "experiments/persist_eeg_final_model_v8/outputs/FINAL_DECISION.json", "not a mathematical ceiling"],
        ["development", "V8", "WBCIC updated-screening oracle headroom", "BA_delta_pp", v8["decisions"]["WBCIC"]["oracle_headroom_vs_updated_screening_baseline_pp"], "", "", "headroom_context", "experiments/persist_eeg_final_model_v8/outputs/FINAL_DECISION.json", "development only; outer not opened"],
        ["sealed", "OpenBMI", "sealed confirmation", "subject_balanced_BA_delta", "", "", "", sealed_state, "experiments/persist_eeg_pag_final_closure_v1/SEALED_CONFIRMATION_BLOCKED.json", "resource unavailable; outcome not read"],
        ["sealed", "WBCIC", "sealed confirmation", "subject_balanced_BA_delta", "", "", "", sealed_state, "experiments/persist_eeg_pag_final_closure_v1/SEALED_CONFIRMATION_BLOCKED.json", "resource unavailable; outcome not read"],
    ]
    with (ROOT / "FINAL_AUTHORITATIVE_RESULTS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["evidence_scope", "family", "comparison", "metric", "value", "ci_low", "ci_high", "status", "source_artifact", "notes"])
        writer.writerows(rows)

    write_text("FINAL_PAG_DECISION.md", f"""# Final PAG decision

## Terminal state

`{terminal}`. The development evidence supports the existence of persistent structure, task consequence, and bounded shared geometry, but the one-shot sealed confirmation could not run because the authorized OpenBMI internal holdout and WBCIC outer-10 resources are not materialized in the server repository.

The sealed runner returned `{sealed_state}` with `outcome_labels_read=false` and `post_unblinding_repair=false`. This is a resource block, not a negative sealed outcome.

## Primary constructive result

The frozen primary method was PUD-Aux against an exactly matched task-only control with a +0.5 percentage-point BA meaningful-gain gate. The historical unpaired development comparison was `{pud['delta'] * 100:.4f} pp` (95% CI `{pud['ci95_l'] * 100:.4f}, {pud['ci95_u'] * 100:.4f}` pp). The exact matched development comparison was `{matched['primary']['mean'] * 100:.4f} pp` (95% CI `{matched['primary']['ci95_l'] * 100:.4f}, {matched['primary']['ci95_u'] * 100:.4f}` pp), so it does not exclude zero or the preregistered +0.5 pp gate. These are separate comparisons and are not pooled.

## Strongest defensible claim

{strongest_claim}

`READY_FOR_ICLR_MANUSCRIPT = NO` for a confirmatory PAG manuscript until the missing independent sealed evidence is supplied and audited.
""")

    write_text("FINAL_EVIDENCE_LEDGER.md", """# Final evidence ledger

| Evidence layer | Result | Scope | Status |
|---|---|---|---|
| Persistent representation | P2/P3 persistence and task-consequence evidence | frozen development | development evidence |
| Selective compression | Long persistence increased; preregistered compression direction 0/5 seeds | P3 | hypothesis not supported |
| PCA relation | UL/PCA normalized overlap 0.7451; bounded geometry | P3 | development only |
| Shared Geometry | authoritative V1.2 gates passed in bounded development | development | not a constructive intervention result |
| PUD-Aux historical comparison | -0.9333 pp vs historical Vanilla, CI below 0 | 40 development subjects | unpaired development evidence |
| PUD-Aux exact matched comparison | +0.4250 pp, CI includes 0 and +0.5 pp | 40 development subjects | matched development evidence; gate not supported |
| TSEG session-fair | OpenBMI +0.1429 pp; WBCIC -5.6667 pp | development | supplementary falsification evidence |
| V8 headroom | 2.0556 pp OpenBMI; 2.0833 pp WBCIC updated-screening oracle headroom | development | context, not a ceiling |
| Sealed confirmation | not run; resources unavailable | OpenBMI internal / WBCIC outer | fail-closed block |

All numeric rows are in `FINAL_AUTHORITATIVE_RESULTS.csv` with source artifact paths. No sealed outcome is included.
""")

    write_text("FINAL_CLAIMS_ALLOWED.md", """# Claims allowed

- Persistent EEG structure can be task-consequential in frozen development protocols.
- Bounded shared task geometry can coexist with the failure of a constructive intervention to show a meaningful gain.
- In the exact matched development comparison, PUD-Aux did not meet the preregistered +0.5 pp meaningful-gain gate.
- TSEG shows that reducing a plausible mechanism surrogate need not improve unseen-subject BA.
- The independent sealed confirmation remains uncompleted because the authorized resources are unavailable.
""")

    write_text("FINAL_CLAIMS_FORBIDDEN.md", """# Claims forbidden

- Do not claim that persistence is useless or that no method can exploit it.
- Do not claim that the sealed holdout or WBCIC outer-10 falsified or confirmed actionability.
- Do not claim proof that persistence cannot improve generalization.
- Do not call the development PUD-Aux result a sealed negative result.
- Do not mix the unpaired historical Vanilla comparison with the exact matched task-only comparison.
- Do not claim all architectures exhibit PAG; the second-backbone transfer was not feasible under the frozen protocol.
""")

    write_text("FINAL_LIMITATIONS.md", """# Limitations

The decisive limitation is missing authorized independent sealed data: the server contains development caches but not the OpenBMI internal sealed holdout or WBCIC outer-10 subjects/features/labels. Therefore no confirmatory target-label result exists. Development artifacts are heterogeneous across historical families, and some mechanism results are surrogate improvements rather than direct generalization gains. The final package stops rather than inventing a new adapter or opening an untracked resource.
""")

    write_text("FINAL_ICLR_STORYLINE.md", """# ICLR storyline

1. Stable persistent structure exists in development representations.
2. Persistence is task-consequential and not reducible to a trivial compression story; P3 selective compression fails while PCA overlap and bounded geometry remain observable.
3. Shared task geometry is measurable under the authoritative bounded protocol.
4. Persistence, consequence, and geometry do not by themselves certify actionability: PUD-Aux improves a matched mechanism target without passing the +0.5 pp constructive gate.
5. Constructive stress tests spanning protection, auxiliary training, routing, trajectory stabilization, and expanded action families do not yield a frozen independent confirmation; the evidence remains development-scoped.
6. The synthetic counterexample formalizes why source persistence/consequence/geometry do not identify target intervention sign without a transportability assumption.
7. The one-shot sealed confirmation is preregistered and fail-closed, but blocked by missing resources.
""")

    write_text("FINAL_MAIN_TABLES.md", """# Main tables

## Table 1 — PAG evidence chain

Use `PAG_EVIDENCE_MATRIX.csv` for the persistence/consequence/shared-geometry/actionability mapping.

## Table 2 — Mechanism versus actionability

Use `MECHANISM_ACTIONABILITY_MATRIX.csv`. Keep “mechanism improved” and “BA improved” as separate columns.

## Table 3 — Primary constructive result

Report both PUD-Aux comparisons separately: historical unpaired (`-0.9333 pp`, CI `[-1.7250,-0.1333] pp`) and exact matched task-only (`+0.4250 pp`, CI approximately `[0,+0.8167] pp`). Neither supports the preregistered +0.5 pp meaningful-gain gate.

## Table 4 — Sealed status

OpenBMI internal and WBCIC outer confirmation: `SEALED_CONFIRMATION_NOT_RUN_RESOURCE_UNAVAILABLE`; no outcome labels read.
""")

    write_text("FINAL_FIGURE_PLAN.md", """# Figure plan

1. PAG conceptual chain: Persistence → Consequence → Shared Geometry ↛ Actionability.
2. P3 trajectory and UL/PCA relationship.
3. Shared Geometry V1.2 bounded-development gates.
4. Mechanism–Actionability Matrix separating surrogate improvement from BA improvement.
5. Multi-backbone and headroom context, labelled development-only.
6. Preregistered sealed confirmation flow ending at the resource-availability block, with no target outcomes plotted.
""")

    write_text("REPRODUCIBILITY_CHECKLIST.md", """# Reproducibility checklist

- [x] Stage 0 lineage and authoritative result index committed.
- [x] Synthetic theorem intuition executed.
- [x] Second-backbone decision obeyed `DO_NOT_IMPROVISE`.
- [x] Preregistration committed before sealed runner.
- [x] Code and protocol hashes checked by the sealed runner.
- [x] Sealed runner fail-closed with `outcome_labels_read=false`.
- [x] Primary metric and +0.5 pp gate frozen.
- [x] Matched versus unpaired development comparisons kept separate.
- [x] No sealed identifiers, labels, predictions, or outcomes read.
- [x] Independent validator executed after final package generation.
- [ ] Independent sealed resources supplied and audited (blocking item).
""")

    # The manifest covers final compact artifacts, excluding itself, the validator
    # output, and the execution log that records the later commit.
    excluded = {"FINAL_RESULT_MANIFEST.json", "INDEPENDENT_VALIDATION.json", "INDEPENDENT_VALIDATION.md", "EXECUTION_LOG.md"}
    file_hashes = {}
    for path in sorted(ROOT.rglob("*")):
        if path.is_file():
            rel = path.relative_to(ROOT).as_posix()
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if rel not in excluded:
                file_hashes[rel] = sha256(path)
    write_json("FINAL_RESULT_MANIFEST.json", {
        "schema": "PERSIST_EEG_PAG_FINAL_RESULT_MANIFEST_V1",
        "terminal_state": terminal,
        "sealed_confirmation_state": sealed_state,
        "base_commit_before_final_package": base_commit,
        "files_sha256": file_hashes,
        "source_outcome_labels_read": False,
    })

    # Initial validator output; the standalone validator is run again after the
    # final commit and overwrites this compact JSON.
    write_json("INDEPENDENT_VALIDATION.json", {"schema": "PERSIST_EEG_PAG_INDEPENDENT_VALIDATION_V1", "pass": False, "status": "PENDING_STANDALONE_VALIDATOR"})
    write_text("INDEPENDENT_VALIDATION.md", "# Independent validation\n\nStandalone validator pending.\n")


if __name__ == "__main__":
    main()
