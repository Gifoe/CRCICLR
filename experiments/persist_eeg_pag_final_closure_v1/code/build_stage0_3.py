from __future__ import annotations
import csv, hashlib, json, subprocess
from pathlib import Path

REPO = Path(r"D:\nips-temp\TotalP\P1\CRCICLR_NESTED_OOF_V1")
EXP = REPO / "experiments" / "persist_eeg_pag_final_closure_v1"
CODE = EXP / "code"
EXP.mkdir(parents=True, exist_ok=True); CODE.mkdir(parents=True, exist_ok=True)

def run(*args):
    return subprocess.check_output(list(args), cwd=REPO, text=True, errors="replace").strip()
def write_json(name, obj):
    (EXP/name).write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=True)+"\n", encoding="utf-8")
def write_text(name, value):
    (EXP/name).write_text(value.rstrip()+"\n", encoding="utf-8")
def commit_for(path):
    try: return run("git","log","--all","--format=%H","-1","--",path) or "NOT_RECOVERABLE"
    except Exception: return "NOT_RECOVERABLE"
def tracked(path):
    try: run("git","ls-files","--error-unmatch",path); return True
    except Exception: return False

# Compact family definitions. The cited artifacts contain the detailed protocols and numbers.
defs = [
 ("A Persistence definition / P1 / P2","codex/persist-eeg-p3-p4; codex/persist-eeg-persist-net-source-only-diagnostic-v1","experiments/persist_eeg_p3closure_p4; experiments/persist_eeg_persist_net_source_only_diagnostic_v1","P2_PASS_MULTI_SEED_PERSISTENCE_UTILITY","AUTHORITATIVE_PRIMARY","persistent task-consequential structure"),
 ("B P3 trajectory / compression","codex/persist-eeg-p3-p4","experiments/persist_eeg_p3closure_p4","SELECTIVE_COMPRESSION_NOT_SUPPORTED","NEGATIVE_AUTHORITATIVE","selective compression failed in 0/5 seeds"),
 ("C PCA / high-variance relationship","codex/persist-eeg-p3-p4; codex/persist-eeg-exp4-openbmi-final-v1","experiments/persist_eeg_p3closure_p4; experiments/persist_eeg_exp4_openbmi_final_v1","PCA_OVERLAP_HIGH_BUT_NOT_IDENTITY","AUTHORITATIVE_SUPPORTING","PCA overlap is not actionability"),
 ("D identity interpretation / matched identity causal audit","codex/persist-eeg-matched-identity-causal-v1-2; codex/persist-eeg-final-closure-repair-v1","experiments/persist_eeg_matched_identity_causal_v1_2; experiments/persist_eeg_final_closure_repair_v1","IDENTITY_INTERPRETATION_BOUNDED","AUTHORITATIVE_SUPPORTING","identity interpretation is metric-bounded"),
 ("E shared geometry / causal geometry","codex/persist-eeg-p4-shared-geometry; codex/persist-eeg-p4-signed-v3-1","experiments/persist_eeg_p4_shared_geometry; experiments/persist_eeg_p4_signed_v3_1","SHARED_GEOMETRY_V1_2_PASS","AUTHORITATIVE_SUPPORTING","bounded geometry premise"),
 ("F PERSIST-PB / PERSIST-SI","codex/persist-eeg-p3-p4; codex/persist-eeg-p4-selective-invariance","experiments/persist_eeg_p3closure_p4; experiments/persist_eeg_p4_selective_invariance","P4_SELECTIVE_INVARIANCE_NOT_SUPPORTED","NEGATIVE_AUTHORITATIVE","selective intervention gates failed"),
 ("G protected / nuisance / GRL","codex/persist-eeg-p4c-suppression-safety-validation-v1; codex/persist-eeg-p4d-method-level-bridge-v1","experiments/persist_eeg_p4c_suppression_safety_validation_v1; experiments/persist_eeg_p4d_method_level_bridge_v1","SAFETY_OR_ACTIONABILITY_GATE_NOT_SUPPORTED","NEGATIVE_AUTHORITATIVE","nuisance suppression did not certify transfer"),
 ("H PERSIST-Net","codex/persist-eeg-persist-net-final-v1","experiments/persist_eeg_persist_net_final_v1","PERSIST_NET_CONSTRUCTIVE_HYPOTHESIS_NOT_SUPPORTED","NEGATIVE_AUTHORITATIVE","protected mechanism without BA gain"),
 ("I PUD / PUD-Aux","codex/persist-eeg-final-failure-localization-and-aux-v1; codex/persist-eeg-final-closure-repair-v1","experiments/persist_eeg_final_failure_localization_and_aux_v1; experiments/persist_eeg_final_closure_repair_v1","PUD_AUX_CONSTRUCTIVE_HYPOTHESIS_NOT_SUPPORTED","NEGATIVE_AUTHORITATIVE","matched development BA declined"),
 ("J SSPG / prospective gradient signal","codex/persist-eeg-stable-subject-prospective-guard-seed0-v1","experiments/persist_eeg_stable_subject_prospective_guard_seed0_v1; experiments/persist_eeg_cumulative_subject_decision_drift_audit_v1","PROSPECTIVE_SIGNAL_NOT_ACTIONABLE_OR_UNCERTAIN","AUTHORITATIVE_SUPPORTING","seed-0/source-only bounded signal"),
 ("K transport / GeoSR / prototype / relation","codex/persist-eeg-geosr-final-v1; codex/persist-eeg-incremental-relation-pilot-v3","experiments/persist_eeg_persistence_geometry_transfer_risk_audit_v1; experiments/persist_eeg_geosr_final_v1; experiments/persist_eeg_incremental_relation_pilot_v3","NO_STABLE_GENERAL_ACTIONABILITY_SIGNAL","EXPLORATORY_ONLY","fold-limited transport pilots"),
 ("L PDA / U-PDA / TEA / routing / selectors","codex/persist-eeg-prospective-action-policy-v2-1; codex/persist-eeg-utility-certified-pda-final","experiments/persist_eeg_prospective_action_policy_v2_1; experiments/persist_eeg_utility_certified_pda_final; experiments/persist_eeg_router","UTILITY_OR_ROUTING_NOT_CERTIFIED","NEGATIVE_AUTHORITATIVE","selectors did not pass utility gates"),
 ("M nested OOF / correction / utility gate","codex/persist-eeg-nested-oof-error-audit-v1; codex/persist-eeg-prospective-utility-gate-v1","experiments/persist_eeg_nested_oof_error_audit_v1; experiments/persist_eeg_prospective_utility_gate_v1","NO_CERTIFIED_PROSPECTIVE_UTILITY","AUTHORITATIVE_SUPPORTING","historical OOF claims corrected"),
 ("N multi-backbone closure","codex/persist-eeg-multibackbone-final-closure","experiments/persist_eeg_multibackbone_final_closure","FINAL_MULTIBACKBONE_FALSIFICATION_CLOSURE","NEGATIVE_AUTHORITATIVE","no jointly surviving target; FBCNet competence-failed"),
 ("O SCST / competence-generality","codex/persist-eeg-scst-competence-generality-v1; codex/persist-eeg-scst-utility-stage1","experiments/persist_eeg_scst_competence_generality_v1; experiments/persist_eeg_scst_utility_stage1","SCST_UTILITY_NOT_SUPPORTED_IN_NEAR_ADMISSIBLE_SPACE","NEGATIVE_AUTHORITATIVE","source-only secondary negative"),
 ("P V4-V8 / headroom","codex/persist-eeg-final-model-v4..v8","experiments/persist_eeg_final_model_v4; experiments/persist_eeg_final_model_v5; experiments/persist_eeg_final_model_v6; experiments/persist_eeg_final_model_v7; experiments/persist_eeg_final_model_v8","V8_SCIENTIFIC_EXHAUSTION_PHASE_A_HEADROOM","AUTHORITATIVE_SUPPORTING","realistic remaining headroom about 2 pp"),
 ("Q GroupDRO / MLDG / Fishr / style extrapolation","codex/persist-eeg-route-b-foundation-screen-v1; codex/persist-eeg-mldg-robustness-confirm-v1","experiments/persist_eeg_route_b_foundation_screen_v1; experiments/persist_eeg_mldg_robustness_confirm_v1","GENERIC_SUBJECT_DG_NOT_CERTIFIED","NEGATIVE_AUTHORITATIVE","generic DG is not a PAG solution"),
 ("R MLDG robustness / per-seed early stopping","codex/persist-eeg-mldg-perseed-earlystop-check-v1","experiments/persist_eeg_mldg_perseed_earlystop_check_v1; experiments/persist_eeg_mldg_robustness_confirm_v1","MLDG_PER_SEED_EARLY_STOP_PARTIAL_ONLY","NEGATIVE_AUTHORITATIVE","fixed-epoch interpretation superseded"),
 ("S TSEG / trajectory stability","codex/persist-eeg-tseg-session-fairness-v1","experiments/persist_eeg_tseg_session_fairness_v1","TWO_TRAJECTORY_GAIN_EXPLAINED_OR_NOT_GENERAL","NEGATIVE_AUTHORITATIVE","mechanism reduction without BA gain"),
 ("T additional PERSIST experiments","multiple historical refs","experiments/persist_eeg_exp3_decision_grounding_closure_v1; experiments/persist_eeg_dda_v1; experiments/persist_eeg_external_actionability_v1","EMPIRICAL_CHAIN_INCOMPLETE","AUTHORITATIVE_SUPPORTING","decision/actionability prerequisites bounded"),
]
base={"protocol":"see cited frozen protocol","dataset":"OpenBMI/WBCIC development unless artifact says otherwise","task":"persistence/consequence/actionability","backbone":"EEGNet or frozen roster","seeds_folds":"protocol-specific","outer_usage":"sealed/outer not used","metric":"BA and mechanism metrics","integrity":"artifact audit required","later":"consolidated by PAG closure","confound":"see cited artifact","usable":"bounded by role"}
rows=[]
for fam,branch,paths,terminal,role,hyp in defs:
 commits=sorted({commit_for(p) for p in paths.split('; ')})
 rows.append({"Family":fam,"branch":branch,"commit":"; ".join(commits),"experiment_path":paths,"hypothesis":hyp,"protocol":base["protocol"],"dataset":base["dataset"],"task":base["task"],"backbone":base["backbone"],"seeds_folds":base["seeds_folds"],"outer_heldout_usage":base["outer_usage"],"main_metric":base["metric"],"terminal_decision":terminal,"integrity_status":base["integrity"],"later_superseded":("yes" if fam.startswith(("C ","F ","R ","S ")) else "no"),"superseded_by":"PAG closure or later repaired artifact","leakage_or_confound":base["confound"],"usable_in_paper":base["usable"],"evidence_role":role,"tracked_in_current_base":any(tracked(p) for p in paths.split('; '))})
cols=list(rows[0])
with (EXP/"RESEARCH_LINEAGE_LEDGER.csv").open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(rows)
lines=["# Research lineage ledger","","This is grouped by hypothesis family, not by branch. Exploratory and superseded rows are not confirmatory evidence.","","| Family | Role | Terminal | Experiment path |","|---|---|---|---|"]
for r in rows: lines.append(f"| {r['Family']} | {r['evidence_role']} | `{r['terminal_decision']}` | `{r['experiment_path']}` |")
lines += ["","## Explicit supersession decisions","","- Shared Geometry V1 and V1.1 are superseded by V1.2; the earlier sampler/provenance was invalid or blocked.","- Early TSEG results with session exposure confounding are superseded by the session-fair closure.","- Fixed-epoch MLDG interpretations are superseded by the per-seed early-stop audit.","- Initial unpaired PUD-Aux numbers are not causal confirmation; matched repair is the usable development comparison.","- Historical best numbers are never cherry-picked as confirmatory evidence."]
write_text("RESEARCH_LINEAGE_LEDGER.md","\n".join(lines))
write_json("AUTHORITATIVE_RESULT_INDEX.json",{"schema":"PERSIST_EEG_PAG_AUTHORITATIVE_RESULT_INDEX_V1","families":[{"family":r["Family"],"role":r["evidence_role"],"terminal":r["terminal_decision"],"artifact_paths":r["experiment_path"].split('; '),"commit":r["commit"]} for r in rows if r["evidence_role"] in {"AUTHORITATIVE_PRIMARY","AUTHORITATIVE_SUPPORTING","NEGATIVE_AUTHORITATIVE"}]})
write_text("SUPERSEDED_AND_INVALID_RESULTS.md","""# Superseded and invalid results

- Shared Geometry V1: invalid process-dependent sampler; superseded by V1.2.
- Shared Geometry V1.1: blocked by upstream provenance; not a scientific mechanism failure.
- Early TSEG: session exposure confound; superseded by session-fair closure.
- Fixed-epoch MLDG: interpretation superseded by per-seed early stopping.
- Initial unpaired PUD-Aux: not an exact matched causal control; superseded by matched repair.
- Any result selected after target outcomes is exploratory only.
""")
matrix=[
 ("P2 persistence utility","supported","supported","not tested","persistence utility","not an intervention claim","persistent task-consequential structure exists"),
 ("P3 compression/PCA","supported","supported","bounded","compression failed; PCA overlap measured","not established","persistence is not certified as selective-compression target"),
 ("Shared Geometry V1.2","supported","supported","supported in bounded development","geometry transfer","not an intervention result","shared geometry does not by itself certify actionability"),
 ("PUD-Aux matched","supported","supported","not required","teacher/erasure consequence","no; matched BA delta negative","mechanism consequence did not yield constructive gain"),
 ("PERSIST-Net / protected families","supported","supported","not decisive","protected drift/erasure","no","protected intervention not actionable"),
 ("Multi-backbone closure","supported","mixed","mixed","no jointly surviving target","no certified target","breadth did not close actionability"),
 ("TSEG session-fair","supported","mechanism surrogate reduced","not primary premise","trajectory disagreement","no; WBCIC -5.6667 pp vs ERM","mechanism reduction can fail to improve BA"),
 ("V8 headroom","supported","candidate breadth","multi-backbone correctness","oracle headroom","no deployable action bank","realistic remaining headroom about 2 pp in development"),
]
with (EXP/"PAG_EVIDENCE_MATRIX.csv").open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(["Family","Persistence","Consequence","Shared geometry","Intended mechanism improved","Unseen-subject BA improved","Final interpretation"]); w.writerows(matrix)
mech=[
 ("PUD-Aux","intended mechanism improved","BA did not improve","mechanism_improved_BA_not_improved","experiments/persist_eeg_final_failure_localization_and_aux_v1/results/pud_aux_statistics.json"),
 ("PERSIST-Net","protected drift/erasure consequence observed","BA decreased","mechanism_improved_BA_not_improved","experiments/persist_eeg_persist_net_final_v1/FINAL_REPORT.json"),
 ("Shared Geometry V1.2","geometry gates passed in bounded development","no constructive intervention test","mechanism_improved_BA_not_tested","experiments/persist_eeg_p4_shared_geometry/results_v1_2/SHARED_GEOMETRY_FINAL_REPORT.json"),
 ("TSEG session-fair","trajectory disagreement decreased","OpenBMI +0.1429 pp; WBCIC -5.6667 pp","mechanism_improved_BA_not_improved","experiments/persist_eeg_tseg_session_fairness_v1/GO_GATE.json"),
 ("P4 selective invariance","some nuisance gates passed in early versions","stable generalization gate failed","mechanism_improved_BA_not_improved","experiments/persist_eeg_p4_selective_invariance/results/P4_SI_FINAL_REPORT.json"),
 ("MLDG","not robust under per-seed selection","not stable","protocol_reinterpreted","experiments/persist_eeg_mldg_perseed_earlystop_check_v1/PER_SEED_GATE.json"),
 ("GeoSR/Route-B","transport signal exploratory","no stable positive confirmation","BA_unstable","experiments/persist_eeg_geosr_final_v1/FROZEN_PROTOCOL.json"),
]
with (EXP/"MECHANISM_ACTIONABILITY_MATRIX.csv").open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(["Family","mechanism_status","actionability_status","classification","artifact"]); w.writerows(mech)
write_text("PAG_EMPIRICAL_CHAIN.md","""# PAG empirical chain

The bounded repository evidence supports `P (persistence) -> C (task/intervention consequence) -> G (shared geometry) -X-> A (actionability)`. This is not a causal theorem about EEG.

P2/P3 support repeated-measure persistence and task consequence. P3 V2 rejects selective compression (0/5 seeds). UL/PCA overlap is substantial but does not identify a safe intervention. Matched identity audits are bounded and do not justify the claim that persistence is unrelated to identity. Shared Geometry V1.2 passes its bounded development protocol; V1/V1.1 are superseded. PUD-Aux and PERSIST-Net show measurable mechanism consequences without constructive BA gains. Multi-backbone closure, V8 headroom, and session-fair TSEG fail to certify a beneficial unseen-subject intervention. All numbers trace to the index and cited artifacts.
""")
write_text("SECOND_BACKBONE_PROTOCOL.md","""# Second-backbone geometry check

Candidate: DeepConvNet, because it is competent in the existing multi-backbone closure.

Decision: `DO_NOT_IMPROVISE`. The existing DeepConvNet closure uses a different representation/block and candidate-roster protocol than Shared Geometry V1.2. There is no unambiguous one-to-one mapping for protected coordinates, random controls, and geometry gates. A new mapping would change the frozen protocol, so no second-backbone training is executed.
""")
write_json("SECOND_BACKBONE_DECISION.json",{"schema":"PERSIST_EEG_PAG_SECOND_BACKBONE_V1","candidate":"DeepConvNet","decision":"SECOND_BACKBONE_GEOMETRY_NOT_FEASIBLE_UNDER_FROZEN_PROTOCOL","training_started":False})
write_json("STAGE0_AUDIT.json",{"schema":"PERSIST_EEG_PAG_STAGE0_AUDIT_V1","family_count":len(rows),"base_head":run("git","rev-parse","HEAD"),"sealed_outcomes_read":False,"status":"COMPLETE"})
print("STAGE0_1_3_BUILT",len(rows))
