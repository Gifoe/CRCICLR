from __future__ import annotations

import pandas as pd

import common as c


def main() -> None:
    state=c.read_json(c.RUNTIME/"PRE_OUTCOME_SELECTION_COMPLETE.json")
    if not state.get("complete") or state.get("S2_or_S3_accessed") is not False: raise RuntimeError("pre-outcome selection incomplete")
    anchors=pd.read_csv(c.RESULTS/"FM_SOURCE_VALIDATION_SELECTION.csv"); scaa=pd.read_csv(c.RESULTS/"SCAA_S1_ADAPTATION_SELECTION.csv")
    code_hashes={str(p.relative_to(c.EXP)).replace("\\","/"):c.sha256(p) for p in sorted(c.HERE.glob("*.py"))}
    lock={"schema":"FM_RESCUE_STAGE0_PROTOCOL_LOCK_V1","created_at_code_commit":c.git_head(),"frozen_before_primary_outcomes":True,
        "data_access_lock_sha256":c.sha256(c.PROTOCOL/"DATA_ACCESS_LOCK.json"),"fm_input_protocol_lock_sha256":c.sha256(c.PROTOCOL/"FM_INPUT_PROTOCOL_LOCK.json"),"fm_input_protocol_amendment_sha256":c.sha256(c.PROTOCOL/"FM_INPUT_PROTOCOL_AMENDMENT_V1.json"),"code_hashes":code_hashes,
        "FMs":list(c.FMS),"datasets":list(c.DATASETS),"folds":list(c.FOLDS),"seeds":list(c.SEEDS),"source_sessions":c.SOURCE_SESSIONS,
        "selected_anchor_lr":{f"{r.fm}/{r.dataset}":float(r.lr) for r in anchors.itertuples()},"task_head":"new Linear(200,2)","fine_tuning":"all encoder and head parameters; official pretrained initialization",
        "optimizer":{"name":"AdamW","weight_decay":c.WEIGHT_DECAY,"max_epochs":c.MAX_EPOCHS,"minimum_epochs":c.MIN_EPOCHS,"patience":c.PATIENCE,"label_smoothing":c.LABEL_SMOOTHING,"batch_size":c.BATCH_SIZE,"precision":"BF16"},
        "competence":{"specialist_anchors":c.SPECIALIST_ANCHORS,"thresholds":c.COMPETENCE_THRESHOLDS,"clearly_above_chance":True},
        "representation":"LaBraM official final encoder mean pool (200-D); CBraMod official BCIC-IV-2a all_patch_reps downstream penultimate projection (200-D), fixed by pre-outcome competence repair; no layer search",
        "D_vs_I":{"directions":8,"source_definition":"cross-session persistent directions from model-fit subjects","identity":"symmetric cross-session identity-skill drop after direction erasure on validation subjects","decision":"exact centered clean-vs-erased task-logit RMS on model-fit trials","consequence":"held-out outcome-subject CE increase after direction erasure","models":{"M0":["persistence","geometry_strength","rank"],"MI":["M0","identity"],"MD":["M0","decision"],"MID":["M0","identity","decision"]},"fit":"ridge alpha=1; leave-one-run-out","primary":"RMSE_I-RMSE_D"},
        "SCAA":{"dataset":"WBCIC","roles":{"S1":0,"S2":1,"S3":2},"adapter":"classification head only; encoder and normalization frozen","S1_split":"within-class chronological 70/30","selected_lr":{r.fm:float(r.lr) for r in scaa.itertuples()},"certificate":"Delta2>0","threshold_search":False,"bootstrap":10000},
        "SCST":{"layer":"final encoder 200-D","definition":"Repair-2 fixed/source-support-constrained class-conditional residual transport","alpha_grid":[i/64 for i in range(17)],"alpha_max":.25,"gates":{"affinity_ci_lower":0,"random_advantage":0,"class_accuracy_loss_max":.02,"class_true_log_probability_change_min":-.05,"independent_session_3NN_ratio_max":1.25,"off_manifold_excess_vs_random_max":.02},"fold_seed_units":True},
        "statistics":{"primary_unit":"subject or historical run","bootstrap_draws":10000,"trials_independent":False,"seeds_independent_people":False},
        "rescue_gates":"exactly Sections 15,19,22 of task prompt","confirmatory_trigger":"only overall strong constructive candidate or mixed with one strong predeclared route","matched_controls_trigger":"any strong constructive rescue signal",
        "sealed":{"OpenBMI_holdout":"UNTOUCHED_UNENUMERATED_UNEVALUATED","WBCIC_outer10":"UNTOUCHED_UNENUMERATED_UNEVALUATED","STEEGFORMER":"NOT_ACCESSED"},"S2_or_S3_accessed_before_freeze":False,"primary_outcomes_accessed_before_freeze":False}
    c.write_json(c.PROTOCOL/"FM_RESCUE_STAGE0_PROTOCOL_LOCK.json",lock)
    c.write_text(c.EXP/"FM_TASK_COMPETENCE.md","# FM task competence\n\nThe source-validation recipes are frozen. Held-out outcome BA is intentionally unavailable at this freeze and will be compared with the predeclared thresholds only after this lock is committed.")
    with (c.EXP/"FM_TRAINING_LEDGER.md").open("a",encoding="utf-8") as f:
        f.write("\n## Source-validation selection complete\n\n"+anchors.to_markdown(index=False)+"\n\nS1-only head adaptation (no S2/S3 access):\n\n"+scaa.to_markdown(index=False)+"\n")
    with (c.EXP/"FM_ITERATION_LEDGER.md").open("a",encoding="utf-8") as f:
        f.write("\n## V0 decision\n\nThe globally selected source-validation recipes and S1-only head recipes were retained. No layer, channel, outcome, S2 or S3 search occurred. The primary protocol is now frozen.\n")
    print("FM_RESCUE_STAGE0_PROTOCOL_FREEZE_COMPLETE",flush=True)


if __name__ == "__main__": main()
