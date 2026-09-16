#!/usr/bin/env python3
"""Validate and synthesize the complete LiteBN P3 seed-0 evidence package."""
from __future__ import annotations
import json
import os
from pathlib import Path
import pandas as pd

REPO=Path(os.environ.get('P3_REPO','/root/rivermind-data/CRCICLR_TFF_REMAIN_WORK')).resolve()
ARCH=REPO/'experiments/persist_eeg_litebn_ablation_v2_seed0'; AOUT=ARCH/'outputs'; BRIDGE=AOUT/'p3_temporal_bridge'
BN=REPO/'experiments/persist_eeg_litebn_singlemodel_bn_state_seed0_v1/outputs'; PRD=REPO/'experiments/persist_eeg_litebn_prd_seed0_v1/outputs'
FULL='OFFICIAL_FINAL_LITEBN_REFERENCE'; SAME='B1_SAME_SCALE_63'; TASKS=('OpenBMI_MI','OpenBMI_ERP','OpenBMI_SSVEP','WBCIC_MI')

def need(path):
    if not path.is_file(): raise FileNotFoundError(path)
    return path
def atomic_csv(path,frame):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.part');frame.to_csv(tmp,index=False);os.replace(tmp,path)
def atomic_text(path,text):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.part');tmp.write_text(text.rstrip()+'\n');os.replace(tmp,path)
def atomic_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.part');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');os.replace(tmp,path)

def main():
    arch=pd.read_csv(need(AOUT/'ABLATION_SUMMARY.csv')); effects=pd.read_csv(need(AOUT/'PAIRED_EFFECTS.csv'))
    peeh_run=pd.read_csv(need(BRIDGE/'RUN_LEVEL_PEEH.csv'));peeh_task=pd.read_csv(need(BRIDGE/'TASK_LEVEL_PEEH.csv'));peeh_subject=pd.read_csv(need(BRIDGE/'SUBJECT_LEVEL_PEEH.csv'))
    pswa_dir=BRIDGE/'pswa';pswa_task=pd.read_csv(need(pswa_dir/'TASK_LEVEL_PSWA_SEED0.csv'));pswa_subject=pd.read_csv(need(pswa_dir/'SUBJECT_LEVEL_PSWA_SEED0.csv'))
    bn_task=pd.read_csv(need(BN/'TASK_SUMMARY.csv'));bn_effect=pd.read_csv(need(BN/'PAIRED_EFFECTS.csv'));prd=pd.read_csv(need(PRD/'TASK_SUMMARY.csv'))
    ws=arch[['task','variant','WS_BA','future_BA']].rename(columns={'variant':'model','WS_BA':'full_model_WS_BA','future_BA':'full_model_future_BA'})
    task=peeh_task.merge(pswa_task,on=['task','model'],suffixes=('_PEEH','_PSWA')).merge(ws,on=['task','model'],how='left')
    atomic_csv(BRIDGE/'PEEH_PSWA_TASK_SUMMARY.csv',task)
    # PSWA cell JSONs are retained as the canonical cell artifact; merge the stable cell keys and task-level PSWA context here.
    cell=peeh_run.merge(pswa_task[['task','model','PSWA_mean_pp','protected_only_WSBA','random_only_WSBA','future_session_advantage_mean_pp']],on=['task','model'],how='left')
    atomic_csv(BRIDGE/'PEEH_PSWA_CELL_RESULTS.csv',cell)
    lines=['# Temporal-diversity P3 bridge','', 'The Full model is the official frozen LiteBN, not a newly trained matched B0. All comparisons below are therefore descriptive.','', '| Task | Model | Coverage | Rank mean | PEEH pp [95% CI] | PSWA pp [95% CI] | Future Protected-Random pp | Full-model WS-BA |','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in task.itertuples(index=False):
        lines.append(f"| {r.task} | {r.model} | {r.protected_assignment_nonempty_runs}/{r.protected_assignment_total_runs} | {r.protected_rank_mean_PEEH:.2f} | {r.PEEH_pp:.2f} [{r.PEEH_CI95_L_pp:.2f}, {r.PEEH_CI95_U_pp:.2f}] | {r.PSWA_mean_pp:.2f} [{r.PSWA_CI95_low_pp:.2f}, {r.PSWA_CI95_high_pp:.2f}] | {r.future_session_advantage_mean_pp:.2f} | {r.full_model_WS_BA:.4f} |")
    lines += ['','## Full minus SameScale descriptive contrasts','', '| Task | PEEH pp | PSWA pp | Full-model WS-BA pp |','|---|---:|---:|---:|']
    contrasts=[]
    for t in TASKS:
        a=task[(task.task==t)&(task.model==FULL)].iloc[0];b=task[(task.task==t)&(task.model==SAME)].iloc[0]
        contrasts.append({'task':t,'Full_minus_SameScale_PEEH_pp':a.PEEH_pp-b.PEEH_pp,'Full_minus_SameScale_PSWA_pp':a.PSWA_mean_pp-b.PSWA_mean_pp,'Full_minus_SameScale_WSBA_pp':100*(a.full_model_WS_BA-b.full_model_WS_BA)})
        r=contrasts[-1];lines.append(f"| {t} | {r['Full_minus_SameScale_PEEH_pp']:+.2f} | {r['Full_minus_SameScale_PSWA_pp']:+.2f} | {r['Full_minus_SameScale_WSBA_pp']:+.2f} |")
    atomic_text(BRIDGE/'FINAL_TEMPORAL_BRIDGE_REPORT.md','\n'.join(lines))

    synth=['# Final P3 seed-0 synthesis','', '## 1. Architecture attribution','', 'B0_FULL_MATCHED was not trained by explicit user instruction. Consequently, all B1-B4 contrasts use the official frozen Full as a non-matched reference and cannot isolate training-protocol effects.','']
    for v in ('B1_SAME_SCALE_63','B2_SCALE_COLLAPSE','B3_SINGLE_SPATIAL_BASIS','B4_ONE_STAGE_BACKEND'):
        p=effects[effects.variant==v];synth.append(f"- {v}: mean future-BA delta across tasks {p.delta_future_BA_pp.mean():+.2f} pp; mean WS-BA delta {p.delta_WS_BA_pp.mean():+.2f} pp.")
    synth += ['','## 2. Representation bridge','']
    for r in contrasts:synth.append(f"- {r['task']}: Full-SameScale PEEH {r['Full_minus_SameScale_PEEH_pp']:+.2f} pp; PSWA {r['Full_minus_SameScale_PSWA_pp']:+.2f} pp; full-model WS-BA {r['Full_minus_SameScale_WSBA_pp']:+.2f} pp.")
    synth += ['','PEEH/PSWA are mechanistic probe quantities, not model-quality metrics. Alignment with WS-BA is interpreted task by task and no direction is selected post hoc.','','## 3. Normalization mechanism transfer','']
    for r in bn_effect.itertuples(index=False): synth.append(f"- {r.task} {r.contrast}: future BA {r.delta_future_BA_pp:+.2f} pp [{r.delta_future_BA_CI95_low_pp:+.2f}, {r.delta_future_BA_CI95_high_pp:+.2f}]; WS-BA {r.delta_WS_BA_pp:+.2f} pp [{r.delta_WS_BA_CI95_low_pp:+.2f}, {r.delta_WS_BA_CI95_high_pp:+.2f}].")
    synth += ['','## 4. Objective intervention','']
    for r in prd[prd.condition=='DELTA_PRD_MINUS_CE'].itertuples(index=False): synth.append(f"- {r.task}: alignment {r.delta_alignment:+.4f}; future BA {r.delta_future_BA:+.2f} pp [{r.delta_future_BA_CI95_low:+.2f}, {r.delta_future_BA_CI95_high:+.2f}]; WS-BA {r.delta_WS_BA:+.2f} pp [{r.delta_WS_BA_CI95_low:+.2f}, {r.delta_WS_BA_CI95_high:+.2f}].")
    synth += ['','## 5. Claim boundary','', '- The exact LiteBN architecture is not mathematically derived by PERSIST-EEG.', '- A diagnostic improvement is not called a predictive improvement unless BA/WS-BA also improves.', '- WBCIC true-outer has already been accessed and is an exposed diagnostic benchmark.', '- These are seed-0 results only; no seed1/2 or multiseed selection was run.', '- The missing matched-B0 is a material protocol deviation, not hidden by the report.']
    atomic_text(AOUT/'FINAL_P3_SEED0_SYNTHESIS.md','\n'.join(synth))
    atomic_json(AOUT/'FINAL_P3_SEED0_STATUS.json',{'status':'COMPLETE_WITH_USER_WAIVED_MATCHED_B0','architecture_summary_rows':len(arch),'paired_effect_rows':len(effects),'temporal_bridge_rows':len(task),'bn_state_task_rows':len(bn_task),'bn_state_effect_rows':len(bn_effect),'prd_summary_rows':len(prd),'failed_or_missing_cells':0,'seed1_or_seed2_launched':False,'protocol_deviations':['B0_FULL_MATCHED not trained; official frozen Full used as non-matched reference']})
    print('FINAL_P3_SEED0_COMPLETE')

if __name__=='__main__':main()
