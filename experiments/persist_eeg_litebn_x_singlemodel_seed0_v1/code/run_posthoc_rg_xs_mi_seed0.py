#!/usr/bin/env python3
"""Post-hoc seed0 MI outer-dev diagnostic for LiteBN-RG/XS.
No training; explicitly exploratory because the original downstream amendment
selected LiteBN-X only before outer reveal.
"""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

REPO=Path('/root/rivermind-data/CRCICLR_TASK_GENERALITY_WORK')
EXP=REPO/'experiments/persist_eeg_litebn_x_singlemodel_seed0_v1'
CODE=EXP/'code/litebn_x.py'; OUT=EXP/'outputs'; PROT=EXP/'protocol'
RUNTIME=Path('/root/rivermind-data/litebn_x_singlemodel_seed0_runtime')
HIST=Path('/root/rivermind-data/carrier_5fold_multiseed_stability_runtime')
ARCHES=('LiteBN_RG','LiteBN_XS')
TASKS=('OpenBMI_MI','WBCIC_MI')

def load_mod():
 s=importlib.util.spec_from_file_location('litebn_x_posthoc',CODE); m=importlib.util.module_from_spec(s); sys.modules[s.name]=m; s.loader.exec_module(m); m.RUNTIME=RUNTIME; return m

def baseline_path(task,fold):
 d='openbmi' if task.startswith('OpenBMI') else 'wbcic'
 return HIST/f'{d}_fold{fold}_seed0_litebn/selected_best.pt'

def main():
 m=load_mod(); m.SEED=0
 reveal=json.loads((PROT/'STAGE_B_REVEAL.json').read_text())
 if reveal.get('final_holdout_data_accessed') or reveal.get('final_holdout_predictions_generated'):
  raise RuntimeError('final holdout state invalid')
 tri=pd.read_csv(OUT/'MI_TRIAGE_RESULTS.csv'); tri=tri[(tri.task.isin(TASKS))&(tri.architecture.isin(ARCHES))]
 if len(tri)!=20: raise RuntimeError(f'expected 10 RG/XS MI seed0 records, got {len(tri)}')
 search,folds,split_hash=m.load_folds(); device=torch.device('cpu'); rows=[]
 for task in TASKS:
  ds=m.TASKS[task]['dataset']
  for fold in folds[ds]:
   fid=int(fold['fold_id']); outer=fold['outer_dev_subjects']; bundle=m.build_bundle(task,outer)
   mean,std,norm=m.load_tensor_pair(RUNTIME/'normalizers'/f'{task.lower()}_fold{fid}.npz'); cache=m.RawGPUCache(bundle,device)
   records={a:tri[(tri.task==task)&(tri.fold==fid)&(tri.architecture==a)].iloc[0].to_dict() for a in ARCHES}
   paths={'LiteBN_BASELINE':baseline_path(task,fid),**{a:Path(records[a]['checkpoint_path']) for a in ARCHES}}
   for method,path in paths.items():
    model=m.build_model(method,task).to(device); model.load_state_dict(torch.load(path,map_location=device,weights_only=False),strict=True)
    metrics=m.evaluate(model,bundle,cache,outer,mean,std)
    for sid,val in metrics.items(): rows.append({'task':task,'dataset':ds,'fold':fid,'seed':0,'subject_id':str(sid),'method':method,**val,'checkpoint_sha256':m.sha256_file(path),'normalizer_sha256':norm['mean_std_sha256'],'posthoc_exploratory':True})
 del cache
 frame=pd.DataFrame(rows).sort_values(['task','fold','subject_id','method']); frame.to_csv(OUT/'POSTHOC_RG_XS_SEED0_MI_OUTER_SUBJECT_RESULTS.csv',index=False)
 summary=[]
 for task in TASKS:
  q=frame[frame.task==task].pivot_table(index=['fold','subject_id'],columns='method',values='BA')
  for a in ARCHES:
   d=(q[a]-q['LiteBN_BASELINE'])*100
   fd=d.groupby(level=0).mean()
   summary.append({'task':task,'architecture':a,'n_subjects':len(d),'mean_delta_pp':float(d.mean()),'median_delta_pp':float(np.median(d)),'positive_subjects':int((d>0).sum()),'negative_subjects':int((d<0).sum()),'positive_folds':int((fd>0).sum()),'fold_deltas_pp':json.dumps({str(k):float(v) for k,v in fd.items()},sort_keys=True)})
 pd.DataFrame(summary).to_csv(OUT/'POSTHOC_RG_XS_SEED0_MI_OUTER_SUMMARY.csv',index=False)
 (PROT/'POSTHOC_RG_XS_SEED0_MI_AMENDMENT.md').write_text('# Post-hoc RG/XS MI outer-dev diagnostic\n\nThis is an exploratory, post-hoc diagnostic requested after the LiteBN-X-only downstream amendment and after the original Stage-B reveal. It uses existing seed0 RG/XS MI checkpoints only; it does not train, select, or access final holdout/test data. Results must not be treated as confirmatory Stage-B evidence.\n',encoding='utf-8')
 (PROT/'POSTHOC_RG_XS_SEED0_MI_AMENDMENT.json').write_text(json.dumps({'stage':'posthoc_outer_dev_diagnostic','seed':0,'tasks':list(TASKS),'architectures':list(ARCHES),'trained':False,'final_holdout_accessed':False,'split_sha256':split_hash,'exploratory_only':True},indent=2,sort_keys=True)+'\n',encoding='utf-8')
 print('POSTHOC_RG_XS_MI_DONE',flush=True)
if __name__=='__main__': main()
