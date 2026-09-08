"""Aggregate locked BNLOCK Phase A and apply preregistered gates."""
from __future__ import annotations
import json,math,os
from pathlib import Path
import numpy as np,pandas as pd
EXP=Path(__file__).resolve().parents[1];REPO=Path(os.environ.get('R2EEG_REPO',EXP.parents[1])).resolve();OUT=EXP/'outputs';RUN=Path(os.environ.get('BNLOCK_RUNTIME','/root/rivermind-data/bnlocked_stage2_runtime')).resolve();OLD=REPO/'experiments/persist_eeg_nrrf_final_model_stage1_v1/outputs'
def clean(v):
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,(np.integer,)):return int(v)
 if isinstance(v,(np.floating,float)):return float(v) if math.isfinite(float(v)) else None
 if isinstance(v,(np.bool_,bool)):return bool(v)
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 return v
def js(p,v):p.write_text(json.dumps(clean(v),indent=2,sort_keys=True)+'\n')
def boot(x):
 r=np.random.default_rng(0);z=r.choice(x,size=(10000,len(x)),replace=True).mean(1);return {'mean_pp':float(x.mean()*100),'median_pp':float(np.median(x)*100),'ci_low_pp':float(np.quantile(z,.025)*100),'ci_high_pp':float(np.quantile(z,.975)*100)}
def main():
 s=pd.read_csv(OUT/'PHASE_A_SUBJECT_RESULTS.csv');s['delta_vs_EEGNet_pp']=(s.BA-s.EEGNet_BA)*100;s['delta_vs_LiteBN_pp']=(s.BA-s.LiteBN_BA)*100;s['delta_vs_FROZEN_LOGIT50_pp']=(s.BA-s.FROZEN_LOGIT50_BA)*100;s.to_csv(OUT/'PHASE_A_SUBJECT_RESULTS.csv',index=False)
 old=pd.read_csv(OLD/'PHASE_A_FOLD_RESULTS.csv');old=old[old.method.isin(['JOINT-CE','NRRF-v1'])][['dataset','fold','method','mean_subject_BA']].rename(columns={'method':'old_method','mean_subject_BA':'old_stage2_BA'});mapper={'BNLOCK-JOINTCE':'JOINT-CE','BNLOCK-NRRF':'NRRF-v1'}
 frows=[];drows=[];harm={}
 for (d,f,m),g in s.groupby(['dataset','fold','method']):
  delta=(g.BA-g.EEGNet_BA).to_numpy();frows.append({'dataset':d,'fold':int(f),'seed':0,'method':m,'mean_subject_BA':float(g.BA.mean()),'mean_subject_macro_F1':float(g.macro_F1.mean()),'mean_subject_accuracy':float(g.accuracy.mean()),'delta_vs_EEGNet_pp':float(delta.mean()*100),'delta_vs_LiteBN_pp':float((g.BA-g.LiteBN_BA).mean()*100),'delta_vs_FROZEN_LOGIT50_pp':float((g.BA-g.FROZEN_LOGIT50_BA).mean()*100),'median_subject_delta_pp':float(np.median(delta)*100),'harm_le_minus_1pp':float((delta<=-.01).mean()),'harm_le_minus_3pp':float((delta<=-.03).mean()),'harm_le_minus_5pp':float((delta<=-.05).mean()),'gain_ge_plus_1pp':float((delta>=.01).mean()),'tie_abs_lt_0_5pp':float((np.abs(delta)<.005).mean()),'worst_subject_delta_pp':float(delta.min()*100),'n_subjects':len(g)})
 folds=pd.DataFrame(frows);folds['old_method']=folds.method.map(mapper);folds=folds.merge(old,on=['dataset','fold','old_method'],how='left');folds['delta_vs_previous_stage2_same_loss_pp']=(folds.mean_subject_BA-folds.old_stage2_BA)*100;folds.to_csv(OUT/'PHASE_A_FOLD_RESULTS.csv',index=False)
 for (d,m),g in s.groupby(['dataset','method']):
  delta=(g.BA-g.EEGNet_BA).to_numpy();b=boot(delta);ff=folds[(folds.dataset==d)&(folds.method==m)];h={f'le_minus_{t}pp':float((delta<=-t/100).mean()) for t in (1,3,5)};harm[f'{d}/{m}']={'vs_EEGNet':h,'gain_ge_plus_1pp':float((delta>=.01).mean()),'tie_abs_lt_0_5pp':float((np.abs(delta)<.005).mean())};drows.append({'dataset':d,'method':m,'EEGNet_BA':float(g.EEGNet_BA.mean()),'LiteBN_BA':float(g.LiteBN_BA.mean()),'Frozen_LOGIT50_BA':float(g.FROZEN_LOGIT50_BA.mean()),'BNLOCK_BA':float(g.BA.mean()),'gain_vs_EEGNet_pp':b['mean_pp'],'median_subject_delta_pp':b['median_pp'],'bootstrap_ci_low_pp':b['ci_low_pp'],'bootstrap_ci_high_pp':b['ci_high_pp'],'gain_vs_LOGIT50_pp':float((g.BA-g.FROZEN_LOGIT50_BA).mean()*100),'positive_folds':int((ff.delta_vs_EEGNet_pp>0).sum()),'worst_fold_delta_pp':float(ff.delta_vs_EEGNet_pp.min()),'harm_le_minus_1pp':h['le_minus_1pp'],'delta_vs_previous_stage2_same_loss_pp':float(ff.delta_vs_previous_stage2_same_loss_pp.mean())})
 ds=pd.DataFrame(drows);ds['old_method']=ds.method.map(mapper);old_ds=pd.read_csv(OLD/'PHASE_A_DATASET_RESULTS.csv');oldrows=[]
 for d in ('OpenBMI','WBCIC'):
  q=old_ds[old_ds.dataset==d].iloc[0];oldrows+=[{'dataset':d,'old_method':'JOINT-CE','old_BA':q['JOINT_CE_BA']},{'dataset':d,'old_method':'NRRF-v1','old_BA':q['NRRF_BA']}]
 ds=ds.merge(pd.DataFrame(oldrows),on=['dataset','old_method'],how='left');ds.to_csv(OUT/'PHASE_A_DATASET_RESULTS.csv',index=False);js(OUT/'HARM_PROFILE.json',harm)
 diags=[]
 for d in ('OpenBMI','WBCIC'):
  for f in range(5):
   for m in ('BNLOCK-JOINTCE','BNLOCK-NRRF'):
    p=RUN/'cells'/f'{d.lower()}_fold{f}_seed0_{m.lower().replace("-","_")}'/'training_diagnostics.json';diags.append(json.loads(p.read_text()))
 js(OUT/'TRAINING_DIAGNOSTICS.json',{'cells':diags})
 def r(d,m):return ds[(ds.dataset==d)&(ds.method==m)].iloc[0]
 o,w=r('OpenBMI','BNLOCK-JOINTCE'),r('WBCIC','BNLOCK-JOINTCE');frozen_harm=float((s[(s.dataset=='WBCIC')&(s.method=='BNLOCK-JOINTCE')].FROZEN_LOGIT50_BA-s[(s.dataset=='WBCIC')&(s.method=='BNLOCK-JOINTCE')].EEGNet_BA<=-.01).mean());added=(w.BNLOCK_BA>=w.Frozen_LOGIT50_BA+.003) or ((frozen_harm-w.harm_le_minus_1pp)>=.05 and w.BNLOCK_BA>=w.Frozen_LOGIT50_BA-.002);open_upside=o.gain_vs_EEGNet_pp>=3 and o.positive_folds==5 and o.gain_vs_LOGIT50_pp>=-.5;openok=open_upside and o.harm_le_minus_1pp<=.10;wok=w.gain_vs_EEGNet_pp>=1 and w.positive_folds>=4 and w.median_subject_delta_pp>0 and w.harm_le_minus_1pp<.15 and w.worst_fold_delta_pp>-2;repaired=bool(w.delta_vs_previous_stage2_same_loss_pp>0 and w.BNLOCK_BA>=w.Frozen_LOGIT50_BA-.002);primary=bool(openok and wok and added)
 n=r('WBCIC','BNLOCK-NRRF');no=r('OpenBMI','BNLOCK-NRRF');nrrf_add=bool(n.BNLOCK_BA>=w.BNLOCK_BA+.003 or (n.harm_le_minus_1pp<=w.harm_le_minus_1pp-.05 and n.BNLOCK_BA>=w.BNLOCK_BA-.002 and no.BNLOCK_BA>=o.BNLOCK_BA-.005))
 if primary:terminal='BNLOCK_STAGE2_SEED0_PASS';preferred='BNLOCK-NRRF' if nrrf_add else 'BNLOCK-JOINTCE'
 elif repaired and open_upside and not added:terminal='BNLOCK_REPAIRS_COLLAPSE_BUT_NO_ADDED_VALUE_STOP';preferred=None
 elif not wok:terminal='BNLOCK_STAGE2_WBCIC_FAIL_STOP';preferred=None
 else:terminal='BNLOCK_OPENBMI_UPSIDE_FAIL_STOP';preferred=None
 gate={'openbmi_primary':bool(openok),'openbmi_upside_preserved':bool(open_upside),'wbcic_primary':bool(wok),'bnlock_repairs_previous_collapse':repaired,'added_value':bool(added),'primary_pass':primary,'nrrf_adds_value':nrrf_add,'frozen_wbcic_harm_le_minus_1pp':frozen_harm,'terminal':terminal,'preferred_method':preferred};js(OUT/'PHASE_A_GATE.json',gate)
 lines=['# BNLOCK Stage-2 Phase-A Decision','','| Metric | OpenBMI | WBCIC |','|---|---:|---:|']
 for label,key in [('EEGNet BA','EEGNet_BA'),('LiteBN BA','LiteBN_BA'),('Frozen LOGIT50 BA','Frozen_LOGIT50_BA'),('Old JOINT-CE BA','old_BA'),('BNLOCK-JOINTCE BA','BNLOCK_BA'),('BNLOCK-JOINT delta vs EEGNet (pp)','gain_vs_EEGNet_pp'),('BNLOCK-JOINT delta vs LOGIT50 (pp)','gain_vs_LOGIT50_pp'),('Positive folds','positive_folds'),('Harm <= -1pp','harm_le_minus_1pp')]:lines.append(f'| {label} | {o[key]:.4f} | {w[key]:.4f} |')
 lines+=['',f'BNLOCK-NRRF BA: OpenBMI {no.BNLOCK_BA:.4f}; WBCIC {n.BNLOCK_BA:.4f}.',f'BNLOCK-NRRF delta vs EEGNet: OpenBMI {no.gain_vs_EEGNet_pp:+.3f} pp; WBCIC {n.gain_vs_EEGNet_pp:+.3f} pp.',f'BN-lock fixed previous collapse: {"YES" if repaired else "NO"}.',f'JOINT adds value beyond frozen fusion: {"YES" if added else "NO"}.',f'NRRF adds value beyond JOINT: {"YES" if nrrf_add else "NO"}.',f'Exact gate: {json.dumps(clean(gate),sort_keys=True)}.',f'Final terminal: **{terminal}**.']
 (OUT/'PHASE_A_DECISION.md').write_text('\n'.join(lines)+'\n');print(terminal)
if __name__=='__main__':main()
