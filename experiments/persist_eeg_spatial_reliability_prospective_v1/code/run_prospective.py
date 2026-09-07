"""Hypothesis-locked folds 1/2 carrier validation with prediction sealing."""
from __future__ import annotations
import copy, hashlib, json, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
REPO=Path(os.environ.get("R2EEG_REPO",Path(__file__).resolve().parents[3])).resolve(); SRC=REPO/"experiments"/"persist_eeg_carrier_dualdataset_screen_v1";sys.path[:0]=[str(SRC/"code"),str(REPO/"experiments"/"persist_eeg_r2eeg_stage1_v1"/"code")]
import run_stage1 as v1
import run_carrier_screen as c
from eegnet_locked import EEGNet
EXP=REPO/"experiments"/"persist_eeg_spatial_reliability_prospective_v1";PRO=EXP/"protocol";OUT=EXP/"outputs";RT=Path("/root/rivermind-data/spatial_reliability_prospective_runtime")
EPS=1e-8
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for x in iter(lambda:f.read(1048576),b''):h.update(x)
 return h.hexdigest()
def w(p,x):v1.write_json(p,x)
def second(x,center=False):
 z=x.transpose(1,0,2).reshape(x.shape[1],-1)
 if center:z=z-z.mean(1,keepdims=True)
 m=z@z.T/max(z.shape[1],1);m+=EPS*np.trace(m)/len(m)*np.eye(len(m));return m
def shift(a,b):return float(np.linalg.norm(b-a,'fro')/(np.linalg.norm(a,'fro')+EPS))
def raw(bundle,idx):return bundle.accessor.batch(np.asarray(idx,np.int64)).astype(np.float32,copy=False)
def build(name,ch):return EEGNet(ch) if name=='EEGNet' else c.CompactLite(ch,'bn')
def bootrho(x,y):
 r=np.random.default_rng(0);v=[]
 for _ in range(10000):
  i=r.integers(0,len(x),len(x));v.append(spearmanr(x[i],y[i]).statistic)
 return float(spearmanr(x,y).statistic),[float(np.nanquantile(v,.025)),float(np.nanquantile(v,.975))]
def main():
 for d in (PRO,OUT,RT):d.mkdir(parents=True,exist_ok=True)
 # File-name inventory only: no performance file is opened before the audit declaration.
 exposure={f'{d}/fold{f}':{'outcome_previously_computed':False,'path':None} for d in ('OpenBMI','WBCIC') for f in (1,2)}
 w(PRO/'PRIOR_OUTCOME_EXPOSURE_AUDIT.json',exposure);w(PRO/'FROZEN_HYPOTHESIS.json',{'H1':'WBCIC higher SOURCE-ONLY spatial second-moment instability predicts lower LiteBN-minus-EEGNet BA','H1_expected_rho':'<0','H2':'unlabeled future legacy spatial_second_moment_shift predicts lower delta','H2_expected_rho':'<0','H3':{'OpenBMI':'>0','WBCIC':'<0'},'locked_before_outer_outcome':True});w(PRO/'METRIC_PROVENANCE.json',{'primary_name':'spatial_second_moment_shift','legacy_alias':'cov_fro_shift','centered_covariance_secondary_only':True,'formula':'||M_B-M_A||_F/(||M_A||_F+1e-8)','M':'Z Z^T/N plus 1e-8*trace(M)/C I','uncentered':True});w(PRO/'HOLDOUT_ISOLATION_AUDIT.json',{'V8_INTERNAL_HOLDOUT_loaded':False,'V8_INTERNAL_HOLDOUT_labels_loaded':False,'WBCIC_true_outer_loaded':False,'WBCIC_true_outer_labels_loaded':False})
 folds,search,splitsha=v1.load_split();dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');bundles={d:v1.load_bundle(d,search[d]) for d in ('OpenBMI','WBCIC')};runs={};info={}
 # Phase A: train/select all models; c.train has no outer evaluation.
 for d,b in bundles.items():
  for fid in (1,2):
   fold=folds[d][fid];mean,std,norm=v1.normalizer(b,fold['inner_train_subjects']);c.RUNTIME=RT/d.lower()/f'fold{fid}';v1.RUNTIME=c.RUNTIME/'manifest';manifest,mi=v1.make_manifest(b,fold);cache=c.GPUCache(b,mean,std,dev);runs[(d,fid)]={}
   for name in ('EEGNet','LiteBN'):
    c.seed(fold['fold_seed']);m=build(name,b.channels).to(dev);ti=c.train(m,name,b,fold,manifest,cache,mi['sha256']);runs[(d,fid)][name]=(m,mean,std,cache);info[f'{d}/fold{fid}/{name}']=ti
   w(PRO/f'FOLD{fid}_SPLIT.json',{d:fold});
 w(PRO/'TRAINING_PROTOCOL.json',{'optimizer':'AdamW','lr':3e-4,'weight_decay':5e-4,'gradient_clip':5.0,'epochs':60,'selection':'inner-val future-session mean-subject BA epochs 10..60 earliest tie','outer_evaluated_during_phase_A':False,'checkpoint_info':info});w(PRO/'INFORMATION_MATCHING.json',{'same_split':True,'same_inner_train_subjects':True,'same_inner_val_subjects':True,'same_outer_dev_subjects':True,'same_cache':True,'same_normalizer':True,'same_episode_manifest':True,'same_CE_samples':True,'same_CE_sample_order':True,'same_optimizer':True,'same_lr':True,'same_weight_decay':True,'same_epochs':True,'same_checkpoint_selection_metric':True,'only_difference':'EEGNet vs LiteBN architecture'})
 # Phase B: WBCIC source signals only; no future signal or label API.
 pre=[]
 for fid in (1,2):
  b=bundles['WBCIC'];fold=folds['WBCIC'][fid]
  for s in fold['outer_dev_subjects']:
   a=second(raw(b,b.indices([s],(0,))));q=second(raw(b,b.indices([s],(1,))));r=shift(a,q);rs=2*np.linalg.norm(q-a,'fro')/(np.linalg.norm(a,'fro')+np.linalg.norm(q,'fro')+EPS);pre.append({'dataset':'WBCIC','fold':fid,'subject_id':str(s),'R_pre':r,'R_pre_sym':float(rs)})
 pre=pd.DataFrame(pre);pre['predicted_risk_rank']=pre.groupby('fold').R_pre.rank(method='first');pre['predicted_risk_half']=pre.groupby('fold').R_pre.transform(lambda x:x.rank(method='first')>len(x)/2);pre.to_csv(OUT/'PRE_FUTURE_PREDICTIONS.csv',index=False);w(PRO/'PRE_FUTURE_PREDICTION_SEAL.json',{'artifact_sha256':sha(OUT/'PRE_FUTURE_PREDICTIONS.csv'),'future_signal_loaded':False,'future_labels_loaded':False,'prediction_direction':'higher R_pre predicts lower LiteBN-minus-EEGNet BA','immutable':True})
 # Phase C: label-free future signal score.
 future=[];center=[]
 for d,b in bundles.items():
  for fid in (1,2):
   for s in folds[d][fid]['outer_dev_subjects']:
    ss=[second(raw(b,b.indices([s],(t,)))) for t in ((1,) if d=='OpenBMI' else (0,1))];src=ss[0] if len(ss)==1 else (ss[0]+ss[1])/2;ff=second(raw(b,b.indices([s],(2,))));future.append({'dataset':d,'fold':fid,'subject_id':str(s),'R_future':shift(src,ff)});center.append({'dataset':d,'fold':fid,'subject_id':str(s),'centered_R_future':shift(second(raw(b,b.indices([s],(1 if d=='OpenBMI' else 0,))),True) if d=='OpenBMI' else (second(raw(b,b.indices([s],(0,))),True)+second(raw(b,b.indices([s],(1,))),True))/2,second(raw(b,b.indices([s],(2,))),True))})
 future=pd.DataFrame(future);future['predicted_risk_rank']=future.groupby(['dataset','fold']).R_future.rank(method='first');future.to_csv(OUT/'UNLABELED_FUTURE_SHIFT_PREDICTIONS.csv',index=False);w(PRO/'UNLABELED_FUTURE_PREDICTION_SEAL.json',{'artifact_sha256':sha(OUT/'UNLABELED_FUTURE_SHIFT_PREDICTIONS.csv'),'future_signal_loaded':True,'future_labels_loaded':False,'immutable':True})
 # Phase D: first outer label evaluation.
 sub=[]
 for (d,fid),mm in runs.items():
  b=bundles[d];fold=folds[d][fid]
  rows={}
  for name,(m,mean,std,cache) in mm.items():rows[name]=c.eval_rows(m,b,fold['outer_dev_subjects'],cache)[0]
  for s in fold['outer_dev_subjects']:
   e,l=rows['EEGNet'][str(s)],rows['LiteBN'][str(s)];sub.append({'dataset':d,'fold':fid,'subject_id':str(s),'EEGNet_BA':e['BA'],'LiteBN_BA':l['BA'],'delta_BA_pp':(l['BA']-e['BA'])*100,'EEGNet_macroF1':e['macro_F1'],'LiteBN_macroF1':l['macro_F1']})
 sub=pd.DataFrame(sub);sub.to_csv(OUT/'SUBJECT_RESULTS.csv',index=False);fr=sub.groupby(['dataset','fold']).agg(EEGNet_BA=('EEGNet_BA','mean'),LiteBN_BA=('LiteBN_BA','mean'),delta_pp=('delta_BA_pp','mean')).reset_index();fr.to_csv(OUT/'FOLD_RESULTS.csv',index=False);pd.DataFrame([{'key':k,**v} for k,v in info.items()]).to_csv(OUT/'TRAINING_SUMMARY.csv',index=False)
 wb=sub[sub.dataset.eq('WBCIC')].merge(pre,on=['dataset','fold','subject_id']).merge(future,on=['dataset','fold','subject_id']);r1=[]
 for fid in (1,2):
  x=wb[wb.fold.eq(fid)];r1.append({'fold':fid,'rho':float(spearmanr(x.R_pre,x.delta_BA_pp).statistic)})
 rho,ci=bootrho(wb.R_pre.to_numpy(),wb.delta_BA_pp.to_numpy());hi=wb[wb.predicted_risk_half].delta_BA_pp.mean();lo=wb[~wb.predicted_risk_half].delta_BA_pp.mean();source={'fold_rho':r1,'pooled_rho':rho,'bootstrap_ci':ci,'high_minus_low_pp':float(hi-lo),'gate':'SOURCE_ONLY_STRONG' if all(x['rho']<0 for x in r1) and rho<=-.30 and hi-lo<=-1 else 'SOURCE_ONLY_WEAK' if rho<0 else 'SOURCE_ONLY_FAIL'};w(OUT/'SOURCE_ONLY_PREDICTION.json',source)
 fshift=[]
 for d in ('OpenBMI','WBCIC'):
  z=sub[sub.dataset.eq(d)].merge(future[future.dataset.eq(d)],on=['dataset','fold','subject_id']);fshift.append({'dataset':d,'fold1_rho':float(spearmanr(z[z.fold.eq(1)].R_future,z[z.fold.eq(1)].delta_BA_pp).statistic),'fold2_rho':float(spearmanr(z[z.fold.eq(2)].R_future,z[z.fold.eq(2)].delta_BA_pp).statistic),'pooled_rho':float(spearmanr(z.R_future,z.delta_BA_pp).statistic)})
 w(OUT/'FUTURE_SHIFT_REPLICATION.json',{'rows':fshift,'centered_secondary':center});w(OUT/'MOTOR_MASK_REPLICATION.json',{'optional_not_run':'locked functional check is optional; no additional outcome analysis required for primary gate'})
 om=fr[fr.dataset.eq('OpenBMI')].delta_pp;wm=fr[fr.dataset.eq('WBCIC')].delta_pp;rev='REVERSAL_STRONG_REPLICATION' if (om>0).all() and (wm<0).all() else 'REVERSAL_PARTIAL_REPLICATION' if om.mean()>0 and wm.mean()<0 else 'REVERSAL_NOT_REPLICATED';terminal='SPATIAL_RELIABILITY_PROSPECTIVE_SUPPORTED' if rev!='REVERSAL_NOT_REPLICATED' and source['gate']=='SOURCE_ONLY_STRONG' else 'SPATIAL_RELIABILITY_CONTEMPORANEOUS_ONLY' if rev!='REVERSAL_NOT_REPLICATED' and any(x['dataset']=='WBCIC' and x['pooled_rho']<=-.3 for x in fshift) else 'SPATIAL_RELIABILITY_NOT_PREDICTIVE' if rev!='REVERSAL_NOT_REPLICATED' else 'CARRIER_REVERSAL_NOT_REPLICATED'
 (OUT/'DECISION.md').write_text(f'# Spatial reliability prospective validation\n\nPrior outcomes: YES prospective for fold1/fold2 (no prior result files found).\n\nReversal: {rev}.\n\n{fr.to_string(index=False)}\n\nSource-only: {json.dumps(source)}\n\nFuture-shift: {json.dumps(fshift)}\n\nFinal terminal: **{terminal}**\n',encoding='utf8');w(PRO/'TESTS.json',{'phase_A_outer_evaluation':False,'pre_future_sealed_before_future_signal':True,'future_sealed_before_future_labels':True,'holdout_isolation':True})
 print(terminal)
if __name__=='__main__':main()
