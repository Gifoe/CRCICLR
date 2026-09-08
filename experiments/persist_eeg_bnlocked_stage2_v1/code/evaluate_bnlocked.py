"""Outer-development evaluation after all 20 BNLOCK cells are complete."""
from __future__ import annotations
import json,os,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
EXP=Path(__file__).resolve().parents[1];REPO=Path(os.environ.get('R2EEG_REPO',EXP.parents[1])).resolve();RUN=Path(os.environ.get('BNLOCK_RUNTIME','/root/rivermind-data/bnlocked_stage2_runtime')).resolve();SRC=Path(os.environ.get('CARRIER_5FOLD_RUNTIME','/root/rivermind-data/carrier_5fold_multiseed_stability_runtime')).resolve();OUT=EXP/'outputs';CEXP=REPO/'experiments/persist_eeg_carrier_5fold_multiseed_stability_v1'
sys.path[:0]=[str(EXP/'code'),str(REPO/'experiments/persist_eeg_carrier_dualdataset_screen_v1/code'),str(REPO/'experiments/persist_eeg_r2eeg_stage1_v1/code')]
import run_carrier_screen as carrier  # noqa
import run_stage1 as v1  # noqa
from eegnet_locked import EEGNet  # noqa
from bnlocked_model import ALPHA,assert_buffers,bn_buffers,frozen  # noqa
METHODS=('BNLOCK-JOINTCE','BNLOCK-NRRF')
def source(d,f,m):return SRC/f'{d.lower()}_fold{f}_seed0_{m.lower()}'/'selected_best.pt'
def make(m,c,p,dev):
 z=EEGNet(c) if m=='EEGNet' else carrier.CompactLite(c,'bn');z=z.to(dev);z.load_state_dict(torch.load(p,map_location=dev,weights_only=False),strict=True);return frozen(z)
def met(y,z):
 p=z.argmax(1);return {'BA':float(balanced_accuracy_score(y,p)),'macro_F1':float(f1_score(y,p,average='macro',zero_division=0)),'accuracy':float(accuracy_score(y,p))}
def logits(m,idx,cache):
 q=[]
 with torch.no_grad():
  for i in range(0,len(idx),128):x,_=cache.batch(idx[i:i+128]);q.append(m(x)[0].float().cpu().numpy())
 return np.concatenate(q)
def main():
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');sp=json.loads((CEXP/'protocol/FIVEFOLD_SPLIT.json').read_text());missing=[f'{d}/{f}/{m}' for d in ('OpenBMI','WBCIC') for f in range(5) for m in METHODS if not (RUN/'cells'/f'{d.lower()}_fold{f}_seed0_{m.lower().replace("-","_")}'/'epoch20_ema.pt').is_file()]
 if missing:raise RuntimeError('outer evaluation forbidden before all Phase-A training cells complete: '+str(missing))
 rows=[];raw=[];invariants=[]
 for d in ('OpenBMI','WBCIC'):
  bundle=v1.load_bundle(d,list(map(str,sp['search_subjects'][d])))
  for fs in sp['folds'][d]:
   f=int(fs['fold_id']);mean,std,_=v1.normalizer(bundle,list(map(str,fs['inner_train_subjects'])));cache=carrier.GPUCache(bundle,mean,std,dev);a=make('EEGNet',bundle.channels,source(d,f,'EEGNet'),dev);l=make('LiteBN',bundle.channels,source(d,f,'LiteBN'),dev);models={}
   for method in METHODS:
    for variant in ('ema','raw'):
     p=RUN/'cells'/f'{d.lower()}_fold{f}_seed0_{method.lower().replace("-","_")}'/f'epoch20_{variant}.pt';q=torch.load(p,map_location=dev,weights_only=False);z=carrier.CompactLite(bundle.channels,'bn').to(dev);z.load_state_dict(q['expressive_state'],strict=True);z=frozen(z);assert_buffers(z,q['source_bn_buffers']);invariants.append({'dataset':d,'fold':f,'method':method,'variant':variant,'final_buffers_equal_source':True,'all_parameters_frozen_for_eval':not any(x.requires_grad for x in z.parameters())});models[(method,variant)]=z
   for s in map(str,fs['outer_dev_subjects']):
    idx=bundle.indices([s],(2,));y=bundle.labels(idx);za,ze=logits(a,idx,cache),logits(l,idx,cache);base={'EEGNet':met(y,za),'LiteBN':met(y,ze),'FROZEN-LOGIT50':met(y,.5*za+.5*ze)}
    for method in METHODS:
     for variant in ('ema','raw'):
      zn=.5*za+.5*logits(models[(method,variant)],idx,cache);item={'dataset':d,'fold':f,'seed':0,'subject_id':s,'method':method,'variant':variant,'trials':len(y),'EEGNet_BA':base['EEGNet']['BA'],'LiteBN_BA':base['LiteBN']['BA'],'FROZEN_LOGIT50_BA':base['FROZEN-LOGIT50']['BA'],**met(y,zn)}
      (rows if variant=='ema' else raw).append(item)
   del cache;torch.cuda.empty_cache()
 OUT.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(OUT/'PHASE_A_SUBJECT_RESULTS.csv',index=False);(OUT/'RAW_EPOCH20_SECONDARY.json').write_text(json.dumps(raw,indent=2)+'\n');(OUT/'BN_BUFFER_INVARIANTS.json').write_text(json.dumps(invariants,indent=2)+'\n');print('BNLOCK_PHASE_A_OUTER_EVALUATION_COMPLETE')
if __name__=='__main__':main()
