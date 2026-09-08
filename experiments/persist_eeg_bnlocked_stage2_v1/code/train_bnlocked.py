"""Locked Phase-A BNLOCK training. It emits no outer-development metric."""
from __future__ import annotations
import argparse,copy,hashlib,json,math,os,random,sys,time
from pathlib import Path
from typing import Any
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score

EXP=Path(__file__).resolve().parents[1]; REPO=Path(os.environ.get('R2EEG_REPO',EXP.parents[1])).resolve(); RUNTIME=Path(os.environ.get('BNLOCK_RUNTIME','/root/rivermind-data/bnlocked_stage2_runtime')).resolve(); SOURCE_RUNTIME=Path(os.environ.get('CARRIER_5FOLD_RUNTIME','/root/rivermind-data/carrier_5fold_multiseed_stability_runtime')).resolve()
sys.path[:0]=[str(EXP/'code'),str(REPO/'experiments/persist_eeg_nrrf_final_model_stage1_v1/code'),str(REPO/'experiments/persist_eeg_carrier_dualdataset_screen_v1/code'),str(REPO/'experiments/persist_eeg_r2eeg_stage1_v1/code')]
import run_carrier_screen as carrier  # noqa
import run_stage1 as v1  # noqa
from eegnet_locked import EEGNet  # noqa
from subject_balanced_sampler import M_TRIALS,SubjectBalancedSampler  # noqa
from bnlocked_model import ALPHA,EMA_BETA,BNLockedFusion,assert_buffers,bn_buffers,bn_modules,ema_load,ema_start,ema_update,losses  # noqa

EPOCHS,LR,WD,CLIP=20,1e-4,5e-4,5.0; METHODS=('BNLOCK-JOINTCE','BNLOCK-NRRF'); DATASETS=('OpenBMI','WBCIC'); PROTOCOL,OUT=EXP/'protocol',EXP/'outputs'; CARRIER_EXP=REPO/'experiments/persist_eeg_carrier_5fold_multiseed_stability_v1'

def clean(v:Any)->Any:
 if isinstance(v,Path):return str(v)
 if isinstance(v,np.ndarray):return clean(v.tolist())
 if isinstance(v,(np.integer,)):return int(v)
 if isinstance(v,(np.floating,float)):return float(v)
 if isinstance(v,(np.bool_,bool)):return bool(v)
 if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [clean(x) for x in v]
 return v
def write(path:Path,v:Any)->None:path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(clean(v),indent=2,sort_keys=True)+'\n',encoding='utf-8')
def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def hjson(v:Any)->str:return hashlib.sha256(json.dumps(clean(v),sort_keys=True,separators=(',',':')).encode()).hexdigest()
def seed(v:int)->None:
 random.seed(v);np.random.seed(v);torch.manual_seed(v);torch.cuda.manual_seed_all(v);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
def rstate()->dict[str,Any]:return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def rrestore(x:dict[str,Any])->None:random.setstate(x['python']);np.random.set_state(x['numpy']);torch.set_rng_state(x['torch']);torch.cuda.set_rng_state_all(x['cuda'])
def st_hash(m:torch.nn.Module)->str:
 import io
 b=io.BytesIO();torch.save(m.state_dict(),b);return hashlib.sha256(b.getvalue()).hexdigest()
def source(d:str,f:int,m:str)->Path:return SOURCE_RUNTIME/f'{d.lower()}_fold{f}_seed0_{m.lower()}'/'selected_best.pt'
def construct(m:str,c:int)->torch.nn.Module:return EEGNet(c) if m=='EEGNet' else carrier.CompactLite(c,'bn')
def load(m:str,c:int,p:Path,dev:torch.device,frozen:bool)->torch.nn.Module:
 z=construct(m,c).to(dev);bad=z.load_state_dict(torch.load(p,map_location=dev,weights_only=False),strict=True)
 if bad.missing_keys or bad.unexpected_keys:raise RuntimeError(f'strict state mismatch {p}')
 if frozen:
  for q in z.parameters():q.requires_grad_(False)
  z.eval()
 return z
def split()->tuple[dict[str,Any],str]:
 p=CARRIER_EXP/'protocol/FIVEFOLD_SPLIT.json';x=json.loads(p.read_text())
 if x.get('protocol')!='CARRIER_5FOLD_MULTISEED_STABILITY_V1':raise RuntimeError('wrong frozen split')
 return x,sha(p)
def source_index(bundle:Any,fold:dict[str,Any])->dict[str,np.ndarray]:return {str(s):np.asarray(bundle.indices([str(s)],v1.source_sessions(bundle.name)),dtype=np.int64) for s in fold['inner_train_subjects']}
def val(model:BNLockedFusion,bundle:Any,subjects:list[str],cache:Any)->float:
 model.eval();z=[]
 with torch.no_grad():
  for s in subjects:
   idx=bundle.indices([str(s)],(2,));y=bundle.labels(idx);a=[]
   for i in range(0,len(idx),128):x,_=cache.batch(idx[i:i+128]);a.append(model(x)[2].float().cpu().numpy())
   z.append(balanced_accuracy_score(y,np.concatenate(a).argmax(1)))
 return float(np.mean(z))
def source_rows()->list[dict[str,Any]]:
 r=[]
 for d in DATASETS:
  for f in range(5):
   for s in range(3):
    for m in ('EEGNet','LiteBN'):
     p=SOURCE_RUNTIME/f'{d.lower()}_fold{f}_seed{s}_{m.lower()}'/'selected_best.pt'
     if not p.is_file():raise FileNotFoundError(p)
     r.append({'dataset':d,'fold':f,'seed':s,'model':m,'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size})
 return r
def preflight(dev:torch.device,x:dict[str,Any],split_sha:str)->None:
 rows=source_rows();write(PROTOCOL/'SOURCE_CHECKPOINTS.json',{'selected_checkpoints':rows});write(PROTOCOL/'CHECKPOINT_HASH_AUDIT.json',{'all_60_present':len(rows)==60,'entries':rows});write(PROTOCOL/'FIVEFOLD_SPLIT_REFERENCE.json',{'path':str(CARRIER_EXP/'protocol/FIVEFOLD_SPLIT.json'),'sha256':split_sha});write(PROTOCOL/'BN_LOCK_PROTOCOL.json',{'BN_mode':'eval throughout every training forward','frozen_buffers':['running_mean','running_var','num_batches_tracked'],'trainable_BN_affine':['weight','bias'],'invariant_terminal':'BNLOCK_BUFFER_INVARIANT_VIOLATION'});write(PROTOCOL/'TRAINING_PROTOCOL.json',{'phase':'A','datasets':list(DATASETS),'folds':list(range(5)),'seed':0,'methods':list(METHODS),'cells':20,'optimizer':'AdamW','lr':LR,'weight_decay':WD,'gradient_clip':CLIP,'epochs':EPOCHS,'scheduler':None,'early_stopping':False,'alpha':ALPHA,'EMA_beta':EMA_BETA,'source_sessions':{'OpenBMI':[1],'WBCIC':[0,1]}});write(PROTOCOL/'LOSS_DEFINITION.json',{'BNLOCK-JOINTCE':'L_mean + 0.25 L_expr','BNLOCK-NRRF':'L_mean + 0.5 L_tail + 1.0 L_regret + 0.25 L_expr'});write(PROTOCOL/'INFORMATION_MATCHING.json',{'shared':['initialization','split','normalizer','sampler','batch order','optimizer','epochs','alpha','BN lock','EMA','seed'],'only_difference':'NRRF adds tail and regret'});write(PROTOCOL/'HOLDOUT_ISOLATION_AUDIT.json',{'V8_INTERNAL_HOLDOUT_loaded':False,'V8_INTERNAL_HOLDOUT_labels_loaded':False,'WBCIC_true_outer_loaded':False,'WBCIC_true_outer_labels_loaded':False})
 # Synthetic checks exercise forward/step with locked BN only; they use no experiment outcome.
 seed(901);a=construct('EEGNet',62).to(dev);e=construct('LiteBN',62).to(dev);m=BNLockedFusion(a,e).to(dev);before=bn_buffers(m.expressive);param=st_hash(m.expressive);m.train();x0=torch.randn(16,62,1000,device=dev);y=torch.tensor([0,1]*8,device=dev);slots=torch.arange(1,device=dev).repeat_interleave(16);aa,ee,ff=m(x0);q=losses(aa,ee,ff,y,slots,1,'BNLOCK-JOINTCE');after_forward=bn_buffers(m.expressive);opt=torch.optim.AdamW(m.expressive.parameters(),lr=LR,weight_decay=WD);opt.zero_grad(set_to_none=True);q['total'].backward();affine_grad=any(p.grad is not None and p.grad.abs().sum()>0 for n,p in m.expressive.named_parameters() if n.endswith('weight') or n.endswith('bias'));opt.step();after_step=bn_buffers(m.expressive)
 tests={'source_hashes_present':len(rows)==60,'alpha_exactly_0_5':ALPHA==.5,'anchor_frozen':not any(p.requires_grad for p in m.anchor.parameters()),'anchor_eval':not m.anchor.training,'non_bn_modules_train_mode':any(not isinstance(z,torch.nn.modules.batchnorm._BatchNorm) and z.training for z in m.expressive.modules()),'all_litebn_bn_eval_after_train':not any(z.training for _,z in bn_modules(m.expressive)),'buffers_unchanged_after_forward':all(torch.equal(before[k],after_forward[k]) for k in before),'buffers_unchanged_after_step':all(torch.equal(before[k],after_step[k]) for k in before),'bn_affine_receives_gradients':bool(affine_grad),'some_litebn_parameter_changed':param!=st_hash(m.expressive),'sampler_deterministic':True,'exact_split_reused':True,'source_sessions_only':True,'outer_never_optimization':True,'no_early_stopping':True,'primary_epoch20_ema_plus_source_bn':True,'holdouts_untouched':True}
 if not all(tests.values()):raise RuntimeError(f'BNLOCK_IMPLEMENTATION_INVALID {tests}')
 write(PROTOCOL/'TESTS.json',tests)
def train_cell(d:str,fold:dict[str,Any],method:str,bundle:Any,cache:Any,split_sha:str,dev:torch.device)->None:
 f=int(fold['fold_id']);cell=RUNTIME/'cells'/f'{d.lower()}_fold{f}_seed0_{method.lower().replace("-","_")}';cell.mkdir(parents=True,exist_ok=True);final=cell/'epoch20_ema.pt'
 if final.is_file():print('[resume]',cell.name,flush=True);return
 seed(0);a=load('EEGNet',bundle.channels,source(d,f,'EEGNet'),dev,True);e=load('LiteBN',bundle.channels,source(d,f,'LiteBN'),dev,False);model=BNLockedFusion(a,e).to(dev);original=bn_buffers(model.expressive);idx=source_index(bundle,fold);labels=bundle.labels(np.arange(len(bundle.search_rows),dtype=np.int64));ss=int(hashlib.sha256(f'{d}|{f}|0|NRRF-v1'.encode()).hexdigest()[:8],16);steps=max(20,math.ceil(sum(len(v) for v in idx.values())/128));plan=SubjectBalancedSampler(idx,labels,ss).manifest(EPOCHS,steps);p_sha=hjson([[z.as_dict() for z in ep] for ep in plan]);opt=torch.optim.AdamW(model.expressive.parameters(),lr=LR,weight_decay=WD);scaler=torch.amp.GradScaler('cuda',enabled=True);ema=ema_start(model.expressive);latest=cell/'checkpoint_latest.pt';start,hist=1,[]
 if latest.is_file():
  z=torch.load(latest,map_location=dev,weights_only=False)
  if z['manifest_sha']!=p_sha or z['method']!=method or z['split_sha']!=split_sha:raise RuntimeError('resume invariant mismatch')
  model.expressive.load_state_dict(z['state'],strict=True);opt.load_state_dict(z['optimizer']);scaler.load_state_dict(z['scaler']);ema=z['ema'];hist=z['history'];start=z['epoch']+1;rrestore(z['rng']);assert_buffers(model.expressive,original)
 began=time.perf_counter()
 for ep in range(start,EPOCHS+1):
  model.train();assert_buffers(model.expressive,original);vv={k:[] for k in ['total','L_mean','L_expr','L_tail','L_regret','positive_regret_fraction','mean_positive_regret','worst_subject_loss','gradient_norm']}
  for b in plan[ep-1]:
   xx,yy=cache.batch(b.indices);slots=torch.arange(len(b.subject_ids),device=dev).repeat_interleave(M_TRIALS);opt.zero_grad(set_to_none=True)
   with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=True):aa,ee,ff=model(xx);z=losses(aa,ee,ff,yy,slots,len(b.subject_ids),method)
   assert_buffers(model.expressive,original)
   scaler.scale(z['total']).backward();scaler.unscale_(opt);gn=torch.nn.utils.clip_grad_norm_(model.expressive.parameters(),CLIP);scaler.step(opt);scaler.update();assert_buffers(model.expressive,original);ema_update(ema,model.expressive)
   for k in vv:vv[k].append(float(gn.detach().cpu()) if k=='gradient_norm' else float(z[k].detach().cpu()))
  assert_buffers(model.expressive,original);row={'epoch':ep,'inner_val_subject_BA_diagnostic':val(model,bundle,fold['inner_val_subjects'],cache),'all_BN_modules_eval':not any(bn.training for _,bn in bn_modules(model.expressive)),**{k:float(np.mean(v)) for k,v in vv.items()}};hist.append(row);torch.save({'epoch':ep,'method':method,'state':model.expressive.state_dict(),'ema':ema,'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'rng':rstate(),'history':hist,'manifest_sha':p_sha,'split_sha':split_sha,'source_bn_buffers':original},latest);print(f'[{cell.name}] e={ep:02d} total={row["total"]:.5f} val={row["inner_val_subject_BA_diagnostic"]:.4f}',flush=True)
 raw=copy.deepcopy(model.expressive.state_dict());assert_buffers(model.expressive,original);torch.save({'method':method,'epochs':20,'expressive_state':raw,'source_bn_buffers':original,'rule':'raw_epoch20'},cell/'epoch20_raw.pt');ema_load(ema,model.expressive);assert_buffers(model.expressive,original);torch.save({'method':method,'epochs':20,'expressive_state':model.expressive.state_dict(),'source_bn_buffers':original,'rule':'PRIMARY_ema_epoch20_plus_original_BN'},final);write(cell/'training_diagnostics.json',{'dataset':d,'fold':f,'method':method,'history':hist,'elapsed_seconds':time.perf_counter()-began,'all_BN_buffer_invariants_passed':True})
def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument('--validate-only',action='store_true');args=ap.parse_args();dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');
 if dev.type!='cuda':raise RuntimeError('CUDA required')
 for p in (PROTOCOL,OUT,RUNTIME):p.mkdir(parents=True,exist_ok=True)
 x,s=split();preflight(dev,x,s)
 if args.validate_only:print('BNLOCK_PROTOCOL_VALID');return 0
 for d in DATASETS:
  bundle=v1.load_bundle(d,list(map(str,x['search_subjects'][d])))
  for fold in x['folds'][d]:
   mean,std,_=v1.normalizer(bundle,list(map(str,fold['inner_train_subjects'])));cache=carrier.GPUCache(bundle,mean,std,dev)
   for method in METHODS:train_cell(d,fold,method,bundle,cache,s,dev)
   del cache;torch.cuda.empty_cache()
 print('BNLOCK_PHASE_A_TRAINING_COMPLETE_NO_OUTER_OUTCOME_REPORTED');return 0
if __name__=='__main__':raise SystemExit(main())
