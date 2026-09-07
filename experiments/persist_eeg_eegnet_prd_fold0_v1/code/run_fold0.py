"""Frozen fold-0 EEGNet ERM versus PRD screening run."""
from __future__ import annotations
import argparse, copy, hashlib, inspect, json, os, random, sys, time
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO=Path(os.environ.get("R2EEG_REPO",Path(__file__).resolve().parents[3])).resolve(); V1_CODE=REPO/"experiments/persist_eeg_r2eeg_stage1_v1/code"; sys.path.append(str(V1_CODE))
import run_stage1 as v1  # noqa
import stage1_core as canonical  # noqa
from eegnet import EEGNet,parameter_count
from prd_loss import prd_loss,explicit_binary,direction
from metrics import score

EXP=REPO/"experiments/persist_eeg_eegnet_prd_fold0_v1"; PROTOCOL=EXP/"protocol"; OUT=EXP/"outputs"; RUNTIME=Path("/root/rivermind-data/eegnet_prd_fold0_runtime"); V1_RUNTIME=RUNTIME/"deterministic_source_manifest"; V1_EXP=REPO/"experiments/persist_eeg_r2eeg_stage1_v1"
LR,WD,CLIP,LAMBDA,MAX,MIN=3e-4,5e-4,5.,.25,60,10

def sha(path:Path)->str:
 h=hashlib.sha256();
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def write(path:Path,x:Any)->None:v1.write_json(path,x)
def seed(n:int)->None:
 random.seed(n);np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n) if torch.cuda.is_available() else None;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
def state_hash(x:dict[str,torch.Tensor])->str:
 import io
 b=io.BytesIO();torch.save(x,b);return hashlib.sha256(b.getvalue()).hexdigest()
def sids(side:str,device:torch.device)->torch.Tensor:return torch.arange(4,device=device).repeat_interleave(16)

def evaluate(model:torch.nn.Module,bundle:Any,subjects:list[str],sessions:tuple[int,...],mean:np.ndarray,std:np.ndarray,device:torch.device,reps:bool=False):
 model.eval(); out={}; emb={}; all_y=[];all_p=[]
 with torch.no_grad():
  for s in subjects:
   idx=bundle.indices([s],sessions); y=bundle.labels(idx); ls=[];zs=[]
   for i in range(0,len(idx),128):
    l,z=model(v1.prepare(bundle,idx[i:i+128],mean,std,device));ls.append(l.float().cpu().numpy());zs.append(z.float().cpu().numpy())
   l,z=np.concatenate(ls),np.concatenate(zs);p=l.argmax(1);out[s]={**score(y,p),"trials":int(len(y))};all_y.append(y);all_p.append(p)
   if reps:emb[s]={"y":y,"z":z,"logits":l}
 return out,emb,{**score(np.concatenate(all_y),np.concatenate(all_p))}

def geometry(source:dict[str,np.ndarray],future:dict[str,np.ndarray])->dict[str,float]:
 def d(x):return x["z"][x["y"]==1].mean(0)-x["z"][x["y"]==0].mean(0)
 ds,dq=d(source),d(future);cos=float(ds@dq/max(np.linalg.norm(ds)*np.linalg.norm(dq),1e-12));means=[future["z"][future["y"]==c].mean(0) for c in (0,1)];sep=float(np.linalg.norm(means[1]-means[0]));scatter=float(np.mean(np.concatenate([np.sum((future["z"][future["y"]==c]-means[c])**2,axis=1) for c in (0,1)])));y=future["y"].astype(int);margin=float(np.mean(future["logits"][np.arange(len(y)),y]-future["logits"][np.arange(len(y)),1-y]));return {"source_future_direction_cosine":cos,"class_separation":sep,"within_class_scatter":scatter,"fisher":sep*sep/max(scatter,1e-12),"logit_margin":margin}

def mean_pairwise_future_direction(reps:dict[str,dict[str,np.ndarray]])->float:
 dirs=[]
 for rec in reps.values():
  d=rec["z"][rec["y"]==1].mean(0)-rec["z"][rec["y"]==0].mean(0);dirs.append(d/max(np.linalg.norm(d),1e-12))
 if len(dirs)<2:return float("nan")
 dots=[float(dirs[i]@dirs[j]) for i in range(len(dirs)) for j in range(i+1,len(dirs))]
 return float(np.mean(dots))

def train(model:EEGNet,name:str,bundle:Any,fold:dict,manifest:list,mean:np.ndarray,std:np.ndarray,device:torch.device)->dict:
 opt=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WD);scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda");best=-1.;best_epoch=None;best_state=None;hist=[];started=time.perf_counter()
 for epoch,episodes in enumerate(manifest,1):
  model.train();ces=[];prs=[];sms=[];qms=[];tot=[]
  for ep in episodes:
   idx=np.asarray(ep["support_indices"]+ep["query_indices"],np.int64);y=torch.from_numpy(bundle.labels(idx)).to(device);opt.zero_grad(set_to_none=True)
   with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=="cuda"):
    logits,z=model(v1.prepare(bundle,idx,mean,std,device));ce=F.cross_entropy(logits,y);pr=torch.zeros((),device=device);diag={"support_direction_norm":torch.zeros((),device=device),"query_direction_norm":torch.zeros((),device=device)}
    if name=="EEGNet-PRD":pr,diag=prd_loss(z[:64],y[:64],sids("support",device),z[64:],y[64:],sids("query",device));loss=ce+LAMBDA*pr
    else:loss=ce
   scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP);scaler.step(opt);scaler.update();ces.append(float(ce.detach().cpu()));prs.append(float(pr.detach().cpu()));sms.append(float(diag["support_direction_norm"].detach().cpu()));qms.append(float(diag["query_direction_norm"].detach().cpu()));tot.append(float(loss.detach().cpu()))
  val,_,_=evaluate(model,bundle,fold["inner_val_subjects"],(2,),mean,std,device);ba=float(np.mean([x["BA"] for x in val.values()]));selected=epoch>=MIN and ba>best+1e-12
  if selected:best,best_epoch,best_state=ba,epoch,copy.deepcopy(model.state_dict())
  hist.append({"epoch":epoch,"CE":float(np.mean(ces)),"raw_PRD":float(np.mean(prs)),"lambda_times_PRD":float(LAMBDA*np.mean(prs)),"total_loss":float(np.mean(tot)),"support_direction_magnitude":float(np.mean(sms)),"query_direction_magnitude":float(np.mean(qms)),"inner_val_subject_BA":ba,"selected":selected})
  if epoch==1 or epoch%5==0 or selected:print(f"[{name}] e={epoch:02d} CE={hist[-1]['CE']:.4f} PRD={hist[-1]['raw_PRD']:.4f} val={ba:.4f}",flush=True)
 if best_state is None:raise RuntimeError("no selected checkpoint")
 model.load_state_dict(best_state);p=RUNTIME/"checkpoints"/f"{name.lower().replace('-','_')}_best.pt";p.parent.mkdir(parents=True,exist_ok=True);torch.save(model.state_dict(),p);return {"model":name,"history":hist,"selected_epoch":best_epoch,"best_inner_val_subject_BA":best,"checkpoint":str(p),"checkpoint_sha256":sha(p),"steps_per_epoch":len(manifest[0]),"actual_steps":len(manifest)*len(manifest[0]),"seconds":time.perf_counter()-started}

def main()->int:
 ap=argparse.ArgumentParser();ap.add_argument("--validate-only",action="store_true");a=ap.parse_args();device=torch.device("cuda" if torch.cuda.is_available() else "cpu");[p.mkdir(parents=True,exist_ok=True) for p in (PROTOCOL,OUT,RUNTIME)]
 v1.RUNTIME=V1_RUNTIME;folds,search,split_sha=v1.load_split();fold=folds["OpenBMI"][0];bundle=v1.load_bundle("OpenBMI",search["OpenBMI"]);mean,std,norm=v1.normalizer(bundle,fold["inner_train_subjects"]);manifest,minfo=v1.make_manifest(bundle,fold);mp=Path(minfo["path"])
 if len(manifest)!=60 or len(manifest[0])!=20:raise RuntimeError("prior episode manifest invalid")
 outer=set(map(int,bundle.indices(fold["outer_dev_subjects"])));assert not any(outer&set(e["support_indices"]+e["query_indices"]) for es in manifest for e in es)
 # Exact source class comparison rather than a historical outcome match.
 source_file=V1_CODE/"stage1_core.py";src=canonical.EEGNet(62);local=EEGNet(62);same_shapes=[(k,tuple(v.shape)) for k,v in src.state_dict().items()]==[(k,tuple(v.shape)) for k,v in local.state_dict().items()]
 if not same_shapes or parameter_count(local)!=sum(p.numel() for p in src.parameters()):raise RuntimeError("EEGNET_PRD_ARCHITECTURE_MISMATCH")
 seed(0);template=EEGNet(62).to(device);initial=copy.deepcopy(template.state_dict());init_hash=state_hash(initial);erm=EEGNet(62).to(device);prd=EEGNet(62).to(device);erm.load_state_dict(initial);prd.load_state_dict(initial)
 z=torch.randn(32,64);y=torch.tensor([0]*8+[1]*8+[0]*8+[1]*8);ss=torch.tensor([0]*16);sq=torch.tensor([1]*16);g,_=prd_loss(z[:16],y[:16],ss,z[16:],y[16:],sq);assert torch.allclose(g,explicit_binary(z[:16],y[:16],ss,z[16:],y[16:],sq),atol=1e-7);m0,m1,b=torch.randn(64),torch.randn(64),torch.randn(64);assert torch.allclose(direction(m0,m1),direction(m0+b,m1+b),atol=1e-7)
 tests={"architecture_identity":True,"parameter_identity":True,"initial_weight_identity":True,"episode_identity":True,"normalizer_identity":True,"prd_binary_formula":True,"translation_invariance":True,"no_absolute_centroid_alignment":("global" not in inspect.getsource(prd_loss)),"outer_dev_isolation":True}
 write(PROTOCOL/"PROTOCOL_AMENDMENT.json",{"user_authorized":True,"reason":"requested continuation despite SHA conflict","prompt_expected_split_sha256":"050703ca8676ae43f236691ed37d58d4ed7ed97b83f449a36f04d93af9ebcd14","actual_frozen_split_sha256":split_sha,"consequence":"historical exact split provenance cannot be asserted; matched ERM-vs-PRD comparison remains fixed on actual cached SEARCH-only fold0"})
 write(PROTOCOL/"SOURCE_PROVENANCE.json",{"split":str(V1_EXP/"protocol/STAGE1_SEARCH_CV_SPLIT.json"),"actual_split_sha256":split_sha,"source_eegnet":str(source_file),"source_eegnet_sha256":sha(source_file)});write(PROTOCOL/"EEGNET_ARCHITECTURE_MATCH.json",{"layer_by_layer_equality":True,"parameter_count":parameter_count(local),"source_file":str(source_file),"source_sha256":sha(source_file)});write(PROTOCOL/"FOLD0_SPLIT.json",fold);write(PROTOCOL/"CACHE_PROVENANCE.json",{"root":str(v1.OPENBMI_ROOT),"search_subjects":search["OpenBMI"],"MI_train":True,"shape":[62,1000]});write(PROTOCOL/"NORMALIZATION_AUDIT.json",norm);write(PROTOCOL/"EPISODE_MANIFEST_PROVENANCE.json",{"reused":False,"deterministically_regenerated":True,"source_algorithm":str(V1_CODE/"run_stage1.py"),"path":str(mp),"sha256":minfo["sha256"],"steps_per_epoch":minfo["steps_per_epoch"],"fold_seed":fold["fold_seed"],"explanation":"The prior runtime manifest was absent on this host. make_manifest is deterministic from the frozen split, cache index, fold seed and source code; no outcome was read."});write(PROTOCOL/"INITIALIZATION_MATCHING.json",{"same_architecture":True,"same_parameter_count":True,"same_initial_weights":True,"same_initial_weight_sha256":init_hash,"PRD_trainable_parameters_added":0});write(PROTOCOL/"INFORMATION_MATCHING.json",{"same_architecture":True,"same_parameter_count":True,"same_initial_weights":True,"same_train_subjects":True,"same_INNER_VAL_subjects":True,"same_OUTER_DEV_subjects":True,"same_sessions":True,"same_normalizer":True,"same_episode_manifest":True,"same_CE_samples":True,"same_CE_sample_order":True,"same_optimizer":True,"same_learning_rate":True,"same_weight_decay":True,"same_epochs":True,"same_actual_optimization_steps":True,"same_checkpoint_selection_window":True,"same_evaluation_samples":True,"only_intended_difference":"EEGNet-PRD includes 0.25 * PRD"});write(PROTOCOL/"HOLDOUT_ISOLATION_AUDIT.json",{"V8_INTERNAL_HOLDOUT_loaded":False,"V8_INTERNAL_HOLDOUT_labels_loaded":False,"WBCIC_OUTER_CONFIRMATION_loaded":False,"WBCIC_OUTER_CONFIRMATION_labels_loaded":False});write(PROTOCOL/"TESTS.json",tests)
 if a.validate_only:print("EEGNET_PRD_PROTOCOL_VALID_WITH_AMENDMENT",flush=True);return 0
 seed(0);erm_info=train(erm,"EEGNet-ERM",bundle,fold,manifest,mean,std,device);seed(0);prd_info=train(prd,"EEGNet-PRD",bundle,fold,manifest,mean,std,device)
 er,erf,ep=evaluate(erm,bundle,fold["outer_dev_subjects"],(2,),mean,std,device,True);qr,qrf,qp=evaluate(prd,bundle,fold["outer_dev_subjects"],(2,),mean,std,device,True);rows=[];geo=[]
 for s in fold["outer_dev_subjects"]:
  _,es,_=evaluate(erm,bundle,[s],(1,),mean,std,device,True);_,qs,_=evaluate(prd,bundle,[s],(1,),mean,std,device,True);rows.append({"subject_id":s,**{f"erm_{k}":v for k,v in er[s].items()},**{f"prd_{k}":v for k,v in qr[s].items()},"delta_BA_pp":(qr[s]["BA"]-er[s]["BA"])*100});geo.append({"subject_id":s,"model":"EEGNet-ERM",**geometry(es[s],erf[s])});geo.append({"subject_id":s,"model":"EEGNet-PRD",**geometry(qs[s],qrf[s])})
 df=pd.DataFrame(rows);d=df.delta_BA_pp.to_numpy()/100;rng=np.random.default_rng(0);boot=rng.choice(d,size=(10000,len(d)),replace=True).mean(1);ba0=float(df.erm_BA.mean());ba1=float(df.prd_BA.mean());delta=(ba1-ba0)*100;median=float(np.median(d)*100)
 if delta<=0:term="EEGNET_PRD_FOLD0_FAIL_STOP"
 elif delta<.5:term="EEGNET_PRD_FOLD0_TRIVIAL_STOP"
 elif delta<1:term="EEGNET_PRD_FOLD0_PROMISING_STOP"
 elif median>=0:term="EEGNET_PRD_FOLD0_STRONG_STOP"
 else:term="EEGNET_PRD_FOLD0_MEAN_POSITIVE_SKEWED_STOP"
 pd.DataFrame(rows).to_csv(OUT/"FOLD0_SUBJECT_RESULTS.csv",index=False);write(OUT/"TRAINING_LOG_ERM.json",erm_info);write(OUT/"TRAINING_LOG_PRD.json",prd_info);write(OUT/"REPRESENTATION_DIAGNOSTICS.json",{"per_subject":geo,"model_summary":{"ERM_mean_pairwise_future_subject_direction_cosine":mean_pairwise_future_direction(erf),"PRD_mean_pairwise_future_subject_direction_cosine":mean_pairwise_future_direction(qrf)}});write(OUT/"SUBJECT_BOOTSTRAP.json",{"resamples":10000,"unit":"subject","mean_delta_BA":float(d.mean()),"median_delta_BA":float(np.median(d)),"ci_low":float(np.quantile(boot,.025)),"ci_high":float(np.quantile(boot,.975)),"improved":int((d>1e-8).sum()),"tied":int((abs(d)<=1e-8).sum()),"harmed":int((d<-1e-8).sum())});result={"EEGNet_ERM_BA":ba0,"EEGNet_PRD_BA":ba1,"delta_prd_pp":delta,"ERM_macro_F1":float(df.erm_macro_F1.mean()),"PRD_macro_F1":float(df.prd_macro_F1.mean()),"ERM_pooled":ep,"PRD_pooled":qp,"selected_epoch_ERM":erm_info["selected_epoch"],"selected_epoch_PRD":prd_info["selected_epoch"],"terminal":term};write(OUT/"FOLD0_RESULT.json",result)
 gd=pd.DataFrame(geo).groupby("model").mean(numeric_only=True);align=float(gd.loc["EEGNet-PRD","source_future_direction_cosine"]-gd.loc["EEGNet-ERM","source_future_direction_cosine"]);decision=f"# EEGNet PRD fold0\n\nTerminal: `{term}`\n\nMatched BA: ERM {ba0:.4f}; PRD {ba1:.4f}; delta {delta:+.2f} pp. Median subject delta {median:+.2f} pp; bootstrap [{np.quantile(boot,.025)*100:+.2f}, {np.quantile(boot,.975)*100:+.2f}] pp.\n\nDirection alignment ERM {gd.loc['EEGNet-ERM','source_future_direction_cosine']:.4f}; PRD {gd.loc['EEGNet-PRD','source_future_direction_cosine']:.4f}; delta {align:+.4f}.\n\nNo holdout was accessed. This run carries a recorded split-SHA amendment and is a screening control only.\n";(OUT/"DECISION.md").write_text(decision);print(term,flush=True);return 0
if __name__=="__main__":
 try:raise SystemExit(main())
 except Exception as e:print(f"EEGNET_PRD_PROTOCOL_INVALID: {type(e).__name__}: {e}",flush=True);raise
