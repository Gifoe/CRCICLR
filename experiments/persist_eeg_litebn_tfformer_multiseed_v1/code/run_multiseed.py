#!/usr/bin/env python3
"""Run the frozen TFFormer architecture for one requested multiseed replicate."""
from __future__ import annotations
import copy, gc, hashlib, importlib.util, json, math, os, random, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from cached_tfformer import CachedTFFormer

REPO=Path(os.environ.get("TFFREM_REPO","/root/rivermind-data/CRCICLR_TFF_REMAIN_WORK")).resolve()
EXP=REPO/"experiments/persist_eeg_litebn_tfformer_multiseed_v1"
SEED=int(os.environ.get("TFFREM_SEED","1"))
if SEED not in (1,2):raise RuntimeError("TFFREM_SEED must be 1 or 2")
OUT=EXP/"outputs"/f"seed{SEED}";RUN=EXP/"runtime"/f"seed{SEED}";PROTOCOL=EXP/"protocol"/f"seed{SEED}"
BASE_RUNNER=REPO/"experiments/persist_eeg_linux_w0r0_screen_v1/code/run_w0r0_seed0.py";TASKS=("OpenBMI_SSVEP","OpenBMI_ERP","OpenBMI_MI","WBCIC_MI");FS=250;EPOCHS=60
HELDOUT={"OpenBMI":["4","12","13","17","18","24","25","29","36","37","39","42","51","54"],"WBCIC":["sub-2","sub-3","sub-17","sub-19","sub-21","sub-25","sub-31","sub-33","sub-38","sub-42"]}
BENCH={"OpenBMI_SSVEP":.9234,"OpenBMI_ERP":.8557,"OpenBMI_MI":.7555,"WBCIC_MI":.7918}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(8<<20),b""):h.update(b)
 return h.hexdigest()
def js(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+".part");q.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+"\n");os.replace(q,p)
def csv(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+".part");pd.DataFrame(x).to_csv(q,index=False);os.replace(q,p)
def seed(x):random.seed(x);np.random.seed(x);torch.manual_seed(x);torch.cuda.manual_seed_all(x)
def rng():return {"py":random.getstate(),"np":np.random.get_state(),"th":torch.get_rng_state(),"cu":torch.cuda.get_rng_state_all()}
def restore(x):random.setstate(x["py"]);np.random.set_state(x["np"]);torch.set_rng_state(x["th"].cpu());torch.cuda.set_rng_state_all([v.cpu() for v in x["cu"]])
def state(m):return {k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
def load():
 os.environ["W0R0_REPO"]=str(REPO);s=importlib.util.spec_from_file_location("tffrem_base",BASE_RUNNER);r=importlib.util.module_from_spec(s);sys.modules[s.name]=r;s.loader.exec_module(r);b,_=r.load_modules();b.RUNTIME=RUN;b.SEED=SEED;return b,r
base,runner=load()
def baseline_path(task,fold):
 if task in ("OpenBMI_ERP","OpenBMI_SSVEP"):
  short="erp" if task.endswith("ERP") else "ssvep";return runner.TASK_RUNTIME/f"{short}_fold{fold}_seed{SEED}_litebn/selected_best.pt"
 short="openbmi" if task=="OpenBMI_MI" else "wbcic";return runner.CARRIER_RUNTIME/f"{short}_fold{fold}_seed{SEED}_litebn/selected_best.pt"
def model(task,fold,device):
 p=baseline_path(task,fold);b=base.build_model("LiteBN_BASELINE",task);b.load_state_dict(torch.load(p,map_location="cpu",weights_only=False),strict=True);seed(SEED);return CachedTFFormer(b,FS).to(device),p

class Cache:
 def __init__(self,raw,mean,std,device):
  self.raw,self.mean,self.std,self.device,self.n=raw,mean,std,device,int(raw.x.shape[0]);self.parts=None;t=time.perf_counter()
  with torch.no_grad():
   for start in range(0,self.n,32):
    ids=np.arange(start,min(start+32,self.n));x,_=raw.batch(ids,mean,std);cur=[]
    for sec in (.25,.5,1.0):
     w=round(FS*sec);n=1<<(w-1).bit_length();z=torch.stft(x.float().reshape(-1,x.shape[-1]),n_fft=n,hop_length=max(1,round(w/4)),win_length=w,window=torch.hann_window(w,device=device),center=True,return_complex=True).abs().log1p();z=z.reshape(x.shape[0],x.shape[1],z.shape[-2],z.shape[-1]);f=torch.fft.rfftfreq(n,1/FS).to(device);cur.append(z[:,:,(f>=1)&(f<=45)].cpu())
    if self.parts is None:self.parts=[torch.empty((self.n,*z.shape[1:]),dtype=z.dtype) for z in cur]
    for a,z in zip(self.parts,cur):a[start:start+len(ids)].copy_(z)
  self.seconds=time.perf_counter()-t;self.bytes=sum(x.numel()*x.element_size() for x in self.parts)
 def batch(self,ids):
  ids=np.asarray(ids,dtype=np.int64);x,y=self.raw.batch(ids,self.mean,self.std);ii=torch.from_numpy(ids);return x,y,[z.index_select(0,ii).to(self.device) for z in self.parts]

def evalm(m,b,c,subjects):
 m.eval();rows={};pred={}
 with torch.no_grad():
  for s in subjects:
   ids=b.indices([s],(int(base.TASKS[b.task]["future_session"]),));zs=[]
   for q in range(0,len(ids),128):x,_,st=c.batch(ids[q:q+128]);zs.append(m(x,st)[0].float().cpu().numpy())
   y=b.labels(ids);z=np.concatenate(zs);rows[str(s)]=base.classification_metrics(y,z);pred[str(s)]=(y,z.argmax(1))
 return float(np.mean([x["BA"] for x in rows.values()])),rows,pred
def evalb0(m,b,raw,subjects,mean,std):
 m.eval();rows={}
 with torch.no_grad():
  for s in subjects:
   ids=b.indices([s],(int(base.TASKS[b.task]["future_session"]),));zs=[]
   for q in range(0,len(ids),128):x,_=raw.batch(ids[q:q+128],mean,std);zs.append(m(x)[0].float().cpu().numpy())
   rows[str(s)]=base.classification_metrics(b.labels(ids),np.concatenate(zs))
 return rows
def groups(m):
 new=[];high=[];stem=[]
 for n,p in m.named_parameters():
  if not p.requires_grad:continue
  if n.startswith(("spectral.","cross.","g1.","g2.","conv.")):new.append(p)
  elif n.startswith(("base.temporal","base.spatial")):stem.append(p)
  else:high.append(p)
 return new,high,stem
def train(task,fold,b,c,weight,device):
 fi=int(fold["fold_id"]);d=RUN/"checkpoints"/task/f"fold{fi}";latest=d/"latest.pt";selected=d/"selected.pt";m,bp=model(task,fi,device);new,high,stem=groups(m);opt=torch.optim.AdamW([{"params":new,"lr":3e-4},{"params":high,"lr":5e-5},{"params":stem,"lr":1e-5}],weight_decay=5e-4);inv={"task":task,"fold":fi,"seed":SEED,"base_sha":sha(bp),"source_sha":sha(Path(__file__).with_name("cached_tfformer.py"))};start=1;hist=[];best=-1.;be=None;bs=None
 if latest.is_file():
  q=torch.load(latest,map_location="cpu",weights_only=False)
  if q["inv"]!=inv:raise RuntimeError("resume mismatch")
  m.load_state_dict(q["state"]);opt.load_state_dict(q["opt"]);restore(q["rng"]);start=q["epoch"]+1;hist=q["hist"];best=q["best"];be=q["be"];bs=q["bs"]
 trainids=b.indices(fold["inner_train_subjects"],base.TASKS[task]["source_sessions"]);episodes=base.mi_manifest(b,fold,task)[0] if base.TASKS[task]["mi_protocol"] else None;w=None if weight is None else weight.to(device);t=time.perf_counter();early_stopped=False;last_epoch=start-1
 for e in range(start,EPOCHS+1):
  last_epoch=e
  m.train();fac=e/5 if e<=5 else .05+.95*.5*(1+math.cos(math.pi*(e-5)/55))
  for g,lr in zip(opt.param_groups,(3e-4,5e-5,1e-5)):g["lr"]=lr*fac
  batches=episodes[e-1] if episodes is not None else base.task_epoch_batches(trainids,task,fi,e);losses=[];gn=[]
  for ids in batches:
   x,y,st=c.batch(ids);opt.zero_grad(set_to_none=True);z,_=m(x,st);loss=F.cross_entropy(z,y,weight=w);loss.backward();gn.append(float(torch.nn.utils.clip_grad_norm_(new+high+stem,5)));opt.step();losses.append(float(loss.detach()))
  va,_,_=evalm(m,b,c,fold["inner_val_subjects"]);chose=va>best+1e-12
  if chose:best,be,bs=va,e,state(m)
  hist.append({"seed":SEED,"task":task,"fold":fi,"epoch":e,"train_loss":float(np.mean(losses)),"inner_val_BA":va,"selected":chose,"lr_N":opt.param_groups[0]["lr"],"lr_B":opt.param_groups[1]["lr"],"lr_S":opt.param_groups[2]["lr"],"grad_norm":float(np.mean(gn)),"cross_entropy":float(m.cross.ent),"gqa1_entropy":float(m.g1.ent),"gqa2_entropy":float(m.g2.ent)})
  if e==1 or e%5==0 or chose:print(f"TFFREM seed{SEED} {task} f{fi} e{e:02d} BA={va:.6f}",flush=True)
  if e%5==0:
   d.mkdir(parents=True,exist_ok=True);tmp=latest.with_suffix(".pt.part");torch.save({"epoch":e,"state":state(m),"opt":opt.state_dict(),"rng":rng(),"hist":hist,"best":best,"be":be,"bs":bs,"inv":inv},tmp);os.replace(tmp,latest)
  if e>=10 and be is not None and e-be>=8:
   early_stopped=True;print(f"TFFREM_EARLY_STOP seed{SEED} {task} f{fi} epoch={e} best_epoch={be} patience=8",flush=True);break
 m.load_state_dict(bs);replay,_,_=evalm(m,b,c,fold["inner_val_subjects"])
 if abs(replay-best)>1e-12:raise RuntimeError("replay mismatch")
 tmp=selected.with_suffix(".pt.part");torch.save({"state_dict":state(m),"epoch":be,"inner_val_BA":best,"inv":inv},tmp);os.replace(tmp,selected);return {"seed":SEED,"task":task,"fold":fi,"selected_epoch":be,"inner_val_BA":best,"epochs_ran":last_epoch,"early_stopped":early_stopped,"checkpoint":str(selected),"checkpoint_sha":sha(selected),"elapsed_seconds":time.perf_counter()-t,"history":hist}

def heldout(task,folds,records,device):
 subjects=HELDOUT[base.TASKS[task]["dataset"]];rows=[]
 for f in folds:
  fi=int(f["fold_id"]);b=base.build_bundle(task,subjects);mean,std,_=base.load_tensor_pair(runner.normalizer_source(task,fi));raw=base.RawGPUCache(b,device);c=Cache(raw,mean,std,device);m,_=model(task,fi,device);q=torch.load(RUN/f"checkpoints/{task}/fold{fi}/selected.pt",map_location=device,weights_only=False);m.load_state_dict(q["state_dict"]);_,mr,_=evalm(m,b,c,subjects);b0=base.build_model("LiteBN_BASELINE",task);b0.load_state_dict(torch.load(baseline_path(task,fi),map_location="cpu",weights_only=False),strict=True);br=evalb0(b0.to(device),b,raw,subjects,mean,std)
  for s in subjects:rows.append({"seed":SEED,"task":task,"fold":fi,"subject_id":s,"TFFormer_BA":mr[s]["BA"],"LiteBN_BA":br[s]["BA"],"delta_pp":100*(mr[s]["BA"]-br[s]["BA"])})
  print(f"TFFREM_HELDOUT seed{SEED} {task} f{fi}",flush=True);del m,b0,c,raw,b;gc.collect();torch.cuda.empty_cache()
 frame=pd.DataFrame(rows);per=frame.groupby("subject_id",as_index=False).mean(numeric_only=True);return rows,{"seed":SEED,"task":task,"TFFormer":float(per.TFFormer_BA.mean()),"LiteBN":float(per.LiteBN_BA.mean()),"delta_vs_LiteBN_pp":float((per.TFFormer_BA-per.LiteBN_BA).mean()*100),"benchmark":BENCH[task],"delta_vs_benchmark_pp":float((per.TFFormer_BA.mean()-BENCH[task])*100),"PASS":bool(per.TFFormer_BA.mean()>BENCH[task])}
def report(results):
 cols=("seed","task","TFFormer","LiteBN","delta_vs_LiteBN_pp","benchmark","delta_vs_benchmark_pp","PASS")
 lines=[f"# TFFormer seed{SEED} four-task evaluation","","| "+" | ".join(cols)+" |","|"+"|".join(["---"]*len(cols))+"|"]
 for row in results:lines.append("| "+" | ".join(str(row[c]) for c in cols)+" |")
 lines += ["","Internal heldout diagnostic only. New sealed test was not accessed.",""]
 (OUT/"FINAL_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
def main():
 if not torch.cuda.is_available():raise RuntimeError("CUDA required")
 for p in (OUT,RUN,PROTOCOL):p.mkdir(parents=True,exist_ok=True)
 js(PROTOCOL/"SCOPE.json",{"seed":SEED,"tasks":TASKS,"status":"user-requested four-task multiseed extension","checkpoint_selection":"inner validation only","heldout_after_five_checkpoints":True,"effective_fs":250,"early_stopping":{"applies_to":"all seed1/2 folds","minimum_epochs":10,"patience_without_strict_inner_val_BA_improvement":8,"user_revision":True}});device=torch.device("cuda");_,foldmap,_=base.load_folds();audit=[]
 for task in TASKS:
  for f in foldmap[base.TASKS[task]["dataset"]]:
   fi=int(f["fold_id"]);p=baseline_path(task,fi);b0=base.build_model("LiteBN_BASELINE",task);b0.load_state_dict(torch.load(p,map_location="cpu",weights_only=False),strict=True);audit.append({"seed":SEED,"task":task,"fold":fi,"path":str(p),"sha256":sha(p),"strict_load":True})
 csv(PROTOCOL/"BASELINE_AUDIT.csv",audit);rp=RUN/"TRAINING_LOGS.json";records=json.loads(rp.read_text()) if rp.is_file() else [];results=[];heldrows=[]
 for task in TASKS:
  folds=foldmap[base.TASKS[task]["dataset"]]
  for f in folds:
   if any(r["task"]==task and int(r["fold"])==int(f["fold_id"]) for r in records):continue
   b=base.build_bundle(task,f["inner_train_subjects"]+f["inner_val_subjects"]);mean,std,_=base.load_tensor_pair(runner.normalizer_source(task,int(f["fold_id"])));raw=base.RawGPUCache(b,device);c=Cache(raw,mean,std,device);w,_=base.class_weights(b,f["inner_train_subjects"]);print(f"TFFREM_CACHE seed{SEED} {task} f{f['fold_id']} GiB={c.bytes/2**30:.3f}",flush=True);r=train(task,f,b,c,w,device);records.append(r);js(rp,records);csv(OUT/"TRAINING_TRAJECTORY.csv",[x for r in records for x in r["history"]]);csv(OUT/"CHECKPOINT_SELECTION.csv",[{k:v for k,v in r.items() if k!="history"} for r in records]);del c,raw,b;gc.collect();torch.cuda.empty_cache()
  rows,result=heldout(task,folds,records,device);heldrows+=rows;results.append(result);csv(OUT/"HELDOUT_SUBJECT_RESULTS.csv",heldrows);csv(OUT/f"SEED{SEED}_HELDOUT_RESULTS.csv",results);print("TFFREM_TASK_DONE "+json.dumps(result),flush=True)
 decision={"terminal":f"TFFORMER_SEED{SEED}_FOUR_TASK_COMPLETE","seed":SEED,"results":results,"original_SSVEP_gate_overridden_for_user_requested_multiseed_diagnostic":True,"new_sealed_test_accessed":False};js(OUT/"FINAL_DECISION.json",decision);report(results);print("TFFREM_DONE "+json.dumps(decision),flush=True)
if __name__=="__main__":main()
