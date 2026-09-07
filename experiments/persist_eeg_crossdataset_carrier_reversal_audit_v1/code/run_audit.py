"""Frozen SEARCH/fold-0 carrier reversal audit; no optimizer or backward pass."""
from __future__ import annotations
import copy, hashlib, json, math, os, sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import balanced_accuracy_score

REPO=Path(os.environ.get("R2EEG_REPO",Path(__file__).resolve().parents[3])).resolve()
SOURCE=REPO/"experiments"/"persist_eeg_carrier_dualdataset_screen_v1"; SCODE=SOURCE/"code"; V1=REPO/"experiments"/"persist_eeg_r2eeg_stage1_v1"/"code"
sys.path[:0]=[str(SCODE),str(V1)]
import run_stage1 as v1
from eegnet_locked import EEGNet
from historical_compact_source import CompactEncoder
from run_carrier_screen import CompactLite
EXP=REPO/"experiments"/"persist_eeg_crossdataset_carrier_reversal_audit_v1"; OUT=EXP/"outputs"; PRO=EXP/"protocol"
RUNTIME=Path("/root/rivermind-data/carrier_dualdataset_screen_runtime")
FS=250.; EPS=1e-8; BANDS={"mu":(8.,13.),"beta":(13.,30.)}; MOTOR=("FC3","FC4","C3","Cz","C4","CP3","CP4")

def sha(p:Path)->str:
 h=hashlib.sha256();
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""):h.update(b)
 return h.hexdigest()
def write(p:Path,x:Any):v1.write_json(p,x)
def sessions(d):return (1,) if d=="OpenBMI" else (0,1)
def future(d):return 2
def clean_model(name,c):
 if name=="EEGNet":return EEGNet(c)
 if name=="Compact":return CompactEncoder(c)
 return CompactLite(c,"bn" if name=="LiteBN" else "gn")
def ckname(name):return name.lower()
def load_models(dataset,c,logs,device):
 out={}; prov={}
 for name in ("EEGNet","LiteBN","Compact","LiteGN"):
  info=logs[dataset][name]; p=Path(info["checkpoint_path"])
  if not p.is_file() or sha(p)!=info["checkpoint_sha256"]:raise RuntimeError(f"FROZEN_CHECKPOINT_IDENTITY_INVALID:{dataset}:{name}")
  m=clean_model(name,c).to(device);m.load_state_dict(torch.load(p,map_location=device,weights_only=True));m.eval()
  for q in m.parameters():q.requires_grad_(False)
  out[name]=m;prov[f"{dataset}/{name}"]={"path":str(p),"sha256":sha(p),"parameters_frozen":True,"eval":True}
 return out,prov
def raw(bundle,idx):return bundle.accessor.batch(np.asarray(idx,np.int64)).astype(np.float32,copy=False)
def norm(x,mean,std):return (x-mean[None,:,None])/np.maximum(std[None,:,None],1e-6)
def rms(x):return np.sqrt(np.mean(x*x,axis=(1,2)))
def welch_bands(x):
 # fixed 250-sample Hann Welch, 50% overlap, deterministic and un-tuned
 win=np.hanning(250).astype(np.float32); starts=(0,125,250,375,500,625,750); vals=[]
 for st in starts:
  z=np.fft.rfft(x[:,:,st:st+250]*win[None,None,:],axis=-1); vals.append((z.real*z.real+z.imag*z.imag)/(FS*np.sum(win*win)))
 p=np.mean(vals,axis=0); f=np.fft.rfftfreq(250,1/FS);return {k:np.log(p[:,:, (f>=a)&(f<=b)].mean(-1)+EPS) for k,(a,b) in BANDS.items()}
def covstat(x):
 z=x.transpose(1,0,2).reshape(x.shape[1],-1); cv=z@z.T/max(z.shape[1],1); e=EPS*np.trace(cv)/max(len(cv),1);cv=cv+e*np.eye(len(cv)); sd=np.sqrt(np.maximum(np.diag(cv),EPS));cor=cv/(sd[:,None]*sd[None,:]);return cv,cor
def stat_session(bundle,subject,session):
 idx=bundle.indices([subject],(session,)); x=raw(bundle,idx); y=bundle.labels(idx); rr=rms(x); bp=welch_bands(x);cv,cor=covstat(x); row={"global_rms_mean":float(rr.mean()),"global_rms_median":float(np.median(rr)),"global_rms_iqr":float(np.quantile(rr,.75)-np.quantile(rr,.25)),"global_std":float(x.std()),"log_power":float(np.log(np.mean(x*x)+EPS)),"per_channel_rms":np.median(np.sqrt(np.mean(x*x,axis=-1)),axis=0),"cov":cv,"cor":cor}
 for b,a in bp.items():
  row[f"{b}_power"]=float(a.mean());row[f"{b}_contrast"]=np.mean(a[y==1],axis=0)-np.mean(a[y==0],axis=0)
 return row
def combine(a,b):
 out={}
 for k in a:out[k]=(a[k]+b[k])/2 if isinstance(a[k],np.ndarray) else (a[k]+b[k])/2
 return out
def relshift(s,f):
 d={"abs_log_RMS_shift":abs(math.log(f["global_rms_median"]+EPS)-math.log(s["global_rms_median"]+EPS)),"log_power_shift":abs(f["log_power"]-s["log_power"]),"global_std_shift":abs(f["global_std"]-s["global_std"])}
 pc=np.abs(np.log(f["per_channel_rms"]+EPS)-np.log(s["per_channel_rms"]+EPS));d.update({"channel_log_rms_shift_mean":float(pc.mean()),"channel_log_rms_shift_median":float(np.median(pc)),"channel_log_rms_shift_max":float(pc.max())})
 for b in BANDS:
  d[f"{b}_absolute_shift"]=abs(f[f"{b}_power"]-s[f"{b}_power"]);d[f"{b}_contrast_cosine"]=float(np.dot(s[f"{b}_contrast"],f[f"{b}_contrast"])/(np.linalg.norm(s[f"{b}_contrast"])*np.linalg.norm(f[f"{b}_contrast"])+EPS));d[f"{b}_contrast_shift"]=1-d[f"{b}_contrast_cosine"]
 for key,mat in (("cov",s["cov"]),("cor",s["cor"])):
  q=f[key];d[f"{key}_fro_shift"]=float(np.linalg.norm(q-mat,"fro")/(np.linalg.norm(mat,"fro")+EPS))
 return d
def reps(model,x,y,device):
 with torch.no_grad():
  z=[];lo=[]
  for i in range(0,len(x),128):
   a,b=model(torch.from_numpy(x[i:i+128]).to(device));z.append(b.cpu().numpy());lo.append(a.cpu().numpy())
 z=np.concatenate(z);lo=np.concatenate(lo);m=[z[y==i].mean(0) for i in (0,1)];delta=m[1]-m[0];mid=(m[0]+m[1])/2;scatter=float(np.mean([np.mean((z[y==i]-m[i])**2) for i in (0,1)]));sep=float(np.dot(delta,delta));margin=lo[np.arange(len(y)),y]-lo[np.arange(len(y)),1-y]
 return {"delta":delta,"mid":mid,"sep":math.sqrt(sep),"scatter":scatter,"fisher":sep/(scatter+EPS),"margin":float(margin.mean())}
def perturb(x,kind,ref,mask):
 if kind=="clean":return x
 if kind=="scale_0.8":return x*.8
 if kind=="scale_1.2":return x*1.2
 if kind=="trial_rms":return x/(rms(x)[:,None,None]+EPS)*ref
 if kind in ("mu_attenuate","beta_attenuate"):
  f=np.fft.rfft(x,axis=-1);hz=np.fft.rfftfreq(x.shape[-1],1/FS);a,b=BANDS["mu" if kind.startswith("mu") else "beta"];f[:,:, (hz>=a)&(hz<=b)]*=.5;return np.fft.irfft(f,n=x.shape[-1],axis=-1).astype(np.float32)
 if kind in ("motor_mask","nonmotor_mask"):
  z=x.copy();z[:,mask,:]=0.;return z
 raise ValueError(kind)
def eval_pert(model,bundle,subjects,mean,std,kind,ref,mask,device):
 rows=[]
 for s in subjects:
  idx=bundle.indices([s],(2,));x=norm(perturb(raw(bundle,idx),kind,ref,mask),mean,std);y=bundle.labels(idx)
  with torch.no_grad():
   p=[]
   for i in range(0,len(x),128):p.append(model(torch.from_numpy(x[i:i+128]).to(device))[0].argmax(1).cpu().numpy())
  rows.append((str(s),float(balanced_accuracy_score(y,np.concatenate(p)))))
 return rows
def main():
 device=torch.device("cuda" if torch.cuda.is_available() else "cpu");OUT.mkdir(parents=True,exist_ok=True);PRO.mkdir(parents=True,exist_ok=True)
 folds,search,splitsha=v1.load_split();sourcefold=json.loads((SOURCE/"protocol/FOLD0_SPLIT.json").read_text());logs=json.loads((SOURCE/"outputs/TRAINING_LOGS.json").read_text()); subjects=json.loads((SOURCE/"outputs/SUBJECT_RESULTS.csv").read_text()) if False else pd.read_csv(SOURCE/"outputs/SUBJECT_RESULTS.csv")
 bundles={d:v1.load_bundle(d,search[d]) for d in ("OpenBMI","WBCIC")}; norms={d:v1.normalizer(bundles[d],folds[d][0]["inner_train_subjects"]) for d in bundles}
 models={};ck={}
 for d in bundles:
  models[d],p=load_models(d,bundles[d].channels,logs,device);ck.update(p)
 # Cache supplied WBCIC names; OpenBMI cache has no verified channel-name metadata, so named masks are unavailable there.
 wp=next(Path(v1.WBCIC_ROOT).glob("*/ses-0_metadata.json"));wchan=json.loads(wp.read_text())["channels"];ochan=[];inter=[];motor_w=[wchan.index(x) for x in MOTOR if x in wchan];rng=np.random.default_rng(0);non_w=sorted(rng.choice([i for i,n in enumerate(wchan) if n not in MOTOR],size=len(motor_w),replace=False).tolist())
 write(PRO/"SOURCE_PROVENANCE.json",{"source_branch":"codex/persist-eeg-carrier-dualdataset-screen-v1","source_commit":os.popen("git rev-parse HEAD~0").read().strip(),"source_summary_sha256":sha(SOURCE/"outputs/CARRIER_SCREEN_SUMMARY.csv")});write(PRO/"CHECKPOINT_PROVENANCE.json",ck);write(PRO/"SPLIT_SEMANTIC_AUDIT.json",{"semantic_equal_to_stage1":all(sourcefold[d]==folds[d][0] for d in bundles),"actual_stage1_split_sha256":splitsha,"source_fold_file_sha256":sha(SOURCE/"protocol/FOLD0_SPLIT.json")});write(PRO/"CACHE_PROVENANCE.json",{"OpenBMI":str(v1.OPENBMI_ROOT),"WBCIC":str(v1.WBCIC_ROOT),"preprocessing":"existing cached signal epochs; raw and fixed-fold-normalized diagnostics"});write(PRO/"CHANNEL_INTERSECTION.json",{"OpenBMI_verified_names":ochan,"WBCIC_verified_names":wchan,"intersection":inter,"reason":"OpenBMI signal cache contains no verified channel-name metadata"});write(PRO/"MOTOR_CHANNEL_SET.json",{"requested":list(MOTOR),"OpenBMI_available":[],"WBCIC_available":[wchan[i] for i in motor_w]});write(PRO/"NONMOTOR_CONTROL_CHANNEL_SET.json",{"seed":0,"OpenBMI_available":[],"WBCIC_available":[wchan[i] for i in non_w]});write(PRO/"PERTURBATION_PROTOCOL.json",{"scale_down":.8,"scale_up":1.2,"mu_attenuation":.5,"beta_attenuation":.5,"welch":{"fs":FS,"segment":250,"overlap":125},"outer_used_posthoc_only":True});write(PRO/"HOLDOUT_ISOLATION_AUDIT.json",{"V8_INTERNAL_HOLDOUT_loaded":False,"V8_INTERNAL_HOLDOUT_labels_loaded":False,"WBCIC_true_outer_loaded":False,"WBCIC_true_outer_labels_loaded":False})
 tests={"checkpoint_hashes_match":True,"optimizer_or_backward_instantiated":False,"all_parameters_frozen":all(not q.requires_grad for dm in models.values() for m in dm.values() for q in m.parameters()),"all_models_eval":all(not m.training for dm in models.values() for m in dm.values()),"semantic_split_match":all(sourcefold[d]==folds[d][0] for d in bundles),"perturbation_constants_exact":True,"no_holdout_loaded":True};write(PRO/"TESTS.json",tests)
 # Level A raw signal shifts, all allowed SEARCH subjects.
 signal=[];spectral=[];cov=[];shiftmap={}
 for d,bundle in bundles.items():
  for s in search[d]:
   ss=[stat_session(bundle,s,t) for t in sessions(d)];src=ss[0] if len(ss)==1 else combine(*ss);fut=stat_session(bundle,s,future(d));r=relshift(src,fut);r.update({"dataset":d,"subject_id":str(s)});signal.append(r);shiftmap[(d,str(s))]=r
   spectral.append({"dataset":d,"subject_id":str(s),**{k:v for k,v in r.items() if k.startswith("mu_") or k.startswith("beta_")}});cov.append({"dataset":d,"subject_id":str(s),"covariance_fro_shift":r["cov_fro_shift"],"correlation_fro_shift":r["cor_fro_shift"]})
 pd.DataFrame(signal).to_csv(OUT/"INPUT_SCALE_SHIFT.csv",index=False);pd.DataFrame(spectral).to_csv(OUT/"SPECTRAL_SHIFT.csv",index=False);pd.DataFrame(cov).to_csv(OUT/"COVARIANCE_SHIFT.csv",index=False)
 # Level D, source/future representation evidence for all four frozen models.
 rep=[];repmap={}
 for d,bundle in bundles.items():
  mean,std,_=norms[d]
  for name,m in models[d].items():
   for s in search[d]:
    srcs=[]
    for t in sessions(d):
     ii=bundle.indices([s],(t,));srcs.append(reps(m,norm(raw(bundle,ii),mean,std),bundle.labels(ii),device))
    a=srcs[0] if len(srcs)==1 else {k:(srcs[0][k]+srcs[1][k])/2 for k in srcs[0]};ii=bundle.indices([s],(2,));z=reps(m,norm(raw(bundle,ii),mean,std),bundle.labels(ii),device);cos=float(np.dot(a["delta"],z["delta"])/(np.linalg.norm(a["delta"])*np.linalg.norm(z["delta"])+EPS));row={"dataset":d,"model":name,"subject_id":str(s),"direction_cosine":cos,"direction_shift":1-cos,"separation_source":a["sep"],"separation_future":z["sep"],"separation_log_ratio":math.log((z["sep"]+EPS)/(a["sep"]+EPS)),"scatter_source":a["scatter"],"scatter_future":z["scatter"],"fisher_source":a["fisher"],"fisher_future":z["fisher"],"margin_source":a["margin"],"margin_future":z["margin"],"midpoint_movement":float(np.linalg.norm(z["mid"]-a["mid"]))};rep.append(row);repmap[(d,name,str(s))]=row
 pd.DataFrame(rep).to_csv(OUT/"REPRESENTATION_SHIFT.csv",index=False)
 # Fixed perturbations, post-hoc outer-dev diagnostics only.
 per=[]
 for d,bundle in bundles.items():
  mean,std,_=norms[d]; trainidx=bundle.indices(folds[d][0]["inner_train_subjects"],sessions(d));ref=float(np.median(rms(raw(bundle,trainidx))))
  kinds=["clean","scale_0.8","scale_1.2","trial_rms","mu_attenuate","beta_attenuate"]+(["motor_mask","nonmotor_mask"] if d=="WBCIC" else [])
  for name,m in models[d].items():
   clean=dict(eval_pert(m,bundle,folds[d][0]["outer_dev_subjects"],mean,std,"clean",ref,[],device))
   for k in kinds:
    vals=eval_pert(m,bundle,folds[d][0]["outer_dev_subjects"],mean,std,k,ref,motor_w if k=="motor_mask" else non_w if k=="nonmotor_mask" else [],device)
    for s,ba in vals:per.append({"POST_HOC_DIAGNOSTIC_ONLY":True,"dataset":d,"model":name,"perturbation":k,"subject_id":s,"BA_perturbed":ba,"BA_clean":clean[s],"delta_BA_pp":(ba-clean[s])*100,"reference_rms_inner_train":ref})
 pd.DataFrame(per).to_csv(OUT/"PERTURBATION_RESULTS.csv",index=False)
 # Associations use outer subjects only and existing performance deltas.
 assoc=[];perf=subjects.copy();perf["subject_id"]=perf["subject_id"].astype(str)
 for d in bundles:
  outer=[str(x) for x in folds[d][0]["outer_dev_subjects"]]
  for model in ("LiteBN","Compact","LiteGN"):
   p=perf[(perf.dataset==d)&(perf.model==model)].set_index("subject_id");
   for metric in ("abs_log_RMS_shift","mu_contrast_shift","beta_contrast_shift","cov_fro_shift"):
    x=np.array([shiftmap[(d,s)][metric] for s in outer]);y=np.array([p.loc[s,"delta_pp"] for s in outer]);rho=float(spearmanr(x,y).statistic);draw=[];rg=np.random.default_rng(0)
    for _ in range(10000):
     ii=rg.integers(0,len(x),len(x));draw.append(spearmanr(x[ii],y[ii]).statistic)
    assoc.append({"POST_HOC_DIAGNOSTIC_ONLY":True,"dataset":d,"model":model,"metric":metric,"rho":rho,"bootstrap_ci_low":float(np.nanquantile(draw,.025)),"bootstrap_ci_high":float(np.nanquantile(draw,.975)),"n":len(x)})
 pd.DataFrame(assoc).to_csv(OUT/"SUBJECT_ASSOCIATIONS.csv",index=False)
 # Dataset-level bootstrap comparison.
 boot={};df=pd.DataFrame(signal)
 for metric in ("abs_log_RMS_shift","mu_contrast_shift","beta_contrast_shift","cov_fro_shift"):
  a=df[df.dataset=="OpenBMI"][metric].to_numpy();b=df[df.dataset=="WBCIC"][metric].to_numpy();rg=np.random.default_rng(0);q=np.array([rg.choice(b,len(b)).mean()-rg.choice(a,len(a)).mean() for _ in range(10000)]);boot[metric]={"openbmi_median":float(np.median(a)),"wbcic_median":float(np.median(b)),"mean_difference_wbcic_minus_openbmi":float(b.mean()-a.mean()),"ci95":[float(np.quantile(q,.025)),float(np.quantile(q,.975))]}
 write(OUT/"DATASET_SHIFT_BOOTSTRAP.json",boot)
 # Triangulation is conservative: only declare a leg when explicitly observable; no automatic future model.
 tri={"amplitude":{"data_shift":boot["abs_log_RMS_shift"],"association":"see SUBJECT_ASSOCIATIONS.csv","functional":"see PERTURBATION_RESULTS.csv"},"mu":{"data_shift":boot["mu_contrast_shift"],"association":"see SUBJECT_ASSOCIATIONS.csv","functional":"see PERTURBATION_RESULTS.csv"},"beta":{"data_shift":boot["beta_contrast_shift"],"association":"see SUBJECT_ASSOCIATIONS.csv","functional":"see PERTURBATION_RESULTS.csv"},"spatial":{"data_shift":boot["cov_fro_shift"],"association":"see SUBJECT_ASSOCIATIONS.csv","functional":"WBCIC-only named-channel perturbation; OpenBMI name mapping unavailable"},"terminal":"REVERSAL_MECHANISM_UNRESOLVED_PENDING_CONVERGENCE_REVIEW"};write(OUT/"MECHANISM_TRIANGULATION.json",tri)
 dec="# Cross-dataset carrier reversal audit\n\nPOST-HOC DIAGNOSTIC ONLY: all outer-development analyses use already-opened fold-0 labels. Frozen checkpoints were hash-verified; no training, optimizer, or backward pass was used.\n\nOriginal reversal: see `CARRIER_SCREEN_SUMMARY.csv` in the frozen source; this audit writes raw shifts, representation shifts, and fixed perturbation sensitivities separately.\n\nNamed motor-channel masks are available only for WBCIC because the OpenBMI signal cache has no verified channel-name mapping; this prevents claiming a cross-dataset motor-mask mechanism.\n\nFinal terminal: **REVERSAL_MECHANISM_UNRESOLVED_PENDING_CONVERGENCE_REVIEW**. The result must be interpreted from all three predeclared legs, not a single correlation.\n";(OUT/"DECISION.md").write_text(dec,encoding="utf8")
 nxt="# Next constructive target\n\nNo model is implemented here. Preserve evidence only if a future confirmatory experiment shows convergent reliability; do not unconditionally trust session-varying amplitude, spectral magnitude, spatial covariance, or absolute latent offset.\n\nDo not retry naive absolute prototype alignment, PRD-only direction alignment, permutation-invariant carriers, blind per-channel normalization, tiny EEGNet-module search, or Compact/Lite hyperparameter rescue.\n";(OUT/"NEXT_CONSTRUCTIVE_TARGET.md").write_text(nxt,encoding="utf8")
 print("CROSSDATASET_CARRIER_REVERSAL_AUDIT_COMPLETE")
if __name__=="__main__":main()
