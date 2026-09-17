#!/usr/bin/env python3
"""Frozen three-seed PERSIST and rank-matched inference; no ablations."""
from __future__ import annotations
import argparse, importlib.util, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ROOT=Path(os.environ.get("BNCI_P4_ROOT","/root/p4_bnci2015_001_seed0_matched_episodic_v1"))
PRIMARY=ROOT/"three_seed_nonablation/primary"; OUT=ROOT/"three_seed_nonablation"
BASE=ROOT/"code/run_seed0_matched_episodic.py"; SEED0=ROOT/"code/run_external_seed0_evidence.py"
SEEDS=(0,1,2)

def imp(name,path):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); sys.modules[name]=m; s.loader.exec_module(m); return m
def csv(path, rows):
 path.parent.mkdir(parents=True,exist_ok=True); t=path.with_suffix(path.suffix+".part"); pd.DataFrame(rows).to_csv(t,index=False); os.replace(t,path)
def js(path,value):
 path.parent.mkdir(parents=True,exist_ok=True); t=path.with_suffix(path.suffix+".part"); t.write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+"\n"); os.replace(t,path)
def records(seed):
 p=ROOT/"seed0_training_records.json" if seed==0 else PRIMARY/"runtime"/f"seed{seed}"/"seed0_training_records.json"
 d=json.loads(p.read_text()); r={(x["model"],int(x["fold"])):x for x in d["records"]}
 if len(r)!=10 or any(int(x["seed"])!=seed for x in r.values()): raise RuntimeError(f"bad records seed{seed}")
 return r
def run():
 base=imp("ms_base",BASE); ev=imp("ms_seed0",SEED0); peeh=imp("ms_peeh",ev.PEEH_SRC); pswa=imp("ms_pswa",ev.PSWA_SRC); rank=imp("ms_rank",ev.RANK_SRC); rs=imp("ms_rank_summary",ev.RANK_SUMMARY_SRC)
 ev.source_audit(); stage,eeg,sire,_=base.load_authoritative(); base.audit_models(eeg,sire); bundle,raw=base.load_bundle(stage); folds,_=base.make_splits(); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 normal={}
 for f in folds: normal[f["fold_id"]]=base.normalize(raw,bundle,f["inner_train_subjects"])
 prows=[]; srows=[]
 for seed in SEEDS:
  rec=records(seed)
  for f in folds:
   fid=int(f["fold_id"]); mean,std,norm=normal[fid]; cache=base.GPUCache(raw,bundle,mean,std,dev); ti=bundle.indices(f["inner_train_subjects"],(1,2)); ei=bundle.indices(f["outer_test_subjects"],(2,)); ai=bundle.indices(f["outer_test_subjects"],(1,2)); tm,em,am=ev.metadata(bundle,ti),ev.metadata(bundle,ei),ev.metadata(bundle,ai)
   for name in ("EEGNet","SIRE-EEG"):
    r=rec[(name,fid)]; model=ev.frozen_model(base,eeg,sire,name,r,dev); th,eh,ah=ev.extract(model,cache,ti),ev.extract(model,cache,ei),ev.extract(model,cache,ai)
    emb={"train_h":th,"train_y":tm.label.to_numpy(int),"train_subject":tm.subject_id.astype(str).to_numpy(),"train_session":tm.session_id.to_numpy(int),"eval_h":eh,"eval_y":em.label.to_numpy(int),"eval_subject":em.subject_id.astype(str).to_numpy(),"eval_session":em.session_id.to_numpy(int),"metadata":{"checkpoint_sha256":r["checkpoint_sha256"],"normalizer_sha256":norm["mean_std_sha256"]}}
    one=ev.empty_peeh(peeh.run_one(emb,"BNCI2015_001",name,fid,seed,2))
    for q in one["subject_rows"]: prows.append({**q,"seed":seed,"checkpoint_sha256":r["checkpoint_sha256"]})
    two=ev.pswa_for_fold(pswa,one,th,tm,ah,am,f["outer_test_subjects"],name,fid)
    for q in two["subject_rows"]: srows.append({**q,"seed":seed,"checkpoint_sha256":r["checkpoint_sha256"]})
    del model
   del cache
   if dev.type=="cuda": torch.cuda.empty_cache()
 csv(OUT/"diagnostics/multiseed_peeh_subject_results.csv",prows); csv(OUT/"diagnostics/multiseed_pswa_subject_results.csv",srows)
 summary=[]
 for name in ("EEGNet","SIRE-EEG"):
  for kind,frame,col,boot in (("PEEH",pd.DataFrame(prows),"PEEH_pp",peeh.bootstrap),("PSWA",pd.DataFrame(srows),"PSWA_pp",pswa.bootstrap)):
   subject_column="subject_id" if kind=="PEEH" else "subject"
   x=frame[frame.model==name].groupby(subject_column)[col].mean().to_numpy(float); ranks=[int(z) for z in frame[frame.model==name].groupby(["seed","fold"]).protected_rank.first()]
   if kind=="PEEH": b=boot(x,peeh.stable_seed("BNCI-3seed",kind,name),20000); mean,lo,hi=b["mean"],b["ci95"][0],b["ci95"][1]
   else: lo,hi=boot(x,pswa.stable_seed("BNCI-3seed",kind,name)); mean=float(x.mean())
   summary.append({"diagnostic":kind,"model":name,"mean_pp":mean,"CI95_low_pp":lo,"CI95_high_pp":hi,"biological_subjects":len(x),"nonempty_fold_checkpoint_coverage":f"{sum(z>0 for z in ranks)}/15","protected_ranks_seed_fold":ranks,"bootstrap_draws":20000})
 csv(OUT/"diagnostics/multiseed_diagnostic_summary.csv",summary)
 # Every replay cell is checked against the just-aggregated frozen Full outputs before intervention.
 expected=pd.read_csv(PRIMARY/"multiseed_subject_metrics.csv"); expected=expected[expected.model=="SIRE-EEG"]
 with np.load(ev.RANK_PROTOCOL/"random_projectors.npz",allow_pickle=False) as z: mats=np.concatenate((z["identity"][None],z["scale"][None],z["projectors"])).astype(np.float32)
 mt=torch.as_tensor(mats,device=dev); replay=[]; cells=[]; identity=0.
 for seed in SEEDS:
  rec=records(seed)
  for f in folds:
   fid=int(f["fold_id"]); mean,std,norm=normal[fid]; cache=base.GPUCache(raw,bundle,mean,std,dev); r=rec[("SIRE-EEG",fid)]; model=ev.frozen_model(base,eeg,sire,"SIRE-EEG",r,dev)
   with torch.inference_mode():
    for subj in f["outer_test_subjects"]:
     for label,sid in (("S1",1),("S2",2)):
      ids=bundle.indices([subj],(sid,)); y=bundle.labels(ids); old=expected[(expected.seed==seed)&(expected.fold==fid)&(expected.subject==subj)&(expected.session==label)].iloc[0]; direct=[]; pred=[[] for _ in range(102)]
      for lo in range(0,len(ids),32):
       x,_=cache.batch(ids[lo:lo+32]); original=model(x)[0]; direct.append(original.argmax(-1).cpu().numpy()); zz=rank.branch_output(model,x); first=rank.projected_logits(model,zz,mt[:2]); identity=max(identity,float((first[0]-original).abs().max())); pred[0].append(first[0].argmax(-1).cpu().numpy()); pred[1].append(first[1].argmax(-1).cpu().numpy())
       for st in range(2,102,20):
        v=rank.projected_logits(model,zz,mt[st:st+20]).argmax(-1).cpu().numpy()
        for j,a in enumerate(v): pred[st+j].append(a)
      ba,f1=ev.metrics(base,y,np.concatenate(direct)); ok=abs(ba-float(old.BA))<=1e-7 and abs(f1-float(old.Macro_F1))<=1e-7; replay.append({"seed":seed,"fold":fid,"subject":subj,"session":label,"pass":ok,"abs_BA_difference":abs(ba-float(old.BA)),"abs_Macro_F1_difference":abs(f1-float(old.Macro_F1))})
      if not ok: raise RuntimeError("replay gate failed")
      for j,a in enumerate(pred):
       ba,f1=ev.metrics(base,y,np.concatenate(a)); cells.append({"seed":seed,"fold":fid,"subject":subj,"session":label,"condition":"Intact" if j==0 else ("ScaleCollapse" if j==1 else "RandomRank16"),"projector_id":-1 if j<2 else j-2,"BA":ba,"Macro_F1":f1})
   del model,cache
   if dev.type=="cuda": torch.cuda.empty_cache()
 if identity>=1e-5: raise RuntimeError("identity wrapper failed")
 csv(OUT/"rankmatched/replay_audit.csv",replay); frame=pd.DataFrame(cells); out=[]; effects=[]
 for subj in base.SUBJECTS:
  part=frame[frame.subject==subj]; vals={}
  for cond in ("Intact","ScaleCollapse"):
   q=part[part.condition==cond]; vals[cond]=(float(q[q.session=="S2"].BA.mean()),float(q.groupby(["seed","session"]).BA.mean().unstack().min(axis=1).mean()))
  rnd=part[part.condition=="RandomRank16"]
  rb=rnd[rnd.session=="S2"].groupby("projector_id").BA.mean().sort_index().to_numpy(); rw=rnd.groupby(["projector_id","seed","session"]).BA.mean().unstack().min(axis=1).groupby("projector_id").mean().sort_index().to_numpy()
  out.append({"subject":subj,"intact_BA":vals["Intact"][0],"ScaleCollapse_BA":vals["ScaleCollapse"][0],"mean_random_BA":float(rb.mean()),"intact_WSBA":vals["Intact"][1],"ScaleCollapse_WSBA":vals["ScaleCollapse"][1],"mean_random_WSBA":float(rw.mean())})
  for metric,a in (("future_S2_BA",rb),("WS_BA",rw)):
   for j,v in enumerate(a): effects.append({"subject":subj,"metric":metric,"projector_id":j,"random_harm_pp":100*((vals["Intact"][0] if metric=="future_S2_BA" else vals["Intact"][1])-v)})
 csv(OUT/"rankmatched/multiseed_subject_metrics.csv",out); ef=pd.DataFrame(effects); sums=[]
 for metric,ic,sc,rc in (("future_S2_BA","intact_BA","ScaleCollapse_BA","mean_random_BA"),("WS_BA","intact_WSBA","ScaleCollapse_WSBA","mean_random_WSBA")):
  q=pd.DataFrame(out); excess=100*(q[rc]-q[sc]); mean,lo,hi=rs.bootstrap(excess,"BNCI-3seed",metric); sh=float((100*(q[ic]-q[sc])).mean()); rh=ef[ef.metric==metric].groupby("projector_id").random_harm_pp.mean().to_numpy(); sums.append({"metric":metric,"ScaleCollapse_harm_pp":sh,"random_rank16_harm_pp":float(rh.mean()),"excess_harm_pp":mean,"CI95_low_pp":lo,"CI95_high_pp":hi,"percentile":float(100*(np.mean(rh<sh)+.5*np.mean(rh==sh)),),"bootstrap_draws":20000})
 csv(OUT/"rankmatched/multiseed_summary.csv",sums); js(OUT/"PROVENANCE.json",{"seeds":[0,1,2],"seed0":"frozen reused","seed1_2":"new final-semantics checkpoints","no_architecture_ablations":True,"replay_cells":len(replay),"identity_max_abs_logit_difference":identity,"projector_bank":str(ev.RANK_PROTOCOL/"random_projectors.npz")})
 (OUT/"MULTISEED_NONABLATION_SUMMARY.md").write_text("# BNCI2015-001 three-seed non-ablation continuation\n\nFull-model seeds 0/1/2, frozen PEEH/PSWA, and frozen rank-matched ScaleCollapse were run. B1--B4 ablations were not run beyond seed 0. Three-seed rows average seeds within biological subject before the 20,000-draw subject bootstrap.\n",encoding="utf-8")
 print("BNCI_THREE_SEED_DIAGNOSTICS_RANK_COMPLETE",flush=True)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--run",action="store_true");a=p.parse_args();
 if not a.run: raise RuntimeError("pass --run")
 run()
