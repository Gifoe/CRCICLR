from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
import p4_persist_ct as base
import p4_persist_ct_v2 as v2

ROOT=base.ROOT; OUT=ROOT/"outputs"/"persist_eeg_p4_ct_v2_1"
TASKS=base.TASKS; CLASSES=base.CLASSES; FOLDS=(0,1,2); SEEDS=(0,1)
INNER_SPLITS=5; RANDOM_DRAWS=100; BOOT_DRAWS=10000

def write_json(p,v):
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(base.clean(v),indent=2)+"\n",encoding="utf-8")
def risk(Xf,yf,Xe,ye,c):
    pack=base.ridge_probe(np.asarray(Xf),yf,c); pred,prob=base.probe_predict(np.asarray(Xe),pack,c)
    ba=float(np.mean([np.mean(pred[np.asarray(ye)==k]==k) for k in range(c) if np.any(np.asarray(ye)==k)]))
    yy=np.asarray(ye,dtype=np.int64); pp=np.asarray(prob,dtype=np.float64); ce=float(-np.mean(np.log(np.clip(pp[np.arange(len(yy)),yy],1e-12,1.0))))
    return ce, ba
def hier_boot(vals,seed):
    vals=np.asarray(vals,float)
    if len(vals)==0:return {"mean":None,"ci95":[None,None],"sign_probability":None,"draws":BOOT_DRAWS,"n_unique_subjects":0}
    g=np.random.default_rng(seed); d=g.choice(vals,size=(BOOT_DRAWS,len(vals)),replace=True).mean(1)
    return {"mean":float(vals.mean()),"ci95":[float(np.quantile(d,.025)),float(np.quantile(d,.975))],"sign_probability":float(np.mean(d>0)),"draws":BOOT_DRAWS,"n_unique_subjects":int(len(vals))}

def run(fold,seed,device):
    m=base.load_manifest(); split=next(x for x in base.load_splits() if int(x["fold"])==fold)
    ckpt,mean,std=base.historical(fold,seed); model=base.load_model(ckpt,m,device)
    trm,trh,try_=base.extract(model,m,split["train_subjects"],mean,std,device,190000+fold*101+seed,cap=3000)
    vam,vah,vay=base.extract(model,m,split["validation_subjects"],mean,std,device,200000+fold*101+seed,cap=1500)
    spec=v2.build_spectrum_v2(trm,trh,30000+fold*101+seed)
    subjects=v2.subject_sort(split["train_subjects"]); rows=[]; split_rows=[]; assignments={}
    for task in TASKS:
      trows=[]
      for bi,block in enumerate(spec["blocks"]):
        obs_by_sub={}; null_by_sub={}; supported=bool(spec["audit"]["persistence_support"][bi]["persistence_supported"])
        for inner in range(INNER_SPLITS):
          fit,ev=v2.split_subjects(subjects,inner,seed)
          fi=base.sample_positions(trm,task,fit,500,31000+inner); ei=base.sample_positions(trm,task,ev,500,32000+inner)
          if len(fi)<20 or len(ei)<20: continue
          yf,ye=v2.labels(trm,task,fi),v2.labels(trm,task,ei)
          b_risk,_=risk(trh[fi],yf,trh[ei],ye,CLASSES[task]); ef=base.erase(trh[fi],spec,block); ee=base.erase(trh[ei],spec,block)
          em=trm.iloc[ei].reset_index(drop=True)
          cand=np.setdiff1d(np.arange(len(spec["rho"])),np.asarray(block)); rng=np.random.default_rng(41000+fold*1000+seed*100+inner*10+bi)
          choices=[rng.choice(cand if len(cand)>=len(block) else np.arange(len(spec["rho"])),size=len(block),replace=False) for _ in range(RANDOM_DRAWS)]
          for subj,g in em.groupby(em.subject_id.astype(str),sort=True):
            loc=g.index.to_numpy();
            if len(loc)<2: continue
            rr,_=risk(trh[fi],yf,trh[ei][loc],ye[loc],CLASSES[task]); rre,_=risk(ef,yf,ee[loc],ye[loc],CLASSES[task]); obs_by_sub.setdefault(str(subj),[]).append(float(rre-rr))
            ns=[]
            for ch in choices:
              er=base.erase(trh[fi],spec,ch); evv=base.erase(trh[ei][loc],spec,ch); rn,_=risk(er,yf,evv,ye[loc],CLASSES[task]); ns.append(float(rn-rr))
            null_by_sub.setdefault(str(subj),[]).append(float(np.mean(ns)))
          split_rows.append({"fold":fold,"seed":seed,"task":task,"block":bi,"inner_split":inner,"n_eval_subjects":int(em.subject_id.astype(str).nunique()),"random_draws":RANDOM_DRAWS,"random_rank":len(block),"persistence_supported":supported})
        cal={s:float(np.mean(obs_by_sub[s])-np.mean(null_by_sub.get(s,[0.]))) for s in obs_by_sub if s in null_by_sub}
        bs=hier_boot(np.array(list(cal.values())),70000+fold*101+seed*11+bi)
        row={"fold":fold,"seed":seed,"task":task,"block":bi,"dimensions":len(block),"persistence_supported":supported,"n_unique_subjects":len(cal),"n_split_observations":sum(len(x) for x in obs_by_sub.values()),"bootstrap_hierarchy":"unique_subject_aggregate_across_inner_splits_then_subject_bootstrap","calibrated_utility_mean_CE":bs["mean"],"calibrated_utility_ci95":bs["ci95"],"calibrated_utility_sign_probability":bs["sign_probability"],"bootstrap":bs,"random_draws_per_subject_split":RANDOM_DRAWS,"probe":"full_128D_ridge"}
        rows.append(row); trows.append(row)
      prot=[r["block"] for r in trows if r["persistence_supported"] and r["calibrated_utility_ci95"][0] is not None and r["calibrated_utility_ci95"][0]>0]
      nuis=[r["block"] for r in trows if r["persistence_supported"] and r["calibrated_utility_ci95"][0] is not None and r["calibrated_utility_ci95"][0]>=-.005 and r["calibrated_utility_ci95"][1]<=.005]
      assignments[task]={"protected":prot,"nuisance":nuis,"uncertain":[r["block"] for r in trows if r["block"] not in prot+nuis]}
    harms={t:{"protected":0.,"nuisance":0.,"random_same_total_rank":0.,"random_block_same_rank":0.} for t in TASKS}
    for task in TASKS:
      ti=np.flatnonzero((trm.paradigm==task).to_numpy()); vi=np.flatnonzero((vam.paradigm==task).to_numpy()); ytr,yv=try_[ti],vay[vi]; br,bba=risk(trh[ti],ytr,vah[vi],yv,CLASSES[task])
      dims={c:len(set(sum((spec["blocks"][b] for b in assignments[task][c]),[]))) for c in ("protected","nuisance")}
      for c in ("protected","nuisance"):
        ids=sorted(set(sum((spec["blocks"][b] for b in assignments[task][c]),[])))
        if ids: _,ba=risk(base.erase(trh[ti],spec,ids),ytr,base.erase(vah[vi],spec,ids),yv,CLASSES[task]); harms[task][c]=float(bba-ba)
        if dims[c]:
          rng=np.random.default_rng(80000+fold*100+seed); vals=[]
          for _ in range(RANDOM_DRAWS):
            ch=rng.choice(np.arange(len(spec["rho"])),size=dims[c],replace=False); _,ba=risk(base.erase(trh[ti],spec,ch),ytr,base.erase(vah[vi],spec,ch),yv,CLASSES[task]); vals.append(bba-ba)
          harms[task]["random_same_total_rank"] = float(np.mean(vals))
      rng=np.random.default_rng(81000+fold*100+seed); vals=[]; rank=len(spec["blocks"][0])
      for _ in range(RANDOM_DRAWS):
        ch=rng.choice(np.arange(len(spec["rho"])),size=rank,replace=False); _,ba=risk(base.erase(trh[ti],spec,ch),ytr,base.erase(vah[vi],spec,ch),yv,CLASSES[task]); vals.append(bba-ba)
      harms[task]["random_block_same_rank"]=float(np.mean(vals))
    empty=np.zeros((2,128),np.float32); sel=spec["blocks"][0][:min(2,len(spec["blocks"][0]))]; h0=np.random.default_rng(1).normal(size=(2,128)).astype(np.float32); hi=base.erase(h0,spec,sel); hn=base.erase(h0,spec,[]); q0=base.coords(h0,spec); qi=base.coords(hi,spec); unit={"empty_intervention_error":float(np.max(np.abs(hn-h0))),"non_selected_residual_error":float(np.max(np.abs((qi-q0)[:,np.setdiff1d(np.arange(len(spec["rho"])),sel)]))),"selected_dimensions":len(sel)}
    r={"fold":fold,"seed":seed,"mi_harm":harms["mi"],"mi_difference":harms["mi"]["protected"]-harms["mi"]["nuisance"],"harms":harms,"assignments":assignments,"rows":rows,"split_rows":split_rows,"unit_tests":unit,"outer_test_used":False}
    d=OUT/"audit"/f"fold-{fold}"/f"seed-{seed}"; d.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(d/"INTERVENTION_UTILITY_V2_1.csv",index=False); pd.DataFrame(split_rows).to_csv(d/"CROSS_FITTING_SPLITS_V2_1.csv",index=False); write_json(d/"AUDIT_V2_1.json",r); return r

def main():
  ap=argparse.ArgumentParser(); ap.add_argument('--fold',type=int,default=None); ap.add_argument('--seed',type=int,default=None); a=ap.parse_args()
  folds=(a.fold,) if a.fold is not None else FOLDS; seeds=(a.seed,) if a.seed is not None else SEEDS
  dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); results=[]
  try:
    for f in folds:
      for s in seeds: print(f"[V2.1] fold={f} seed={s}",flush=True); results.append(run(f,s,dev))
  except Exception:
    import traceback
    (ROOT/"v21_exception.txt").write_text(traceback.format_exc(),encoding="utf-8")
    raise
  dif=np.array([r["mi_difference"] for r in results]); rng=np.random.default_rng(991337); boot=rng.choice(dif,size=(BOOT_DRAWS,len(dif)),replace=True).mean(1); ci=[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]; pos=int(np.sum(dif>0)); prot=np.array([r["mi_harm"]["protected"] for r in results]); nuis=np.array([r["mi_harm"]["nuisance"] for r in results]); rnd=np.array([r["mi_harm"]["random_same_total_rank"] for r in results]); gate={"mean_positive":float(dif.mean())>0,"at_least_5_of_6":pos>=5,"hierarchical_ci_lcb_gt_zero":ci[0]>0,"protected_gt_same_rank_random":float(prot.mean())>float(rnd.mean()),"nuisance_near_below_same_rank_random":float(nuis.mean())<=float(rnd.mean())}; status="P4_INTERVENTION_GUIDED_ADAPTATION_AUDIT_V2_1_PASS" if all(gate.values()) else "P4_INTERVENTION_GUIDED_ADAPTATION_NOT_SUPPORTED"; payload={"status":status,"gate":gate,"mean_mi_difference":float(dif.mean()),"positive_runs":pos,"hierarchical_bootstrap":{"draws":BOOT_DRAWS,"ci95":ci,"sign_probability":float(np.mean(boot>0))},"runs":[{"fold":r["fold"],"seed":r["seed"],"difference":r["mi_difference"],"mi_harm":r["mi_harm"],"outer_test_used":False} for r in results],"outer_test_used":False}; write_json(OUT/"P4_INTERVENTION_GUIDED_ADAPTATION_AUDIT_V2_1.json",payload); write_json(OUT/"P4_INTERVENTION_GUIDED_ADAPTATION_NOT_SUPPORTED.json" if status.endswith("NOT_SUPPORTED") else OUT/"P4_INTERVENTION_GUIDED_ADAPTATION_AUDIT_V2_1_PASS.json",payload); print(json.dumps(base.clean(payload),indent=2))
if __name__=="__main__": main()
