from pathlib import Path
import pandas as pd, numpy as np, json
ROOT=Path("/root/rivermind-data/CRCICLR_TASK_GENERALITY_WORK/experiments/persist_eeg_xs_erp_seed12_stability_v1")
OUT=ROOT/"outputs"
def agg(f,seed):
    p=f.pivot_table(index=["fold","subject_id"],columns="method",values=["BA","macro_F1","accuracy"])
    d=p[("BA","LiteBN_X")]-p[("BA","LiteBN_BASELINE")]
    rows=[]
    for fid,g in f.groupby("fold"):
        q=g.pivot_table(index="subject_id",columns="method",values=["BA","macro_F1","accuracy"]); dd=q[("BA","LiteBN_X")]-q[("BA","LiteBN_BASELINE")]
        rows.append({"seed":int(seed),"fold":int(fid),"n_subjects":len(dd),"LiteBN_BA":q[("BA","LiteBN_BASELINE")].mean(),"XS_BA":q[("BA","LiteBN_X")].mean(),"delta_pp":100*dd.mean(),"LiteBN_macro_F1":q[("macro_F1","LiteBN_BASELINE")].mean(),"XS_macro_F1":q[("macro_F1","LiteBN_X")].mean(),"positive_subjects":int((dd>0).sum()),"harmed_subjects":int((dd<0).sum())})
    fold=pd.DataFrame(rows)
    overall={"seed":int(seed),"n_subjects":len(p),"LiteBN_BA":p[("BA","LiteBN_BASELINE")].mean(),"XS_BA":p[("BA","LiteBN_X")].mean(),"delta_pp":100*d.mean(),"LiteBN_macro_F1":p[("macro_F1","LiteBN_BASELINE")].mean(),"XS_macro_F1":p[("macro_F1","LiteBN_X")].mean(),"positive_folds":int((fold.delta_pp>0).sum()),"negative_folds":int((fold.delta_pp<0).sum()),"positive_subjects":int((d>0).sum()),"harmed_subjects":int((d<0).sum())}
    return fold,overall
all_o=[]; all_f=[]
for seed in (0,1,2):
    p=OUT/("SEED0_EXISTING_ERP_OUTER_SUBJECT_RESULTS.csv" if seed==0 else "SEED{}_OUTER_SUBJECT_RESULTS.csv".format(seed))
    f=pd.read_csv(p); fold,o=agg(f,seed); all_o.append(o)
    if seed: all_f.append(fold)
pd.DataFrame(all_o).to_csv(OUT/"ERP_SEED012_SUMMARY.csv",index=False)
pd.concat(all_f,ignore_index=True).to_csv(OUT/"ERP_SEED12_FOLD_SUMMARY.csv",index=False)
d={int(x["seed"]):float(x["delta_pp"]) for x in all_o}
if d[1]>=-0.05 and d[2]>=-0.05: interp="SEED0_DEGRADATION_LIKELY_RANDOM"
elif d[1]<-0.20 and d[2]<-0.20: interp="ERP_DEGRADATION_STABLE_FAILURE"
else: interp="ERP_DEGRADATION_MIXED"
lines=["# OpenBMI ERP LiteBN-XS seed stability (outer-development)","","Only OpenBMI ERP was evaluated. Existing canonical folds were reused; no new inner split was created.","LiteBN baseline checkpoints were verified and reused. XS seed1/2 were trained independently with the original architecture and selection rule.","Outer-development rows were accessed for development/stability analysis. Final heldout was not accessed.","","| Seed | LiteBN outer BA | XS outer BA | delta XS-LiteBN (pp) | LiteBN macro-F1 | XS macro-F1 | positive folds |","|---:|---:|---:|---:|---:|---:|---:|"]
for x in all_o: lines.append("| {} | {:.6f} | {:.6f} | {:+.3f} | {:.6f} | {:.6f} | {} |".format(x["seed"],x["LiteBN_BA"],x["XS_BA"],x["delta_pp"],x["LiteBN_macro_F1"],x["XS_macro_F1"],x["positive_folds"]))
lines += ["","Seed1/2 mean delta: {:+.3f} pp.".format(np.mean([d[1],d[2]])),"Positive seeds (0/1/2): {}/3.".format(sum(d[s]>0 for s in (0,1,2))),"Interpretation: {}.".format(interp),"","Selection: max 60 epochs; canonical inner-validation subject-mean BA; eligibility starts epoch 10; patience 10; best checkpoint restored.","","FINAL_HELDOUT_ACCESSED = NO","OUTER_DEVELOPMENT_ACCESSED = YES (development/stability analysis only)"]
(OUT/"ERP_SEED12_STABILITY_REPORT.md").write_text("\n".join(lines)+"\n")
(OUT/"ERP_SEED12_STABILITY_METADATA.json").write_text(json.dumps({"experiment":"persist_eeg_xs_erp_seed12_stability_v1","task":"OpenBMI_ERP","seeds":[0,1,2],"split_sha256":"d313f8e1f2c105d0b387c607fb781b61d9a05d92875759e51ed0d4c6d889f42a","selection_rule":"canonical inner-val subject-mean BA; min_epoch=10; patience=10","max_epochs":60,"lr":3e-4,"weight_decay":5e-4,"baseline_reused":True,"new_inner_splits_created":False,"outer_development_accessed":True,"final_heldout_accessed":False,"interpretation":interp,"seed_deltas_pp":d},indent=2)+"\n")
print("FINALIZED",interp,flush=True)
