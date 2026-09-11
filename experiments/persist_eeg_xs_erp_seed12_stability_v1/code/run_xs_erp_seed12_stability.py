#!/usr/bin/env python3
from pathlib import Path
import importlib.util, sys, json, hashlib, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO=Path("/root/rivermind-data/CRCICLR_TASK_GENERALITY_WORK")
RUNNER_PATH=REPO/"experiments/persist_eeg_litebn_x_singlemodel_seed0_v1/code/run_x_multiseed_x_only.py"
EXP=REPO/"experiments/persist_eeg_xs_erp_seed12_stability_v1"
OUT=EXP/"outputs_correct_xs"
RUNTIME_ROOT=Path("/root/rivermind-data/xs_erp_seed12_stability_runtime_correct_xs")
TASK="OpenBMI_ERP"
SEEDS=(1,2)

def load_runner():
    spec=importlib.util.spec_from_file_location("xs_erp_runner",RUNNER_PATH)
    mod=importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod); return mod

def sha256(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()

def aggregate(frame,seed):
    p=frame.pivot_table(index=["fold","subject_id"],columns="method",values=["BA","macro_F1","accuracy"])
    d=p[("BA","LiteBN_XS")]-p[("BA","LiteBN_BASELINE")]
    rows=[]
    for fid,g in frame.groupby("fold"):
        q=g.pivot_table(index="subject_id",columns="method",values=["BA","macro_F1","accuracy"])
        dd=q[("BA","LiteBN_XS")]-q[("BA","LiteBN_BASELINE")]
        rows.append({"seed":int(seed),"fold":int(fid),"n_subjects":int(len(dd)),
                     "LiteBN_BA":float(q[("BA","LiteBN_BASELINE")].mean()),
                     "XS_BA":float(q[("BA","LiteBN_XS")].mean()),
                     "delta_pp":float(100*dd.mean()),
                     "LiteBN_macro_F1":float(q[("macro_F1","LiteBN_BASELINE")].mean()),
                     "XS_macro_F1":float(q[("macro_F1","LiteBN_XS")].mean()),
                     "positive_subjects":int((dd>0).sum()),"harmed_subjects":int((dd<0).sum())})
    fold=pd.DataFrame(rows)
    return fold,{"seed":int(seed),"n_subjects":int(len(p)),
        "LiteBN_BA":float(p[("BA","LiteBN_BASELINE")].mean()),
        "XS_BA":float(p[("BA","LiteBN_XS")].mean()),"delta_pp":float(100*d.mean()),
        "LiteBN_macro_F1":float(p[("macro_F1","LiteBN_BASELINE")].mean()),
        "XS_macro_F1":float(p[("macro_F1","LiteBN_XS")].mean()),
        "positive_folds":int((fold.delta_pp>0).sum()),"negative_folds":int((fold.delta_pp<0).sum()),
        "positive_subjects":int((d>0).sum()),"harmed_subjects":int((d<0).sum())}


M=None
MAX_EPOCHS=60
MIN_EPOCH=10
PATIENCE=10
LR=3e-4
WEIGHT_DECAY=5e-4
CLIP=5.0
TOL=1e-12

def train_one_patience(model, architecture, task, fold, bundle, cache, mean, std, normalizer_meta, batch_info, class_weight, class_weight_meta, device):
    global M
    if architecture=="LiteBN_X":
        architecture="LiteBN_XS"
    latest=M.checkpoint_path(task,int(fold["fold_id"]),architecture,"checkpoint_latest.pt")
    selected=M.checkpoint_path(task,int(fold["fold_id"]),architecture,"selected_best.pt")
    initial_hash=M.state_hash(model); source_hash=sha256(RUNNER_PATH)
    invariants={"task":task,"fold":int(fold["fold_id"]),"architecture":architecture,"seed":M.SEED,
                "initial_sha256":initial_hash,"source_sha256":source_hash,
                "normalizer_sha256":normalizer_meta["mean_std_sha256"],
                "batch_manifest_sha256":batch_info.get("manifest_sha256"),
                "class_weight_info":class_weight_meta,"max_epochs":MAX_EPOCHS,
                "min_selection_epoch":MIN_EPOCH,"patience":PATIENCE,
                "lr":LR,"weight_decay":WEIGHT_DECAY,"clip":CLIP}
    optimizer=torch.optim.AdamW(model.parameters(),lr=LR,weight_decay=WEIGHT_DECAY)
    amp=device.type=="cuda"; scaler=torch.amp.GradScaler("cuda",enabled=amp)
    start=1; history=[]; best=-float("inf"); best_f1=-float("inf"); best_epoch=None; best_state=None; bad=0
    if latest.is_file():
        saved=torch.load(latest,map_location=device,weights_only=False)
        if saved.get("invariants")!=invariants: raise RuntimeError("resume invariant mismatch {}".format(latest))
        model.load_state_dict(saved["current_state"],strict=True); optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"]); M.restore_rng(saved["rng"])
        start=int(saved["epoch"])+1; history=list(saved["history"]); best=float(saved["best"])
        best_f1=float(saved["best_f1"]); best_epoch=saved["best_epoch"]; best_state=saved["best_state"]; bad=int(saved["bad"])
    train_indices=bundle.indices(fold["inner_train_subjects"],M.TASKS[task]["source_sessions"])
    weight=None if class_weight is None else class_weight.to(device)
    started=time.perf_counter()
    for epoch in range(start,MAX_EPOCHS+1):
        model.train(); losses=[]
        batches=batch_info["episodes"][epoch-1] if M.TASKS[task]["mi_protocol"] else M.task_epoch_batches(train_indices,task,int(fold["fold_id"]),epoch)
        for indices in batches:
            value,labels=cache.batch(np.asarray(indices,dtype=np.int64),mean,std)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                logits,_=model(value); loss=F.cross_entropy(logits,labels,weight=weight)
            if not torch.isfinite(loss): raise RuntimeError("nonfinite CE {}/{}".format(task,architecture))
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP); scaler.step(optimizer); scaler.update()
            losses.append(float(loss.detach().cpu()))
        val=M.evaluate(model,bundle,cache,fold["inner_val_subjects"],mean,std)
        val_ba=float(np.mean([x["BA"] for x in val.values()])); val_f1=float(np.mean([x["macro_F1"] for x in val.values()]))
        eligible=epoch>=MIN_EPOCH
        improved=bool(eligible and (val_ba>best+TOL or (abs(val_ba-best)<=TOL and val_f1>best_f1+TOL)))
        if improved:
            best,best_f1,best_epoch=val_ba,val_f1,epoch
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; bad=0
        elif eligible: bad+=1
        stop=bool(eligible and bad>=PATIENCE)
        history.append({"epoch":int(epoch),"cross_entropy":float(np.mean(losses)),
            "inner_val_subject_BA":val_ba,"inner_val_subject_macro_F1":val_f1,
            "selected":improved,"bad_epochs":bad,"eligible":eligible,"stopped_early":stop,"batches":len(losses)})
        M.atomic_torch_save(latest,{"epoch":epoch,"history":history,"best":best,"best_f1":best_f1,
            "best_epoch":best_epoch,"best_state":best_state,"current_state":model.state_dict(),
            "optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),"rng":M.rng_state(),
            "bad":bad,"invariants":invariants})
        if epoch==1 or epoch%5==0 or improved: print("[XS seed{} ERP f{}] epoch={:02d} valBA={:.5f} best={:.5f} bad={}/{}".format(M.SEED,fold["fold_id"],epoch,val_ba,best,bad,PATIENCE),flush=True)
        if stop: print("[XS seed{} ERP f{}] EARLY_STOP epoch={} best={}".format(M.SEED,fold["fold_id"],epoch,best_epoch),flush=True); break
    if best_state is None: raise RuntimeError("no checkpoint selected")
    model.load_state_dict(best_state,strict=True); M.atomic_torch_save(selected,model.state_dict())
    rec={"task":task,"dataset":M.TASKS[task]["dataset"],"fold":int(fold["fold_id"]),"architecture":architecture,
         "seed":int(M.SEED),"parameter_count":M.parameter_count(model),"selected_epoch":int(best_epoch),
         "best_inner_val_BA":float(best),"best_inner_val_macro_F1":float(best_f1),
         "checkpoint_path":str(selected),"checkpoint_sha256":M.sha256_file(selected),
         "normalizer_sha256":normalizer_meta["mean_std_sha256"],"batch_manifest_sha256":batch_info.get("manifest_sha256"),
         "class_weighted_ce":bool(class_weight_meta["weighted_cross_entropy"]),
         "elapsed_seconds_this_invocation":float(time.perf_counter()-started),"epochs_completed":len(history),
         "history":history,"source_sha256":source_hash}
    return rec


def evaluate_only_erp(mod,seed,records,folds,device):
    runtime=RUNTIME_ROOT/"seed{}".format(seed); mod.RUNTIME=runtime
    byfold={int(r["fold"]):r for r in records}; rows=[]
    for fold in folds:
        fid=int(fold["fold_id"]); outer=fold["outer_dev_subjects"]
        bundle=mod.build_bundle(TASK,outer)
        mean,std,norm=mod.load_tensor_pair(runtime/"normalizers"/"openbmi_erp_fold{}.npz".format(fid))
        cache=mod.RawGPUCache(bundle,device)
        paths={"LiteBN_BASELINE":Path("/root/rivermind-data/openbmi_task_generality_runtime/erp_fold{}_seed{}_litebn/selected_best.pt".format(fid,seed)),
               "LiteBN_XS":Path(byfold[fid]["checkpoint_path"])}
        for method,path in paths.items():
            if not path.is_file(): raise FileNotFoundError(str(path))
            model=mod.build_model(method,TASK).to(device)
            model.load_state_dict(torch.load(path,map_location=device,weights_only=False),strict=True)
            metrics=mod.evaluate(model,bundle,cache,outer,mean,std)
            for subject,val in metrics.items():
                rows.append({"seed":int(seed),"task":TASK,"dataset":"OpenBMI","fold":fid,
                    "subject_id":str(subject),"method":method,**val,
                    "checkpoint_sha256":sha256(path),"normalizer_sha256":norm["mean_std_sha256"],
                    "baseline_reused":method=="LiteBN_BASELINE"})
            del model
        del cache,bundle
        if device.type=="cuda": torch.cuda.empty_cache()
    frame=pd.DataFrame(rows).sort_values(["fold","subject_id","method"]).reset_index(drop=True)
    expected=2*sum(len(f["outer_dev_subjects"]) for f in folds)
    if len(frame)!=expected: raise RuntimeError("outer cardinality {} != {}".format(len(frame),expected))
    frame.to_csv(OUT/"SEED{}_OUTER_SUBJECT_RESULTS.csv".format(seed),index=False)
    return frame

def seed0_reference(mod):
    src=REPO/"experiments/persist_eeg_litebn_x_singlemodel_seed0_v1/outputs/POSTHOC_RG_XS_SEED0_ERP_OUTER_SUBJECT_RESULTS.csv"
    f=pd.read_csv(src); f=f[f.task.eq(TASK) & f.method.isin(["LiteBN_BASELINE","LiteBN_XS"])].copy(); f["seed"]=0; f["baseline_reused"]=f.method.eq("LiteBN_BASELINE")
    f.to_csv(OUT/"SEED0_EXISTING_ERP_OUTER_SUBJECT_RESULTS.csv",index=False)
    return aggregate(f,0)[1]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    runner=load_runner()
    mod=runner.load_module()
    global M
    M=mod
    mod.train_one=train_one_patience
    original_build=mod.build_model
    original_checkpoint_path=mod.checkpoint_path
    def build_model_xs(architecture, task):
        return original_build("LiteBN_XS" if architecture=="LiteBN_X" else architecture, task)
    def checkpoint_path_xs(task, fold_id, architecture, which):
        return original_checkpoint_path(task, fold_id, "LiteBN_XS" if architecture=="LiteBN_X" else architecture, which)
    mod.build_model=build_model_xs
    mod.checkpoint_path=checkpoint_path_xs
    mod.TASK_ORDER=(TASK,)
    mod.OUT=OUT
    mod.RUNTIME=RUNTIME_ROOT
    runner.RUNTIME_ROOT=RUNTIME_ROOT
    runner.OUT=OUT
    search,foldmap,split_hash=mod.load_folds()
    folds=list(foldmap["OpenBMI"])
    prov=runner.verify_baselines(mod)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overalls=[]; foldframes=[]
    for seed in SEEDS:
        recs=runner.train_seed(mod,seed)
        frame=evaluate_only_erp(mod,seed,recs,folds,device)
        fold,overall=aggregate(frame,seed); fold.to_csv(OUT/"SEED{}_FOLD_SUMMARY.csv".format(seed),index=False)
        overalls.append(overall); foldframes.append(fold)
        print("SEED{}_ERP_COMPLETE delta_pp={:+.4f}".format(seed,overall["delta_pp"]),flush=True)
    ref=seed0_reference(mod); overalls.insert(0,ref)
    pd.DataFrame(overalls).to_csv(OUT/"ERP_SEED012_SUMMARY.csv",index=False)
    pd.concat(foldframes,ignore_index=True).to_csv(OUT/"ERP_SEED12_FOLD_SUMMARY.csv",index=False)
    dd={int(r["seed"]):float(r["delta_pp"]) for r in overalls}
    if dd[1]>=-0.05 and dd[2]>=-0.05: interp="SEED0_DEGRADATION_LIKELY_RANDOM"
    elif dd[1]<-0.20 and dd[2]<-0.20: interp="ERP_DEGRADATION_STABLE_FAILURE"
    else: interp="ERP_DEGRADATION_MIXED"
    report=["# OpenBMI ERP LiteBN-XS seed stability (outer-development)","",
      "Only OpenBMI ERP was evaluated. Existing canonical folds were reused; no new inner split was created.",
      "LiteBN baseline checkpoints were verified and reused. XS seed1/2 were trained in a separate runtime with the original architecture and selection rule.",
      "Outer-development rows were accessed for this development/stability analysis. Final heldout was not accessed.","",
      "| Seed | LiteBN outer BA | XS outer BA | delta XS-LiteBN (pp) | LiteBN macro-F1 | XS macro-F1 | positive folds |",
      "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in overalls:
        report.append("| {} | {:.6f} | {:.6f} | {:+.3f} | {:.6f} | {:.6f} | {} |".format(
          r["seed"],r["LiteBN_BA"],r["XS_BA"],r["delta_pp"],r["LiteBN_macro_F1"],r["XS_macro_F1"],r.get("positive_folds","—")))
    report += ["","Seed1/2 mean delta: {:+.3f} pp.".format(float(np.mean([dd[1],dd[2]]))),
      "Positive seeds (0/1/2): {}/3.".format(sum(dd[s]>0 for s in (0,1,2))),
      "Interpretation: {}.".format(interp),"",
      "Selection: max 60 epochs; canonical inner-validation subject-mean BA; eligibility starts epoch 10; same rule for LiteBN-X seed1/2; best checkpoint restored.",
      "","FINAL_HELDOUT_ACCESSED = NO","OUTER_DEVELOPMENT_ACCESSED = YES (development/stability analysis only)"]
    (OUT/"ERP_SEED12_STABILITY_REPORT.md").write_text("\n".join(report)+"\n")
    (OUT/"ERP_SEED12_STABILITY_METADATA.json").write_text(json.dumps({
      "experiment":"persist_eeg_xs_erp_seed12_stability_v1","task":TASK,"seeds":[0,1,2],
      "split_sha256":split_hash,"selection_rule":"canonical inner-val subject-mean BA; min_epoch=10; patience=10",
      "max_epochs":60,"lr":3e-4,"weight_decay":5e-4,"baseline_reused":True,
      "new_inner_splits_created":False,"outer_development_accessed":True,
      "final_heldout_accessed":False,"interpretation":interp,"seed_deltas_pp":dd},indent=2)+"\n")
    print("ERP_SEED12_STABILITY_COMPLETE {}".format(interp),flush=True)

if __name__=="__main__": main()
