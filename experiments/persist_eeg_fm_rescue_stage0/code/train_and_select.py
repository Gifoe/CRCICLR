from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import torch

import common as c


def main() -> None:
    c.ensure_dirs(); lock=c.read_json(c.PROTOCOL/"DATA_ACCESS_LOCK.json")
    if lock.get("primary_outcomes_inspected") is not False or lock.get("S2_or_S3_utility_inspected") is not False: raise RuntimeError("outcome-clean lock required")
    c.prepare_inputs("OpenBMI", include_future=False); c.prepare_inputs("WBCIC", include_future=False)
    device=torch.device("cuda"); rows=[]
    for fm in c.FMS:
        for dataset in c.DATASETS:
            for lr in c.LR_GRIDS[fm]:
                for fold in c.FOLDS:
                    record=c.train_anchor(fm,dataset,fold,0,lr,c.search_checkpoint(fm,dataset,fold,lr),device); rows.append(record)
            grid=pd.DataFrame([r for r in rows if r["fm"]==fm and r["dataset"]==dataset]).groupby("lr",as_index=False).agg(mean_validation_BA=("validation_mean_subject_BA","mean"),minimum_fold_BA=("validation_mean_subject_BA","min"),folds=("fold","nunique"))
            selected=grid.sort_values(["mean_validation_BA","minimum_fold_BA","lr"],ascending=[False,False,True]).iloc[0]; lr=float(selected.lr)
            for fold in c.FOLDS:
                target=c.model_checkpoint(fm,dataset,fold,0); target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(c.search_checkpoint(fm,dataset,fold,lr),target)
                for seed in (1,2): rows.append(c.train_anchor(fm,dataset,fold,seed,lr,c.model_checkpoint(fm,dataset,fold,seed),device))
            print(f"[selected] {fm} {dataset} lr={lr:g} valBA={float(selected.mean_validation_BA):.5f}",flush=True)
    frame=pd.DataFrame(rows).drop_duplicates(["fm","dataset","fold","seed","lr"]); c.write_csv(c.RUNTIME/"FM_SOURCE_VALIDATION_RUNS.csv",frame.drop(columns="history"))
    selected_rows=[]
    for fm in c.FMS:
        for dataset in c.DATASETS:
            sub=frame[(frame.fm==fm)&(frame.dataset==dataset)&(frame.seed==0)].groupby("lr",as_index=False).agg(mean_validation_BA=("validation_mean_subject_BA","mean"),minimum_fold_BA=("validation_mean_subject_BA","min"),folds=("fold","nunique"))
            best=sub.sort_values(["mean_validation_BA","minimum_fold_BA","lr"],ascending=[False,False,True]).iloc[0]
            selected_rows.append({"fm":fm,"dataset":dataset,**best.to_dict()})
    selection=pd.DataFrame(selected_rows); c.write_csv(c.RESULTS/"FM_SOURCE_VALIDATION_SELECTION.csv",selection)
    scaa_rows=[]; lr_grid=(1e-4,3e-4,1e-3)
    data=c.load_data("WBCIC")
    for fm in c.FMS:
        for fold in c.FOLDS:
            targets=c.fold_roles("WBCIC",fold)["outcome"]; indices=c.row_indices(data.metadata,targets,(0,))
            for seed in c.SEEDS:
                model=c.load_anchor(fm,"WBCIC",fold,seed,device); ex=c.infer(model,"WBCIC",indices,device)
                weight=model.head.weight.detach().float().cpu().numpy(); bias=model.head.bias.detach().float().cpu().numpy()
                for subject in targets:
                    m=ex["subjects"].astype(str)==subject; feats=ex["features"][m]; labels=ex["labels"][m]; anchor=ex["logits"][m]; tr,va=c.chronological_class_split(labels)
                    for lr in lr_grid:
                        adapted=c.adapt_linear_head(feats,labels,tr,va,weight,bias,lr,c.stable_seed("scaa-s1",fm,fold,seed,subject,lr)); a=c.metrics(labels[va],anchor[va]); b=c.metrics(labels[va],adapted["logits"][va])
                        scaa_rows.append({"fm":fm,"fold":fold,"seed":seed,"subject_id":subject,"lr":lr,"anchor_S1_validation_BA":a["BA"],"adapted_S1_validation_BA":b["BA"],"delta":b["BA"]-a["BA"],"adapted_NLL":b["NLL"],"prediction_change":float(np.mean(anchor[va].argmax(1)!=adapted["logits"][va].argmax(1))),"parameter_relative_change":adapted["parameter_relative_change"],"S2_or_S3_accessed":False})
                del model; torch.cuda.empty_cache(); print(f"[S1-only] {fm} fold={fold} seed={seed}",flush=True)
    cells=pd.DataFrame(scaa_rows); c.write_csv(c.RUNTIME/"SCAA_S1_ONLY_CELLS.csv",cells)
    summary=cells.groupby(["fm","lr"],as_index=False).agg(cells=("subject_id","size"),anchor_BA=("anchor_S1_validation_BA","mean"),adapted_BA=("adapted_S1_validation_BA","mean"),delta=("delta","mean"),NLL=("adapted_NLL","mean"),prediction_change=("prediction_change","mean"),parameter_change=("parameter_relative_change","mean"))
    chosen=[]
    for fm in c.FMS:
        best=summary[summary.fm==fm].sort_values(["adapted_BA","delta","NLL","lr"],ascending=[False,False,True,True]).iloc[0]; chosen.append(best.to_dict())
    c.write_csv(c.RESULTS/"SCAA_S1_ADAPTATION_SELECTION.csv",pd.DataFrame(chosen)); c.write_csv(c.RESULTS/"SCAA_S1_ADAPTATION_GRID.csv",summary)
    c.write_json(c.RUNTIME/"PRE_OUTCOME_SELECTION_COMPLETE.json",{"complete":True,"source_validation_only":True,"S2_or_S3_accessed":False,"primary_outcomes_accessed":False,"anchor_selection":selected_rows,"SCAA_S1_selection":chosen})
    print("PRE_OUTCOME_TRAINING_AND_SELECTION_COMPLETE",flush=True)


if __name__ == "__main__": main()
