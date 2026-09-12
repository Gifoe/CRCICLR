"""Evaluate all frozen task carriers on the predefined SEARCH outer-development subjects."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

from task_datasets import (OUTPUTS, PROTOCOL, TASKS, RawGPUCache, build_model, load_bundle, normalizer, sha256)


def score(task:str,y:np.ndarray,logits:np.ndarray)->dict[str,Any]:
    p=logits.argmax(1); result={"BA":float(balanced_accuracy_score(y,p)),"macro_F1":float(f1_score(y,p,average="macro",zero_division=0)),"accuracy":float(accuracy_score(y,p))}
    if task=="ERP":
        prob=torch.softmax(torch.from_numpy(logits),dim=1).numpy()[:,1]
        result.update({"AUROC":float(roc_auc_score(y,prob)),"AUPRC":float(average_precision_score(y,prob))})
    else: result.update({"AUROC":None,"AUPRC":None})
    return result


def logits(model:torch.nn.Module,cache:RawGPUCache,indices:np.ndarray,mean:np.ndarray,std:np.ndarray)->np.ndarray:
    chunks=[]
    with torch.no_grad():
        for start in range(0,len(indices),128): chunks.append(model(cache.batch(indices[start:start+128],mean,std)[0])[0].float().cpu().numpy())
    return np.concatenate(chunks)


def main() -> int:
    provenance=json.loads((PROTOCOL/"CHECKPOINT_PROVENANCE.json").read_text(encoding="utf-8"))
    if not provenance.get("pass") or len(provenance["records"])!=60: raise RuntimeError("full frozen carrier grid required")
    split=json.loads((PROTOCOL/"SUBJECT_SPLIT_REFERENCE.json").read_text(encoding="utf-8")); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    by={(x["task"],int(x["fold"]),int(x["seed"]),x["model"]):x for x in provenance["records"]}; rows=[]
    for task in ("ERP","SSVEP"):
        bundle=load_bundle(task,split["search_subjects"]); cache=RawGPUCache(bundle,device)
        for fold in split["folds"]:
            mean,std,_=normalizer(bundle,fold["inner_train_subjects"])
            for seed in (0,1,2):
                models={}
                for name in ("EEGNet","LiteBN"):
                    record=by[(task,int(fold["fold_id"]),seed,name)]; path=Path(record["checkpoint_path"])
                    if sha256(path)!=record["checkpoint_sha256"]: raise RuntimeError("checkpoint mutated after training")
                    model=build_model(name,task).to(device); model.load_state_dict(torch.load(path,map_location=device,weights_only=False),strict=True); model.eval()
                    for parameter in model.parameters(): parameter.requires_grad_(False)
                    models[name]=model
                for subject in fold["outer_dev_subjects"]:
                    ii=bundle.indices([subject],(2,)); y=bundle.labels(ii); ze=logits(models["EEGNet"],cache,ii,mean,std); zl=logits(models["LiteBN"],cache,ii,mean,std); z50=(ze+zl)/2
                    for name,z in (("EEGNet",ze),("LiteBN",zl),("LOGIT50",z50)):
                        rows.append({"task":task,"subject_id":str(subject),"fold":int(fold["fold_id"]),"seed":seed,"method":name,"trials":int(len(y)),**score(task,y,z)})
                del models
                if device.type=="cuda": torch.cuda.empty_cache()
        del cache
    frame=pd.DataFrame(rows); expected=40*3*3*2
    if len(frame)!=expected or frame.duplicated(["task","subject_id","fold","seed","method"]).any(): raise RuntimeError("incomplete SEARCH evaluation")
    OUTPUTS.mkdir(parents=True,exist_ok=True); frame.to_csv(OUTPUTS/"SEARCH_REPLICATE_RESULTS.csv",index=False)
    print("OPENBMI_TASK_SEARCH_EVALUATION_COMPLETE",flush=True)
    return 0


if __name__=="__main__": raise SystemExit(main())
