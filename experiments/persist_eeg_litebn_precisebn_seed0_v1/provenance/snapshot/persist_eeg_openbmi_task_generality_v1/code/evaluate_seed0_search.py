"""SEARCH-only evaluation for the seed-0 preliminary pilot; never opens heldout cache."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from evaluate_search import logits, score
from task_datasets import OUTPUTS, PROTOCOL, RawGPUCache, build_model, load_bundle, normalizer, sha256


def main() -> int:
    p = json.loads((PROTOCOL / "SEED0_CHECKPOINT_PROVENANCE.json").read_text(encoding="utf-8"))
    if not p.get("pass") or len(p.get("records", [])) != 20 or p.get("heldout_labels_opened"):
        raise RuntimeError("seed0 checkpoint provenance invalid")
    split = json.loads((PROTOCOL / "SUBJECT_SPLIT_REFERENCE.json").read_text(encoding="utf-8")); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    by={(r["task"],int(r["fold"]),r["model"]):r for r in p["records"]}; rows=[]
    for task in ("ERP","SSVEP"):
        bundle=load_bundle(task,split["search_subjects"]); cache=RawGPUCache(bundle,device)
        for fold in split["folds"]:
            mean,std,_=normalizer(bundle,fold["inner_train_subjects"]); models={}
            for name in ("EEGNet","LiteBN"):
                r=by[(task,int(fold["fold_id"]),name)]; path=Path(r["checkpoint_path"])
                if sha256(path)!=r["checkpoint_sha256"]: raise RuntimeError("checkpoint mutation")
                m=build_model(name,task).to(device); m.load_state_dict(torch.load(path,map_location=device,weights_only=False),strict=True); m.eval()
                for q in m.parameters(): q.requires_grad_(False)
                models[name]=m
            for subject in fold["outer_dev_subjects"]:
                ix=bundle.indices([subject],(2,)); y=bundle.labels(ix); ze=logits(models["EEGNet"],cache,ix,mean,std); zl=logits(models["LiteBN"],cache,ix,mean,std)
                for name,z in (("EEGNet",ze),("LiteBN",zl),("LOGIT50",(ze+zl)/2.0)):
                    rows.append({"task":task,"subject_id":subject,"fold":int(fold["fold_id"]),"seed":0,"method":name,"trials":int(len(y)),**score(task,y,z)})
            del models
            if device.type=="cuda": torch.cuda.empty_cache()
        del cache
    out=pd.DataFrame(rows)
    if len(out)!=240 or out.duplicated(["task","subject_id","fold","method"]).any(): raise RuntimeError("seed0 SEARCH cardinality invalid")
    OUTPUTS.mkdir(parents=True,exist_ok=True); out.to_csv(OUTPUTS/"SEED0_SEARCH_REPLICATE_RESULTS.csv",index=False)
    print("OPENBMI_TASK_SEED0_SEARCH_COMPLETE")
    return 0

if __name__=="__main__": raise SystemExit(main())