from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import random

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

from hsc_tta.models import make_token_head


ARCHITECTURES = {
    "hmc": ("old_mean_mlp", "temporal_attention_head"),
    "eegmmidb": ("old_mean_mlp", "channel_temporal_head", "official_downstream_head"),
}


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def _path(root: Path, dataset: str, subject: str) -> Path:
    return root / "data" / "embeddings_tokens_v2" / dataset / f"{subject.split(':',1)[1]}.h5"


def _read(root: Path, dataset: str, subject: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(_path(root, dataset, subject), "r") as handle:
        if not bool(handle.attrs["complete"]): raise RuntimeError("incomplete token embedding")
        return handle["token_embeddings"][...], handle["labels"][...].astype(np.int64)


def _state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _evaluate(model: torch.nn.Module, root: Path, dataset: str, subjects: list[str], device: str,
              batch_size: int, n_classes: int) -> pd.DataFrame:
    rows=[]; model.eval(); labels_all=np.arange(n_classes)
    with torch.inference_mode():
        for subject in subjects:
            x,y=_read(root,dataset,subject); predictions=[]
            for start in range(0,len(y),batch_size):
                xb=torch.as_tensor(x[start:start+batch_size],dtype=torch.float32,device=device)
                predictions.append(model(xb).argmax(1).cpu().numpy())
            pred=np.concatenate(predictions)
            rows.append({"subject_id":subject,"macro_f1":f1_score(y,pred,labels=labels_all,average="macro",zero_division=0),
                         "balanced_accuracy":balanced_accuracy_score(y,pred),"n_windows":len(y),
                         "classes_predicted":int(len(np.unique(pred)))})
    return pd.DataFrame(rows)


def train_candidate(root: Path, dataset: str, seed: int, architecture: str, device: str="cuda",
                    max_epochs: int=12) -> dict[str, object]:
    split=json.loads((root/"data/splits_internal"/dataset/f"seed_{seed}.json").read_text())
    fit,val=split["task_head_fit"],split["task_head_val"]
    n_classes=5 if dataset=="hmc" else 4
    counts=np.zeros(n_classes,float)
    for subject in fit:
        with h5py.File(_path(root,dataset,subject),"r") as handle:
            counts += np.bincount(handle["labels"][...].astype(int),minlength=n_classes)
    weights=counts.sum()/np.maximum(counts*n_classes,1)
    _seed(seed*100+list(ARCHITECTURES[dataset]).index(architecture))
    model=make_token_head(architecture,n_classes).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
    criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device))
    batch_size=48 if architecture=="official_downstream_head" else 256
    best=(-1.0,-1.0); best_state=None; stale=0; logs=[]
    for epoch in range(1,max_epochs+1):
        model.train(); losses=[]
        subject_order=np.random.default_rng(seed*1000+epoch).permutation(fit)
        for subject in subject_order:
            x,y=_read(root,dataset,str(subject)); order=np.random.default_rng(seed*10000+epoch).permutation(len(y))
            for start in range(0,len(y),batch_size):
                idx=order[start:start+batch_size]
                xb=torch.as_tensor(x[idx],dtype=torch.float32,device=device)
                yb=torch.as_tensor(y[idx],dtype=torch.long,device=device)
                optimizer.zero_grad(set_to_none=True); loss=criterion(model(xb),yb)
                if not torch.isfinite(loss): raise FloatingPointError("nonfinite source-head loss")
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step()
                losses.append(float(loss.detach()))
        metrics=_evaluate(model,root,dataset,val,device,batch_size,n_classes)
        current=(float(metrics.macro_f1.mean()),float(metrics.balanced_accuracy.mean()))
        logs.append({"epoch":epoch,"loss":float(np.mean(losses)),"macro_f1":current[0],"balanced_accuracy":current[1]})
        if current>best:
            best=current; best_state=copy.deepcopy(model.state_dict()); stale=0
        else:
            stale+=1
            if stale>=3: break
    assert best_state is not None
    model.load_state_dict(best_state); validation=_evaluate(model,root,dataset,val,device,batch_size,n_classes)
    parameters=sum(p.numel() for p in model.parameters())
    output=root/"outputs/v2_joint_certified/source_models"/dataset/f"seed_{seed}"/architecture
    output.mkdir(parents=True,exist_ok=True); state_hash=_state_hash(best_state)
    torch.save({"state_dict":best_state,"architecture":architecture,"n_classes":n_classes,
                "fit_subjects":fit,"val_subjects":val,"state_hash":state_hash,"parameters":parameters},output/"model.pt")
    pd.DataFrame(logs).to_csv(output/"training_log.csv",index=False)
    validation.to_parquet(output/"validation_subject_metrics.parquet",index=False)
    return {"dataset":dataset,"seed":seed,"architecture":architecture,"macro_f1":validation.macro_f1.mean(),
            "balanced_accuracy":validation.balanced_accuracy.mean(),"classes_predicted_min":validation.classes_predicted.min(),
            "parameters":parameters,"state_hash":state_hash,"model_path":str(output/"model.pt")}


def qualify_source_models(root: str|Path, device: str="cuda", resume: bool=True) -> pd.DataFrame:
    root=Path(root); rows=[]
    for dataset,architectures in ARCHITECTURES.items():
        for seed in range(5):
            for architecture in architectures:
                model_file=root/"outputs/v2_joint_certified/source_models"/dataset/f"seed_{seed}"/architecture/"model.pt"
                metrics_file=model_file.parent/"validation_subject_metrics.parquet"
                if resume and model_file.exists() and metrics_file.exists():
                    payload=torch.load(model_file,map_location="cpu",weights_only=False); metrics=pd.read_parquet(metrics_file)
                    row={"dataset":dataset,"seed":seed,"architecture":architecture,"macro_f1":metrics.macro_f1.mean(),
                         "balanced_accuracy":metrics.balanced_accuracy.mean(),"classes_predicted_min":metrics.classes_predicted.min(),
                         "parameters":payload["parameters"],"state_hash":payload["state_hash"],"model_path":str(model_file)}
                else: row=train_candidate(root,dataset,seed,architecture,device)
                rows.append(row); torch.cuda.empty_cache()
    frame=pd.DataFrame(rows)
    selected=[]
    for (dataset,seed),group in frame.groupby(["dataset","seed"]):
        chosen=group.sort_values(["macro_f1","balanced_accuracy","parameters","architecture"],ascending=[False,False,True,True]).iloc[0]
        selected.append(chosen.to_dict())
        target=root/"outputs/v2_joint_certified/source_models"/dataset/f"seed_{seed}"/"selected.json"
        target.write_text(json.dumps(chosen.to_dict(),indent=2),encoding="utf-8")
    selected_frame=pd.DataFrame(selected)
    out=root/"outputs/v2_joint_certified/source_models"
    frame.to_csv(out/"SOURCE_HEAD_BY_SEED.csv",index=False)
    comparison=frame.groupby(["dataset","architecture"]).agg(macro_f1_mean=("macro_f1","mean"),macro_f1_std=("macro_f1","std"),
        balanced_accuracy_mean=("balanced_accuracy","mean"),parameters=("parameters","first"),classes_predicted_min=("classes_predicted_min","min")).reset_index()
    comparison.to_csv(out/"SOURCE_HEAD_COMPARISON.csv",index=False)
    mi=selected_frame[selected_frame.dataset=="eegmmidb"]
    qualified=bool((mi.macro_f1>.30).all() and (mi.balanced_accuracy>.30).all() and (mi.classes_predicted_min>=3).all())
    report="# Source model qualification\n\n"+comparison.to_markdown(index=False)+"\n\n"
    report+=("EEGMMIDB qualification: **PASS**.\n" if qualified else "EEGMMIDB qualification: **HARD BLOCKER**; the legal source model did not establish stable signal above the preregistered threshold.\n")
    report+=("Official audit: CBraMod commit `0ff6be9`. The official PhysioNet 64-channel `all_patch_reps_twolayer` classifier is implemented as `official_downstream_head`. "
             "The official ISRUC head assumes six channels and 30-epoch sequences and is incompatible with the frozen formal single-C4 episode, so it is not mislabeled as an official HMC implementation. "
             "No partial fine-tuning was selected; all compared heads use the frozen backbone.\n")
    (out/"SOURCE_MODEL_QUALIFICATION.md").write_text(report,encoding="utf-8")
    if not qualified:
        blocker=root/"outputs/v2_joint_certified/HARD_BLOCKER_REPORT.md"
        blocker.write_text(report+"\nSleep-task development may continue, but EEGMMIDB cannot be interpreted as selector evidence.\n",encoding="utf-8")
    return selected_frame
