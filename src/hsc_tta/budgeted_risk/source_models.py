from __future__ import annotations

import copy
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score, f1_score

from hsc_tta.models import make_token_head

from hsc_tta.contextual_risk.io import atomic_json, atomic_parquet, sha256_file

CACHE_SCHEMA = "budgeted-risk-stage0-cache-v1"


def _seed(value: int) -> None:
    random.seed(value); np.random.seed(value); torch.manual_seed(value); torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True


def _state_hash(state:dict[str,torch.Tensor])->str:
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _subject_path(root:Path,dataset:str,subject:str)->Path:
    return root/"data/embeddings_tokens_v2"/dataset/f"{subject.split(':',1)[1]}.h5"


def _read_all(root:Path,dataset:str,subjects:list[str])->tuple[np.ndarray,np.ndarray]:
    xs=[];ys=[]
    for subject in subjects:
        with h5py.File(_subject_path(root,dataset,subject),"r") as h:
            xs.append(h["token_embeddings"][...]);ys.append(h["labels"][...].astype(np.int64))
    return np.concatenate(xs),np.concatenate(ys)


def _validation(model:torch.nn.Module,x:np.ndarray,y:np.ndarray,device:str,batch_size:int,n_classes:int)->dict[str,float]:
    prediction=[];model.eval()
    with torch.inference_mode():
        for start in range(0,len(y),batch_size):
            xb=torch.as_tensor(x[start:start+batch_size],dtype=torch.float32,device=device)
            prediction.append(model(xb).argmax(1).cpu().numpy())
    pred=np.concatenate(prediction)
    return {"macro_f1":float(f1_score(y,pred,labels=np.arange(n_classes),average="macro",zero_division=0)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),"classes_predicted":int(len(np.unique(pred)))}


def _split_fit_val(subjects:list[str],dataset:str,fold:int)->tuple[list[str],list[str]]:
    ranked=sorted(subjects,key=lambda s:(hashlib.sha256(f"{dataset}|{fold}|{s}|clean-head-val-v1".encode()).hexdigest(),s))
    n_val=max(3,int(np.ceil(.15*len(ranked))))
    return ranked[n_val:],ranked[:n_val]


def train_clean_head(root:Path,repo:Path,dataset:str,fold:int,seed:int,meta_subjects:list[str],config:dict[str,Any],device:str)->dict[str,Any]:
    architecture=str(config["head_architectures"][dataset]);n_classes=5 if dataset=="hmc" else 4
    output=repo/"outputs/budgeted_risk/source_models/stage0"/dataset/f"fold_{fold}"/f"seed_{seed}";model_path=output/"model.pt";manifest_path=output/"manifest.json"
    fit_subjects,val_subjects=_split_fit_val(meta_subjects,dataset,fold)
    if model_path.exists() and manifest_path.exists():
        manifest=json.loads(manifest_path.read_text());payload=torch.load(model_path,map_location="cpu",weights_only=False)
        if manifest.get("state_hash")==_state_hash(payload["state_dict"]) and manifest.get("fit_subjects")==fit_subjects and manifest.get("val_subjects")==val_subjects:return manifest
        raise RuntimeError(f"invalid existing clean head: {model_path}")
    x_train,y_train=_read_all(root,dataset,fit_subjects);x_val,y_val=_read_all(root,dataset,val_subjects)
    counts=np.bincount(y_train,minlength=n_classes).astype(float);weights=counts.sum()/np.maximum(counts*n_classes,1)
    _seed(20260805+fold*100+seed);model=make_token_head(architecture,n_classes).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=float(config["head_learning_rate"]),weight_decay=float(config["head_weight_decay"]));criterion=torch.nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device))
    batch_size=256 if dataset=="hmc" else 48;best=(-np.inf,-np.inf);best_state=None;stale=0;logs=[]
    rng=np.random.default_rng(20260805+fold*1000+seed)
    for epoch in range(1,int(config["head_max_epochs"])+1):
        model.train();losses=[];order=rng.permutation(len(y_train))
        for start in range(0,len(order),batch_size):
            idx=order[start:start+batch_size];xb=torch.as_tensor(x_train[idx],dtype=torch.float32,device=device);yb=torch.as_tensor(y_train[idx],dtype=torch.long,device=device)
            optimizer.zero_grad(set_to_none=True);loss=criterion(model(xb),yb)
            if not torch.isfinite(loss):raise FloatingPointError("nonfinite clean-head loss")
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),float(config["head_gradient_clip"]));optimizer.step();losses.append(float(loss.detach()))
        metrics=_validation(model,x_val,y_val,device,batch_size,n_classes);current=(metrics["macro_f1"],metrics["balanced_accuracy"]);logs.append({"epoch":epoch,"loss":float(np.mean(losses)),**metrics})
        if current>best:best=current;best_state=copy.deepcopy(model.state_dict());stale=0
        else:
            stale+=1
            if stale>=int(config["head_patience"]):break
    if best_state is None:raise RuntimeError("clean head produced no checkpoint")
    state_hash=_state_hash(best_state);output.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":best_state,"architecture":architecture,"n_classes":n_classes,"fit_subjects":fit_subjects,"val_subjects":val_subjects,"state_hash":state_hash},model_path)
    pd.DataFrame(logs).to_csv(output/"training_log.csv",index=False)
    manifest={"schema_version":"budgeted-risk-clean-head-v1","dataset":dataset,"fold":fold,"seed":seed,"architecture":architecture,"fit_subjects":fit_subjects,"val_subjects":val_subjects,"meta_subjects":sorted(meta_subjects),"state_hash":state_hash,"checkpoint_sha256":sha256_file(model_path),"model_path":str(model_path),"validation_macro_f1":best[0],"validation_balanced_accuracy":best[1],"backbone_frozen":True,"token_source":"frozen CBraMod embeddings"}
    atomic_json(manifest,manifest_path);del model,x_train,x_val
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return manifest


def _load_model(manifest:dict[str,Any],device:str)->torch.nn.Module:
    payload=torch.load(manifest["model_path"],map_location="cpu",weights_only=False);model=make_token_head(payload["architecture"],int(payload["n_classes"]));model.load_state_dict(payload["state_dict"]);return model.requires_grad_(False).to(device).eval()


def _infer(model:torch.nn.Module,tokens:np.ndarray,device:str,batch_size:int)->tuple[np.ndarray,np.ndarray]:
    logits=[];hidden=[]
    with torch.inference_mode():
        for start in range(0,len(tokens),batch_size):
            current,current_hidden=model(torch.as_tensor(tokens[start:start+batch_size],dtype=torch.float32,device=device),return_hidden=True);logits.append(current.float().cpu().numpy());hidden.append(current_hidden.float().cpu().numpy())
    return np.concatenate(logits),np.concatenate(hidden)


def cache_stage0_predictions(root:Path,repo:Path,dataset:str,fold:int,seed:int,manifest:dict[str,Any],subjects:list[str],roles:dict[str,str],device:str)->list[dict[str,Any]]:
    model=_load_model(manifest,device);episodes=pd.read_parquet(root/"data/episodes_contextual_risk"/dataset/f"seed_{seed}.parquet").set_index("subject_id");rows=[];batch_size=256 if dataset=="hmc" else 48
    for subject in subjects:
        episode=episodes.loc[subject];target=repo/"outputs/budgeted_risk/source_cache/stage0"/dataset/f"fold_{fold}"/f"seed_{seed}"/f"{subject.replace(':','_')}.npz"
        if target.exists():
            with np.load(target,allow_pickle=False) as z:valid=str(z["schema_version"])==CACHE_SCHEMA and str(z["source_model_hash"])==manifest["checkpoint_sha256"] and str(z["episode_hash"])==episode.episode_hash
            if valid:rows.append({"dataset":dataset,"fold":fold,"seed":seed,"subject_id":subject,"rotation_role":roles[subject],"cache_path":str(target),"status":"reused"});continue
            raise RuntimeError(f"invalid stage0 cache {target}")
        ci=np.asarray(episode.context_indices,int);fi=np.asarray(episode.future_indices,int)
        with h5py.File(_subject_path(root,dataset,subject),"r") as h:
            cx=h["token_embeddings"][ci];fx=h["token_embeddings"][fi];cy=h["labels"][ci].astype(np.int16);fy=h["labels"][fi].astype(np.int16)
        cl,ch=_infer(model,cx,device,batch_size);fl,fh=_infer(model,fx,device,batch_size);target.parent.mkdir(parents=True,exist_ok=True);temporary=target.with_suffix(".npz.part")
        with temporary.open("wb") as handle:np.savez_compressed(handle,context_sample_indices=ci,context_logits=cl.astype(np.float32),context_probabilities=softmax(cl.astype(np.float64),axis=1).astype(np.float32),context_embeddings=ch.astype(np.float16),context_labels_guarded=cy,future_sample_indices=fi,future_logits=fl.astype(np.float32),future_probabilities=softmax(fl.astype(np.float64),axis=1).astype(np.float32),future_labels_guarded=fy,episode_hash=np.asarray(episode.episode_hash),source_model_hash=np.asarray(manifest["checkpoint_sha256"]),schema_version=np.asarray(CACHE_SCHEMA),rotation_role=np.asarray(roles[subject]))
        os.replace(temporary,target);rows.append({"dataset":dataset,"fold":fold,"seed":seed,"subject_id":subject,"rotation_role":roles[subject],"cache_path":str(target),"status":"created"})
    del model
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return rows


def build_clean_stage0_models_and_cache(project_root:str|Path,config:dict[str,Any],device:str="cuda")->tuple[pd.DataFrame,pd.DataFrame]:
    root=Path(project_root);repo=root/"repo";cohorts=pd.read_parquet(repo/"outputs/contextual_risk/cohorts/MASTER_SUBJECT_COHORTS.parquet");dev=cohorts[cohorts.master_cohort=="method_development"]
    manifests=[];cache_rows=[]
    for dataset in ("hmc","eegmmidb"):
        current=dev[dev.dataset==dataset]
        fold_map=current.set_index("subject_id").screening_fold.astype(int).to_dict();subjects=sorted(fold_map)
        formal=set(cohorts[(cohorts.dataset==dataset)&(cohorts.master_cohort=="formal_calibration")].subject_id);final=set(cohorts[(cohorts.dataset==dataset)&(cohorts.master_cohort=="internal_final_evaluation")].subject_id)
        for fold in range(5):
            evaluation={s for s in subjects if fold_map[s]==fold};calibration={s for s in subjects if fold_map[s]==(fold+1)%5};meta=set(subjects)-evaluation-calibration
            roles={s:("evaluation" if s in evaluation else "calibration" if s in calibration else "meta") for s in subjects}
            for seed in range(5):
                manifest=train_clean_head(root,repo,dataset,fold,seed,sorted(meta),config,device);train=set(manifest["fit_subjects"])|set(manifest["val_subjects"])
                manifest.update({"evaluation_subjects":sorted(evaluation),"calibration_subjects":sorted(calibration),"formal_overlap":len(train&formal),"internal_final_overlap":len(train&final),"evaluation_overlap":len(train&evaluation),"calibration_overlap":len(train&calibration),"cap_overlap":0})
                if any(manifest[k] for k in ("formal_overlap","internal_final_overlap","evaluation_overlap","calibration_overlap","cap_overlap")):raise RuntimeError(f"clean head contamination: {dataset}/{fold}/{seed}")
                atomic_json(manifest,Path(manifest["model_path"]).parent/"manifest.json");manifests.append(manifest)
                cache_rows.extend(cache_stage0_predictions(root,repo,dataset,fold,seed,manifest,subjects,roles,device))
    manifest_frame=pd.DataFrame(manifests);cache_frame=pd.DataFrame(cache_rows);atomic_parquet(manifest_frame,repo/"outputs/budgeted_risk/source_models/STAGE0_SOURCE_MODEL_MANIFEST.parquet");atomic_parquet(cache_frame,repo/"outputs/budgeted_risk/source_cache/STAGE0_CACHE_MANIFEST.parquet")
    return manifest_frame,cache_frame
