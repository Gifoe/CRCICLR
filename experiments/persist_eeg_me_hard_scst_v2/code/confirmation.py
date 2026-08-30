"""Frozen ATCNet-Official/EEGNeX confirmation, callable only after discovery passes."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, f1_score

import v2_common as c
from candidate_engine import margins, upper_tail_loss
from discovery import random_geometry
from source_search import build_geometry
from training_components import EMATeacher, configure_scope, primary_total_loss


MODELS=("ATCNet-Official","EEGNeX")
METHODS=("ERM","Factorized-HardRandom","ME-HardSCST")


def indices(fold:int,*,lock_verified:bool):
    raw,metadata,_=c.load_development_data("WBCIC");role=c.roles("WBCIC",fold)
    # Confirmation training must use only the pre-registered WBCIC
    # model-fit/session-0 partition; validation/session-1 is not a training
    # source for the prospective S3 comparison.
    source=tuple(sorted(role["model_fit"]))
    train=c.S1.row_indices(metadata,source,(c.SOURCE_TRAIN_SESSION["WBCIC"],));outcome=c.discovery_indices(fold,lock_verified=lock_verified)
    if np.intersect1d(train,outcome).size:raise RuntimeError("CONFIRMATION_SPLIT_OVERLAP")
    return raw,metadata,train,outcome


@torch.no_grad()
def extract(model,net,raw,rows,device):
    net.eval();values=[]
    for start in range(0,len(rows),c.BATCH_SIZE):
        x=torch.from_numpy(c.normalize_raw(raw[rows[start:start+c.BATCH_SIZE]])).to(device)
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):values.append(c.model_features(model,net,x).float().cpu().numpy())
    return np.concatenate(values).astype(np.float32)


def batch_features(model,net,raw,rows,scope,fixed,device,local=None):
    if scope=="A":return torch.from_numpy(fixed[local]).to(device)
    x=torch.from_numpy(c.normalize_raw(raw[rows])).to(device);return c.model_features(model,net,x)


@torch.no_grad()
def evaluate(model,net,raw,metadata,rows,scope,fixed,fold,seed,method,device):
    net.eval();logits=[]
    for start in range(0,len(rows),c.BATCH_SIZE):
        local=np.arange(start,min(len(rows),start+c.BATCH_SIZE));h=batch_features(model,net,raw,rows[local],scope,fixed,device,local)
        logits.append(c.feature_logits(model,net,h).float().cpu().numpy())
    logits=np.concatenate(logits);labels=metadata.iloc[rows].label.to_numpy(int);subjects=metadata.iloc[rows].subject_id.astype(str).to_numpy();pred=logits.argmax(1);out=[]
    for subject in c.subject_sort(np.unique(subjects)):
        mask=subjects==subject;out.append({"model":model,"method":method,"fold":fold,"seed":seed,"subject_id":subject,"BA":float(balanced_accuracy_score(labels[mask],pred[mask])),"macro_F1":float(f1_score(labels[mask],pred[mask],average="macro",zero_division=0))})
    return pd.DataFrame(out)


def train(model,method,fold,seed,lock,device):
    directory=c.RUNTIME/"confirmation_units"/model/method/f"fold-{fold}"/f"seed-{seed}";result=directory/"per_subject.csv"
    if result.is_file():return pd.read_csv(result)
    raw,metadata,train_rows,outcome_rows=indices(fold,lock_verified=True);labels=metadata.iloc[train_rows].label.to_numpy(int);subjects=metadata.iloc[train_rows].subject_id.astype(str).to_numpy()
    net,_=c.load_anchor(model,"WBCIC",fold,seed,device);scope=str(lock["scope"]);params=configure_scope(model,net,scope);optimizer=torch.optim.AdamW(params,lr=c.LEARNING_RATE,weight_decay=c.WEIGHT_DECAY)
    fixed_train=extract(model,net,raw,train_rows,device) if scope=="A" else None;fixed_outcome=extract(model,net,raw,outcome_rows,device) if scope=="A" else None
    dynamic=method!="ERM";teacher=EMATeacher(net,c.EMA_DECAY) if dynamic else None;q=float(lock["q"]);lam=float(lock["lambda_H"]);rng=np.random.default_rng(c.stable_seed("confirmation-order",model,fold,seed));geometry=random=None
    for epoch in range(c.EPOCHS):
        if dynamic:
            teacher_features=fixed_train if scope=="A" else extract(model,teacher.model,raw,train_rows,device)
            geometry=build_geometry(teacher_features,labels,subjects,train_rows,model,fold,seed,device,factorized=True)
            random=random_geometry(geometry,teacher_features,labels,subjects,train_rows,fold,seed,device)
        order=rng.permutation(len(train_rows));net.train()
        for start in range(0,len(order),c.BATCH_SIZE):
            pos=order[start:start+c.BATCH_SIZE];y=torch.as_tensor(labels[pos],dtype=torch.long,device=device);optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                h=batch_features(model,net,raw,train_rows[pos],scope,fixed_train,device,pos);clean_logits=c.feature_logits(model,net,h);loss=F.cross_entropy(clean_logits.float(),y)
                if dynamic:
                    chosen=random if method=="Factorized-HardRandom" else geometry;other=geometry if method=="Factorized-HardRandom" else random;offset=torch.from_numpy(chosen.offsets[pos]).to(device).detach();other_offset=torch.from_numpy(other.offsets[pos]).to(device).detach();teacher_clean=torch.from_numpy(teacher_features[pos]).to(device)
                    owner=torch.arange(len(pos),device=device).repeat_interleave(offset.shape[1]);owner_y=y[owner];teacher_logits=c.feature_logits(model,teacher.model,(teacher_clean[:,None]+offset).flatten(0,1));other_logits=c.feature_logits(model,teacher.model,(teacher_clean[:,None]+other_offset).flatten(0,1))
                    valid=torch.from_numpy(chosen.base_valid[pos]).to(device).reshape(-1)&teacher_logits.detach().argmax(1).eq(owner_y)&margins(teacher_logits.detach(),owner_y).gt(0);other_valid=torch.from_numpy(other.base_valid[pos]).to(device).reshape(-1)&other_logits.detach().argmax(1).eq(owner_y)&margins(other_logits.detach(),owner_y).gt(0)
                    matched=torch.zeros_like(valid)
                    for local in range(len(pos)):
                        here=owner.eq(local);left=torch.nonzero(valid&here).flatten();right=torch.nonzero(other_valid&here).flatten();count=min(len(left),len(right))
                        if count:
                            pick=np.random.default_rng(c.stable_seed("confirmation-match",model,method,fold,seed,epoch,int(train_rows[pos[local]]))).choice(left.cpu().numpy(),count,replace=False);matched[torch.as_tensor(pick,device=device)]=True
                    logits=c.feature_logits(model,net,(h[:,None]+offset).flatten(0,1));cf,_=upper_tail_loss(clean_logits,logits,y,owner,valid&matched,q=q);loss=primary_total_loss(clean_logits,y,cf,lam)
            loss.backward();torch.nn.utils.clip_grad_norm_(params,3.0);optimizer.step()
            if teacher is not None:teacher.update(net)
        print(f"[confirmation] {model} {method} f={fold} s={seed} e={epoch+1}",flush=True)
    frame=evaluate(model,net,raw,metadata,outcome_rows,scope,fixed_outcome,fold,seed,method,device);directory.mkdir(parents=True,exist_ok=True);c.write_csv(result,frame);torch.save({"state_dict":{k:v.detach().cpu() for k,v in net.state_dict().items()}},directory/"model.pt");return frame


def aggregate():
    files=sorted((c.RUNTIME/"confirmation_units").rglob("per_subject.csv"));frame=pd.concat([pd.read_csv(path) for path in files],ignore_index=True);c.write_csv(c.RESULTS/"CONFIRMATION_PER_SUBJECT.csv",frame);rows=[];model_pass={}
    rng=np.random.default_rng(c.stable_seed("confirmation-bootstrap"))
    for model,group in frame.groupby("model"):
        pivot=group.pivot_table(index=["fold","seed","subject_id"],columns="method",values="BA").reset_index();subject=pivot.groupby("subject_id",as_index=False).mean(numeric_only=True)
        for method in METHODS:
            values=subject[method].to_numpy(float);rows.append({"model":model,"method":method,"BA":float(values.mean())})
        comparisons={}
        for control in ("ERM","Factorized-HardRandom"):
            values=(subject["ME-HardSCST"]-subject[control]).to_numpy(float);draws=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(1);fold_delta=pivot.groupby("fold").apply(lambda x:float((x["ME-HardSCST"]-x[control]).mean()),include_groups=False);comparisons[control]={"delta_BA":float(values.mean()),"CI95_L":float(np.quantile(draws,.025)),"CI95_U":float(np.quantile(draws,.975)),"positive_folds":int((fold_delta>0).sum())}
        model_pass[model]=bool(comparisons["ERM"]["delta_BA"]>0 and comparisons["ERM"]["CI95_L"]>0 and comparisons["ERM"]["positive_folds"]>=3 and comparisons["Factorized-HardRandom"]["CI95_L"]>0);rows.extend({"model":model,"method":f"ME-HardSCST-{control}",**value} for control,value in comparisons.items())
    c.write_csv(c.RESULTS/"CONFIRMATION_SUMMARY.csv",pd.DataFrame(rows));stats={"models":model_pass,"cross_arch_supported":bool(any(model_pass.values())),"outer_or_sealed_opened":False};c.write_json(c.RESULTS/"CONFIRMATION_STATISTICS.json",stats);return stats


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--model",choices=MODELS,required=True);parser.add_argument("--fold",type=int,choices=c.FOLDS,required=True);parser.add_argument("--seed",type=int,choices=c.SEEDS,required=True);parser.add_argument("--aggregate",action="store_true");args=parser.parse_args();lock=c.verify_lock(c.PROTOCOL/"ME_HARD_SCST_V2_LOCK.json");stats=c.read_json(c.RESULTS/"STATISTICS.json")
    if stats.get("discovery_supported") is not True:raise RuntimeError("DISCOVERY_GATE_NOT_PASSED")
    device=torch.device("cuda")
    for method in METHODS:train(args.model,method,args.fold,args.seed,lock,device)
    if args.aggregate:print(json.dumps(aggregate(),indent=2))


if __name__=="__main__":main()
