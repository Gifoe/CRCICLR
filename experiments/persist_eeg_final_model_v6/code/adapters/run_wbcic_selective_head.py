"""Cross-history selective subject head on the WBCIC large EEGNet."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from backbones import train_wbcic_large_eegnet as large
from common import ABLATIONS, CACHE, DIAGNOSTICS, LEADERBOARD, PROTOCOL, V6_SEED, logit, sigmoid, stable_seed, wbcic_source_root, write_csv, write_json
from evaluation.metrics import summarize
from protocol.datasets import load_wbcic_fold


CONFIGS = tuple({"C": c_value, "alpha": alpha} for c_value in (0.001, 0.01, 0.1, 1.0) for alpha in (0.25, 0.5, 0.75, 1.0))


def _head(c_value, seed):
    return Pipeline([("scale", StandardScaler()), ("head", LogisticRegression(C=c_value, class_weight="balanced", solver="liblinear", max_iter=2_000, random_state=seed))])


def _extract(model, x, device):
    features=[]; logits=[]
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(x),128):
            xb=torch.as_tensor(x[start:start+128],dtype=torch.float32,device=device)
            feature=model.forward_features(xb)
            value=model.head(feature)
            features.append(feature.cpu().numpy()); logits.append((value[:,1]-value[:,0]).cpu().numpy())
    return np.concatenate(features),np.concatenate(logits)


def _sessions(subject):
    root=wbcic_source_root()/"experiments"/"persist_eeg_wbcic_actionability_v2"/"outputs"/"cache"/"wbcic_epochs"/subject
    result=[]
    for session in (0,1,2):
        x=np.asarray(np.load(root/f"ses-{session}_epochs.npy",mmap_mode="r",allow_pickle=False),dtype=np.float32)
        y=np.load(root/f"ses-{session}_labels.npy",allow_pickle=False).astype(int)
        result.append((x,y))
    return result


def _select(features, logits, labels, sessions, seed):
    folds=[(sessions==0,sessions==1),(sessions==1,sessions==0)]
    anchor_ba=[]; anchor_nll=[]
    for _,validation in folds:
        p=sigmoid(logits[validation]); anchor_ba.append(balanced_accuracy_score(labels[validation],p>=.5)); anchor_nll.append(log_loss(labels[validation],p,labels=[0,1]))
    rows=[{"configuration":{"C":0.0,"alpha":0.0},"history_cv_BA":float(np.mean(anchor_ba)),"history_cv_NLL":float(np.mean(anchor_nll)),"history_cv_worst_delta_vs_anchor":0.0,"order":-1}]
    for order,configuration in enumerate(CONFIGS):
        bas=[]; nlls=[]; deltas=[]
        for fold_index,(fit,validation) in enumerate(folds):
            model=_head(configuration["C"],seed+order*17+fold_index); model.fit(features[fit],labels[fit])
            hp=model.predict_proba(features[validation])[:,1]
            p=sigmoid((1-configuration["alpha"])*logits[validation]+configuration["alpha"]*logit(hp))
            ba=balanced_accuracy_score(labels[validation],p>=.5); bas.append(ba); nlls.append(log_loss(labels[validation],p,labels=[0,1])); deltas.append(ba-anchor_ba[fold_index])
        rows.append({"configuration":dict(configuration),"history_cv_BA":float(np.mean(bas)),"history_cv_NLL":float(np.mean(nlls)),"history_cv_worst_delta_vs_anchor":float(np.min(deltas)),"order":order})
    selected=max(rows,key=lambda row:(row["history_cv_BA"],-row["history_cv_NLL"],-row["configuration"]["alpha"],-row["order"]))
    improvement=selected["history_cv_BA"]-rows[0]["history_cv_BA"]
    accept=improvement>=.01 and selected["history_cv_worst_delta_vs_anchor"]>=-.05
    return selected,rows,{"anchor_history_cv_BA":rows[0]["history_cv_BA"],"selected_improvement":improvement,"persist_accept":accept}


def _apply(configuration,hf,hy,ff,raw_future_logit,seed):
    if configuration["alpha"]==0:
        return sigmoid(raw_future_logit)
    model=_head(configuration["C"],seed); model.fit(hf,hy); hp=model.predict_proba(ff)[:,1]
    return sigmoid((1-configuration["alpha"])*raw_future_logit+configuration["alpha"]*logit(hp))


def run():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parts=[]; selections=[]
    for fold in range(5):
        payload=torch.load(CACHE/f"WBCIC_LARGE_EEGNET_FOLD_{fold}.pt",map_location="cpu",weights_only=False)
        model=large.LargeEEGNet().to(device); model.load_state_dict(payload["model"],strict=True)
        data=load_wbcic_fold(fold)
        for subject in data.outcome_subjects:
            session_data=_sessions(subject)
            feature_parts=[]; logit_parts=[]; label_parts=[]; session_parts=[]
            for session,(x,y) in enumerate(session_data):
                feature,value=_extract(model,x,device); feature_parts.append(feature); logit_parts.append(value); label_parts.append(y); session_parts.append(np.full(len(y),session))
            features=np.concatenate(feature_parts); logits=np.concatenate(logit_parts); labels=np.concatenate(label_parts); sessions=np.concatenate(session_parts)
            history=sessions<2; future=sessions==2
            selected,rows,audit=_select(features[history],logits[history],labels[history],sessions[history],stable_seed(V6_SEED,"WBCIC-selective-head",fold,subject))
            generic=_apply(selected["configuration"],features[history],labels[history],features[future],logits[future],stable_seed(V6_SEED,"WBCIC-selective-head-fit",fold,subject))
            persist_config=selected["configuration"] if audit["persist_accept"] else {"C":0.0,"alpha":0.0}
            persist=_apply(persist_config,features[history],labels[history],features[future],logits[future],stable_seed(V6_SEED,"WBCIC-selective-head-persist",fold,subject))
            uid=np.asarray([f"WBCIC_nm000348_dev:{subject}:S3:{index}" for index in range(future.sum())])
            for method_id,p,used in (("LARGE_MI_GENERIC_HISTORY_HEAD",generic,selected["configuration"]["alpha"]>0),("PERSIST_LARGE_MI_SELECTIVE_HISTORY_HEAD",persist,persist_config["alpha"]>0)):
                parts.append(pd.DataFrame({"benchmark":data.benchmark,"method_id":method_id,"trial_uid":uid,"subject_id":subject,"outer_fold":fold,"label":labels[future],"probability":p,"prediction":(p>=.5).astype(int),"target_history_labels_used":used,"target_future_labels_used_for_fit":False,"exploratory":True,"OUTER_TEST_USED":False}))
            for row in rows:
                selections.append({"outer_fold":fold,"subject_id":subject,"configuration":json.dumps(row["configuration"],sort_keys=True),"history_cv_BA":row["history_cv_BA"],"history_cv_NLL":row["history_cv_NLL"],"history_cv_worst_delta_vs_anchor":row["history_cv_worst_delta_vs_anchor"],"selected_generic":row is selected,"selected_persist":row["configuration"]==persist_config,**audit,"OUTER_TEST_USED":False})
        print(f"[WBCIC selective head] fold={fold} complete",flush=True)
    predictions=pd.concat(parts,ignore_index=True)
    anchor=large._v5(); aligned=anchor.set_index("trial_uid")
    for method in ("LARGE_MI_GENERIC_HISTORY_HEAD","PERSIST_LARGE_MI_SELECTIVE_HISTORY_HEAD"):
        part=predictions.loc[predictions.method_id.eq(method)].copy(); ap=aligned.loc[part.trial_uid,"probability"].to_numpy(float); cp=part.probability.to_numpy(float)
        blend=part.copy(); bp=sigmoid(.5*(logit(ap)+logit(cp))); blend["method_id"]="V5_FIXED_BLEND__"+method; blend["probability"]=bp; blend["prediction"]=(bp>=.5).astype(int); blend["target_history_labels_used"]=True
        gate=part.copy(); gp=np.where(np.abs(ap-.5)<=.10,cp,ap); gate["method_id"]="V5_UNCERTAINTY_GATE_010__"+method; gate["probability"]=gp; gate["prediction"]=(gp>=.5).astype(int); gate["target_history_labels_used"]=True
        predictions=pd.concat([predictions,blend,gate],ignore_index=True)
    rows=[]; subjects_parts=[]; fold_parts=[]
    ar,asu,af=summarize(anchor); rows.append(ar); subjects_parts.append(asu); fold_parts.append(af)
    for method in predictions.method_id.unique():
        part=predictions.loc[predictions.method_id.eq(method)].copy(); row,subjects,folds=summarize(part,reference=anchor); rows.append(row); subjects_parts.append(subjects); fold_parts.append(folds)
    table=pd.DataFrame(rows).sort_values("mean_subject_BA",ascending=False)
    write_csv(LEADERBOARD/"WBCIC_LARGE_SELECTIVE_HEAD.csv",table); write_csv(DIAGNOSTICS/"WBCIC_LARGE_SELECTIVE_HEAD_PREDICTIONS.csv",predictions); write_csv(DIAGNOSTICS/"WBCIC_LARGE_SELECTIVE_HEAD_SELECTIONS.csv",pd.DataFrame(selections)); write_csv(DIAGNOSTICS/"WBCIC_LARGE_SELECTIVE_HEAD_SUBJECT_RESULTS.csv",pd.concat(subjects_parts,ignore_index=True)); write_csv(DIAGNOSTICS/"WBCIC_LARGE_SELECTIVE_HEAD_FOLD_RESULTS.csv",pd.concat(fold_parts,ignore_index=True)); write_csv(ABLATIONS/"WBCIC_LARGE_SELECTIVE_HEAD_ABLATION.csv",table)
    write_json(PROTOCOL/"WBCIC_LARGE_SELECTIVE_HEAD_AUDIT.json",{"selection":"per-subject S1->S2 and S2->S1 only","PERSIST_gate":"mean history improvement >=1 pp and worst-direction delta >=-5 pp","future_S3_labels_used_for_fit_or_selection":False,"OUTER_TEST_USED":False})
    print(table.to_string(index=False),flush=True)


if __name__=="__main__":
    run()
