from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.metrics import cohen_kappa_score

from hsc_tta.actions import T3A
from hsc_tta.actions_v2 import RobustResidualAdapter
from hsc_tta.models import make_token_head
from hsc_tta.prediction_sets import evaluate_prediction_sets, prediction_sets


ACTIONS=("no_tta","official_t3a","robust_residual_adapter")
ALPHAS=(.10,.20)
LAMBDAS=np.r_[np.linspace(.50,.99,20),1.0]


def _atomic(frame:pd.DataFrame,path:Path)->None:
    path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_suffix(path.suffix+".part")
    frame.to_parquet(temporary,index=False); os.replace(temporary,path)


def load_source_model(root:Path,dataset:str,seed:int,device:str):
    selected=json.loads((root/"outputs/v2_joint_certified/source_models"/dataset/f"seed_{seed}"/"selected.json").read_text())
    payload=torch.load(selected["model_path"],map_location="cpu",weights_only=False)
    model=make_token_head(payload["architecture"],int(payload["n_classes"])); model.load_state_dict(payload["state_dict"])
    return model.to(device).eval(),payload


def _tokens(root:Path,dataset:str,subject:str,indices:np.ndarray)->torch.Tensor:
    path=root/"data/embeddings_tokens_v2"/dataset/f"{subject.split(':',1)[1]}.h5"
    with h5py.File(path,"r") as handle: return torch.from_numpy(handle["token_embeddings"][np.asarray(indices,int)].astype(np.float32))


def _labels(root:Path,dataset:str,subject:str,indices:np.ndarray)->np.ndarray:
    path=root/"data/embeddings_tokens_v2"/dataset/f"{subject.split(':',1)[1]}.h5"
    with h5py.File(path,"r") as handle: return handle["labels"][np.asarray(indices,int)].astype(np.int64)


def _outputs(model:torch.nn.Module,tokens:torch.Tensor,device:str)->tuple[np.ndarray,np.ndarray]:
    logits=[]; hidden=[]
    with torch.inference_mode():
        for start in range(0,len(tokens),128):
            current,current_hidden=model(tokens[start:start+128].to(device),return_hidden=True)
            logits.append(current.float().cpu().numpy()); hidden.append(current_hidden.float().cpu().numpy())
    return np.concatenate(logits),np.concatenate(hidden)


def _entropy(prob:np.ndarray)->np.ndarray:
    return -(prob*np.log(np.maximum(prob,1e-12))).sum(1)


def _features(subject:str,action:str,source:np.ndarray,current:np.ndarray,hidden:np.ndarray,diagnostics:dict[str,object])->dict[str,object]:
    entropy=_entropy(current); confidence=current.max(1); source_prediction=source.argmax(1); prediction=current.argmax(1)
    result={"subject_id":subject,"action":action,"entropy_q10":np.quantile(entropy,.1),"entropy_q50":np.quantile(entropy,.5),
        "entropy_q90":np.quantile(entropy,.9),"confidence_q10":np.quantile(confidence,.1),"confidence_q50":np.quantile(confidence,.5),
        "confidence_q90":np.quantile(confidence,.9),"temporal_stability":1-float(np.mean(prediction[1:]!=prediction[:-1])) if len(prediction)>1 else 1,
        "representation_mean":float(hidden.mean()),"representation_std":float(hidden.std()),
        "action_entropy_change":float(entropy.mean()-_entropy(source).mean()),
        "prediction_agreement":float(np.mean(prediction==source_prediction)),
        "adaptation_source_kl":float(np.mean(np.sum(current*(np.log(np.maximum(current,1e-12))-np.log(np.maximum(source,1e-12))),axis=1))),
        "prototype_shift":float(diagnostics.get("prototype_shift",0)),"support_count":float(diagnostics.get("support_count",0)),
        "adapter_update_norm":float(diagnostics.get("adapter_update_norm",0)),"collapse_score":float(diagnostics.get("collapse_score",0)),
        "action_failure_indicator":int(diagnostics.get("status","ok")!="ok"),"action_cost":ACTIONS.index(action)}
    for cls,value in enumerate(np.bincount(prediction,minlength=current.shape[1])/len(prediction)): result[f"class_proportion_{cls}"]=value
    sets=prediction_sets(current,LAMBDAS); sizes=sets.sum(2)
    for index in range(len(LAMBDAS)):
        result[f"context_set_size_j{index}"]=float(sizes[:,index].mean())
        result[f"context_singleton_j{index}"]=float(np.mean(sizes[:,index]==1))
    return result


def evaluate_subject(root:Path,dataset:str,seed:int,subject:str,context_indices:np.ndarray,future_indices:np.ndarray,
                     model:torch.nn.Module,device:str)->tuple[list[dict[str,object]],list[dict[str,object]]]:
    context=_tokens(root,dataset,subject,context_indices); future=_tokens(root,dataset,subject,future_indices); labels=_labels(root,dataset,subject,future_indices)
    context_logits,context_hidden=_outputs(model,context,device); future_logits,future_hidden=_outputs(model,future,device)
    context_source=softmax(context_logits.astype(np.float64),axis=1); future_source=softmax(future_logits.astype(np.float64),axis=1)
    context_prob={"no_tta":context_source}; future_prob={"no_tta":future_source}; diagnostics={"no_tta":{"status":"ok"}}
    t3a=T3A(model.classifier.weight.detach().cpu().numpy(),filter_k=20,confidence=None)
    initial=t3a.prototypes.copy(); t3a.adapt(context_hidden,context_logits)
    context_prob["official_t3a"]=t3a.predict_proba(context_hidden); future_prob["official_t3a"]=t3a.predict_proba(future_hidden)
    diagnostics["official_t3a"]={"status":"ok","prototype_shift":float(np.linalg.norm(t3a.prototypes-initial)),
                                  "support_count":len(t3a.supports)}
    robust=RobustResidualAdapter(model,steps=3,learning_rate=5e-5,beta=.5,gamma=.1,eta=1e-3,reliability_quantile=.2,device=device)
    robust.adapt_on_context(context)
    status=robust.failure_status(); diagnostics["robust_residual_adapter"]={"status":status,**robust.diagnostics()}
    if status=="ok":
        context_prob["robust_residual_adapter"]=robust.predict_context(context); robust.freeze_state()
        future_prob["robust_residual_adapter"]=robust.predict_future(future)
    else:
        context_prob["robust_residual_adapter"]=context_source; future_prob["robust_residual_adapter"]=future_source
    feature_rows=[]; outcome_rows=[]; no_error=float(np.mean(future_source.argmax(1)!=labels))
    for action in ACTIONS:
        feature=_features(subject,action,context_source,context_prob[action],context_hidden,diagnostics[action])
        feature.update({"dataset":dataset,"seed":seed,"action_available":diagnostics[action]["status"]=="ok"})
        feature_rows.append(feature)
        curve=evaluate_prediction_sets(future_prob[action],labels,LAMBDAS)
        action_error=float(curve[0]["argmax_error"]); gain=no_error-action_error
        for alpha in ALPHAS:
            critical=next((int(row["lambda_index"]) for row in curve if row["future_risk"]<=alpha),20)
            selected=curve[critical]
            outcome={"dataset":dataset,"seed":seed,"subject_id":subject,"action":action,"alpha":alpha,
                "true_critical_index":critical,"true_benefit":gain,"argmax_error":action_error,
                "future_risk":selected["future_risk"],"future_average_set_size":selected["average_set_size"],
                "future_singleton_rate":selected["singleton_rate"],"macro_f1":selected["macro_f1"],
                "balanced_accuracy":selected["balanced_accuracy"],"cohen_kappa":cohen_kappa_score(labels,future_prob[action].argmax(1)),
                "action_available":diagnostics[action]["status"]=="ok"}
            for index,curve_row in enumerate(curve):
                outcome[f"risk_j{index}"]=curve_row["future_risk"]
                outcome[f"set_size_j{index}"]=curve_row["average_set_size"]
                outcome[f"singleton_j{index}"]=curve_row["singleton_rate"]
            outcome_rows.append(outcome)
    return feature_rows,outcome_rows


def build_development_surfaces(root:str|Path,device:str="cuda",resume:bool=True)->tuple[pd.DataFrame,pd.DataFrame]:
    root=Path(root); out=root/"outputs/v2_joint_certified/actions"; feature_path=out/"DEVELOPMENT_CONTEXT_FEATURES.parquet"; outcome_path=out/"DEVELOPMENT_ACTION_SURFACE.parquet"
    completed=set()
    features=pd.read_parquet(feature_path).to_dict("records") if resume and feature_path.exists() else []
    outcomes=pd.read_parquet(outcome_path).to_dict("records") if resume and outcome_path.exists() else []
    completed={(r["dataset"],int(r["seed"]),r["subject_id"]) for r in features}
    for dataset in ("hmc","eegmmidb"):
        for seed in range(5):
            model,_=load_source_model(root,dataset,seed,device)
            split=json.loads((root/"data/splits"/dataset/f"seed_{seed}.json").read_text()); roles=split["roles"]
            subjects=sorted(set(roles["meta_risk_train"])|set(roles["conformal_calibration"]))
            episodes=pd.read_parquet(root/"data/episodes_main120"/dataset/f"seed_{seed}.parquet").set_index("subject_id")
            for subject in subjects:
                if (dataset,seed,subject) in completed: continue
                row=episodes.loc[subject]; current_features,current_outcomes=evaluate_subject(root,dataset,seed,subject,
                    np.asarray(row.context_indices,int),np.asarray(row.future_indices,int),model,device)
                features.extend(current_features); outcomes.extend(current_outcomes); completed.add((dataset,seed,subject))
                _atomic(pd.DataFrame(features),feature_path); _atomic(pd.DataFrame(outcomes),outcome_path)
                torch.cuda.empty_cache()
    return pd.DataFrame(features),pd.DataFrame(outcomes)
