"""Source-only mechanism diagnostics for the final compact package."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression

import persist_re_core as c


def load_fit(dataset: str, fold: int, seed: int, method: str, recipe: dict[str, object], device: torch.device):
    fd = c.load_fold(dataset, fold, seed)
    train = c.concat_rep(fd.model_fit, fd.validation)
    subjects, mapping = c.subject_index(train["subjects"])
    key = f"{c.FIT_VERSION}_{dataset}_{method}_r{recipe['rank']}_lr{recipe['lambda_R']}_lp{recipe['lambda_P']}_f{fold}_s{seed}_n{len(train['indices'])}"
    path = c.RUNTIME / "fits" / f"{key}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device, weights_only=False)
    model = c.PERSISTRE(train["features"].shape[1], len(subjects), int(recipe["rank"])).to(device)
    model.load_state_dict(payload["state_dict"]); model.eval()
    return fd, train, mapping, model


def identity_probe(dataset: str, recipe: dict[str, object], device: torch.device) -> list[dict[str, object]]:
    fd, train, mapping, model = load_fit(dataset, 0, 0, "ERM", recipe, device)
    rows=[]
    for method in ("ERM", "PERSIST-RE"):
        _, train_m, mapping_m, m = load_fit(dataset, 0, 0, method, recipe, device)
        with torch.no_grad(): z=m.encode(torch.as_tensor(train_m["features"],dtype=torch.float32,device=device)).cpu().numpy()
        y=np.asarray([mapping_m[str(s)] for s in train_m["subjects"]])
        rng=np.random.default_rng(c.stable_seed("identity-probe",dataset,method)); tr=[]; te=[]
        for sid in np.unique(y):
            idx=np.flatnonzero(y==sid); rng.shuffle(idx); cut=max(1,int(.7*len(idx))); tr.extend(idx[:cut]); te.extend(idx[cut:])
        clf=LogisticRegression(max_iter=300,multi_class="auto",random_state=0).fit(z[tr],y[tr]); acc=float((clf.predict(z[te])==y[te]).mean()); rows.append({"dataset":dataset,"method":method,"fold":0,"seed":0,"identity_probe_accuracy":acc,"subjects":len(np.unique(y)),"source_only":True})
    return rows


def decision_rows(dataset: str, recipe: dict[str, object], device: torch.device) -> list[dict[str, object]]:
    out=[]
    for method in ("ERM","PERSIST-RE"):
        fd,train,mapping,m=load_fit(dataset,0,0,method,recipe,device)
        pred=c.predict(m,fd.outcome,mapping,device); logits=pred["population_logits"]; labels=fd.outcome["labels"]; subjects=fd.outcome["subjects"].astype(str); margin=logits[:,1]-logits[:,0]
        vals=[]
        for cls in (0,1):
            means=[float(margin[(subjects==s)&(labels==cls)].mean()) for s in c.subject_sort(np.unique(subjects)) if np.any((subjects==s)&(labels==cls))]
            vals.append({"dataset":dataset,"method":method,"class":cls,"decision_margin_subject_variance":float(np.var(means)),"subjects":len(means),"source_only":True})
        out.extend(vals)
    return out


def random_effect_rows(dataset: str, recipe: dict[str, object], device: torch.device) -> list[dict[str, object]]:
    fd,train,mapping,m=load_fit(dataset,0,0,"PERSIST-RE",recipe,device)
    features=torch.as_tensor(train["features"],dtype=torch.float32,device=device); labels=torch.as_tensor(train["labels"],dtype=torch.long,device=device); sid=torch.as_tensor([mapping[str(s)] for s in train["subjects"]],dtype=torch.long,device=device)
    with torch.no_grad(): p,e,_=m(features,sid,True); ce_pop=float(torch.nn.functional.cross_entropy(p,labels).cpu()); ce_mix=float(torch.nn.functional.cross_entropy(p+e,labels).cpu()); e0,a0=m.centered_effects();
    return [{"dataset":dataset,"center_e_mean_norm":float(e0.mean(0).norm().cpu()),"center_a_mean_norm":float(a0.mean(0).norm().cpu()),"random_effect_variance":float(e0.var().cpu()+a0.var().cpu()),"population_nll":ce_pop,"mixed_nll":ce_mix,"mixed_minus_population_nll":ce_mix-ce_pop,"source_only":True}]


def gradient_rows(dataset: str, recipe: dict[str, object], device: torch.device) -> list[dict[str, object]]:
    rows=[]
    for method in ("ERM","PERSIST-RE"):
        fd,train,mapping,m=load_fit(dataset,0,0,method,recipe,device); features=torch.as_tensor(train["features"],dtype=torch.float32,device=device); labels=torch.as_tensor(train["labels"],dtype=torch.long,device=device); sid=np.asarray([mapping[str(s)] for s in train["subjects"]]); vals=[]
        for subject in sorted(np.unique(sid)):
            mask=torch.as_tensor(sid==subject,device=device); m.zero_grad(set_to_none=True); p,_,_=m(features[mask],None,False); torch.nn.functional.cross_entropy(p,labels[mask]).backward(); vals.append(m.population_head.weight.grad.detach().flatten().cpu().numpy())
        rows.append({"dataset":dataset,"method":method,"population_head_gradient_variance":float(np.var(np.stack(vals),axis=0).mean()),"subjects":len(vals),"source_only":True})
    return rows


def main() -> None:
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); identity=[]; decision=[]; random_effect=[]; gradients=[]
    for dataset in c.DATASETS:
        path=c.RESULTS/f"SOURCE_RECIPE_SELECTION_{dataset}.csv"; recipe=pd.read_csv(path).iloc[0].to_dict(); recipe={"rank":int(recipe["rank"]),"lambda_R":float(recipe["lambda_R"]),"lambda_P":float(recipe["lambda_P"])}
        identity.extend(identity_probe(dataset,recipe,device)); decision.extend(decision_rows(dataset,recipe,device)); random_effect.extend(random_effect_rows(dataset,recipe,device)); gradients.extend(gradient_rows(dataset,recipe,device))
    c.write_csv(c.RESULTS/"IDENTITY_PROBE.csv",pd.DataFrame(identity)); c.write_csv(c.RESULTS/"DECISION_HETEROGENEITY.csv",pd.DataFrame(decision)); c.write_csv(c.RESULTS/"CROSS_SESSION_RE_STABILITY.csv",pd.DataFrame([{"dataset":d,"status":"NOT_AUTHORIZED_BEFORE_CONFIRMATION","future_session_used":False} for d in c.DATASETS])); c.write_csv(c.RESULTS/"GRADIENT_VARIANCE.csv",pd.DataFrame(gradients)); c.write_csv(c.RESULTS/"RANDOM_EFFECT_MECHANISM.csv",pd.DataFrame(random_effect)); print(pd.DataFrame(identity).to_string(index=False),flush=True)


if __name__=="__main__": main()

