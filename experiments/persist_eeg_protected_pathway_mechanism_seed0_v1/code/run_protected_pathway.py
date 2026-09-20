"""Frozen final-Protected pathway mechanism audit.

The only Protected object in this program is the final classifier-input
canonical coordinate set fixed by ``persist_eeg_native_protected_utilization``.
Intermediate layers are never used to redefine it.  Train-only linear maps
recover that same final target; their low-rank column spaces are frozen before
outer-development causal interventions.
"""
from __future__ import annotations

import argparse, csv, hashlib, importlib.util, json, os, sys, time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

EXP = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("PERSIST_SOURCE_REPO", str(EXP.parents[1]))).resolve()
UPSTREAM_EXP = ROOT / "experiments" / "persist_eeg_native_protected_utilization_seed0_v1"
OUT, PROTOCOL = EXP / "outputs", EXP / "protocol"
RUNTIME = Path(os.environ.get("PATHWAY_RUNTIME", str(ROOT.parent / "protected_pathway_mechanism_runtime")))

MODELS = ("EEGNet", "EEGConformer", "FBCNet")
TASKS = ("OpenBMI_MI", "OpenBMI_SSVEP")
FOLDS = tuple(range(5)); SEED = 0
RANDOM_DRAWS, RIDGE_ALPHA, CROSS_FITS, CAP, PATCH_CAP = 100, 1.0, 5, 32, 4
EPS = 1e-12


def load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None: raise ImportError(path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod


UP = load("pathway_upstream_native_protected", UPSTREAM_EXP / "code" / "run_native_protected_utilization.py")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""): h.update(block)
    return h.hexdigest()


def stable_seed(*x: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, x)).encode()).digest()[:8], "little") % (2**32 - 1)


def clean(v: Any) -> Any:
    if isinstance(v, Path): return str(v)
    if isinstance(v, np.ndarray): return clean(v.tolist())
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating, float)): return float(v) if np.isfinite(v) else None
    if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    return v


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(clean(value), indent=2, sort_keys=True) + "\n", encoding="utf-8"); os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); keys = list(dict.fromkeys(k for r in rows for k in r)) or ["status"]
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows([{k: clean(r.get(k, "")) for k in keys} for r in rows])
    os.replace(tmp, path)


def path(model: str, task: str, fold: int) -> Path:
    return RUNTIME / "cells" / model.lower() / task.lower() / f"fold{fold}_seed0.json"


def natural(v: np.ndarray | list[str]) -> list[str]: return UP.natural(v)
def centered(z: np.ndarray) -> np.ndarray: return z - z.mean(1, keepdims=True)
def margin(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    own = z[np.arange(len(y)), y]; b = z.copy(); b[np.arange(len(y)), y] = -np.inf; return own - b.max(1)


class Stages:
    """Exact manual stage boundaries plus exact frozen downstream replays."""
    def __init__(self, model: torch.nn.Module, head: torch.nn.Module, model_name: str, device: torch.device):
        self.m, self.head, self.name, self.device = model, head, model_name, device
        self.fb = UP.import_file("pathway_filterbank", UP.EXT.BASELINE_CODE / "filterbank.py") if model_name == "FBCNet" else None
        if model_name == "EEGNet":
            self.names = ("temporal_bn", "spatial_elu_pool1", "depth_point_elu_pool2", "embedding_64d", "classifier_input")
        elif model_name == "EEGConformer":
            self.names = ("patch_tokenizer", "conformer_block2", "conformer_block4", "conformer_block6", "classifier_256_elu", "classifier_32_elu", "classifier_input")
        elif model_name == "FBCNet":
            self.names = ("grouped_spatial", "batch_norm", "swish_gate", "segmented_variance", "log_variance_flatten", "classifier_input")
        else: raise KeyError(model_name)

    def tensor(self, x: np.ndarray) -> torch.Tensor:
        v = np.ascontiguousarray(x, dtype=np.float32)
        if self.fb is not None: v = self.fb.transform(v)
        return torch.from_numpy(v).to(self.device)

    def native_head(self, h: torch.Tensor) -> torch.Tensor: return self.head(h)

    def all(self, x: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        m = self.m
        if self.name == "EEGNet":
            a = m.bn1(m.temporal(x.unsqueeze(1)))
            b = m.drop1(m.pool1(F.elu(m.bn2(m.spatial(a)))))
            c = m.drop2(m.pool2(F.elu(m.bn3(m.point(m.depth(b))))))
            h = m.embedding(c.flatten(1)); z = self.native_head(h)
            return {"temporal_bn":a,"spatial_elu_pool1":b,"depth_point_elu_pool2":c,"embedding_64d":h,"classifier_input":h}, h, z
        if self.name == "EEGConformer":
            a = m.patch(x.unsqueeze(1)).squeeze(2).transpose(1, 2)
            v = a; out = {"patch_tokenizer":a}
            for i, block in enumerate(m.encoder, 1):
                v = block(v)
                if i in (2,4,6): out[f"conformer_block{i}"] = v
            f = v.flatten(1); c0 = m.classifier[0](f); d = m.classifier[1](c0); out["classifier_256_elu"] = d
            d = m.classifier[2](d); d = m.classifier[3](d); d = m.classifier[4](d); out["classifier_32_elu"] = d
            h = m.classifier[5](d); z = self.native_head(h); out["classifier_input"] = h
            return out, h, z
        a = m.spatial(x); b = m.bn(a); c = b * torch.sigmoid(b)
        if m.samples % 4 == 0: d = c.reshape(len(c), 9 * 32, 4, m.samples // 4).var(dim=-1, unbiased=True)
        else: d = torch.stack([q.var(dim=-1, unbiased=True) for q in torch.tensor_split(c, 4, dim=-1)], dim=-1)
        h = torch.log(d.clamp(1e-6, 1e6)).flatten(1); z = self.native_head(h)
        return {"grouped_spatial":a,"batch_norm":b,"swish_gate":c,"segmented_variance":d,"log_variance_flatten":h,"classifier_input":h}, h, z

    def from_stage(self, name: str, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.m
        if self.name == "EEGNet":
            if name == "temporal_bn": v = m.drop1(m.pool1(F.elu(m.bn2(m.spatial(a))))) ; v = m.drop2(m.pool2(F.elu(m.bn3(m.point(m.depth(v)))))); h = m.embedding(v.flatten(1))
            elif name == "spatial_elu_pool1": v = m.drop2(m.pool2(F.elu(m.bn3(m.point(m.depth(a)))))); h = m.embedding(v.flatten(1))
            elif name == "depth_point_elu_pool2": h = m.embedding(a.flatten(1))
            elif name in ("embedding_64d", "classifier_input"): h = a
            else: raise KeyError(name)
            return h, self.native_head(h)
        if self.name == "EEGConformer":
            if name == "patch_tokenizer":
                v = a
                for block in m.encoder: v = block(v)
                f = v.flatten(1); h = m.classifier[5](m.classifier[4](m.classifier[3](m.classifier[2](m.classifier[1](m.classifier[0](f))))))
            elif name.startswith("conformer_block"):
                k = int(name[-1]); v = a
                for block in list(m.encoder)[k:]: v = block(v)
                f = v.flatten(1); h = m.classifier[5](m.classifier[4](m.classifier[3](m.classifier[2](m.classifier[1](m.classifier[0](f))))))
            elif name == "classifier_256_elu": h = m.classifier[5](m.classifier[4](m.classifier[3](m.classifier[2](a))))
            elif name == "classifier_32_elu": h = m.classifier[5](a)
            elif name == "classifier_input": h = a
            else: raise KeyError(name)
            return h, self.native_head(h)
        if name == "grouped_spatial":
            b = m.bn(a); c = b * torch.sigmoid(b)
        elif name == "batch_norm": c = a * torch.sigmoid(a)
        elif name == "swish_gate": c = a
        elif name == "segmented_variance":
            h = torch.log(a.clamp(1e-6,1e6)).flatten(1); return h, self.native_head(h)
        elif name in ("log_variance_flatten", "classifier_input"): return a, self.native_head(a)
        else: raise KeyError(name)
        if m.samples % 4 == 0: d = c.reshape(len(c), 9*32,4,m.samples//4).var(dim=-1,unbiased=True)
        else: d = torch.stack([q.var(dim=-1,unbiased=True) for q in torch.tensor_split(c,4,dim=-1)],dim=-1)
        h = torch.log(d.clamp(1e-6,1e6)).flatten(1); return h, self.native_head(h)


def stage_check(r: Stages, x: np.ndarray) -> None:
    with torch.inference_mode():
        a,h,z = r.all(r.tensor(x[:min(4,len(x))])); direct = r.m(r.tensor(x[:min(4,len(x))]))
        if not torch.allclose(z, direct, rtol=1e-5, atol=1e-6): raise RuntimeError("manual native forward mismatch")
        for n in r.names:
            hh,zz = r.from_stage(n,a[n])
            if not torch.allclose(zz,z,rtol=1e-5,atol=1e-6) or not torch.allclose(hh,h,rtol=1e-5,atol=1e-6): raise RuntimeError(f"downstream replay mismatch at {n}")


def capped_outer(data: dict[str, Any], model: str, task: str, fold: int, session: str) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    if session == "source": x,y,s,se = data["outer_source_x"],data["outer_source_y"],data["outer_source_subjects"],data["source_session"]
    else: x,y,s,se = data["outer_future_x"],data["outer_future_y"],data["outer_future_subjects"],data["future_session"]
    ix = UP.helper(model).capped_indices(s,y,int(se),task,fold,"pathway-outer-"+session)
    return x[ix], y[ix], s[ix].astype(str)


def groups(subjects: np.ndarray, sessions: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray,list[tuple[str,int,int]]]:
    key = [(str(s),int(se),int(y)) for s,se,y in zip(subjects,sessions,labels)]
    order = sorted(set(key), key=lambda x:(int(x[0]) if x[0].isdigit() else x[0],x[1],x[2])); pos={k:i for i,k in enumerate(order)}
    return np.asarray([pos[k] for k in key]), order


def train_centroids(r: Stages, data: dict[str, Any], model: str, task: str, fold: int) -> dict[str, Any]:
    x,y,s,se = UP.capped_train(data,task,model,fold); gi,keys = groups(s,se,y); count=np.bincount(gi,minlength=len(keys)).astype(np.float32)
    sums: dict[str,np.ndarray] = {}; hsum=None; htrials=[]
    with torch.inference_mode():
        for start in range(0,len(x),16):
            batch=gi[start:start+16]; vals,h,_=r.all(r.tensor(x[start:start+16]))
            if hsum is None:
                hsum=np.zeros((len(keys),h.numel()//len(h)),np.float32)
                for n,v in vals.items(): sums[n]=np.zeros((len(keys),v.numel()//len(v)),np.float32)
            hcpu=h.float().cpu().numpy(); np.add.at(hsum,batch,hcpu); htrials.append(hcpu)
            for n,v in vals.items(): np.add.at(sums[n],batch,v.reshape(len(v),-1).float().cpu().numpy())
    if hsum is None: raise RuntimeError("no TRAIN activations")
    return {"keys":keys,"subjects":np.asarray([x[0] for x in keys]),"sessions":np.asarray([x[1] for x in keys]),"labels":np.asarray([x[2] for x in keys]),"h":hsum/count[:,None],"h_trials":np.concatenate(htrials).astype(np.float32),"acts":{n:v/count[:,None] for n,v in sums.items()},"shapes":{n:list(r.all(r.tensor(x[:1]))[0][n].shape[1:]) for n in r.names}}


def outer_stage(r: Stages, x: np.ndarray, name: str) -> dict[str,np.ndarray]:
    aa=[]; hh=[]; zz=[]
    with torch.inference_mode():
        for start in range(0,len(x),16):
            vals,h,z=r.all(r.tensor(x[start:start+16])); aa.append(vals[name].reshape(len(h),-1).float().cpu().numpy()); hh.append(h.float().cpu().numpy()); zz.append(z.float().cpu().numpy())
    return {"a":np.concatenate(aa).astype(np.float32),"h":np.concatenate(hh).astype(np.float32),"z":np.concatenate(zz).astype(np.float32)}


def metrics(y: np.ndarray, p: np.ndarray) -> tuple[float,float,float]:
    mse=float(np.mean((y-p)**2)); var=float(np.mean((y-y.mean(0,keepdims=True))**2)); r2=1-mse/max(var,EPS); corr=[]
    for j in range(y.shape[1]):
        if np.std(y[:,j])>EPS and np.std(p[:,j])>EPS: corr.append(float(np.corrcoef(y[:,j],p[:,j])[0,1]))
    return r2,mse/max(var,EPS),float(np.mean(corr)) if corr else float("nan")


def standardise(a: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    mu=a.mean(0,dtype=np.float64).astype(np.float32); sd=a.std(0,dtype=np.float64).astype(np.float32); sd=np.maximum(sd,1e-6); return (a-mu)/sd,mu,sd


def ridge_predict(fit_a: np.ndarray, fit_y: np.ndarray, test_a: np.ndarray) -> np.ndarray:
    af,mu,sd=standardise(fit_a); at=(test_a-mu)/sd; dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.inference_mode():
        x=torch.from_numpy(np.ascontiguousarray(af)).to(dev); q=torch.from_numpy(np.ascontiguousarray(fit_y)).to(dev); t=torch.from_numpy(np.ascontiguousarray(at)).to(dev)
        k=(x@x.T)/max(x.shape[1],1); eye=torch.eye(len(x),device=dev,dtype=x.dtype); alpha=torch.linalg.solve(k+RIDGE_ALPHA*eye,q); out=(t@x.T/max(x.shape[1],1))@alpha
    return out.cpu().numpy().astype(np.float32)


def score_crossfit(a: np.ndarray, qall: np.ndarray, subjects: np.ndarray, sessions: np.ndarray, direction: int, model: str, task: str, fold: int) -> list[dict[str,Any]]:
    """Fit one session, evaluate held subjects in same or opposite session."""
    src, dst = (1,1) if direction == 0 else (2,2) if direction == 1 else (1,2) if direction == 2 else (2,1)
    subs=natural(subjects); out=[]
    for k in range(CROSS_FITS):
        held=np.asarray([s for i,s in enumerate(subs) if i % CROSS_FITS == k])
        fit=(sessions==src)&~np.isin(subjects,held); test=(sessions==dst)&np.isin(subjects,held)
        if fit.sum()<4 or not test.any(): continue
        pred=ridge_predict(a[fit],qall[fit],a[test]); yy=qall[test]; ss=subjects[test]
        for s in natural(ss):
            ix=np.flatnonzero(ss==s); p=pred[ix]; y=yy[ix]; rank=qall.shape[1]//(RANDOM_DRAWS+1)
            pr,pn,pc=metrics(y[:,:rank],p[:,:rank]); rr=[]
            for j in range(RANDOM_DRAWS): rr.append(metrics(y[:,rank*(j+1):rank*(j+2)],p[:,rank*(j+1):rank*(j+2)])[0])
            label=("within_s1" if direction==0 else "within_s2" if direction==1 else "s1_to_s2" if direction==2 else "s2_to_s1")
            out.append({"subject_id":s,"direction":label,"protected_R2":pr,"random_R2":float(np.mean(rr)),"R2_excess":pr-float(np.mean(rr)),"normalized_MSE":pn,"coordinate_correlation":pc})
    return out


class PathFit:
    def __init__(self,a:np.ndarray):
        self.x,self.mean,self.std=standardise(a); self.dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.inference_mode():
            self.xt=torch.from_numpy(np.ascontiguousarray(self.x)).to(self.dev); self.k=(self.xt@self.xt.T)/max(self.xt.shape[1],1); self.chol=torch.linalg.cholesky(self.k+RIDGE_ALPHA*torch.eye(len(self.xt),device=self.dev))
    def q(self,y:np.ndarray)->tuple[np.ndarray,np.ndarray]:
        with torch.inference_mode():
            yy=torch.from_numpy(np.ascontiguousarray(y)).to(self.dev); alpha=torch.cholesky_solve(yy,self.chol); b=(self.xt.T@alpha)/max(self.xt.shape[1],1)
        b=b.cpu().numpy().astype(np.float32); u,s,_=np.linalg.svd(b,full_matrices=False); r=max(1,int(np.sum(s>max(float(s[0])*1e-6,1e-8)))) if len(s) else 0
        if not r: raise RuntimeError("empty ridge pathway")
        return u[:,:r].astype(np.float32),s[:r].astype(np.float32)


def overlap(a:np.ndarray,b:np.ndarray)->float:
    if not len(a) or not len(b): return float("nan")
    return float(np.linalg.norm(a.T@b,"fro")**2/max(min(a.shape[1],b.shape[1]),1))


def sketch(a:np.ndarray, model:str, task:str, fold:int, stage:str)->np.ndarray:
    d=a.shape[1]; rng=np.random.default_rng(stable_seed("matching-sketch",model,task,fold,stage)); ids=np.sort(rng.choice(d,min(32,d),replace=False)); x=a[:,ids]; return (x-x.mean(0))/np.maximum(x.std(0),1e-6)


def cross_label_pairs(a:np.ndarray,y:np.ndarray,s:np.ndarray,model:str,task:str,fold:int,stage:str)->list[tuple[int,int]]:
    v=sketch(a,model,task,fold,stage); pairs=[]
    for sub in natural(s):
        ix=np.flatnonzero(s==sub)
        for lab in sorted(np.unique(y[ix])):
            rec=ix[y[ix]==lab]; rng=np.random.default_rng(stable_seed("cross-label-pairs",model,task,fold,stage,sub,int(lab))); rec=np.sort(rng.choice(rec,min(PATCH_CAP,len(rec)),replace=False))
            for i in rec:
                for donor_lab in sorted(np.unique(y[ix])):
                    if donor_lab==y[i]: continue
                    cand=ix[y[ix]==donor_lab]; j=cand[np.argmin(np.sum((v[cand]-v[i])**2,axis=1))]; pairs.append((int(i),int(j)))
    return pairs


def session_pairs(a1:np.ndarray,y1:np.ndarray,s1:np.ndarray,a2:np.ndarray,y2:np.ndarray,s2:np.ndarray,model:str,task:str,fold:int,stage:str)->list[tuple[int,int,int]]:
    v1,v2=sketch(a1,model,task,fold,stage+"-s1"),sketch(a2,model,task,fold,stage+"-s2"); out=[]
    for side,(va,ya,sa,vb,yb,sb) in enumerate(((v1,y1,s1,v2,y2,s2),(v2,y2,s2,v1,y1,s1))):
        for sub in natural(sa):
            ix=np.flatnonzero(sa==sub)
            for lab in sorted(np.unique(ya[ix])):
                rec=ix[ya[ix]==lab]; rng=np.random.default_rng(stable_seed("session-pairs",model,task,fold,stage,side,sub,int(lab))); rec=np.sort(rng.choice(rec,min(PATCH_CAP,len(rec)),replace=False))
                cand=np.flatnonzero((sb==sub)&(yb==lab))
                for i in rec:
                    if len(cand): out.append((side,int(i),int(cand[np.argmin(np.sum((vb[cand]-va[i])**2,axis=1))])))
    return out


def canonical(h:np.ndarray,spec:dict[str,Any])->np.ndarray: return UP.helper("EEGNet").canonical(h,spec)


def patch_effect(r:Stages,name:str,qpath:np.ndarray,fit:PathFit,spec:dict[str,Any],dims:np.ndarray,base:dict[str,np.ndarray],y:np.ndarray,s:np.ndarray,pairs:list[tuple[int,int]], interaction:bool=False)->dict[str,dict[str,float]]:
    if not pairs: return {}
    ii=np.asarray([x[0] for x in pairs]); jj=np.asarray([x[1] for x in pairs]); d=np.setdiff1d(np.arange(spec["rank"]),dims)
    qdev=torch.from_numpy(qpath).to(r.device); mu=torch.from_numpy(fit.mean).to(r.device); sd=torch.from_numpy(fit.std).to(r.device)
    row=defaultdict(lambda:defaultdict(list))
    with torch.inference_mode():
        for start in range(0,len(ii),8):
            a,b=ii[start:start+8],jj[start:start+8]; ai=torch.from_numpy(base["a"][a]).to(r.device); aj=torch.from_numpy(base["a"][b]).to(r.device)
            xi=(ai-mu)/sd; xj=(aj-mu)/sd; pi=(xi@qdev)@qdev.T; pj=(xj@qdev)@qdev.T
            x10=pj+(xi-pi); h10,z10=r.from_stage(name,x10*sd+mu); q10=canonical(h10.float().cpu().numpy(),spec)
            z0=base["z"][a]; q0=canonical(base["h"][a],spec); donor=y[b]; dz=centered(z10.float().cpu().numpy()-z0)
            if interaction:
                x01=pi+(xj-pj); _,z01=r.from_stage(name,x01*sd+mu); z11=base["z"][b]; inter=np.linalg.norm(centered(z0-z10.float().cpu().numpy()-z01.float().cpu().numpy()+z11),axis=1)
            for k,owner in enumerate(s[a].astype(str)):
                row[owner]["delta_qP"].append(float(np.sqrt(np.mean((q10[k,dims]-q0[k,dims])**2))))
                row[owner]["delta_qnonP"].append(float(np.sqrt(np.mean((q10[k,d]-q0[k,d])**2))) if len(d) else 0.0)
                row[owner]["logit_rms"].append(float(np.sqrt(np.mean(dz[k]**2))))
                row[owner]["donor_margin_transfer"].append(float(z10[k,donor[k]].item()-z0[k,donor[k]]))
                row[owner]["flip_rate"].append(float(z10[k].argmax().item()!=z0[k].argmax()))
                row[owner]["true_margin_loss"].append(float(margin(z0[[k]],y[a][[k]])[0]-margin(z10.float().cpu().numpy()[[k]],y[a][[k]])[0]))
                row[owner]["donor_label_transfer"].append(float(z10[k].argmax().item()==donor[k]))
                if interaction: row[owner]["interaction"].append(float(inter[k]))
    return {sub:{k:float(np.mean(v)) for k,v in vals.items()} for sub,vals in row.items()}


def merge_random(rows:list[dict[str,dict[str,float]]])->dict[str,dict[str,float]]:
    out={}
    for sub in set().union(*(r.keys() for r in rows)):
        vals=[r[sub] for r in rows if sub in r]; out[sub]={k:float(np.mean([x[k] for x in vals])) for k in vals[0]}
    return out


def session_effect(r:Stages,name:str,qpath:np.ndarray,fit:PathFit,spec:dict[str,Any],dims:np.ndarray,a1:dict[str,np.ndarray],y1:np.ndarray,s1:np.ndarray,a2:dict[str,np.ndarray],y2:np.ndarray,s2:np.ndarray,pairs:list[tuple[int,int,int]])->dict[str,dict[str,float]]:
    ret=defaultdict(lambda:defaultdict(list)); qdev=torch.from_numpy(qpath).to(r.device); mu=torch.from_numpy(fit.mean).to(r.device); sd=torch.from_numpy(fit.std).to(r.device)
    d=np.setdiff1d(np.arange(spec["rank"]),dims)
    for side in (0,1):
        ps=[x for x in pairs if x[0]==side]
        if not ps: continue
        rec,don=np.asarray([x[1] for x in ps]),np.asarray([x[2] for x in ps]); base,other,yy,ss=(a1,a2,y1,s1) if side==0 else (a2,a1,y2,s2)
        with torch.inference_mode():
            for start in range(0,len(rec),8):
                ii,jj=rec[start:start+8],don[start:start+8]; ai=torch.from_numpy(base["a"][ii]).to(r.device); aj=torch.from_numpy(other["a"][jj]).to(r.device)
                xi=(ai-mu)/sd; xj=(aj-mu)/sd; pi=(xi@qdev)@qdev.T; pj=(xj@qdev)@qdev.T; h,z=r.from_stage(name,(pj+(xi-pi))*sd+mu)
                q=canonical(h.float().cpu().numpy(),spec); q0=canonical(base["h"][ii],spec); z0=base["z"][ii]; zz=z.float().cpu().numpy()
                for k,sub in enumerate(ss[ii].astype(str)):
                    ret[sub][f"qP_{side}"].append(float(np.sqrt(np.mean((q[k,dims]-q0[k,dims])**2))))
                    ret[sub][f"logit_{side}"].append(float(np.sqrt(np.mean(centered((zz[k]-z0[k])[None])**2))))
                    ret[sub][f"margin_{side}"].append(float(margin(z0[[k]],yy[ii][[k]])[0]-margin(zz[[k]],yy[ii][[k]])[0]))
                    ret[sub][f"consistent_{side}"].append(float(zz[k].argmax()==z0[k].argmax()))
    out={}
    for sub,v in ret.items():
        flat={k:float(np.mean(x)) for k,x in v.items()}; x=np.asarray([flat.get("qP_0",np.nan),flat.get("logit_0",np.nan),flat.get("margin_0",np.nan)]); y=np.asarray([flat.get("qP_1",np.nan),flat.get("logit_1",np.nan),flat.get("margin_1",np.nan)])
        flat["effect_cosine"]=float(np.dot(x,y)/(max(np.linalg.norm(x)*np.linalg.norm(y),EPS))) if np.isfinite(x).all() and np.isfinite(y).all() else float("nan"); out[sub]=flat
    return out


def output_evidence_stability(h1:np.ndarray,y1:np.ndarray,s1:np.ndarray,h2:np.ndarray,y2:np.ndarray,s2:np.ndarray,spec:dict[str,Any],dims:np.ndarray,head:torch.nn.Module)->dict[str,dict[str,float]]:
    """Same-subject/class source--future stability of the exact final P evidence."""
    q1,q2=canonical(h1,spec),canonical(h2,spec); raw=UP.raw_base(spec); w=head.weight.detach().float().cpu().numpy(); zp1=(q1[:,dims]@raw[dims])@w.T; zp2=(q2[:,dims]@raw[dims])@w.T; out={}
    for sub in sorted(set(s1.astype(str)) & set(s2.astype(str)), key=lambda x:int(x) if x.isdigit() else x):
        a=[]; b=[]
        for lab in sorted(set(y1[s1.astype(str)==sub]) & set(y2[s2.astype(str)==sub])):
            a.append(centered(zp1[(s1.astype(str)==sub)&(y1==lab)]).mean(0)); b.append(centered(zp2[(s2.astype(str)==sub)&(y2==lab)]).mean(0))
        if not a: continue
        x,y=np.concatenate(a),np.concatenate(b); out[sub]={"output_P_evidence_cosine":float(np.dot(x,y)/max(np.linalg.norm(x)*np.linalg.norm(y),EPS)),"output_P_evidence_normalized_rms":float(np.sqrt(np.mean((x-y)**2))/max(np.sqrt(np.mean(x*x)),EPS))}
    return out


def decision(h:np.ndarray,z:np.ndarray,y:np.ndarray,s:np.ndarray,spec:dict[str,Any],dims:np.ndarray,head:torch.nn.Module)->tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    if not isinstance(head,torch.nn.Linear): raise RuntimeError("native classification head is not affine Linear")
    raw=UP.raw_base(spec); q=canonical(h,spec); hp=q[:,dims]@raw[dims]; other=np.setdiff1d(np.arange(spec["rank"]),dims); hn=q[:,other]@raw[other] if len(other) else np.zeros_like(h); rr=UP.residual(h,q,spec)
    w=head.weight.detach().float().cpu().numpy(); b=head.bias.detach().float().cpu().numpy() if head.bias is not None else 0.0
    zb=spec["mean"]@w.T+b; zp=hp@w.T; zn=hn@w.T; zr=rr@w.T; rec=zb[None]+zp+zn+zr
    if not np.allclose(z,rec,rtol=2e-5,atol=3e-5): raise RuntimeError("exact affine decision decomposition failed")
    c=z.copy(); c[np.arange(len(y)),y]=-np.inf; comp=c.argmax(1); mp=zp[np.arange(len(y)),y]-zp[np.arange(len(y)),comp]; mn=zn[np.arange(len(y)),y]-zn[np.arange(len(y)),comp]; mr=zr[np.arange(len(y)),y]-zr[np.arange(len(y)),comp]; correct=z.argmax(1)==y; out=[]; conflict=[]
    for sub in natural(s):
        ix=np.flatnonzero(s.astype(str)==sub); co=mn[ix]+mr[ix]
        out.append({"subject_id":sub,"mean_abs_zP":float(np.mean(np.abs(centered(zp[ix])))),"mean_abs_zN":float(np.mean(np.abs(centered(zn[ix])))),"mean_abs_z_residual":float(np.mean(np.abs(centered(zr[ix])))),"m_P":float(np.mean(mp[ix])),"m_N":float(np.mean(mn[ix])),"m_residual":float(np.mean(mr[ix])),"reconstruction_max_abs_error":float(np.max(np.abs(z[ix]-rec[ix]))),"n_trials":len(ix)})
        flags={"P_SUPPORT_COMPLEMENT_SUPPORT":(mp[ix]>0)&(co>0),"P_SUPPORT_COMPLEMENT_OPPOSE":(mp[ix]>0)&(co<0),"P_OPPOSE_COMPLEMENT_SUPPORT":(mp[ix]<0)&(co>0),"P_OPPOSE_COMPLEMENT_OPPOSE":(mp[ix]<0)&(co<0)}
        row={"subject_id":sub}; row.update({k:float(np.mean(v)) for k,v in flags.items()}); row.update({"protected_correct_complement_override_rate":float(np.mean((mp[ix]>0)&~correct[ix]&(co<0))),"complement_rescue_rate":float(np.mean((mp[ix]<0)&correct[ix]&(co>0))),"P_dominant_correct_rate":float(np.mean(correct[ix]&(mp[ix]>0)&(np.abs(mp[ix])>np.abs(co)))),"complement_dominant_correct_rate":float(np.mean(correct[ix]&(co>0)&(np.abs(co)>np.abs(mp[ix])))),"correct_trials":int(correct[ix].sum()),"error_trials":int((~correct[ix]).sum())}); conflict.append(row)
    return out,conflict


def previous(model:str,task:str,fold:int)->tuple[dict[str,Any],dict[str,Any],Path,dict[str,Any]]:
    record,ckpt,stored,prov=UP.provenance_row(model,task,fold); old=UP.target_path(model,task,fold)
    if not old.is_file(): raise FileNotFoundError(f"previous native audit cell missing: {old}")
    prior=json.loads(old.read_text(encoding="utf-8"));
    if prior.get("checkpoint_sha256")!=prov["checkpoint_sha256"] or prior.get("protected_assignment_sha256")!=prov["protected_assignment_sha256"]: raise RuntimeError("previous locked checkpoint/P assignment mismatch")
    return record,stored,ckpt,prior


def cell(model:str,task:str,fold:int)->None:
    target=path(model,task,fold)
    if target.is_file(): print("CELL_CACHED",model,task,fold,flush=True); return
    base={"model":model,"task":task,"fold":fold,"seed":SEED,"training_performed":False,"final_heldout_accessed":False}
    try:
        record,stored,ckpt,prior=previous(model,task,fold)
        if prior.get("status") == "EMPTY_PROTECTED": write_json(target,{**base,"status":"EMPTY_PROTECTED","previous_cell_sha256":sha(UP.target_path(model,task,fold))}); return
        if prior.get("status") != "COMPLETE": raise RuntimeError("previous native cell is not complete")
        data=UP.outer_data(task,fold)
        if data["normalizer"]["mean_std_sha256"] != record["normalizer"]["mean_std_sha256"]: raise RuntimeError("TRAIN normalizer hash mismatch")
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net,head=UP.helper(model).build_model({"Model":model,"Task":task,"fold":fold,"seed":0,"channels":int(record.get("channels") or 62),"samples":int(record.get("samples") or 1000),"classes":int(record["classes"]),"checkpoint_path":str(ckpt),"recipe_name":record.get("recipe",{}).get("name"),"trainable_parameters":int(record.get("trainable_parameters",record.get("parameters",0)))},device)
        runner=Stages(net,head,model,device); tx,ty,ts,tse=UP.capped_train(data,task,model,fold); stage_check(runner,tx)
        train=train_centroids(runner,data,model,task,fold); spec=UP.helper(model).spectrum(train["h_trials"],ty,ts,tse,task,model,fold); dims=np.asarray(stored.get("protected_blocks",[]),int)
        basis_sha=UP.array_sha(spec["mean"],spec["basis"],spec["scale"],spec["directions"])
        if basis_sha != prior.get("basis_sha256") or int(spec["rank"])!=int(stored.get("rank",-1)) or np.any(dims<0) or np.any(dims>=spec["rank"]): raise RuntimeError("frozen final P/basis cannot be reproduced")
        randoms=UP.random_dims(spec["rank"],len(dims),model,task,fold,"native-equal-rank-random")
        qtrain=canonical(train["h"],spec); qall=np.concatenate([qtrain[:,dims]]+[qtrain[:,d] for d in randoms],axis=1).astype(np.float32)
        ox1,oy1,os1=capped_outer(data,model,task,fold,"source"); ox2,oy2,os2=capped_outer(data,model,task,fold,"future")
        final=outer_stage(runner,ox2,"classifier_input"); source_final=outer_stage(runner,ox1,"classifier_input"); evidence_stability=output_evidence_stability(source_final["h"],oy1,os1,final["h"],oy2,os2,spec,dims,head); decomp,conflict=decision(final["h"],final["z"],oy2,os2,spec,dims,head)
        rec_rows=[]; path_rows=[]; causal_rows=[]; inter_rows=[]; session_rows=[]
        for name in runner.names:
            a=train["acts"][name]; shape=train["shapes"][name]; subs=np.asarray(train["subjects"]); ses=np.asarray(train["sessions"])
            cf=score_crossfit(a,qall,subs,ses,0,model,task,fold)+score_crossfit(a,qall,subs,ses,1,model,task,fold)+score_crossfit(a,qall,subs,ses,2,model,task,fold)+score_crossfit(a,qall,subs,ses,3,model,task,fold)
            def avg(rows,key,cond=None):
                z=[r[key] for r in rows if cond is None or r["direction"]==cond]; return float(np.nanmean(z)) if z else float("nan")
            rec_rows.append({"layer_name":name,"activation_shape":shape,"protected_rank":len(dims),"within_session_R2":float(np.nanmean([avg(cf,"protected_R2","within_s1"),avg(cf,"protected_R2","within_s2")])),"cross_session_R2":float(np.nanmean([avg(cf,"protected_R2","s1_to_s2"),avg(cf,"protected_R2","s2_to_s1")])),"random_R2":float(np.nanmean([avg(cf,"random_R2","s1_to_s2"),avg(cf,"random_R2","s2_to_s1")])),"R2_excess":float(np.nanmean([avg(cf,"R2_excess","s1_to_s2"),avg(cf,"R2_excess","s2_to_s1")])),"normalized_MSE":float(np.nanmean([avg(cf,"normalized_MSE","s1_to_s2"),avg(cf,"normalized_MSE","s2_to_s1")])),"coordinate_correlation":float(np.nanmean([avg(cf,"coordinate_correlation","s1_to_s2"),avg(cf,"coordinate_correlation","s2_to_s1")])),"subject_crossfit_rows":cf})
            fit=PathFit(a); qp,sv=fit.q(qtrain[:,dims]); p1=PathFit(a[ses==data["source_session"]]); p2=PathFit(a[ses==data["future_session"]]); q1,_=p1.q(qtrain[ses==data["source_session"]][:,dims]); q2,_=p2.q(qtrain[ses==data["future_session"]][:,dims])
            random_overlap=[]
            for rd in randoms:
                r1,_=p1.q(qtrain[ses==data["source_session"]][:,rd]); r2,_=p2.q(qtrain[ses==data["future_session"]][:,rd]); random_overlap.append(overlap(r1,r2))
            path_rows.append({"layer_name":name,"activation_shape":shape,"B_rank":qp.shape[1],"pathway_dimension":qp.shape[1],"singular_values":sv.tolist(),"session1_session2_subspace_overlap":overlap(q1,q2),"random_overlap":float(np.nanmean(random_overlap)),"overlap_excess":overlap(q1,q2)-float(np.nanmean(random_overlap))})
            # The final classifier-input stage is affine; it is retained for recoverability only.
            nonlinear = name != "classifier_input"
            if nonlinear:
                fut=outer_stage(runner,ox2,name); src=outer_stage(runner,ox1,name); pairs=cross_label_pairs(fut["a"],oy2,os2,model,task,fold,name); spairs=session_pairs(src["a"],oy1,os1,fut["a"],oy2,os2,model,task,fold,name)
                pp=patch_effect(runner,name,qp,fit,spec,dims,fut,oy2,os2,pairs,True); rr=[]; ri=[]; sr=[]
                for rd in randoms:
                    qr,_=fit.q(qtrain[:,rd]); rr.append(patch_effect(runner,name,qr,fit,spec,dims,fut,oy2,os2,pairs,True)); ri.append({s:v.get("interaction",float("nan")) for s,v in rr[-1].items()}); sr.append(session_effect(runner,name,qr,fit,spec,dims,src,oy1,os1,fut,oy2,os2,spairs))
                rm=merge_random(rr); se=session_effect(runner,name,qp,fit,spec,dims,src,oy1,os1,fut,oy2,os2,spairs); sm=merge_random(sr)
                for sub,v in pp.items(): causal_rows.append({"layer_name":name,"subject_id":sub,**{k:v.get(k,float("nan")) for k in ("delta_qP","delta_qnonP","donor_margin_transfer","donor_label_transfer","logit_rms","flip_rate","true_margin_loss")},**{"random_"+k:rm.get(sub,{}).get(k,float("nan")) for k in ("delta_qP","delta_qnonP","donor_margin_transfer","donor_label_transfer","logit_rms","flip_rate","true_margin_loss")}}); inter_rows.append({"layer_name":name,"subject_id":sub,"interaction_P":v.get("interaction",float("nan")),"interaction_random":rm.get(sub,{}).get("interaction",float("nan")),"interaction_excess":v.get("interaction",float("nan"))-rm.get(sub,{}).get("interaction",float("nan"))})
                for sub,v in se.items(): session_rows.append({"layer_name":name,"subject_id":sub,**v,**evidence_stability.get(sub,{}),"random_effect_cosine":sm.get(sub,{}).get("effect_cosine",float("nan")),"random_qP_0":sm.get(sub,{}).get("qP_0",float("nan")),"random_qP_1":sm.get(sub,{}).get("qP_1",float("nan"))})
            else:
                path_rows[-1]["causal_status"]="NOT_APPLICABLE_FINAL_AFFINE_HEAD"
        prov={"checkpoint_sha256":sha(ckpt),"normalizer_sha256":data["normalizer"]["mean_std_sha256"],"basis_sha256":basis_sha,"protected_assignment_sha256":hashlib.sha256(json.dumps(stored.get("protected_assignment",[]),sort_keys=True).encode()).hexdigest(),"split_sha256":data["split_sha256"],"previous_cell_sha256":sha(UP.target_path(model,task,fold)),"protected_dims":dims.tolist(),"random_subsets_sha256":hashlib.sha256(np.concatenate(randoms).tobytes()).hexdigest()}
        write_json(target,{**base,"status":"COMPLETE",**prov,"stage_names":runner.names,"decision":decomp,"conflict":conflict,"recoverability":rec_rows,"pathway":path_rows,"causal":causal_rows,"interaction":inter_rows,"session":session_rows})
        print("CELL_COMPLETE",model,task,fold,flush=True); del net; torch.cuda.empty_cache()
    except Exception as e:
        write_json(target,{**base,"status":"FAIL_CLOSED","reason":f"{type(e).__name__}: {e}"}); print("CELL_FAIL_CLOSED",model,task,fold,str(e),flush=True)


def lock() -> None:
    rows=[]
    for m in MODELS:
        for t in TASKS:
            for f in FOLDS:
                record,stored,ckpt,prior=previous(m,t,f); data=UP.outer_data(t,f)
                rows.append({"model":m,"task":t,"fold":f,"checkpoint_sha256":sha(ckpt),"normalizer_sha256":data["normalizer"]["mean_std_sha256"],"protected_assignment_sha256":hashlib.sha256(json.dumps(stored.get("protected_assignment",[]),sort_keys=True).encode()).hexdigest(),"previous_cell_sha256":sha(UP.target_path(m,t,f)),"previous_status":prior.get("status"),"final_heldout_accessed":False})
    value={"schema":"PERSIST_EEG_PROTECTED_PATHWAY_MECHANISM_SEED0_V1","seed":0,"models":MODELS,"tasks":TASKS,"folds":FOLDS,"random_controls":RANDOM_DRAWS,"ridge_alpha":RIDGE_ALPHA,"subject_crossfit_folds":CROSS_FITS,"mapping_unit":"TRAIN subject-session-class activation centroid computed from deterministic capped trials; equal subject/session/class weighting","per_subject_class_cap":CAP,"patch_pairs_per_subject_class":PATCH_CAP,"final_P_definition":"fixed previous classifier-input canonical Protected coordinates only","intermediate_P_reselection":False,"backbone_training":False,"head_or_probe_refit":False,"final_heldout_accessed":False,"cells":rows,"created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    write_json(PROTOCOL/"PROVENANCE.json",value); (PROTOCOL/"PROVENANCE.sha256").write_text(sha(PROTOCOL/"PROVENANCE.json")+"\n",encoding="utf-8"); print("PROTOCOL_LOCKED",len(rows),flush=True)


def aggregate() -> None:
    cells=[]
    for m in MODELS:
        for t in TASKS:
            for f in FOLDS:
                p=path(m,t,f)
                if not p.is_file(): raise RuntimeError(f"missing {m}/{t}/f{f}")
                cells.append(json.loads(p.read_text(encoding="utf-8")))
    tables=defaultdict(list); summary=[]
    for c in cells:
        base={k:c.get(k) for k in ("model","task","fold","seed","status","checkpoint_sha256","basis_sha256","protected_assignment_sha256","split_sha256")}
        summary.append(base)
        if c.get("status")!="COMPLETE": continue
        for name,key in (("DECISION_DECOMPOSITION","decision"),("DECISION_CONFLICT","conflict"),("LAYERWISE_P_RECOVERABILITY","recoverability"),("LAYERWISE_P_PATHWAY","pathway"),("LAYERWISE_CAUSAL_PATCH","causal"),("LAYERWISE_INTERACTION","interaction"),("SESSION_PATHWAY_STABILITY","session")):
            for r in c.get(key,[]):
                row={**base,**r}; tables[name].append(row)
                if key=="recoverability":
                    for sr in r.get("subject_crossfit_rows",[]): tables["RECOVERABILITY_SUBJECT"].append({**base,"layer_name":r["layer_name"],**sr})
    write_csv(OUT/"CELL_STATUS.csv",summary)
    for name,rows in tables.items(): write_csv(OUT/(name+".csv"),rows)
    model_rows=[]
    for m in MODELS:
        for t in TASKS:
            d=[r for r in tables["DECISION_DECOMPOSITION"] if r["model"]==m and r["task"]==t]; p=[r for r in tables["LAYERWISE_P_PATHWAY"] if r["model"]==m and r["task"]==t]; c=[r for r in tables["LAYERWISE_CAUSAL_PATCH"] if r["model"]==m and r["task"]==t]
            if not d: model_rows.append({"model":m,"task":t,"status":"INCOMPLETE_OR_EMPTY_PROTECTED"}); continue
            model_rows.append({"model":m,"task":t,"status":"COMPLETE","mean_abs_zP":float(np.mean([x["mean_abs_zP"] for x in d])),"mean_mP":float(np.mean([x["m_P"] for x in d])),"max_reconstruction_error":float(max(x["reconstruction_max_abs_error"] for x in d)),"max_recoverability_excess":float(np.nanmax([x["R2_excess"] for x in tables["LAYERWISE_P_RECOVERABILITY"] if x["model"]==m and x["task"]==t])),"max_pathway_overlap_excess":float(np.nanmax([x["overlap_excess"] for x in p])) if p else float("nan"),"max_causal_qP_excess":float(np.nanmax([x["delta_qP"]-x["random_delta_qP"] for x in c])) if c else float("nan")})
    write_csv(OUT/"MODEL_TASK_SUMMARY.csv",model_rows)
    lines=["# Protected pathway mechanism audit","","All mappings target the same frozen final Protected coordinates from the preceding native audit. Intermediate layers never redefine Protected. Backbones, native heads, and task probes were not trained or refit. All evaluation is outer-development only; final-heldout data were not accessed.","", "## Completion", "", f"Cells: {sum(c.get('status')=='COMPLETE' for c in cells)} COMPLETE; {sum(c.get('status')=='EMPTY_PROTECTED' for c in cells)} EMPTY_PROTECTED; {sum(c.get('status')=='FAIL_CLOSED' for c in cells)} FAIL_CLOSED.", "", "`MODEL_TASK_SUMMARY.csv` reports descriptive, subject-first summaries. An emergence claim requires cross-session recoverability excess over the matched random final-coordinate controls; Jacobian magnitude is not used as emergence evidence. Causal and interaction rows are omitted only at the terminal affine head, where such an interaction is mathematically uninformative."]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8"); print("AGGREGATE_COMPLETE",flush=True)


def main() -> None:
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest="cmd",required=True); s.add_parser("lock"); c=s.add_parser("cell"); c.add_argument("model",choices=MODELS); c.add_argument("task",choices=TASKS); c.add_argument("fold",type=int,choices=FOLDS); s.add_parser("run-all"); s.add_parser("aggregate"); a=p.parse_args()
    if a.cmd=="lock": lock()
    elif a.cmd=="cell": cell(a.model,a.task,a.fold)
    elif a.cmd=="run-all":
        for m in MODELS:
            for t in TASKS:
                for f in FOLDS: cell(m,t,f)
    else: aggregate()


if __name__ == "__main__": main()
