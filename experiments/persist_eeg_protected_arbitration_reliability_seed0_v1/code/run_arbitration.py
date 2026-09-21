"""Frozen P/C arbitration reliability audit (seed 0, outer-development only)."""
from __future__ import annotations

import argparse, csv, hashlib, importlib.util, json, os, sys, time
from os import environ
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, f1_score

EXP=Path(__file__).resolve().parents[1]; ROOT=Path(os.environ.get("PERSIST_SOURCE_REPO",str(EXP.parents[1]))).resolve()
BASE_EXP=ROOT/"experiments"/"persist_eeg_protected_emergence_routing_seed0_v1"
OUT,PROTOCOL=EXP/"outputs",EXP/"protocol"; RUNTIME=Path(os.environ.get("ARBITRATION_RUNTIME",str(ROOT.parent/"protected_arbitration_reliability_runtime")))
MODELS=("EEGNet","EEGConformer"); TASKS=("OpenBMI_MI","OpenBMI_SSVEP"); FOLDS=tuple(range(5)); SEED=0
ALPHA=np.asarray((0.,.25,.5,.75,1.),np.float32); SCALE=np.asarray((.5,.75,1.,1.25,1.5),np.float32)
CS=(.001,.01,.1,1.,10.); LAM=(.001,.01,.1,1.,10.); DELTAS=(0.,.01,.02,.05,.1); DRAWS=100; PATH_DRAWS=20; EPS=1e-12; BOOT=2000

def load(name:str,path:Path)->Any:
    s=importlib.util.spec_from_file_location(name,path); assert s and s.loader
    m=importlib.util.module_from_spec(s); sys.modules[name]=m; s.loader.exec_module(m); return m
BR=load("arbitration_base",BASE_EXP/"code"/"run_emergence_routing.py"); PW,UP=BR.PW,BR.UP

def seed(*x:object)->int:return int.from_bytes(hashlib.sha256("|".join(map(str,x)).encode()).digest()[:8],"little")%(2**32-1)
def clean(x:Any)->Any:
    if isinstance(x,Path): return str(x)
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(float,np.floating)): return float(x) if np.isfinite(x) else None
    if isinstance(x,dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)): return [clean(v) for v in x]
    return x
def jwrite(p:Path,x:Any)->None:
    p.parent.mkdir(parents=True,exist_ok=True); q=p.with_suffix(p.suffix+".part");q.write_text(json.dumps(clean(x),indent=2,sort_keys=True)+"\n",encoding="utf-8");os.replace(q,p)
def cwrite(p:Path,rows:list[dict[str,Any]])->None:
    p.parent.mkdir(parents=True,exist_ok=True); keys=list(dict.fromkeys(k for r in rows for k in r)) or ["status"];q=p.with_suffix(p.suffix+".part")
    with q.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:clean(r.get(k,"")) for k in keys} for r in rows])
    os.replace(q,p)
def sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8<<20),b""):h.update(b)
    return h.hexdigest()
def array_digest(*arrays):
    digest=hashlib.sha256()
    for value in arrays:
        a=np.ascontiguousarray(value);digest.update(str(a.dtype).encode());digest.update(str(a.shape).encode());digest.update(memoryview(a).cast('B'))
    return digest.hexdigest()
def tpath(m:str,t:str,f:int)->Path:return RUNTIME/"cells"/m.lower()/t.lower()/f"fold{f}_seed0.json"
def center(z:np.ndarray)->np.ndarray:return z-z.mean(1,keepdims=True)
def sm(z:np.ndarray)->np.ndarray:
    u=z-z.max(1,keepdims=True);v=np.exp(u);return v/np.maximum(v.sum(1,keepdims=True),EPS)
def ce(z:np.ndarray,y:np.ndarray)->np.ndarray:return -np.log(np.clip(sm(z)[np.arange(len(y)),y],EPS,1))
def margin(z:np.ndarray)->np.ndarray:
    a=np.partition(z,-2,axis=1);return a[:,-1]-a[:,-2]
def ba(y:np.ndarray,p:np.ndarray)->float:return float(np.mean([np.mean(p[y==c]==c) for c in np.unique(y)])) if len(y) else float("nan")
def subs(s:np.ndarray)->list[str]:return BR.natural(s)
def cosine(a:np.ndarray,b:np.ndarray)->np.ndarray:return np.sum(a*b,1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),EPS)
def entropy(z:np.ndarray)->np.ndarray:
    p=sm(z);return -np.sum(p*np.log(np.clip(p,EPS,1)),1)
def probability_metrics(target,pred):
    if not len(target):return {'AUROC':None,'AUPRC':None,'Brier':None,'accuracy':None,'ECE':None}
    bins=np.minimum((pred*10).astype(int),9);ece=0.
    for b in range(10):
        ix=bins==b
        if ix.any():ece+=float(ix.mean())*abs(float(pred[ix].mean()-target[ix].mean()))
    return {'AUROC':float(roc_auc_score(target,pred)) if len(np.unique(target))>1 else None,'AUPRC':float(average_precision_score(target,pred)) if target.any() else None,'Brier':float(brier_score_loss(target,pred)),'accuracy':float(np.mean((pred>=.5)==target)),'ECE':ece}

def groups(s:np.ndarray)->list[np.ndarray]:
    return [np.isin(s,g) for g in BR.subject_groups(np.asarray(subs(s))) if len(g)]
def fit_logit(x:np.ndarray,y:np.ndarray,c:float,sd:int):
    mu=x.mean(0,dtype=np.float64).astype(np.float32);st=np.maximum(x.std(0,dtype=np.float64).astype(np.float32),1e-6)
    if len(np.unique(y))<2:return float(y[0]),mu,st
    q=LogisticRegression(C=c,penalty="l2",solver="lbfgs",max_iter=1000,random_state=sd).fit((x-mu)/st,y);return q,mu,st
def prob(q:Any,mu:np.ndarray,st:np.ndarray,x:np.ndarray)->np.ndarray:return np.full(len(x),q,np.float32) if isinstance(q,float) else q.predict_proba((x-mu)/st)[:,1].astype(np.float32)
def pick_logit(x:np.ndarray,y:np.ndarray,s:np.ndarray,tag:tuple[object,...]):
    best=(-np.inf,CS[0])
    for c in CS:
        o=np.zeros(len(y),np.float32)
        for k,held in enumerate(groups(s)):
            fit=~held;q,mu,st=fit_logit(x[fit],y[fit],c,seed(*tag,c,k));o[held]=prob(q,mu,st,x[held])
        score=float(np.mean([ba(y[g],(o[g]>=.5).astype(int)) for g in groups(s)]))
        if score>best[0]:best=(score,c)
    q,mu,st=fit_logit(x,y,best[1],seed(*tag,"full"));return q,mu,st,float(best[1])
def fit_ridge(x:np.ndarray,y:np.ndarray,l:float):
    mu=x.mean(0,dtype=np.float64).astype(np.float32);st=np.maximum(x.std(0,dtype=np.float64).astype(np.float32),1e-6);q=Ridge(alpha=l).fit((x-mu)/st,y);return q,mu,st
def rp(q:Any,mu:np.ndarray,st:np.ndarray,x:np.ndarray)->np.ndarray:return q.predict((x-mu)/st).astype(np.float32)
def pick_ridge(x:np.ndarray,y:np.ndarray,s:np.ndarray,tag:tuple[object,...]):
    best=(np.inf,LAM[0])
    for l in LAM:
        o=np.zeros(len(y),np.float32)
        for g in groups(s):
            fit=~g;q,mu,st=fit_ridge(x[fit],y[fit],l);o[g]=rp(q,mu,st,x[g])
        score=float(np.mean((o-y)**2))
        if score<best[0]:best=(score,l)
    q,mu,st=fit_ridge(x,y,best[1]);return q,mu,st,float(best[1])
def ridge_oof(x:np.ndarray,y:np.ndarray,s:np.ndarray,tag:tuple[object,...]):
    """Leave-subject-out TRAIN predictions plus an all-TRAIN frozen refit."""
    # Every lambda uses the same held-subject split and TRAIN standardization.
    # Reuse those arrays, while retaining separate sklearn fits and their solver.
    predictions=[np.zeros_like(y,dtype=np.float32) for _ in LAM]
    for g in groups(s):
        train_x=x[~g];train_y=y[~g]
        mu=train_x.mean(0,dtype=np.float64).astype(np.float32)
        st=np.maximum(train_x.std(0,dtype=np.float64).astype(np.float32),1e-6)
        fit_x=(train_x-mu)/st;held_x=(x[g]-mu)/st
        for i,l in enumerate(LAM):
            estimator=Ridge(alpha=l).fit(fit_x,train_y)
            predictions[i][g]=estimator.predict(held_x).astype(np.float32)
    best=(np.inf,LAM[0],None)
    for l,o in zip(LAM,predictions):
        v=float(np.mean((o-y)**2))
        if v<best[0]:best=(v,l,o)
    q,mu,st=fit_ridge(x,y,best[1]);return best[2],q,mu,st,float(best[1]),float(best[0])
def logit_oof(x:np.ndarray,y:np.ndarray,s:np.ndarray,tag:tuple[object,...]):
    """Nested held-subject TRAIN selection; held predictions exclude C selection."""
    nested=np.zeros(len(y),np.float32)
    for k,g in enumerate(groups(s)):
        if not (~g).any():raise RuntimeError('insufficient subjects for nested logistic CV')
        q,mu,st,_=pick_logit(x[~g],y[~g],s[~g],(*tag,'inner',k))
        nested[g]=prob(q,mu,st,x[g])
    best=(-np.inf,CS[0],None)
    for c in CS:
        o=np.zeros(len(y),np.float32)
        for k,g in enumerate(groups(s)):
            q,mu,st=fit_logit(x[~g],y[~g],c,seed(*tag,c,k));o[g]=prob(q,mu,st,x[g])
        score=float(np.mean([ba(y[g],(o[g]>=.5).astype(int)) for g in groups(s)]))
        if score>best[0]:best=(score,c,o)
    q,mu,st=fit_logit(x,y,best[1],seed(*tag,"full"));return nested,q,mu,st,float(best[1]),float(np.mean([ba(y[s==sub],(nested[s==sub]>=.5).astype(int)) for sub in subs(s)]))

def nested_gain_predictions(x,y,s,xo):
    """Independent Ridge targets share design matrices, never their alpha choices."""
    def select(xx,yy,ss):
        predictions=np.zeros((len(LAM),*yy.shape),np.float32)
        for g in groups(ss):
            train=xx[~g];mu=train.mean(0,dtype=np.float64).astype(np.float32);st=np.maximum(train.std(0,dtype=np.float64).astype(np.float32),1e-6)
            xf=(train-mu)/st;xt=(xx[g]-mu)/st
            for j,l in enumerate(LAM):predictions[j,g]=Ridge(alpha=l).fit(xf,yy[~g]).predict(xt)
        mse=np.mean((predictions-yy[None])**2,axis=1)
        chosen=np.argmin(mse,axis=0)
        return np.asarray(LAM)[chosen],mse[chosen,np.arange(yy.shape[1])]
    out=np.zeros_like(y)
    for g in groups(s):
        alphas,_=select(x[~g],y[~g],s[~g]);model,mu,st=fit_ridge(x[~g],y[~g],alphas)
        out[g]=rp(model,mu,st,x[g])
    alphas,mse=select(x,y,s);model,mu,st=fit_ridge(x,y,alphas)
    return out,rp(model,mu,st,xo),alphas,mse

def features(z0:np.ndarray,zp:np.ndarray,zc:np.ndarray,z:np.ndarray,q:np.ndarray,cent:np.ndarray|None=None)->np.ndarray:
    k=z.shape[1];pp,pc,pn=(z0+zp).argmax(1),(z0+zc).argmax(1),z.argmax(1);oh=lambda a:np.eye(k,dtype=np.float32)[a];a,b,n=center(zp),center(zc),center(z)
    pgap=zp[np.arange(len(zp)),pp]-zp[np.arange(len(zp)),pc];cgap=zc[np.arange(len(zc)),pp]-zc[np.arange(len(zc)),pc];ngap=z[np.arange(len(z)),pp]-z[np.arange(len(z)),pc]
    switches=[]
    for i in range(len(z)):
        roots=[];aa=pn[i]
        for bb in range(k):
            if bb==aa:continue
            u=(z0[0,aa]-z0[0,bb])+(zp[i,aa]-zp[i,bb]);v=zc[i,aa]-zc[i,bb]
            if abs(v)>EPS and 0<=-u/v<=1:roots.append(float(-u/v))
        roots.sort(reverse=True)
        first=roots[0] if roots else -1.
        after=int((z0[0]+zp[i]+max(0.,first-1e-5)*zc[i]).argmax()) if roots else int(pn[i])
        switches.append((first,len(roots),after,int(after==pp[i])))
    sw=np.asarray(switches,np.float32);base=np.c_[oh(pp),oh(pc),oh(pn),(pp==pc),(pp==pn),(pc==pn),margin(z0+zp),margin(z0+zc),margin(z),entropy(z0+zp),entropy(z0+zc),entropy(z),np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1),np.linalg.norm(a,axis=1)/np.maximum(np.linalg.norm(b,axis=1),EPS),cosine(a,b),cosine(a,n),cosine(b,n),pgap,cgap,ngap,sw[:,0],1-np.maximum(sw[:,0],0),sw[:,1],sw[:,0]<0]
    base=np.c_[base,oh(sw[:,2].astype(int)),sw[:,3]]
    if cent is not None:
        d=np.linalg.norm(q[:,None]-cent[None],axis=2);near=np.partition(d,1,axis=1)[:,:2];base=np.c_[base,near,near[:,1]-near[:,0]]
    return base.astype(np.float32)

def pathway_features_legacy(acts:dict[str,np.ndarray],acts_o:dict[str,np.ndarray],q:np.ndarray,qo:np.ndarray,s:np.ndarray,tag:tuple[object,...])->tuple[np.ndarray,np.ndarray,list[dict[str,Any]]]:
    """Frozen pathway maps: LSO TRAIN map outputs for policy fitting, then all-TRAIN outer maps."""
    cols_t=[];cols_o=[]; audit=[]
    for name,a in acts.items():
        started=time.perf_counter()
        print('PATHWAY_START',tag,name,'shape',a.shape,flush=True)
        pt,qfit,mu,st,l,mse=ridge_oof(a,q,s,(*tag,name,"path"));po=rp(qfit,mu,st,acts_o[name])
        ct,co=cosine(pt,q),cosine(po,qo);et=np.linalg.norm(pt-q,axis=1)/np.maximum(np.linalg.norm(q,axis=1),EPS);eo=np.linalg.norm(po-qo,axis=1)/np.maximum(np.linalg.norm(qo,axis=1),EPS)
        cols_t.extend([ct,et]);cols_o.extend([co,eo]);audit.append({"layer_name":name,"ridge_lambda":l,"train_lso_mse":mse,"train_qP_cosine_mean":float(np.mean(ct)),"outer_qP_cosine_mean":float(np.mean(co)),"outer_qP_norm_error_mean":float(np.mean(eo))})
        print('PATHWAY_DONE',tag,name,'seconds',round(time.perf_counter()-started,2),flush=True)
    return np.stack(cols_t,1).astype(np.float32),np.stack(cols_o,1).astype(np.float32),audit

def shared_pathway_predictions(xfit,yfit,xtest):
    """Share only X kernels/eigendecomposition; preserve each target GEMM shape."""
    n,width=xfit.shape
    mean=xfit.mean(0,dtype=np.float64).astype(np.float32);std=np.maximum(xfit.std(0,dtype=np.float64).astype(np.float32),1e-6)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.inference_mode():
        a=BR.standardised_tensor(xfit,mean,std,dev);t=BR.standardised_tensor(xtest,mean,std,dev)
        kernel=(a@a.T)/max(width,1);kt=(t@a.T)/max(width,1)
        eig,vec=torch.linalg.eigh(kernel);floor=torch.clamp(eig[-1].abs()*1e-8,min=1e-8)
        del a,t,kernel
        predictions=[]
        for j in range(yfit.shape[1]):
            b=torch.from_numpy(np.ascontiguousarray(yfit[:,j].reshape(n,-1))).to(dev);vty=vec.T@b
            parts=[]
            for alpha in BR.RIDGE_ALPHAS:
                den=torch.clamp(eig,min=floor) if alpha==0 else eig+alpha
                parts.append((kt@(vec@(vty/den[:,None]))).cpu().numpy())
            predictions.append(np.stack(parts,0))
        result=np.stack(predictions,2).astype(np.float32)
    return result

def pathway_bank(acts,acts_o,q,qo,s,subsets,raw,w,tag):
    """Reuse the preceding GPU dual solver for P and all 20 fixed controls together.

    Each target retains its own TRAIN nested subject-CV alpha. Expensive kernels
    depend only on layer activations, so their computation is shared across targets.
    """
    class IndexedRows:
        """A row view which streams only feature chunks, never a full host copy."""
        def __init__(self,source,rows):
            self.source=source;self.rows=np.asarray(rows,dtype=np.intp);self.shape=(len(self.rows),source.shape[1]);self.chunk=getattr(BR,'MAPPING_FEATURE_CHUNK',16384)
        def __getitem__(self,key):
            rows,columns=key
            if rows!=slice(None):raise IndexError('IndexedRows supports all-row feature chunks only')
            return np.ascontiguousarray(self.source[self.rows,columns])
        def mean(self,axis,dtype):
            if axis!=0:raise ValueError('IndexedRows supports column statistics only')
            out=np.empty(self.shape[1],dtype=dtype)
            for start in range(0,self.shape[1],self.chunk):
                stop=min(self.shape[1],start+self.chunk)
                out[start:stop]=self.source[self.rows,start:stop].mean(axis=0,dtype=dtype)
            return out
        def std(self,axis,dtype):
            if axis!=0:raise ValueError('IndexedRows supports column statistics only')
            out=np.empty(self.shape[1],dtype=dtype)
            for start in range(0,self.shape[1],self.chunk):
                stop=min(self.shape[1],start+self.chunk)
                out[start:stop]=self.source[self.rows,start:stop].std(axis=0,dtype=dtype)
            return out
    targets=np.stack([q[:,d] for d in subsets],1)
    truth_o=np.stack([qo[:,d] for d in subsets],1)
    evidence_maps=np.stack([raw[d]@w.T for d in subsets])
    cols_t=[[] for _ in subsets];cols_o=[[] for _ in subsets]
    ev_t=[[] for _ in subsets];ev_o=[[] for _ in subsets];audits=[[] for _ in subsets]
    masks=[np.isin(s,g) for g in BR.subject_groups(np.asarray(subs(s))) if len(g)]
    def select(a,y,ss,rows):
        scores=[]
        for held in BR.subject_groups(np.asarray(subs(ss))):
            ix=np.isin(ss,held)
            if not ix.any() or (~ix).sum()<2:continue
            pred=shared_pathway_predictions(IndexedRows(a,rows[~ix]),y[~ix],IndexedRows(a,rows[ix]))
            scores.append(BR.r2_matrix(y[ix],pred,ss[ix]))
        if not scores:raise RuntimeError('insufficient TRAIN subjects for nested pathway CV')
        return np.nanargmax(np.nanmean(scores,axis=0),axis=0)
    for name,a in acts.items():
        if name=='classifier_input':continue  # not one of the requested evidence stages
        begin=time.perf_counter();print('PATHWAY_GPU_START',tag,name,a.shape,flush=True)
        key=hashlib.sha256((sha(Path(__file__))+array_digest(a,acts_o[name],targets)+str(tuple(s))+name).encode()).hexdigest()
        cache=RUNTIME/'mapping_cache'/str(tag[0])/str(tag[1])/str(tag[2])/(name+'_'+key+'.npz')
        if cache.is_file():
            with np.load(cache,allow_pickle=False) as saved:pt=saved['pt'];po=saved['po'];chosen=saved['chosen']
        else:
            pt=np.zeros_like(targets);indices=np.arange(len(subsets))
            all_rows=np.arange(len(a),dtype=np.intp)
            for fold,held in enumerate(masks):
                train_rows,test_rows=all_rows[~held],all_rows[held]
                chosen_inner=select(a,targets[~held],s[~held],train_rows)
                predictions=shared_pathway_predictions(IndexedRows(a,train_rows),targets[~held],IndexedRows(a,test_rows))
                pt[held]=np.stack([predictions[chosen_inner[j],:,j] for j in indices],1)
                print('PATHWAY_GPU_CROSSFIT',name,fold+1,len(masks),flush=True)
            chosen=select(a,targets,s,all_rows)
            predictions=shared_pathway_predictions(IndexedRows(a,all_rows),targets,IndexedRows(acts_o[name],np.arange(len(acts_o[name]),dtype=np.intp)))
            po=np.stack([predictions[chosen[j],:,j] for j in indices],1)
            cache.parent.mkdir(parents=True,exist_ok=True)
            tmp=cache.with_suffix('.part')
            with tmp.open('wb') as stream:np.savez_compressed(stream,pt=pt,po=po,chosen=chosen)
            os.replace(tmp,cache)
        for j in range(len(subsets)):
            for pred,truth,columns,evidence in ((pt[:,j],targets[:,j],cols_t[j],ev_t[j]),(po[:,j],truth_o[:,j],cols_o[j],ev_o[j])):
                zp=truth@evidence_maps[j];zhat=pred@evidence_maps[j]
                columns.extend([cosine(pred,truth),np.linalg.norm(pred-truth,axis=1)/np.maximum(np.linalg.norm(truth,axis=1),EPS),cosine(center(zhat),center(zp)),(zhat.argmax(1)==zp.argmax(1)).astype(np.float32),margin(zhat)-margin(zp)])
                evidence.append(zhat)
            audits[j].append({'layer_name':name,'ridge_lambda':float(BR.RIDGE_ALPHAS[chosen[j]]),'solver':'previous_frozen_GPU_dual_kernel','nested_subject_cv':True,'seconds_shared':round(time.perf_counter()-begin,3)})
        print('PATHWAY_GPU_DONE',name,'seconds',round(time.perf_counter()-begin,2),flush=True)
    for j in range(len(subsets)):
        for ev,cols,truth in ((ev_t[j],cols_t[j],targets[:,j]),(ev_o[j],cols_o[j],truth_o[:,j])):
            zfinal=truth@evidence_maps[j];stack=np.stack(ev,1)
            cols.extend([(stack.argmax(2)==zfinal.argmax(1)[:,None]).mean(1),np.var(stack-stack.mean(2,keepdims=True),axis=1).mean(1),np.var(np.stack([margin(z) for z in ev],1),axis=1),cosine(center(ev[0]),center(ev[-1]))])
    return [(np.stack(cols_t[j],1).astype(np.float32),np.stack(cols_o[j],1).astype(np.float32),audits[j]) for j in range(len(subsets))]

def oracle(z0:np.ndarray,zp:np.ndarray,zc:np.ndarray,y:np.ndarray,s:np.ndarray,grid:np.ndarray)->tuple[list[dict[str,Any]],dict[str,np.ndarray]]:
    zz=np.stack([z0+bp*zp+ac*zc for bp,ac in grid],1); losses=np.stack([ce(zz[:,i],y) for i in range(len(grid))],1);best=losses.argmin(1);native=np.where((grid==1).all(1))[0][0];pred=zz.argmax(2);correct=pred==y[:,None];rows=[]
    competitor=zz.copy();competitor[np.arange(len(y)),:,y]=-np.inf
    true_margin=zz[np.arange(len(y)),:,y]-competitor.max(2)
    native_top=zz[:,native].argmax(1);other=zz.copy();other[np.arange(len(y)),:,native_top]=-np.inf
    native_margin=zz[np.arange(len(y)),:,native_top]-other.max(2)
    margin_best=true_margin.argmax(1)
    for sub in subs(s):
        ix=s.astype(str)==sub;n=zz[ix,native];o=zz[ix,best[ix]]
        harmful=correct[ix,native]&np.any(~correct[ix],axis=1)
        rows.append({"subject_id":sub,"native_BA":ba(y[ix],n.argmax(1)),"oracle_BA":ba(y[ix],o.argmax(1)),"oracle_BA_gain":ba(y[ix],o.argmax(1))-ba(y[ix],n.argmax(1)),"native_CE":float(ce(n,y[ix]).mean()),"oracle_CE":float(ce(o,y[ix]).mean()),"oracle_CE_gain":float((ce(n,y[ix])-ce(o,y[ix])).mean()),"recoverable_error_rate":float(np.mean((n.argmax(1)!=y[ix])&np.any(correct[ix],1))),"harmable_correct_rate":float(harmful.mean())})
    return rows,{"logits":zz,"best":best,"correct":correct,"native":native,"loss":losses,"true_margin":true_margin,"native_top1_margin":native_margin,"margin_best":margin_best,"predicted_class":pred,"entropy":np.stack([entropy(zz[:,i]) for i in range(len(grid))],1)}

def taxonomy(surface:dict[str,np.ndarray],y:np.ndarray,s:np.ndarray,grid:np.ndarray)->list[dict[str,Any]]:
    zz,correct,native=surface["logits"],surface["correct"],surface["native"];rows=[]
    d=np.sqrt(np.log(grid[:,0])**2+np.log(grid[:,1])**2)
    for sub in subs(s):
        ix=np.flatnonzero(s.astype(str)==sub);counts=defaultdict(int);dist=[]
        for i in ix:
            if correct[i,native]:continue
            good=np.flatnonzero(correct[i]);flags={"C_DOWN_ONLY":any(grid[j,0]==1 and grid[j,1]<1 for j in good),"P_UP_ONLY":any(grid[j,0]>1 and grid[j,1]==1 for j in good),"P_DOWN_ONLY":any(grid[j,0]<1 and grid[j,1]==1 for j in good),"C_UP_ONLY":any(grid[j,0]==1 and grid[j,1]>1 for j in good)}
            if not len(good):typ="NOT_REWEIGHT_RECOVERABLE"
            else:
                single=any(flags.values())
                single_good=[j for j in good if grid[j,0]==1 or grid[j,1]==1]
                nearest=min(single_good if single_good else list(good),key=lambda j:(d[j],j))
                b,a=grid[nearest]
                typ=('C_DOWN_ONLY' if a<1 else 'C_UP_ONLY') if b==1 else ('P_UP_ONLY' if b>1 else 'P_DOWN_ONLY') if a==1 else 'JOINT_REWEIGHT'
                dist.append(float(d[good].min()))
            for flag,value in flags.items():counts[flag+'_flag']+=int(value)
            counts[typ]+=1
        rows.append({"subject_id":sub,**dict(counts),**{k:int(counts[k]) for k in ("C_DOWN_ONLY","P_UP_ONLY","P_DOWN_ONLY","C_UP_ONLY","JOINT_REWEIGHT","NOT_REWEIGHT_RECOVERABLE")},"minimum_intervention_distance":float(np.mean(dist)) if dist else float("nan")})
    return rows

def metric_rows(y:np.ndarray,s:np.ndarray,outputs:dict[str,np.ndarray])->list[dict[str,Any]]:
    rows=[]
    for sub in subs(s):
        ix=s.astype(str)==sub;row={"subject_id":sub}
        for name,z in outputs.items():row[name+"_BA"]=ba(y[ix],z[ix].argmax(1));row[name+"_MacroF1"]=float(f1_score(y[ix],z[ix].argmax(1),average="macro",zero_division=0));row[name+"_CE"]=float(ce(z[ix],y[ix]).mean())
        rows.append(row)
    return rows

def disagreement_rows(y,s,z0,zp,zc,z,outputs,surface):
    pp,pc,pn=(z0+zp).argmax(1),(z0+zc).argmax(1),z.argmax(1);d=pp!=pc;rows=[]
    for sub in subs(s):
        base=s==sub
        for group,mask in [('disagreement',d),('P_correct_C_wrong',d&(pp==y)&(pc!=y)),('P_wrong_C_correct',d&(pp!=y)&(pc==y)),('ambiguous',d&(pp!=y)&(pc!=y))]:
            ix=base&mask;n=int(ix.sum())
            row={'subject_id':sub,'subset':group,'n_trials':n,'fraction':float(n/base.sum()),'native_accuracy':float(np.mean(pn[ix]==y[ix])) if n else None,'P_only_accuracy':float(np.mean(pp[ix]==y[ix])) if n else None,'C_only_accuracy':float(np.mean(pc[ix]==y[ix])) if n else None,'oracle_recoverable_fraction':float(np.mean(np.any(surface['correct'][ix],1))) if n else None}
            for name,logits in outputs.items():
                pred=logits.argmax(1);row[name+'_corrected_errors']=int(np.sum(ix&(pn!=y)&(pred==y)));row[name+'_introduced_errors']=int(np.sum(ix&(pn==y)&(pred!=y)))
            rows.append(row)
    return rows

def subject_score(y:np.ndarray,s:np.ndarray,z:np.ndarray)->float:
    return float(np.mean([ba(y[g],z[g].argmax(1)) for g in groups(s)]))
def policy_subject_rows(y:np.ndarray,s:np.ndarray,z:np.ndarray,name:str,meta:dict[str,Any])->list[dict[str,Any]]:
    rows=[]
    for sub in subs(s):
        g=s.astype(str)==sub;pred=z[g].argmax(1)
        rows.append({"subject_id":sub,"policy":name,**meta,"n_trials":int(g.sum()),"BA":ba(y[g],pred),"MacroF1":float(f1_score(y[g],pred,average="macro",zero_division=0)),"CE":float(ce(z[g],y[g]).mean())})
    return rows
def gain_distill(x:np.ndarray,xo:np.ndarray,states:list[np.ndarray],states_o:list[np.ndarray],native:np.ndarray,y:np.ndarray,s:np.ndarray,tag:tuple[object,...],name:str)->tuple[np.ndarray,list[dict[str,Any]],dict[str,Any]]:
    """One Ridge gain predictor per frozen candidate; selection uses only LSO TRAIN predictions."""
    target=np.stack([ce(native,y)-ce(z,y) for z in states[1:]],1)
    ot,oo,lam,mse=nested_gain_predictions(x,target,s,xo)
    pt,po=np.c_[np.zeros(len(y)),ot],np.c_[np.zeros(len(xo)),oo];best=(-np.inf,DELTAS[0])
    for d in DELTAS:
        sel=np.argmax(pt,1);sel[pt[np.arange(len(y)),sel]<=d]=0;zz=np.stack(states,1)[np.arange(len(y)),sel];v=subject_score(y,s,zz)
        if v>best[0]:best=(v,d)
    sel=np.argmax(po,1);sel[po[np.arange(len(sel)),sel]<=best[1]]=0;outer=np.stack(states_o,1)[np.arange(len(sel)),sel]
    return outer,[],{"policy":name,"feature_family":tag[-1],"selected_delta":float(best[1]),"train_nested_subject_BA":float(best[0]),"ridge_lambdas":lam.tolist(),"ridge_lso_mse":mse.tolist(),"outer_selection":sel}
def gain_policy_rows(y:np.ndarray,s:np.ndarray,z:np.ndarray,meta:dict[str,Any])->list[dict[str,Any]]:
    return policy_subject_rows(y,s,z,str(meta["policy"]),{k:v for k,v in meta.items() if k!="outer_selection"})
def reliability_arbitration(x:np.ndarray,xo:np.ndarray,zp:np.ndarray,zc:np.ndarray,zpo:np.ndarray,zco:np.ndarray,z0:np.ndarray,z0o:np.ndarray,y:np.ndarray,yo:np.ndarray,s:np.ndarray,tag:tuple[object,...])->tuple[np.ndarray,list[dict[str,Any]],dict[str,Any]]:
    pp,pc=(z0+zp).argmax(1),(z0+zc).argmax(1)
    opp,opc=(z0o+zpo).argmax(1),(z0o+zco).argmax(1)
    tp=(pp==y).astype(int);tc=(pc==y).astype(int)
    op,qp,mp,sp,cp,vp=logit_oof(x,tp,s,(*tag,"rP"));oc,qc,mc,sc,cc,vc=logit_oof(x,tc,s,(*tag,"rC"))
    oop, ooc=prob(qp,mp,sp,xo),prob(qc,mc,sc,xo); best=(-np.inf,.25,.25)
    for tau in (.25,.5,.75):
      for a in (.25,.5,.75):
        zz=z0+zp+np.where((pp!=pc)&(op-oc>tau),a,1.)[:,None]*zc;v=subject_score(y,s,zz)
        if v>best[0]:best=(v,tau,float(a))
    a=np.where((opp!=opc)&(oop-ooc>best[1]),best[2],1.).astype(np.float32);outer=z0o+zpo+a[:,None]*zco
    rows=[]
    for name,target,pred,c,score in (("rP",(opp==yo).astype(int),oop,cp,vp),("rC",(opc==yo).astype(int),ooc,cc,vc)):
        rows.append({"target":name,"feature_family":tag[-1],"C":c,"train_nested_subject_BA":score,**probability_metrics(target,pred)})
    return outer,rows,{"policy":"ReliabilityArbitration","feature_family":tag[-1],"selected_tau":best[1],"selected_alpha":best[2],"train_lso_BA":best[0]}
def direct_arbiter(x:np.ndarray,xo:np.ndarray,zp:np.ndarray,zc:np.ndarray,zpo:np.ndarray,zco:np.ndarray,y:np.ndarray,yo:np.ndarray,s:np.ndarray,tag:tuple[object,...])->dict[str,Any]:
    disagreement=zp.argmax(1)!=zc.argmax(1);disagreement_o=zpo.argmax(1)!=zco.argmax(1)
    d=disagreement & ((zp.argmax(1)==y) ^ (zc.argmax(1)==y));do=disagreement_o & ((zpo.argmax(1)==yo) ^ (zco.argmax(1)==yo))
    if int(d.sum())<4 or len(np.unique((zp[d].argmax(1)==y[d]).astype(int)))<2:return {"status":"FAIL_CLOSED_INSUFFICIENT_TRAIN_DISAGREEMENT","feature_family":tag[-1]}
    target=(zp[d].argmax(1)==y[d]).astype(int);ot,q,mu,st,c,v=logit_oof(x[d],target,s[d],(*tag,"direct"));p=prob(q,mu,st,xo[do]);outer=(zpo[do].argmax(1)==yo[do]).astype(int)
    return {"status":"COMPLETE","feature_family":tag[-1],"C":c,"train_nested_subject_BA":v,"train_disagreement_n":int(disagreement.sum()),"outer_disagreement_n":int(disagreement_o.sum()),"train_ambiguous_n":int((disagreement&~d).sum()),"outer_ambiguous_n":int((disagreement_o&~do).sum()),"outer_primary_n":int(do.sum()),**probability_metrics(outer,p)}

def reliability(x:np.ndarray,zp:np.ndarray,zc:np.ndarray,y:np.ndarray,s:np.ndarray,xo:np.ndarray,zpo:np.ndarray,zco:np.ndarray,yo:np.ndarray,tag:tuple[object,...])->list[dict[str,Any]]:
    rows=[]
    for name,target,outer in (("rP",(zp.argmax(1)==y).astype(int),(zpo.argmax(1)==yo).astype(int)),("rC",(zc.argmax(1)==y).astype(int),(zco.argmax(1)==yo).astype(int))):
        q,mu,st,c=pick_logit(x,target,s,(*tag,name));p=prob(q,mu,st,xo);rows.append({"target":name,"feature_family":tag[-1],"C":c,"AUROC":float(roc_auc_score(outer,p)) if len(np.unique(outer))>1 else float("nan"),"AUPRC":float(average_precision_score(outer,p)) if len(np.unique(outer))>1 else float("nan"),"Brier":float(brier_score_loss(outer,p)),"accuracy":float(np.mean((p>=.5)==outer))})
    return rows

def cell(m:str,t:str,f:int)->None:
    dst=tpath(m,t,f)
    if dst.is_file():print("CELL_CACHED",m,t,f,flush=True);return
    base={"model":m,"task":t,"fold":f,"seed":0,"backbone_training":False,"head_refit":False,"final_heldout_accessed":False}
    try:
        if not (PROTOCOL/"PROVENANCE.json").is_file():raise RuntimeError("protocol lock required")
        expected_hash=(PROTOCOL/'PROVENANCE.sha256').read_text().strip()
        if sha(PROTOCOL/'PROVENANCE.json')!=expected_hash:raise RuntimeError('protocol lock hash changed')
        frozen=json.loads((PROTOCOL/'PROVENANCE.json').read_text())
        provenance=next(r for r in frozen['cells'] if (r['model'],r['task'],r['fold'])==(m,t,f))
        rec,stored,ckpt,pathway=PW.previous(m,t,f)
        if sha(ckpt)!=provenance['checkpoint_sha256'] or sha(PW.path(m,t,f))!=provenance['previous_pathway_sha256']:raise RuntimeError('frozen upstream source hash mismatch')
        if pathway.get("status")!="COMPLETE":raise RuntimeError("previous pathway cell not COMPLETE")
        data=UP.outer_data(t,f)
        if data["normalizer"]["mean_std_sha256"]!=rec["normalizer"]["mean_std_sha256"]:raise RuntimeError("normalizer mismatch")
        dev=torch.device("cuda" if torch.cuda.is_available() else "cpu");net,head=UP.helper(m).build_model({"Model":m,"Task":t,"fold":f,"seed":0,"channels":int(rec.get("channels") or 62),"samples":int(rec.get("samples") or 1000),"classes":int(rec["classes"]),"checkpoint_path":str(ckpt),"recipe_name":rec.get("recipe",{}).get("name"),"trainable_parameters":int(rec.get("trainable_parameters",rec.get("parameters",0)))},dev)
        run=PW.Stages(net,head,m,dev);tx,ty,ts,tse=UP.capped_train(data,t,m,f);PW.stage_check(run,tx);bh,_,_=UP.hook_representations(net,head,tx,m,dev);spec=UP.helper(m).spectrum(bh,ty,ts,tse,t,m,f);dims=np.asarray(stored.get("protected_blocks",[]),int);basis_sha=UP.array_sha(spec["mean"],spec["basis"],spec["scale"],spec["directions"])
        if basis_sha!=pathway.get("basis_sha256"):raise RuntimeError("frozen canonical basis mismatch")
        x,y,s,se=BR.trial_train(data,m,t,f);acts,h,z=BR.forward_stages(run,x);q,z0,zp,zc=BR.decompose(h,z,spec,dims,head); ox,oy,os=data["outer_future_x"],data["outer_future_y"].astype(int),data["outer_future_subjects"].astype(str);oacts,oh,oz=BR.forward_stages(run,ox);qo,oz0,ozp,ozc=BR.decompose(oh,oz,spec,dims,head)
        exact=float(max(np.max(np.abs(z-(z0+zp+zc))),np.max(np.abs(oz-(oz0+ozp+ozc)))));
        BR.exact_target(h,spec,dims);BR.exact_target(oh,spec,dims)
        if exact>=1e-5:raise RuntimeError(f"FAIL_PROTOCOL exact decomposition {exact}")
        cent=np.asarray([q[y==c].mean(0) for c in range(data["classes"])],np.float32);ga=features(z0,zp,zc,z,q);go=features(oz0,ozp,ozc,oz,qo)
        randoms=UP.random_dims(spec["rank"],len(dims),m,t,f,"native-equal-rank-random");raw=UP.raw_base(spec);w=head.weight.detach().float().cpu().numpy()
        bank=pathway_bank(acts,oacts,q,qo,s,[dims]+list(randoms[:PATH_DRAWS]),raw,w,(m,t,f))
        pft,pfo,pfa=bank[0];qd,cd=q[:,dims],cent[:,dims];od=qo[:,dims];dt=np.sort(np.linalg.norm(qd[:,None]-cd[None],axis=2),axis=1)[:,:2];do=np.sort(np.linalg.norm(od[:,None]-cd[None],axis=2),axis=1)[:,:2]
        fa={"A":ga,"B":np.c_[ga,dt],"C":np.c_[ga,pft],"D":np.c_[ga,dt,pft]};fo={"A":go,"B":np.c_[go,do],"C":np.c_[go,pfo],"D":np.c_[go,do,pfo]}
        grid1=np.c_[np.ones(len(ALPHA)),ALPHA];grid2=np.asarray([(b,a) for b in SCALE for a in SCALE],np.float32);po,sp=oracle(oz0,ozp,ozc,oy,os,grid1);two,sp2=oracle(oz0,ozp,ozc,oy,os,grid2);tax=taxonomy(sp2,oy,os,grid2)
        surface_path=RUNTIME/'surfaces'/m/t/f'fold{f}_seed0.npz';surface_path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(surface_path,grid_1d=grid1,grid_2d=grid2,subject_ids=os,**{'one_d_'+k:v for k,v in sp.items()},**{'two_d_'+k:v for k,v in sp2.items()})
        randoms=UP.random_dims(spec["rank"],len(dims),m,t,f,"native-equal-rank-random");raw=UP.raw_base(spec);w=head.weight.detach().float().cpu().numpy();random=[]
        for j,d in enumerate(randoms):
            rz=(qo[:,d]@raw[d])@w.T;rc=oz-oz0-rz; rr,_=oracle(oz0,rz,rc,oy,os,grid1);rt,_=oracle(oz0,rz,rc,oy,os,grid2);random.append({"draw":j,"suppression_BA_gain":float(np.mean([r["oracle_BA_gain"] for r in rr])),"two_d_BA_gain":float(np.mean([r["oracle_BA_gain"] for r in rt]))})
        pg=float(np.mean([r["oracle_BA_gain"] for r in po]));rv=np.asarray([r["suppression_BA_gain"] for r in random]);specrow={"P_oracle_BA_gain":pg,"random_oracle_mean":float(rv.mean()),"random_oracle_low":float(np.quantile(rv,.025)),"random_oracle_high":float(np.quantile(rv,.975)),"P_percentile":float(np.mean(rv<=pg)),"empirical_p":float((1+(rv>=pg).sum())/(1+len(rv))),"P_minus_random":float(pg-rv.mean())}
        rel=[];policy=[];two_policy=[];arb=[];learned={}
        one_train=[z]+[z0+zp+a*zc for a in ALPHA[:-1]];one_outer=[oz]+[oz0+ozp+a*ozc for a in ALPHA[:-1]]
        two_train=[z]+[z0+b*zp+a*zc for b,a in grid2 if not (b==1 and a==1)];two_outer=[oz]+[oz0+b*ozp+a*ozc for b,a in grid2 if not (b==1 and a==1)]
        for name in ("A","B","C","D"):
            print('RELIABILITY_START',m,t,f,name,flush=True)
            rz,rr,rm=reliability_arbitration(fa[name],fo[name],zp,zc,ozp,ozc,z0,oz0,y,oy,s,(m,t,f,name));rel+=rr;policy+=gain_policy_rows(oy,os,rz,rm);learned[f"reliability_{name}"]=rz
            arb.append(direct_arbiter(fa[name],fo[name],z0+zp,z0+zc,oz0+ozp,oz0+ozc,y,oy,s,(m,t,f,name)))
        for name in ("A","C","D"):
            print('GAIN_START',m,t,f,name,flush=True)
            gz,_,gm=gain_distill(fa[name],fo[name],one_train,one_outer,z,y,s,(m,t,f,name),"one_d_gain");policy+=gain_policy_rows(oy,os,gz,gm);learned[f"gain_alpha_{name}"]=gz
            tz,_,tm=gain_distill(fa[name],fo[name],two_train,two_outer,z,y,s,(m,t,f,name),"two_d_gain");two_policy+=gain_policy_rows(oy,os,tz,tm);learned[f"gain_2d_{name}"]=tz
        outputs={"native":oz,"P_only":oz0+ozp,"C_only":oz0+ozc,**{f"fixed_alpha_{a:g}":oz0+ozp+a*ozc for a in ALPHA},"oracle_suppression":sp["logits"][np.arange(len(oy)),sp["best"]],"oracle_2d":sp2["logits"][np.arange(len(oy)),sp2["best"]],**learned}
        random_policy=[];random_pathway=[]
        # K1: 100 equal-rank random learned alpha policies, trained only on their own TRAIN geometry.
        def random_learned(pair:tuple[int,np.ndarray])->dict[str,Any]:
            j,d=pair
            rzt=(q[:,d]@raw[d])@w.T;rzo=(qo[:,d]@raw[d])@w.T;rct=z-z0-rzt;rco=oz-oz0-rzo
            rga=features(z0,rzt,rct,z,q);rgo=features(oz0,rzo,rco,oz,qo)
            rs=[z]+[z0+rzt+a*rct for a in ALPHA[:-1]];rso=[oz]+[oz0+rzo+a*rco for a in ALPHA[:-1]]
            rz,_,rm=gain_distill(rga,rgo,rs,rso,z,y,s,(m,t,f,j,"randomA"),"random_one_d_gain")
            rr,_,rrm=reliability_arbitration(rga,rgo,rzt,rct,rzo,rco,z0,oz0,y,oy,s,(m,t,f,j,'randomA'))
            return {"draw":j,"control":"equal_rank_random_learned_alpha","mean_subject_BA":float(np.mean([r["BA"] for r in gain_policy_rows(oy,os,rz,rm)])),"reliability_mean_subject_BA":subject_score(oy,os,rr),"selected_delta":rm["selected_delta"],"train_nested_subject_BA":rm["train_nested_subject_BA"],"subject_rows":gain_policy_rows(oy,os,rz,rm),"reliability_subject_rows":gain_policy_rows(oy,os,rr,rrm)}
        # Independent controls are ordered after collection. Each regression itself remains single-threaded.
        workers=max(1,min(4,int(environ.get("ARBITRATION_RANDOM_WORKERS","4"))))
        with ThreadPoolExecutor(max_workers=workers,thread_name_prefix="random_policy") as pool:
            for result in pool.map(random_learned,enumerate(randoms)):
                random_policy.append(result)
                if len(random_policy)%10==0:print('RANDOM_POLICY_DONE',len(random_policy),'of',len(randoms),flush=True)
        # K2: 20 pathway-aware random controls use identical frozen layer maps and ridge protocol.
        for j,d in enumerate(randoms[:PATH_DRAWS]):
            rzt=(q[:,d]@raw[d])@w.T;rzo=(qo[:,d]@raw[d])@w.T;rct=z-z0-rzt;rco=oz-oz0-rzo
            ptt,pto,_=bank[j+1];rga=features(z0,rzt,rct,z,q);rgo=features(oz0,rzo,rco,oz,qo)
            rt_dist=np.sort(np.linalg.norm(q[:,d][:,None]-cent[:,d][None],axis=2),axis=1)[:,:2];ro_dist=np.sort(np.linalg.norm(qo[:,d][:,None]-cent[:,d][None],axis=2),axis=1)[:,:2]
            rf=np.c_[rga,rt_dist,ptt];ro=np.c_[rgo,ro_dist,pto];rs=[z]+[z0+rzt+a*rct for a in ALPHA[:-1]];rso=[oz]+[oz0+rzo+a*rco for a in ALPHA[:-1]]
            rz,_,rm=gain_distill(rf,ro,rs,rso,z,y,s,(m,t,f,j,"random_path_D"),"random_pathway_gain")
            random_pathway.append({"draw":j,"control":"equal_rank_random_pathway_D","mean_subject_BA":float(np.mean([r["BA"] for r in gain_policy_rows(oy,os,rz,rm)])),"selected_delta":rm["selected_delta"],"train_nested_subject_BA":rm["train_nested_subject_BA"],"subject_rows":gain_policy_rows(oy,os,rz,rm)})
        prov={"checkpoint_sha256":sha(ckpt),"normalizer_sha256":data["normalizer"]["mean_std_sha256"],"basis_sha256":basis_sha,"protected_dims":dims.tolist(),"split_sha256":data["split_sha256"],"random_subsets_sha256":hashlib.sha256(np.concatenate(randoms).tobytes()).hexdigest(),"implementation_sha256":sha(Path(__file__)),"surface_sha256":sha(surface_path),"protocol_sha256":expected_hash,"implementation_revision":"protocol_repair_v2"}
        jwrite(dst,{**base,"status":"COMPLETE",**prov,"exact_max_abs":exact,"oracle_specificity":specrow,"response_surface":metric_rows(oy,os,outputs),"taxonomy":tax,"pathway_audit":pfa,"reliability":rel,"direct_arbiter":arb,"disagreement":disagreement_rows(oy,os,oz0,ozp,ozc,oz,outputs,sp2),"gain_distillation":policy,"two_d_arbitration":two_policy,"random_oracle":random,"random_learned_policy":random_policy,"random_pathway_policy":random_pathway,"policy_status":"COMPLETE_TRAIN_ONLY_NESTED_SUBJECT_CV"});print("CELL_COMPLETE",m,t,f,flush=True);del net;torch.cuda.empty_cache()
    except Exception as e:
        import traceback
        jwrite(dst,{**base,"status":"FAIL_CLOSED","reason":f"{type(e).__name__}: {e}"});print("CELL_FAIL_CLOSED",m,t,f,str(e),flush=True);traceback.print_exc();raise SystemExit(1)

def lock()->None:
    rows=[]
    for m in MODELS:
      for t in TASKS:
       for f in FOLDS:
        rec,stored,ckpt,pathway=PW.previous(m,t,f);d=UP.outer_data(t,f);rows.append({"model":m,"task":t,"fold":f,"checkpoint_sha256":sha(ckpt),"normalizer_sha256":d["normalizer"]["mean_std_sha256"],"previous_pathway_sha256":sha(PW.path(m,t,f)),"previous_status":pathway.get("status"),"final_heldout_accessed":False})
    v={"schema":"PERSIST_EEG_PROTECTED_ARBITRATION_RELIABILITY_SEED0_V1","models":MODELS,"tasks":TASKS,"folds":FOLDS,"seed":0,"random_draws":DRAWS,"pathway_random_draws":PATH_DRAWS,"alpha_grid":ALPHA,"two_d_grid":SCALE,"backbone_training":False,"head_refit":False,"protected_reselection":False,"final_heldout_accessed":False,"cells":rows};jwrite(PROTOCOL/"PROVENANCE.json",v);(PROTOCOL/"PROVENANCE.sha256").write_text(sha(PROTOCOL/"PROVENANCE.json")+"\n",encoding="utf-8");print("PROTOCOL_LOCKED",len(rows),flush=True)
def aggregate()->None:
    cells=[]
    for m in MODELS:
     for t in TASKS:
      for f in FOLDS:
       p=tpath(m,t,f)
       if not p.is_file():raise RuntimeError(f"missing {m}/{t}/f{f}")
       cells.append(json.loads(p.read_text(encoding="utf-8")))
    status=[];tables=defaultdict(list)
    for c in cells:
      b={k:c.get(k) for k in ("model","task","fold","seed","status","checkpoint_sha256","basis_sha256","split_sha256")};status.append(b)
      if c.get("status")!="COMPLETE":continue
      if c.get('implementation_revision')!='protocol_repair_v2':raise RuntimeError('reject pre-repair cells')
      if c.get('final_heldout_accessed') is not False or c.get('exact_max_abs',1)>=1e-5:raise RuntimeError('cell integrity validation failed')
      if len(c.get('random_oracle',[]))!=100 or len(c.get('random_learned_policy',[]))!=100 or len(c.get('random_pathway_policy',[]))!=20:raise RuntimeError('missing fixed random controls')
      for name,key in (("P_RANDOM_ORACLE_SPECIFICITY","oracle_specificity"),("PC_RESPONSE_SURFACE_SUMMARY","response_surface"),("PC_ERROR_RESCUE_TAXONOMY","taxonomy"),("TRIAL_GEOMETRY_FEATURE_AUDIT","pathway_audit"),("RELIABILITY_RESULTS","reliability"),("RELIABILITY_RESULTS","direct_arbiter"),("PATHWAY_RELIABILITY_ABLATION","reliability"),("GAIN_DISTILLATION_RESULTS","gain_distillation"),("TWO_D_ARBITRATION_RESULTS","two_d_arbitration"),("DISAGREEMENT_SUBSET_RESULTS","disagreement"),("RANDOM_POLICY_CONTROLS","random_oracle"),("RANDOM_POLICY_CONTROLS","random_learned_policy"),("RANDOM_POLICY_CONTROLS","random_pathway_policy")):
       vals=c.get(key,[]);vals=vals if isinstance(vals,list) else [vals]
       tables[name]+=[{**b,**r} for r in vals]
    cwrite(OUT/"CELL_STATUS.csv",status)
    for n,r in tables.items():cwrite(OUT/(n+".csv"),r)
    summary=[]
    for m in MODELS:
     for t in TASKS:
      rr=[r for r in tables["P_RANDOM_ORACLE_SPECIFICITY"] if r["model"]==m and r["task"]==t];summary.append({"model":m,"task":t,"status":"COMPLETE" if len(rr)==5 else "INCOMPLETE","completed_folds":len(rr),"P_specific_oracle_gain_mean":float(np.mean([r["P_oracle_BA_gain"] for r in rr])) if rr else float("nan"),"P_minus_random_mean":float(np.mean([r["P_minus_random"] for r in rr])) if rr else float("nan")})
    cwrite(OUT/"MODEL_TASK_SUMMARY.csv",summary)
    cwrite(OUT/"PC_EXACT_DECOMPOSITION_AUDIT.csv",[{**r,"exact_max_abs":c.get("exact_max_abs"),"final_heldout_accessed":c.get("final_heldout_accessed")} for r,c in zip(status,cells)])
    completed=sum(c.get("status")=="COMPLETE" for c in cells);failed=len(cells)-completed
    paired=[]
    for m in MODELS:
     for t in TASKS:
      rows=[r for r in tables['PC_RESPONSE_SURFACE_SUMMARY'] if r['model']==m and r['task']==t]
      if not rows:continue
      policies=[k[:-3] for k in rows[0] if k.endswith('_BA') and k!='native_BA']
      for policy_name in policies:
       for metric in ('BA','MacroF1','CE'):
        subject_deltas=defaultdict(list);fold_deltas=defaultdict(list)
        for row in rows:
         delta=float(row[policy_name+'_'+metric])-float(row['native_'+metric]);subject_deltas[str(row['subject_id'])].append(delta);fold_deltas[row['fold']].append(delta)
        values=np.asarray([np.mean(v) for v in subject_deltas.values()]);rng=np.random.default_rng(seed(m,t,policy_name,metric,'subject_bootstrap'))
        boot=np.mean(rng.choice(values,size=(BOOT,len(values)),replace=True),axis=1)
        paired.append({'model':m,'task':t,'policy':policy_name,'metric':metric,'unit':'biological_subject','subjects':len(values),'delta_mean':float(values.mean()),'ci_low':float(np.quantile(boot,.025)),'ci_high':float(np.quantile(boot,.975)),'positive_folds':sum(np.mean(v)>0 for v in fold_deltas.values()),'folds':len(fold_deltas)})
    cwrite(OUT/'PAIRED_SUBJECT_BOOTSTRAP.csv',paired)
    def mt(name,m,t):return [r for r in tables[name] if r['model']==m and r['task']==t]
    def mean(rows,key):
      out=[]
      for r in rows:
       try:
        v=float(r.get(key,'nan'))
        if np.isfinite(v):out.append(v)
       except (TypeError,ValueError):pass
      return float(np.mean(out)) if out else float('nan')
    def stat(m,t,policy):
      return next((r for r in paired if r['model']==m and r['task']==t and r['policy']==policy and r['metric']=='BA'),None)
    def brief(s):return 'not available' if s is None else f'{s["delta_mean"]:.6g} (95% CI [{s["ci_low"]:.6g}, {s["ci_high"]:.6g}]; positive folds {s["positive_folds"]}/{s["folds"]})'
    report=[f'# Protected P/C arbitration reliability audit\n\nCells: {completed}/{len(cells)} COMPLETE; {failed} fail-closed. Final-heldout access: NO. Frozen backbone, native head, bases, Protected dimensions, splits and locked random subsets were reused. Statistical unit: biological subject; repeated fold measurements are averaged per subject before bootstrapping. Oracle rows are label-aware diagnostics only and are never presented as deployable-policy results.\n']
    for row in summary:
      m,t=row['model'],row['task'];oracle=mt('P_RANDOM_ORACLE_SPECIFICITY',m,t);tax=mt('PC_ERROR_RESCUE_TAXONOMY',m,t);rel=mt('RELIABILITY_RESULTS',m,t);dis=mt('DISAGREEMENT_SUBSET_RESULTS',m,t)
      pminus=mean(oracle,'P_minus_random');pp=mean(oracle,'P_percentile');ep=mean(oracle,'empirical_p')
      families={}
      for target in ('rP','rC',''):
       for family in ('A','C','D'):
        rows=[r for r in rel if r.get('target','')==target and r.get('feature_family')==family]
        families[(target or 'arbiter',family)]=(mean(rows,'AUROC'),mean(rows,'AUPRC'),mean(rows,'Brier'),mean(rows,'accuracy'))
      tax_keys=('C_DOWN_ONLY','P_UP_ONLY','P_DOWN_ONLY','C_UP_ONLY','JOINT_REWEIGHT','NOT_REWEIGHT_RECOVERABLE')
      tax_summary=', '.join(f'{k}={mean(tax,k):.3g}' for k in tax_keys)
      dominant=max(tax_keys,key=lambda k:mean(tax,k)) if tax else 'not available'
      drows=[r for r in dis if r.get('subset')=='disagreement'];disrate=mean(drows,'fraction');precover=mean(drows,'oracle_recoverable_fraction')
      alpha=stat(m,t,'gain_alpha_D');two=stat(m,t,'gain_2d_D');reliability=stat(m,t,'reliability_D')
      p_specific='is not supported' if not np.isfinite(pminus) or pminus<=0 else 'is descriptively positive but requires the fixed random-control comparison for attribution'
      report.append(f'''\n## {m} / {t}\n\nCompleted folds: {row["completed_folds"]}/5.\n\n1. **P-specific oracle headroom.** Mean P suppression-oracle BA gain was {row["P_specific_oracle_gain_mean"]:.6g}; P minus the equal-rank-random mean was {pminus:.6g}, mean P percentile was {pp:.6g}, and mean empirical p was {ep:.6g}. Therefore Protected-specific headroom {p_specific}; this report does not attribute generic reweighting headroom to P.\n\n2. **How native errors are reweight-recoverable.** Mean subject-fold rescue counts were {tax_summary}; the largest mean category was {dominant}. Mean minimum log-scale intervention distance was {mean(tax,'minimum_intervention_distance'):.6g}. These are response-surface diagnostics, not policy performance.\n\n3. **P-only and C-only reliability.** FEATURE-D outer metrics (AUROC/AUPRC/Brier) were rP={families[('rP','D')][0]:.6g}/{families[('rP','D')][1]:.6g}/{families[('rP','D')][2]:.6g} and rC={families[('rC','D')][0]:.6g}/{families[('rC','D')][1]:.6g}/{families[('rC','D')][2]:.6g}. These are outer-development evaluations only.\n\n4. **P/C disagreement arbiter.** Disagreement fraction was {disrate:.6g}; the direct arbiter FEATURE-D outer AUROC/AUPRC/Brier/accuracy was {families[('arbiter','D')][0]:.6g}/{families[('arbiter','D')][1]:.6g}/{families[('arbiter','D')][2]:.6g}/{families[('arbiter','D')][3]:.6g}.\n\n5. **Pathway-consistency contribution.** For rP, FEATURE-A to C to D AUROC was {families[('rP','A')][0]:.6g} → {families[('rP','C')][0]:.6g} → {families[('rP','D')][0]:.6g}; for rC it was {families[('rC','A')][0]:.6g} → {families[('rC','C')][0]:.6g} → {families[('rC','D')][0]:.6g}. This ablation is descriptive across the fixed outer folds; no untested causal claim is made.\n\n6. **Where suppression oracle headroom occurs.** On disagreement trials, mean oracle-recoverable fraction was {precover:.6g}; the rescue taxonomy above identifies whether attenuation, P scaling, joint scaling, or no fixed-grid reweighting was involved.\n\n7. **Can TRAIN-only gain models distill suppression benefit?** FEATURE-D gain-alpha BA delta versus native was {brief(alpha)}.\n\n8. **Does the learned alpha policy improve outer subjects?** The result in item 7 is the biological-subject bootstrap estimate; an interval spanning zero is not evidence of a reliable improvement.\n\n9. **Does 2D arbitration clearly exceed suppression-only?** FEATURE-D 2D gain-policy BA delta was {brief(two)}, versus suppression-only {brief(alpha)}. This is a comparison of fixed TRAIN-only policies, not an oracle comparison.\n\n10. **Why learned policies may fail.** FEATURE-D reliability-routing BA delta was {brief(reliability)}. Reliability, disagreement, pathway ablation, random-policy controls and gain-distillation results are all retained in their compact tables; a near-zero or uncertain policy delta must not be attributed to one mechanism without those diagnostics agreeing.\n''')
      report.append('\n### Policy bootstrap summary\n')
      for s in paired:
       if s['model']==m and s['task']==t and s['metric']=='BA':report.append(f'- {s["policy"]}: {brief(s)}')
    (OUT/'REPORT.md').write_text('\n'.join(report),encoding='utf-8');jwrite(OUT/'PROVENANCE.json',json.loads((PROTOCOL/'PROVENANCE.json').read_text()));print('AGGREGATE_COMPLETE',flush=True)
def main()->None:
 p=argparse.ArgumentParser();p.add_argument("mode",choices=("lock","cell","aggregate"));p.add_argument("model",nargs="?");p.add_argument("task",nargs="?");p.add_argument("fold",nargs="?",type=int);a=p.parse_args()
 if a.mode=="lock":lock()
 elif a.mode=="aggregate":aggregate()
 else:
  if a.model not in MODELS or a.task not in TASKS or a.fold not in FOLDS:raise SystemExit("invalid cell")
  cell(a.model,a.task,a.fold)
if __name__=="__main__":main()
