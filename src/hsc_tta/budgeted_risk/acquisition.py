from __future__ import annotations

import hashlib
import numpy as np

from .inclusion_index import risk_index_entropy


def _rank01(values: np.ndarray) -> np.ndarray:
    x=np.asarray(values,float);order=np.lexsort((np.arange(len(x)),x));ranks=np.empty(len(x),float);ranks[order]=np.arange(len(x))
    return ranks/max(len(x)-1,1)


def random_order(n:int, dataset:str, seed:int, subject_id:str, repeat:int)->np.ndarray:
    token=f"{dataset}|{seed}|{subject_id}|{repeat}|budgeted-risk-random-v1".encode();number=int.from_bytes(hashlib.sha256(token).digest()[:8],"big")
    return np.random.default_rng(number).permutation(n)


def temporal_stratified_order(n:int)->np.ndarray:
    if n<=0:return np.empty(0,int)
    # Greedy farthest-point sampling on normalized time.  Every prefix is a
    # nested temporal cover; deterministic lower-index tie breaking makes the
    # acquisition transcript reproducible.
    positions=(np.arange(n,dtype=float)+.5)/n
    remaining=set(range(n));order=[]
    first=min(remaining,key=lambda i:(abs(positions[i]-.5),i))
    order.append(first);remaining.remove(first)
    while remaining:
        chosen=min(
            remaining,
            key=lambda i:(-min(abs(positions[i]-positions[j]) for j in order),i),
        )
        order.append(chosen);remaining.remove(chosen)
    return np.asarray(order,int)


def predictive_entropy_order(probabilities:np.ndarray)->np.ndarray:
    p=np.asarray(probabilities,float);h=-(p*np.log(p+1e-12)).sum(1)
    return np.lexsort((np.arange(len(p)),-h))


def class_balanced_order(probabilities:np.ndarray)->np.ndarray:
    p=np.asarray(probabilities,float);pred=p.argmax(1);h=-(p*np.log(p+1e-12)).sum(1);queues={}
    for cls in sorted(set(pred.tolist())):queues[cls]=list(np.flatnonzero(pred==cls)[np.lexsort((np.flatnonzero(pred==cls),-h[pred==cls]))])
    order=[]
    while any(queues.values()):
        for cls in sorted(queues):
            if queues[cls]:order.append(queues[cls].pop(0))
    return np.asarray(order,int)


def diversity_order(embeddings:np.ndarray)->np.ndarray:
    x=np.asarray(embeddings,float);centroid=x.mean(0);remaining=set(range(len(x)));order=[]
    while remaining:
        if not order:scores=np.linalg.norm(x-centroid,axis=1)
        else:scores=np.min(np.linalg.norm(x[:,None,:]-x[np.asarray(order)][None,:,:],axis=2),axis=1)
        chosen=min(remaining,key=lambda i:(-scores[i],i));order.append(chosen);remaining.remove(chosen)
    return np.asarray(order,int)


def risk_entropy_order(probabilities:np.ndarray)->np.ndarray:
    score=risk_index_entropy(probabilities);return np.lexsort((np.arange(len(score)),-score))


def active_order(probabilities:np.ndarray,embeddings:np.ndarray,time_bins:int=4)->np.ndarray:
    p=np.asarray(probabilities,float);x=np.asarray(embeddings,float);h=_rank01(risk_index_entropy(p));centroid=x.mean(0);remaining=set(range(len(p)));selected=[];counts=np.zeros(time_bins,int);bins=np.minimum(np.arange(len(p))*time_bins//max(len(p),1),time_bins-1)
    while remaining:
        temporal=_rank01(1/(1+counts[bins]));
        if selected:diversity=np.min(np.linalg.norm(x[:,None,:]-x[np.asarray(selected)][None,:,:],axis=2),axis=1)
        else:diversity=np.linalg.norm(x-centroid,axis=1)
        d=_rank01(diversity);score=.5*h+.25*temporal+.25*d
        chosen=min(remaining,key=lambda i:(-score[i],i));selected.append(chosen);remaining.remove(chosen);counts[bins[chosen]]+=1
    return np.asarray(selected,int)


def acquisition_order(strategy:str,probabilities:np.ndarray,embeddings:np.ndarray,*,dataset:str,seed:int,subject_id:str,repeat:int=0)->np.ndarray:
    n=len(probabilities)
    if strategy=="random":return random_order(n,dataset,seed,subject_id,repeat)
    if strategy=="first":return np.arange(n)
    if strategy=="temporal":return temporal_stratified_order(n)
    if strategy=="predictive_entropy":return predictive_entropy_order(probabilities)
    if strategy=="class_balanced":return class_balanced_order(probabilities)
    if strategy=="diversity":return diversity_order(embeddings)
    if strategy=="risk_entropy":return risk_entropy_order(probabilities)
    if strategy=="active":return active_order(probabilities,embeddings)
    raise ValueError(strategy)
