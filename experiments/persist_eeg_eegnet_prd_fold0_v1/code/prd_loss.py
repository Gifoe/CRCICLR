"""Un-detached subject-relative predictive-direction loss."""
from __future__ import annotations
import itertools
import torch
import torch.nn.functional as F


def unit(v: torch.Tensor) -> torch.Tensor: return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)
def direction(mu0: torch.Tensor, mu1: torch.Tensor) -> torch.Tensor: return unit(mu1-mu0)


def _subject_directions(z: torch.Tensor,y: torch.Tensor,sid: torch.Tensor,pair: tuple[int,int]) -> tuple[torch.Tensor,torch.Tensor]:
    ds=[]; magnitudes=[]
    for s in torch.unique(sid,sorted=True):
        select=sid==s; mu0=z[select&(y==pair[0])].mean(0); mu1=z[select&(y==pair[1])].mean(0); raw=mu1-mu0; ds.append(unit(raw)); magnitudes.append(raw.norm())
    return torch.stack(ds),torch.stack(magnitudes)


def prd_loss(zs: torch.Tensor,ys: torch.Tensor,ss: torch.Tensor,zq: torch.Tensor,yq: torch.Tensor,sq: torch.Tensor,num_classes: int=2) -> tuple[torch.Tensor,dict[str,torch.Tensor]]:
    losses=[]; sm=[]; qm=[]
    for pair in itertools.combinations(range(num_classes),2):
        ds,a=_subject_directions(zs,ys,ss,pair); dq,b=_subject_directions(zq,yq,sq,pair); pop=unit(ds.mean(0,keepdim=True)); losses.append((1-F.cosine_similarity(dq,pop.expand_as(dq),dim=1)).mean()); sm.append(a); qm.append(b)
    return torch.stack(losses).mean(),{"support_direction_norm":torch.cat(sm).mean(),"query_direction_norm":torch.cat(qm).mean()}


def explicit_binary(zs: torch.Tensor,ys: torch.Tensor,ss: torch.Tensor,zq: torch.Tensor,yq: torch.Tensor,sq: torch.Tensor) -> torch.Tensor:
    ds,_=_subject_directions(zs,ys,ss,(0,1)); dq,_=_subject_directions(zq,yq,sq,(0,1)); pop=unit(ds.mean(0,keepdim=True)); return (1-F.cosine_similarity(dq,pop.expand_as(dq),dim=1)).mean()
