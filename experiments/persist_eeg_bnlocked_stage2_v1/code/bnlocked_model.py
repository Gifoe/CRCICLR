"""BNLOCK Stage-2 model: frozen anchor and LiteBN with trainable affine only."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

ALPHA, EMA_BETA = .5, .99


def bn_modules(module: nn.Module) -> list[tuple[str, nn.modules.batchnorm._BatchNorm]]:
    answer=[(n,m) for n,m in module.named_modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)]
    if not answer: raise RuntimeError('LiteBN contains no BatchNorm modules')
    return answer


def bn_buffers(module: nn.Module) -> dict[str, torch.Tensor]:
    out={}
    for name,bn in bn_modules(module):
        prefix=f'{name}.' if name else ''
        out[prefix+'running_mean']=bn.running_mean.detach().clone(); out[prefix+'running_var']=bn.running_var.detach().clone(); out[prefix+'num_batches_tracked']=bn.num_batches_tracked.detach().clone()
    return out


def assert_buffers(module: nn.Module, source: dict[str, torch.Tensor]) -> None:
    now=bn_buffers(module)
    if set(now)!=set(source) or any(not torch.equal(now[k].cpu(),source[k].cpu()) for k in source):
        raise RuntimeError('BNLOCK_BUFFER_INVARIANT_VIOLATION')


def frozen(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters(): parameter.requires_grad_(False)
    if module.training or any(parameter.requires_grad for parameter in module.parameters()):
        raise RuntimeError('frozen evaluation invariant failed')
    return module


class BNLockedFusion(nn.Module):
    def __init__(self, anchor: nn.Module, expressive: nn.Module) -> None:
        super().__init__(); self.anchor=anchor; self.expressive=expressive
        for p in self.anchor.parameters(): p.requires_grad_(False)
        self.anchor.eval(); self._lock_bn()

    def _lock_bn(self) -> None:
        for _,bn in bn_modules(self.expressive): bn.eval()

    def train(self, mode: bool=True) -> 'BNLockedFusion':
        self.training=mode; self.expressive.train(mode); self._lock_bn(); self.anchor.eval()
        if any(bn.training for _,bn in bn_modules(self.expressive)) or self.anchor.training: raise RuntimeError('BNLOCK train-mode invariant failed')
        return self

    def forward(self,x:torch.Tensor)->tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
        self.anchor.eval(); self._lock_bn()
        with torch.no_grad(): a,_=self.anchor(x)
        e,_=self.expressive(x); return a,e,ALPHA*a+ALPHA*e


def means(v:torch.Tensor,slots:torch.Tensor,k:int)->torch.Tensor:
    total=torch.zeros(k,device=v.device,dtype=v.dtype); cnt=torch.zeros_like(total); total.scatter_add_(0,slots,v);cnt.scatter_add_(0,slots,torch.ones_like(v))
    if bool((cnt==0).any()): raise RuntimeError('sampled subject absent')
    return total/cnt


def losses(a:torch.Tensor,e:torch.Tensor,f:torch.Tensor,y:torch.Tensor,slots:torch.Tensor,k:int,method:str)->dict[str,torch.Tensor]:
    la=means(F.cross_entropy(a.float(),y,reduction='none').detach(),slots,k).detach(); le=means(F.cross_entropy(e.float(),y,reduction='none'),slots,k); lf=means(F.cross_entropy(f.float(),y,reduction='none'),slots,k)
    mean,expr=lf.mean(),le.mean(); tail=lf.topk(int(math.ceil(.25*k))).values.mean(); regret=torch.relu(lf-la); reg=regret.mean()
    if method=='BNLOCK-JOINTCE': total=mean+.25*expr
    elif method=='BNLOCK-NRRF': total=mean+.5*tail+reg+.25*expr
    else: raise ValueError(method)
    return {'total':total,'L_mean':mean,'L_expr':expr,'L_tail':tail,'L_regret':reg,'positive_regret_fraction':(regret>0).float().mean(),'mean_positive_regret':regret[regret>0].mean() if bool((regret>0).any()) else regret.new_zeros(()),'worst_subject_loss':lf.max()}


def ema_start(module:nn.Module)->dict[str,torch.Tensor]: return {n:p.detach().clone() for n,p in module.named_parameters()}
@torch.no_grad()
def ema_update(ema:dict[str,torch.Tensor],module:nn.Module)->None:
    for n,p in module.named_parameters(): ema[n].mul_(EMA_BETA).add_(p.detach(),alpha=1-EMA_BETA)
@torch.no_grad()
def ema_load(ema:dict[str,torch.Tensor],module:nn.Module)->None:
    for n,p in module.named_parameters():p.copy_(ema[n])
