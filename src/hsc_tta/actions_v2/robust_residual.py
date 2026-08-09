from __future__ import annotations

import copy
import hashlib
import time

import numpy as np
import torch
from torch import nn

from hsc_tta.actions_v2.base import ActionV2


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, bottleneck: int=64):
        super().__init__()
        self.norm=nn.LayerNorm(dim); self.w1=nn.Linear(dim,bottleneck); self.w2=nn.Linear(bottleneck,dim)
        nn.init.zeros_(self.w2.weight); nn.init.zeros_(self.w2.bias)

    def forward(self,z:torch.Tensor)->torch.Tensor:
        return z+self.w2(torch.nn.functional.gelu(self.w1(self.norm(z))))


class RobustResidualAdapter(ActionV2):
    """Subject-reset residual adaptation with explicit reliability filtering and U-only rollback."""

    def __init__(self, source_head: nn.Module, *, steps: int=3, learning_rate: float=5e-5, beta: float=.5,
                 gamma: float=.1, eta: float=1e-3, mu: float=.5, reliability_quantile: float=.2,
                 device: str="cuda"):
        self.source_head=source_head.to(device).eval(); self.device=torch.device(device)
        for p in self.source_head.parameters(): p.requires_grad_(False)
        self.steps=steps; self.lr=learning_rate; self.beta=beta; self.gamma=gamma; self.eta=eta; self.mu=mu
        self.reliability_quantile=reliability_quantile
        self.dim=int(source_head.classifier.in_features)
        self.adapter=ResidualBlock(self.dim,64).to(device)
        self._initial=copy.deepcopy(self.adapter.state_dict()); self._frozen=None; self._status="not_adapted"; self._diagnostics={}

    def reset_subject(self)->None:
        self.adapter.load_state_dict(self._initial); self._frozen=None; self._status="reset"; self._diagnostics={}

    def _source(self,tokens:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
        logits,hidden=self.source_head(tokens,return_hidden=True); return logits,hidden

    def adapt_on_context(self,context:torch.Tensor)->None:
        self.reset_subject(); started=time.perf_counter(); x=context.to(self.device).float()
        with torch.no_grad():
            source_logits,hidden=self._source(x); source_prob=source_logits.softmax(1)
            entropy=-(source_prob*source_prob.clamp_min(1e-8).log()).sum(1)
            confidence=source_prob.max(1).values
            threshold=torch.quantile(entropy,self.reliability_quantile)
            reliable=(entropy<=threshold)&(confidence>=torch.quantile(confidence,1-self.reliability_quantile))
        if reliable.sum()<max(2,source_prob.shape[1]):
            self._status="unavailable_insufficient_reliable_context"; self._diagnostics={"reliable_count":int(reliable.sum())}; return
        optimizer=torch.optim.AdamW(self.adapter.parameters(),lr=self.lr)
        initial_objective=None; best_state=None; best_objective=float("inf")
        for _ in range(self.steps):
            optimizer.zero_grad(set_to_none=True); adapted=self.adapter(hidden[reliable])
            logits=self.source_head.classifier(adapted); prob=logits.softmax(1)
            selective_entropy=-(prob*prob.clamp_min(1e-8).log()).sum(1).mean()
            source_kl=torch.nn.functional.kl_div(logits.log_softmax(1),source_prob[reliable],reduction="batchmean")
            noisy=self.adapter(hidden[reliable]+.01*torch.randn_like(hidden[reliable]))
            consistency=torch.nn.functional.mse_loss(noisy,adapted)
            change=sum((p-p0.to(self.device)).square().sum() for p,p0 in zip(self.adapter.parameters(),self._initial.values()))
            class_mass=prob.mean(0); collapse=torch.relu(.5-class_mass.max()).neg()+torch.relu(class_mass.max()-.8)
            objective=selective_entropy+self.beta*source_kl+self.gamma*consistency+self.eta*change+self.mu*collapse
            if initial_objective is None: initial_objective=float(objective.detach())
            if not torch.isfinite(objective): self._status="unavailable_nonfinite"; self.adapter.load_state_dict(self._initial); return
            objective.backward(); torch.nn.utils.clip_grad_norm_(self.adapter.parameters(),1.0); optimizer.step()
            if float(objective.detach())<best_objective:
                best_objective=float(objective.detach()); best_state=copy.deepcopy(self.adapter.state_dict())
        assert best_state is not None
        self.adapter.load_state_dict(best_state)
        with torch.no_grad():
            final_prob=self.source_head.classifier(self.adapter(hidden)).softmax(1); collapse_score=float(final_prob.mean(0).max())
        update_norm=float(sum((p-p0.to(self.device)).square().sum() for p,p0 in zip(self.adapter.parameters(),self._initial.values())).sqrt())
        if collapse_score>.8 or update_norm>5 or (initial_objective is not None and best_objective>initial_objective+1e-6):
            self.adapter.load_state_dict(self._initial); self._status="unavailable_rollback"
        else: self._status="ok"
        self._diagnostics={"reliable_count":int(reliable.sum()),"objective_before":initial_objective,
            "objective_after":best_objective,"adapter_update_norm":update_norm,"collapse_score":collapse_score,
            "source_kl":float(source_kl.detach()),"adaptation_runtime":time.perf_counter()-started}

    def _predict(self,tokens:torch.Tensor)->np.ndarray:
        if self._status!="ok": raise RuntimeError(f"robust adapter unavailable: {self._status}")
        with torch.inference_mode():
            _,hidden=self._source(tokens.to(self.device).float()); return self.source_head.classifier(self.adapter(hidden)).softmax(1).cpu().numpy()

    def predict_context(self,context:torch.Tensor)->np.ndarray: return self._predict(context)
    def freeze_state(self)->dict[str,torch.Tensor]:
        if self._status!="ok": raise RuntimeError("cannot freeze unavailable action")
        self._frozen={k:v.detach().cpu().clone() for k,v in self.adapter.state_dict().items()}; return self._frozen
    def predict_future(self,future:torch.Tensor)->np.ndarray:
        if self._frozen is None: raise RuntimeError("future prediction requires frozen U-derived state")
        self.adapter.load_state_dict(self._frozen); return self._predict(future)
    def state_hash(self)->str:
        digest=hashlib.sha256()
        for name,value in sorted(self.adapter.state_dict().items()): digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()
    def diagnostics(self)->dict[str,object]: return dict(self._diagnostics)
    def failure_status(self)->str: return self._status
