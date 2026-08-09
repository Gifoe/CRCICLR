import torch
from torch import nn

from hsc_tta.actions_v2.robust_residual import RobustResidualAdapter


class Head(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder=nn.Linear(200,256); self.classifier=nn.Linear(256,4)
    def forward(self,x,return_hidden=False):
        h=self.encoder(x.mean((1,2))); y=self.classifier(h); return (y,h) if return_hidden else y


def test_robust_adapter_parameter_scope_reset_and_u_only_rollback():
    action=RobustResidualAdapter(Head(),device="cpu",steps=1,reliability_quantile=.5)
    source=[p.detach().clone() for p in action.source_head.parameters()]
    initial=action.state_hash(); action.adapt_on_context(torch.randn(20,1,3,200)); action.reset_subject()
    assert action.state_hash()==initial
    assert all(torch.equal(a,b) for a,b in zip(source,action.source_head.parameters()))
    assert all(not p.requires_grad for p in action.source_head.parameters())


def test_action_failure_is_explicit():
    action=RobustResidualAdapter(Head(),device="cpu",steps=1,reliability_quantile=.01)
    action.adapt_on_context(torch.zeros(2,1,3,200))
    assert action.failure_status().startswith("unavailable")
