import numpy as np
import pytest
import torch
from torch import nn

from hsc_tta.v3.actions import AdapterConfig, CorrectedResidualAdapter, FrozenT3AAction, T3AConfig
from hsc_tta.v3.augmentations import TokenNuisanceAugmenter


class Head(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder=nn.Linear(8,8); self.classifier=nn.Linear(8,3)
    def forward(self,x,return_hidden=False):
        hidden=self.encoder(x.mean((1,2))); logits=self.classifier(hidden)
        return (logits,hidden) if return_hidden else logits


def test_nuisances_are_deterministic_and_distinct():
    x=torch.randn(6,1,4,8); augmenter=TokenNuisanceAugmenter()
    first=augmenter.all(x); second=augmenter.all(x)
    assert set(first)==set(augmenter.names)
    assert all(torch.equal(first[name],second[name]) for name in first)
    assert all(not torch.equal(x,value) for value in first.values())


def test_collapse_penalty_is_nonnegative():
    balanced=torch.full((10,3),1/3); collapsed=torch.tensor([[.98,.01,.01]]).repeat(10,1)
    assert CorrectedResidualAdapter.collapse_penalty(balanced,.8).item()==0
    assert CorrectedResidualAdapter.collapse_penalty(collapsed,.8).item()>0


def test_t3a_subject_reset_and_frozen_prediction():
    action=FrozenT3AAction(Head(),T3AConfig(filter_k=5,prototype_interpolation=.5),device="cpu")
    x=torch.randn(20,1,4,8); action.adapt_on_adapt(x); digest=action.freeze_state(); prediction=action.predict_probe(x)
    assert digest==action.state_hash() and prediction.shape==(20,3)
    action.reset_subject()
    with pytest.raises(RuntimeError,match="frozen"): action.predict_future(x)


def test_adapter_freeze_is_immutable_or_failure_explicit():
    torch.manual_seed(4); x=torch.randn(30,1,4,8)
    action=CorrectedResidualAdapter(Head(),AdapterConfig(steps=1,reliability_quantile=.4),device="cpu")
    action.adapt_on_adapt(x)
    if action.failure_status()=="adapted":
        digest=action.freeze_state(); before={k:v.clone() for k,v in action._frozen.items()}
        action.predict_probe(x); action.predict_future(x)
        assert digest==action.state_hash() and all(torch.equal(before[k],action._frozen[k]) for k in before)
    else:
        assert action.failure_status().startswith("unavailable:")
        with pytest.raises(RuntimeError): action.freeze_state()
