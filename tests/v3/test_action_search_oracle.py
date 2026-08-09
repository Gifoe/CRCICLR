import numpy as np
import pandas as pd
import torch
from torch import nn

from hsc_tta.v3.action_search import action_grid, config_id, subject_fold
from hsc_tta.v3.evaluation import safe_point
from hsc_tta.v3.evaluation import SubjectEvaluator


def test_required_action_grid_is_finite_and_complete():
    config = {"t3a": {"filter_k": [5, 10, 20, 50], "confidence_threshold": [None, .7, .8, .9],
                       "prototype_interpolation": [.25, .5, .75, 1.]},
              "adapter": {"steps": [1, 3, 5], "learning_rate": [1e-5, 5e-5, 1e-4],
                          "source_preservation_weight": [.25, .5, 1.], "reliability_quantile": [.1, .2, .3]},
              "fixed": {"consistency_weight": .1, "parameter_weight": .001, "collapse_weight": .5, "collapse_rho": .8}}
    grid = action_grid(config)
    assert len(grid["official_t3a"]) == 64
    assert len(grid["robust_residual_adapter"]) == 81
    assert len({config_id("official_t3a", x) for x in grid["official_t3a"]}) == 64
    assert subject_fold("same-subject", 1) == subject_fold("same-subject", 1)
    with np.testing.assert_raises(ValueError): subject_fold("s", 0, 1)


def test_safe_point_uses_smallest_valid_set():
    probabilities = np.array([[.8, .2], [.55, .45], [.1, .9]])
    labels = np.array([0, 1, 1])
    result = safe_point(probabilities, labels, .20, np.array([.5, .75, 1.]))
    assert result["future_risk"] <= .20
    assert result["lambda_index"] == 1


def test_subject_evaluator_can_delay_future_open(monkeypatch, tmp_path):
    class Head(nn.Module):
        def __init__(self): super().__init__(); self.classifier=nn.Linear(3,2,bias=False)
        def forward(self,x,return_hidden=False):
            hidden=x.mean((1,2)); logits=self.classifier(hidden); return (logits,hidden) if return_hidden else logits
    monkeypatch.setattr("hsc_tta.v3.evaluation.load_source_model",lambda *args:(Head(),{}))
    monkeypatch.setattr("hsc_tta.v3.evaluation._tokens",lambda *args:torch.ones((len(args[-1]),1,1,3)))
    monkeypatch.setattr("hsc_tta.v3.evaluation._labels",lambda *args:np.zeros(len(args[-1]),dtype=int))
    evaluator=SubjectEvaluator(tmp_path,"hmc",0,"cpu")
    row=pd.Series({"adapt_indices":np.arange(3),"probe_indices":np.arange(3,6),"future_indices":np.arange(6,9)},name="hmc:s")
    episode=evaluator.prepare_episode(row,include_future=False)
    assert "future" not in episode and set(evaluator.source(episode))=={"adapt","probe"}
    evaluator.open_future(row,episode); assert len(evaluator.source(episode)["future"])==3
    with np.testing.assert_raises(RuntimeError): evaluator.open_future(row,episode)
