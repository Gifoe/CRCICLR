from __future__ import annotations

import inspect
import sys

import numpy as np
import torch

sys.path.insert(0, "experiments/persist_eeg_persist_re_final/code")
import persist_re_core as c


def _model(n=6, rank=2):
    return c.PERSISTRE(8, n, rank)


def test_random_effect_stop_gradient_and_centering():
    model = _model()
    features = torch.randn(12, 8, requires_grad=True)
    subjects = torch.arange(12) % 6
    effect = model.random_effect(model.encode(features), subjects)
    effect.sum().backward()
    assert features.grad is None or torch.allclose(features.grad, torch.zeros_like(features.grad))
    assert model.U.grad is not None
    e, a = model.centered_effects()
    assert torch.allclose(e.mean(0), torch.zeros(2), atol=1e-7)
    assert torch.allclose(a.mean(0), torch.zeros(2), atol=1e-7)


def test_inference_has_no_subject_id_and_zero_random_effect():
    model = _model()
    rep = {"features": np.random.default_rng(0).normal(size=(5, 8)).astype("float32"), "labels": np.array([0, 1, 0, 1, 0]), "subjects": np.array(["a", "b", "a", "b", "a"])}
    value = c.predict(model, rep, {}, torch.device("cpu"))
    assert np.array_equal(value["random_effect"], np.zeros_like(value["random_effect"]))
    assert "subject" not in inspect.signature(c.predict).parameters or "subject_id" not in inspect.getsource(c.predict)


def test_deterministic_subject_partition_and_rotation():
    subjects = [str(i) for i in range(12)]
    assert c.partition_subjects(subjects, 3, 9) == c.partition_subjects(subjects, 3, 9)
    pseudo = set()
    for epoch in range(48):
        context, unseen = c.partition_subjects(subjects, epoch, 9)
        assert not set(context) & set(unseen)
        assert len(context) + len(unseen) == len(subjects)
        pseudo.update(unseen)
    assert pseudo == set(subjects)


def test_subject_balanced_loss_equalizes_trial_counts():
    logits = torch.tensor([[4.0, 0.0], [4.0, 0.0], [0.0, 4.0], [0.0, 4.0], [4.0, 0.0]])
    labels = torch.tensor([0, 0, 1, 1, 1])
    subjects = torch.tensor([0, 0, 1, 1, 1])
    value, by_subject = c.per_subject_class_ce(logits, labels, subjects)
    assert set(by_subject) == {0, 1}
    assert torch.isclose(value, torch.stack(list(by_subject.values())).mean())


def test_optimizer_scopes_are_disjoint():
    model = _model()
    groups = c.model_parameter_groups(model)
    assert groups["shared"].isdisjoint(groups["random_effect"])
    assert groups["random_effect"] == {"U", "B", "subject_embedding", "subject_intercept"}


def test_primary_has_no_grl_or_transport_and_controls_are_explicit():
    source = inspect.getsource(c.fit_model)
    # GRL is only constructed in the named adversarial control branch.
    assert 'method == "AdversarialMixed"' in source
    assert "PERSIST-RE" in source
    assert "transport" not in source.lower()
    assert "grad_reverse(z" in source


def test_detach_contracts_and_fixed_budget():
    source = inspect.getsource(c.fit_model)
    assert "p_det.detach()" in source
    assert "effect[context_mask].detach()" in source
    assert c.LR_HEAD == 1e-4 and c.LR_FEATURE == 1e-5 and c.LR_RE == 1e-3
    assert c.WEIGHT_DECAY == 1e-3 and c.GRAD_CLIP == 3.0 and c.LAMBDA_Q == 1.0 and c.GAMMA_A == 1.0


def test_source_gate_bootstrap_is_subject_level():
    import run_source

    source = inspect.getsource(run_source.summarize)
    assert 'groupby(["dataset", "method", "subject_id"]' in source
    mean, lo, hi = run_source.bootstrap_delta(np.array([0.1, 0.2, 0.3]), 1)
    assert lo <= mean <= hi

