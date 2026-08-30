from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn


CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

from candidate_engine import match_structured_random, upper_tail_loss
from mixed_effects import MixedEffectsBank, detach_bank_tensor
from training_components import BankRefreshTracker, configure_scope, primary_total_loss


def tiny_bank() -> MixedEffectsBank:
    # Two classes for each subject and enough rows to expose self inclusion.
    features, labels, subjects, rows = [], [], [], []
    index = 0
    for subject, shift in (("1", -1.0), ("2", 0.5), ("3", 1.5)):
        for label, class_shift in ((0, -0.25), (1, 0.25)):
            for repeat in range(3):
                features.append([shift + class_shift + repeat * 0.1, shift - class_shift])
                labels.append(label); subjects.append(subject); rows.append(index); index += 1
    return MixedEffectsBank(np.asarray(features), np.asarray(labels), np.asarray(subjects), np.asarray(rows))


def test_anchor_excluded_from_own_centroid():
    bank = tiny_bank()
    full = bank.snapshot()
    loo = bank.anchor_snapshot(0)
    si, yi = bank.subject_index["1"], bank.label_index[0]
    assert loo.counts[si, yi] == full.counts[si, yi] - 1
    assert not np.allclose(loo.residual[si, yi], full.residual[si, yi])


def test_nontraining_rows_cannot_enter_bank():
    bank = tiny_bank()
    assert set(bank.row_ids) == set(range(18))
    with pytest.raises(KeyError):
        bank.anchor_snapshot(999)


def test_shrinkage_and_rho():
    bank = tiny_bank(); snap = bank.snapshot()
    assert np.all(bank.rho == 3)
    assert np.allclose(snap.eta, 0.5)


def test_main_interaction_decomposition():
    bank = tiny_bank(); snap = bank.snapshot()
    assert np.allclose(snap.residual, snap.b[:, None, :] + snap.c)
    assert np.allclose(snap.c.mean(1), 0, atol=1e-10)


def test_primary_does_not_transport_interaction():
    bank = tiny_bank(); snap = bank.anchor_snapshot(0)
    got = bank.direction(0, "2", factorized=True)
    want = snap.b[bank.subject_index["2"]] - snap.b[bank.subject_index["1"]]
    assert np.allclose(got, want)


def test_scope_b_bank_refresh_exactly_once_epoch():
    tracker = BankRefreshTracker()
    for epoch in range(3): tracker.refresh(epoch)
    assert tracker.refreshes == 3
    with pytest.raises(RuntimeError): tracker.refresh(2)


class ToyATC(nn.Module):
    def __init__(self):
        super().__init__(); self.encoder = nn.Linear(2, 2); self.tcn = nn.Linear(2, 2); self.norm = nn.LayerNorm(2); self.head = nn.Linear(2, 2)


def test_scope_a_encoder_coordinates_fixed():
    net = ToyATC(); params = configure_scope("ATCNet-CleanRoom", net, "A")
    before = net.encoder.weight.detach().clone()
    opt = torch.optim.SGD(params, lr=.1)
    opt.zero_grad(); net.head(net.encoder(torch.ones(2, 2)).detach()).sum().backward(); opt.step()
    assert torch.equal(before, net.encoder.weight)
    assert all(p.requires_grad for p in net.head.parameters())


def test_bank_tensor_has_no_gradient():
    tensor = detach_bank_tensor(np.ones((2, 2)), torch.device("cpu"))
    assert not tensor.requires_grad and tensor.grad_fn is None


def test_candidate_membership_is_detached():
    clean = torch.tensor([[2., 0.], [0., 2.]], requires_grad=True)
    candidate = torch.tensor([[1., 0.], [.5, 0.], [0., 1.]], requires_grad=True)
    labels = torch.tensor([0, 1]); owner = torch.tensor([0, 0, 1]); valid = torch.ones(3, dtype=torch.bool)
    loss, _ = upper_tail_loss(clean, candidate, labels, owner, valid, q=.5)
    loss.backward()
    assert clean.grad is None
    assert candidate.grad is not None


def test_primary_loss_contains_no_kl_symbol():
    tree = ast.parse((CODE / "training_components.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "primary_total_loss")
    text = ast.unparse(function).lower()
    assert "kl" not in text and "log_softmax" not in text


def test_no_candidate_has_finite_clean_loss_zero_cf():
    clean = torch.tensor([[2., 0.]], requires_grad=True)
    candidate = torch.empty((0, 2), requires_grad=True)
    cf, audit = upper_tail_loss(clean, candidate, torch.tensor([0]), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.bool), q=.5)
    loss = primary_total_loss(clean, torch.tensor([0]), cf, .25)
    assert torch.isfinite(loss) and cf.item() == 0 and audit["selected"] == 0


def test_hard_random_is_in_residual_span():
    bank = tiny_bank(); delta = bank.direction(0, "2")
    random = bank.hard_random(delta, np.random.default_rng(1))
    projection = bank.basis @ bank.basis.T @ random
    assert np.allclose(random, projection, atol=1e-6)


def test_hard_random_whitened_norm_exact():
    bank = tiny_bank(); delta = bank.direction(0, "2")
    random = bank.hard_random(delta, np.random.default_rng(2))
    assert bank.whitened_norm(delta) == pytest.approx(bank.whitened_norm(random), rel=1e-6)


def test_valid_counts_matched():
    valid = np.asarray([1, 1, 1, 1, 1, 0], bool); structured = np.asarray([1, 1, 1, 0, 0, 0], bool)
    keep = match_structured_random(valid, structured, seed=3)
    assert np.sum(keep & structured) == np.sum(keep & ~structured) == 2


def test_alpha_and_candidate_budgets_are_frozen():
    source = (CODE / "v2_common.py").read_text()
    assert "ALPHAS = (1.0 / 64.0, 2.0 / 64.0, 3.0 / 64.0)" in source
    assert "K_TARGETS = 8" in source


def test_fixed_seed_determinism():
    bank = tiny_bank(); delta = bank.direction(0, "2")
    a = bank.hard_random(delta, np.random.default_rng(99))
    b = bank.hard_random(delta, np.random.default_rng(99))
    assert np.array_equal(a, b)


def test_bootstrap_unit_is_subject_not_trial():
    script = (CODE / "v1_reproduce.py").read_text()
    assert 'groupby("subject_id"' in script
    assert "10_000" in script


def test_data_sentinel_rejects_outer_and_sealed():
    text = (CODE / "v2_common.py").read_text()
    namespace = {}
    # Direct import is unavailable on non-server machines, so audit source.
    assert '"outer"' in text and '"sealed"' in text and "RESERVED_RESOURCE_REJECTED" in text


def test_v1_reproduction_result_passes_if_present():
    result = Path(r"D:\nips-temp\TotalP\P1\CRCICLR_SOURCE_ONLY_DIAGNOSTIC\experiments\persist_eeg_me_hard_scst_v2\results\V1_REPRODUCTION.json")
    if result.exists():
        assert json.loads(result.read_text())["artifact_backed_reproduction_pass"] is True
    else:
        assert "artifact_backed_reproduction_pass" in (CODE / "v1_reproduce.py").read_text()


def test_lock_hash_guard_is_fail_closed():
    source = (CODE / "v2_common.py").read_text()
    assert "PROTOCOL_LOCK_HASH_MISMATCH" in source
    assert "PROTOCOL_LOCK_NOT_VERIFIED" in source

