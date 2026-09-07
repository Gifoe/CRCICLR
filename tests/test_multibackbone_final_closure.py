from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "experiments" / "persist_eeg_multibackbone_final_closure" / "code"
sys.path.insert(0, str(CODE))

import common  # noqa: E402
import freeze_protocol  # noqa: E402
import models  # noqa: E402


def test_roster_and_budget_are_frozen_before_outcomes():
    roster = json.loads((common.PROTOCOL / "BACKBONE_ROSTER_LOCK.json").read_text(encoding="utf-8"))
    assert roster["exact_roster"] == ["EEGNet", "FBCNet", "EEGConformer", "DeepConvNet", "TeCh"]
    assert roster["no_sixth_backbone"] is True
    assert all(len(value) <= 6 for value in freeze_protocol.CONFIGS.values())
    assert roster["conditional_replication_seeds"] == [20260823, 20260829]


def test_all_models_have_native_32d_linear_head_and_expected_shape():
    x = torch.zeros(1, 58, 1000)
    for backbone, configs in freeze_protocol.CONFIGS.items():
        model = models.build_model(backbone, configs[0]).eval()
        with torch.inference_mode():
            h = model.forward_features(x)
            z = model(x)
        assert h.shape == (1, model.representation_dim)
        assert h.shape[1] >= 32
        assert z.shape == (1, 2)
        assert isinstance(model.head, torch.nn.Linear)
        assert model.head.in_features == model.representation_dim


def test_tech_eval_is_bitwise_deterministic():
    config = freeze_protocol.CONFIGS["TeCh"][1]
    model = models.build_model("TeCh", config).eval()
    x = torch.randn(2, 58, 1000)
    with torch.inference_mode():
        first_h, first_z = model.forward_features(x), model(x)
        second_h, second_z = model.forward_features(x), model(x)
    assert torch.equal(first_h, second_h)
    assert torch.equal(first_z, second_z)


def test_fold_roles_are_subject_disjoint_and_outer_ids_absent():
    scope = json.loads(common.SCOPE_PATH.read_text(encoding="utf-8"))
    assert len(scope["allowed_subjects"]) == 41
    assert scope["outer_subject_ids_present"] is False
    for fold in range(5):
        outcome, discovery, model_fit = common.audit_roles(scope, fold)
        assert not set(outcome) & set(discovery)
        assert not set(outcome) & set(model_fit)
        assert not set(discovery) & set(model_fit)
        assert set(outcome) | set(discovery) | set(model_fit) == set(scope["allowed_subjects"])


def test_same_rank_random_control_and_matched_norm_dynamic_dimension():
    dim, rank = 64, 8
    basis = common.random_bases(dim, rank, "TeCh", 0, "P09_16")[0]
    assert basis.shape == (dim, rank)
    assert np.linalg.norm(basis.T @ basis - np.eye(rank)) < 1e-10
    rng = np.random.default_rng(0)
    residual = rng.normal(size=(11, dim))
    target_basis, _ = np.linalg.qr(rng.normal(size=(dim, rank)))
    target = common.project_rows(residual, target_basis[:, :rank])
    matched = common.exact_matched_delta(residual, target, basis)
    np.testing.assert_allclose(np.linalg.norm(matched, axis=1), np.linalg.norm(target, axis=1), atol=5e-6)


def test_basis_uses_subject_session_centroids_and_is_orthonormal():
    rng = np.random.default_rng(1)
    subjects, dim, trials = 8, 40, 5
    h, sid, ses = [], [], []
    for subject in range(subjects):
        identity = rng.normal(size=dim)
        for session in (0, 1):
            h.append(identity + rng.normal(scale=.1, size=(trials, dim)))
            sid.extend([subject] * trials)
            ses.extend([session] * trials)
    arrays = {
        "subjects": np.asarray([f"s{index}" for index in range(subjects)]),
        "embeddings": np.concatenate(h),
        "subject_index": np.asarray(sid),
        "session": np.asarray(ses),
    }
    basis, _, _, _, diag = common.discovered_basis(arrays)
    assert basis.shape == (dim, 32)
    assert np.linalg.norm(basis.T @ basis - np.eye(32)) < 1e-8
    assert diag["effective_positive_rank"] > 0


def test_agdi_algebra_preserves_protected_and_suppresses_harmful():
    rng = np.random.default_rng(2)
    q, _ = np.linalg.qr(rng.normal(size=(64, 12)))
    protected, harmful = q[:, :4], np.column_stack((q[:, :4], q[:, 4:12]))
    residual, diagnostic = common.residual_harmful(protected, harmful)
    assert diagnostic["residual_harmful_rank"] == 8
    weight = rng.normal(size=(2, 64))
    after = common.agdi_projection(weight, residual, 1.0)
    np.testing.assert_allclose(after @ residual, 0, atol=1e-10)
    np.testing.assert_allclose(after @ protected, weight @ protected, atol=1e-10)


def test_identity_fixed_point_without_actionable_span():
    weight = np.arange(64, dtype=float).reshape(2, 32)
    after = common.agdi_projection(weight, np.empty((32, 0)), 0.0)
    np.testing.assert_array_equal(after, weight)
