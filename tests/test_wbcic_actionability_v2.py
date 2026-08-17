from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "experiments" / "persist_eeg_wbcic_actionability_v2" / "code"


def load_module(name: str, filename: str):
    path = CODE / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PROTOCOL = load_module("test_wbcic_protocol", "protocol.py")
CORE = load_module("core", "core.py")


def test_primary_subject_split_is_disjoint_exhaustive_and_stable() -> None:
    subjects = [f"sub-{index}" for index in range(1, 52)]
    development, outer, folds = PROTOCOL.deterministic_split(subjects)
    again = PROTOCOL.deterministic_split(subjects)
    assert (development, outer, folds) == again
    assert len(development) == 41
    assert len(outer) == 10
    assert sorted(len(fold) for fold in folds) == [8, 8, 8, 8, 9]
    assert not set(development).intersection(outer)
    assert set(development).union(outer) == set(subjects)
    assert len(set().union(*map(set, folds))) == 41
    assert all(
        not set(left).intersection(right)
        for index, left in enumerate(folds)
        for right in folds[index + 1 :]
    )


def test_development_runtime_does_not_open_sealed_outer_split() -> None:
    for filename in ("cache.py", "core.py", "pipeline.py"):
        source = (CODE / filename).read_text(encoding="utf-8")
        assert 'PROTOCOL / "OUTER_SPLIT_LOCK.json"' not in source
        assert "open(OUTER_SPLIT" not in source


def test_eegnet_forward_and_embedding_shape() -> None:
    model = CORE.EEGNet(dropout=0.5).eval()
    x = torch.zeros(3, 58, 1000)
    with torch.inference_mode():
        embedding = model.forward_features(x)
        logits = model(x)
    assert embedding.shape == (3, 32)
    assert logits.shape == (3, 2)
    assert torch.isfinite(embedding).all()
    assert torch.isfinite(logits).all()


def test_exact_matched_random_displacement_norm() -> None:
    rng = np.random.default_rng(11)
    residual = rng.normal(size=(37, 32))
    candidate, _ = np.linalg.qr(rng.normal(size=(32, 8)))
    random, _ = np.linalg.qr(rng.normal(size=(32, 8)))
    target = residual @ candidate[:, :8] @ candidate[:, :8].T
    matched = CORE.exact_matched_delta(residual, target, random[:, :8])
    np.testing.assert_allclose(
        np.linalg.norm(matched, axis=1), np.linalg.norm(target, axis=1), atol=5e-6, rtol=0
    )


def test_agdi_projection_has_protected_hard_guarantee() -> None:
    rng = np.random.default_rng(19)
    q, _ = np.linalg.qr(rng.normal(size=(32, 12)))
    protected = q[:, :4]
    harmful = q[:, 4:12]
    weight = rng.normal(size=(2, 32))
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        after = CORE.protected_projection(weight, harmful, alpha)
        error = CORE.protected_relative_error(weight, after, protected)
        assert error < 1e-12


def test_holm_adjustment_is_monotone() -> None:
    adjusted = CORE.holm({"a": 0.01, "b": 0.03, "c": 0.02, "d": 0.9})
    values = [adjusted[key] for key in ("a", "c", "b", "d")]
    assert values == sorted(values)
    assert all(0 <= value <= 1 for value in values)
