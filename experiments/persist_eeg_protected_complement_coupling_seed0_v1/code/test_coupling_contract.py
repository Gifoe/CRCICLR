"""Synthetic algebra, sensitivity, and TRAIN-only surrogate contract checks."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

path = Path(__file__).with_name("run_coupling.py")
spec = importlib.util.spec_from_file_location("coupling_contract", path)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class SyntheticRunner:
    device = torch.device("cpu")

    def __init__(self, coupled: bool):
        self.coupled = coupled

    def from_stage(self, _stage: str, activation: torch.Tensor):
        flat = activation.reshape(len(activation), -1)
        first = flat[:, 0] + 0.2 * flat[:, 1]
        if self.coupled:
            first = first + flat[:, 0] * flat[:, 1]
        logits = torch.stack((first, -first), dim=1)
        return flat, logits


a = np.asarray([[1., 2.], [2., 3.]], np.float32)
y = np.asarray([0, 1])
subjects = np.asarray(["1", "1"])
sessions = np.asarray([1, 1])
q = np.asarray([[1.], [0.]], np.float32)
qc = np.asarray([[0.], [1.]], np.float32)
mu = np.zeros(2, np.float32)

for coupled in (False, True):
    runner = SyntheticRunner(coupled)
    with torch.inference_mode():
        _, tensor = runner.from_stage("fake", torch.from_numpy(a))
    logits = tensor.numpy()
    rows = module.functional_pairs(runner, "fake", [2], a, logits, y, subjects,
                                   [(0, 1)], q, mu, qc, sessions)
    assert len(rows) == 1
    if coupled:
        assert rows[0]["interaction_norm"] > 0.1
        assert rows[0]["D_JP"] > 0.1
        assert rows[0]["D_JC"] > 0.1
    else:
        assert rows[0]["interaction_norm"] < 1e-6
        assert rows[0]["D_JP"] < 1e-6
        assert rows[0]["D_JC"] < 1e-6

rng = np.random.default_rng(15)
n = 180
p = rng.normal(size=(n, 2)).astype(np.float32)
c = rng.normal(size=(n, 4)).astype(np.float32)
target = np.stack((p[:, 0] * c[:, 0] + p[:, 1] + c[:, 2],
                   -p[:, 0] * c[:, 0] - p[:, 1] - c[:, 2]), axis=1)
subjects = np.repeat(np.arange(18).astype(str), 10)
rank = module.select_bilinear_rank(p, c, target, subjects)
assert rank in (1, 2, 4, 8)
additive = module.Ridge(alpha=1.).fit(np.column_stack((p[:140], c[:140])), target[:140])
bilinear, factors = module.fit_bilinear(p[:140], c[:140], target[:140], rank)
add_loss = np.mean((additive.predict(np.column_stack((p[140:], c[140:]))) - target[140:]) ** 2)
bil_loss = np.mean((bilinear.predict(module.surrogate_design(p[140:], c[140:], "bilinear", factors)) - target[140:]) ** 2)
assert bil_loss < add_loss

raw = rng.normal(size=(50, 24)).astype(np.float32)
qraw = np.linalg.qr(rng.normal(size=(24, 3)))[0].astype(np.float32)
mu = raw.mean(axis=0).astype(np.float32)
context = module.GramPCA(raw, raw[:7], mu, torch.device("cpu"))
_, gram_coordinates, _, _ = context.coordinates(qraw, None, 4)
residual = raw - mu - ((raw - mu) @ qraw) @ qraw.T
reference = PCA(n_components=4, svd_solver="full").fit_transform(residual)
for column in range(4):
    assert abs(np.corrcoef(gram_coordinates[:, column], reference[:, column])[0, 1]) > 0.999

left = rng.normal(size=(24, 3)).astype(np.float32)
right = rng.normal(size=(3, 24)).astype(np.float32) * 0.04
_, oblique_coordinates, _, _ = context.coordinates(qraw, (left, right), 4)
oblique_residual = raw - mu - ((raw - mu) @ left) @ right
oblique_reference = PCA(n_components=4, svd_solver="full").fit_transform(oblique_residual)
for column in range(4):
    assert abs(np.corrcoef(oblique_coordinates[:, column], oblique_reference[:, column])[0, 1]) > 0.999

print("ALL_COUPLING_CONTRACT_TESTS_PASSED")
