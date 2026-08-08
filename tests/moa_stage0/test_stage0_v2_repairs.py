import numpy as np
import torch

from hsc_tta.moa_stage0.models import Stage0Transformer, make_torch_operator
from hsc_tta.moa_stage0.operators import OperatorView


def _view():
    c = np.eye(4) - np.ones((4, 4)) / 4
    psi = np.arange(8, dtype=float).reshape(4, 2) / 10.0
    a = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=float)
    return OperatorView("v2", "subset_car", "eegmmidb_car64", a, a @ c @ psi, a @ c,
                        ("A", "C"), "fixed pool", "test"), c, psi


def test_v2_observation_is_car_then_operator_and_matches_b():
    view, c, psi = _view()
    coordinates = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [-1., -1., -1.]])
    op = make_torch_operator(view, coordinates, np.zeros((2, 3)), 1e-2, c)
    raw = torch.randn(3, 4, 5)
    model = Stage0Transformer("B2", 2, canonical_dim=2, hidden_dim=16, layers=1, heads=4, patch_size=5)
    observed, _ = model._representation(raw, op)
    expected = torch.einsum("or,rc,bct->bot", torch.as_tensor(view.A, dtype=torch.float32), torch.as_tensor(c, dtype=torch.float32), raw) * 1e6
    assert torch.allclose(observed, expected)
    x = np.random.default_rng(3).normal(size=(2, 7))
    assert np.linalg.norm(view.A @ c @ psi @ x - view.B @ x) < 1e-10


def test_b4_features_are_continuous_not_family_one_hot():
    view, c, _ = _view()
    coordinates = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [-1., -1., -1.]])
    op = make_torch_operator(view, coordinates, np.zeros((2, 3)), 1e-2, c)
    model = Stage0Transformer("B4", 2, canonical_dim=2, hidden_dim=16, layers=1, heads=4, patch_size=5)
    _, features = model._representation(torch.zeros(1, 4, 5), op)
    assert features.shape == (2, 11)
    assert model.family_order == ()
