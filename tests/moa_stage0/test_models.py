import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hsc_tta.moa_stage0.models import Stage0Transformer, make_torch_operator
from hsc_tta.moa_stage0.operators import OperatorView


@pytest.mark.parametrize("method", ["B2", "B3", "B4", "B5", "B6", "B7", "B8"])
def test_all_baselines_share_shape_and_core(method):
    rng = np.random.default_rng(3)
    a = np.eye(8); b = rng.normal(size=(8, 32)); coefficients = np.eye(8)
    view = OperatorView("test", "dense_subset", "source", a, b, coefficients, tuple(str(i) for i in range(8)), "test", "train")
    coordinates = rng.normal(size=(8, 3)); coordinates /= np.linalg.norm(coordinates, axis=1, keepdims=True)
    centers = rng.normal(size=(32, 3)); centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    operator = make_torch_operator(view, coordinates, centers, 1e-2)
    model = Stage0Transformer(method, 4, hidden_dim=32, layers=1, heads=4, patch_size=16)
    output = model(torch.randn(2, 8, 64), operator)
    assert output.shape == (2, 4)
    assert isinstance(model.core, torch.nn.Module)
