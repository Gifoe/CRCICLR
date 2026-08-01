import pytest

torch = pytest.importorskip("torch")
from hsc_tta.backbones import module_sha256


def test_parameter_hash_changes_on_mutation():
    module = torch.nn.Linear(2, 2)
    before = module_sha256(module)
    with torch.no_grad(): module.weight.add_(1)
    assert module_sha256(module) != before
