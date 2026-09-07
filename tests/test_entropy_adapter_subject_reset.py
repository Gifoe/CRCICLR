import pytest

torch = pytest.importorskip("torch")
from hsc_tta.actions import EntropyAdapter
from hsc_tta.models import TaskHead


def test_entropy_adapter_reset_hash():
    adapter = EntropyAdapter(TaskHead(200, 3), device="cpu", steps=1)
    before = adapter.initial_hash
    with torch.no_grad(): next(adapter.adapter.parameters()).add_(1)
    adapter.reset()
    assert adapter.parameter_hash() == before
