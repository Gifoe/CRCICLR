import pytest

torch = pytest.importorskip("torch")
from hsc_tta.actions import EntropyAdapter
from hsc_tta.models import TaskHead


def test_only_adapter_is_optimized():
    head = TaskHead(200, 3)
    action = EntropyAdapter(head, device="cpu", steps=1)
    action.adapt(torch.randn(4, 200).numpy())
    assert not any(parameter.requires_grad for parameter in head.parameters())
