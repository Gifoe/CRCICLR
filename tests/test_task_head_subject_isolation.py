import pytest

torch = pytest.importorskip("torch")
from hsc_tta.models import TaskHead


def test_task_head_exposes_hidden():
    logits, hidden = TaskHead(200, 5)(torch.zeros(3, 200), return_hidden=True)
    assert logits.shape == (3, 5) and hidden.shape == (3, 256)
