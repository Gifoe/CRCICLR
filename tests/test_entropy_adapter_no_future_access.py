import inspect
import pytest

pytest.importorskip("torch")
from hsc_tta.actions import EntropyAdapter


def test_entropy_adapter_adapt_has_no_future_argument():
    assert all("future" not in name for name in inspect.signature(EntropyAdapter.adapt).parameters)
