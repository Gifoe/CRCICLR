import inspect
from hsc_tta.actions import T3A


def test_t3a_adapt_has_no_future_argument():
    assert all("future" not in name for name in inspect.signature(T3A.adapt).parameters)
