import inspect
from hsc_tta.actions import T3A


def test_adaptation_signature_has_no_future_argument():
    assert "future" not in inspect.signature(T3A.adapt).parameters

