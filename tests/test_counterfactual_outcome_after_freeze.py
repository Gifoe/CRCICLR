import inspect
import pytest

pytest.importorskip("torch")
from hsc_tta.gpu.formal import evaluate_final_outcomes


def test_counterfactual_gate_precedes_future_read():
    source = inspect.getsource(evaluate_final_outcomes)
    assert source.index("verify_method_freeze") < source.index("read_segment")
