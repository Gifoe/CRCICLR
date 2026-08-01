import inspect
import pytest

pytest.importorskip("torch")
from hsc_tta.gpu.formal import evaluate_final_outcomes


def test_final_outcome_requires_decision_freeze():
    assert "require_decisions=True" not in inspect.getsource(evaluate_final_outcomes)
    assert "verify_method_freeze" in inspect.getsource(evaluate_final_outcomes)
