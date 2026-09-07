import pytest
from hsc_tta.selection.core import _reject_future_columns


def test_gpu_context_schema_rejects_future():
    with pytest.raises(ValueError): _reject_future_columns(["future_risk"])
