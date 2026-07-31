import pytest
from hsc_tta.preprocessing import map_mi_event


def test_run_dependent_event_mapping():
    assert map_mi_event(4,"T1") == 0 and map_mi_event(4,"T2") == 1
    assert map_mi_event(6,"T1") == 2 and map_mi_event(6,"T2") == 3
    assert map_mi_event(10,"T0") is None
    with pytest.raises(ValueError): map_mi_event(5,"T1")

