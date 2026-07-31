from hsc_tta.preprocessing import map_sleep_label


def test_hmc_and_cap_sleep_mapping():
    assert [map_sleep_label(x,"hmc") for x in ["W","N1","N2","N3","REM"]] == [0,1,2,3,4]
    assert map_sleep_label("S4","cap") == 3
    assert map_sleep_label("Movement","cap") is None

