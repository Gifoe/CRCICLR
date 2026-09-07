from datetime import datetime, timezone
from hsc_tta.preprocessing import map_sleep_label
from hsc_tta.preprocessing.annotations import load_sleep_annotations


def test_hmc_and_cap_sleep_mapping():
    assert [map_sleep_label(x,"hmc") for x in ["W","N1","N2","N3","REM"]] == [0,1,2,3,4]
    assert map_sleep_label("S4","cap") == 3
    assert map_sleep_label("Movement","cap") is None


def test_cap_sidecar_clock_alignment(tmp_path):
    signal=tmp_path/"record.edf"; signal.touch()
    (tmp_path/"record.txt").write_text("header\nSleep Stage\tPosition\tTime [hh:mm:ss]\tEvent\tDuration[s]\tLocation\nW\tx\t23.59.30\tSLEEP-S0\t30\tx\nS1\tx\t00.00.00\tSLEEP-S1\t30\tx\n",encoding="latin-1")
    start=datetime(2020,1,1,23,59,0,tzinfo=timezone.utc)
    rows=load_sleep_annotations(signal,"cap",start)
    assert [r["onset"] for r in rows] == [30.0,60.0]


def test_cap_sidecar_accepts_colon_clock_format(tmp_path):
    signal = tmp_path / "record.edf"
    signal.touch()
    (tmp_path / "record.txt").write_text(
        "header\n"
        "Sleep Stage\tPosition\tTime [hh:mm:ss]\tEvent\tDuration[s]\tLocation\n"
        "W\tx\t23:59:30\tSLEEP-S0\t30\tx\n"
        "S1\tx\t00:00:00\tSLEEP-S1\t30\tx\n",
        encoding="latin-1",
    )
    start = datetime(2020, 1, 1, 23, 59, 0, tzinfo=timezone.utc)
    rows = load_sleep_annotations(signal, "cap", start)
    assert [r["onset"] for r in rows] == [30.0, 60.0]
