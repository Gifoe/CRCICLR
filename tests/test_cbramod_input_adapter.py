import numpy as np
import pytest

torch = pytest.importorskip("torch")
from hsc_tta.backbones import CBraModInputAdapter


def test_sleep_c4_not_duplicated_and_mi_shape():
    adapter = CBraModInputAdapter()
    sleep = adapter.adapt("hmc", np.zeros((2, 2, 6000), np.float32), ["EEG C3-M2", "EEG C4-M1"], 200)
    assert sleep.tensor.shape == (2, 1, 30, 200)
    mi = adapter.adapt("eegmmidb", np.zeros((1, 64, 640), np.float32), [f"c{x}" for x in range(64)], 160)
    assert mi.tensor.shape == (1, 64, 4, 200)
