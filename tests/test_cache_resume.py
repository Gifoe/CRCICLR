import numpy as np
import h5py
import pytest
from hsc_tta.preprocessing.storage import write_subject_hdf5


def test_atomic_cache_resume_and_hash(tmp_path):
    p=tmp_path/"s.h5"; arrays={"signal":np.zeros((2,10)),"sampling_rate":np.asarray(200.0)}
    assert write_subject_hdf5(p,arrays,{},"abc") == "written"
    assert write_subject_hdf5(p,arrays,{},"abc") == "resumed"
    assert write_subject_hdf5(p,arrays,{},"def") == "rewritten_config_changed"
    with h5py.File(p,"r") as handle:
        assert handle.attrs["preprocessing_config_hash"] == "def"
