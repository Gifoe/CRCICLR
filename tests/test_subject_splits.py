import pytest
from hsc_tta.splits import make_subject_split, validate_subject_split


def test_deterministic_subject_disjoint_split():
    ids=[f"hmc:{i:03d}" for i in range(151)]
    a=make_subject_split(ids,"hmc",0); b=make_subject_split(ids,"hmc",0)
    assert a == b and [len(a[k]) for k in a] == [70,35,20,26]
    validate_subject_split(a,ids)


def test_leakage_is_rejected():
    with pytest.raises(ValueError): validate_subject_split({"train":["s1"],"test":["s1"]})

