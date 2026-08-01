import pytest
from hsc_tta.splits import make_subject_split, validate_subject_split


def test_deterministic_subject_disjoint_split():
    ids=[f"hmc:{i:03d}" for i in range(151)]
    a=make_subject_split(ids,"hmc",0); b=make_subject_split(ids,"hmc",0)
    assert a == b and [len(a[k]) for k in a] == [70,35,20,26]
    validate_subject_split(a,ids)


def test_leakage_is_rejected():
    with pytest.raises(ValueError): validate_subject_split({"train":["s1"],"test":["s1"]})


def test_cap_calibration_is_pathology_stratified():
    groups = {"brux": 2, "ins": 9, "n": 16, "narco": 5, "nfle": 35, "plm": 10, "rbd": 22, "sdb": 4}
    subjects = [f"cap:{name}{index}" for name, count in groups.items() for index in range(1, count + 1)]
    split = make_subject_split(subjects, "cap", seed=3)
    calibration = split["target_site_calibration"]
    represented = {''.join(character for character in sid.split(':')[-1] if not character.isdigit()) for sid in calibration}
    assert len(calibration) == 25
    assert represented == set(groups)
