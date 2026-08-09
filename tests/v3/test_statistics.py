import pandas as pd
import pytest

from hsc_tta.v3.statistics import average_seeds_by_subject, paired_subject_bootstrap, subject_cluster_bootstrap


def test_seed_average_precedes_subject_bootstrap():
    frame=pd.DataFrame({"dataset":["d"]*4,"subject_id":["a","a","b","b"],"seed":[0,1,0,1],"value":[1,3,2,4]})
    averaged=average_seeds_by_subject(frame,["value"])
    result=subject_cluster_bootstrap(averaged,"value",repetitions=100)
    assert result["n_subjects"]==2 and result["mean"]==2.5
    with pytest.raises(ValueError,match="average repeated"): subject_cluster_bootstrap(frame,"value")


def test_paired_bootstrap_uses_subject_pairs():
    frame=pd.DataFrame({"subject_id":["a","a","b","b"],"policy":["p","b","p","b"],"average_set_size":[1,2,2,3]})
    result=paired_subject_bootstrap(frame,"p","b",repetitions=100)
    assert result["mean_set_size_reduction"]==1 and result["n_subjects"]==2
