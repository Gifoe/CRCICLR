import numpy as np
import pytest
from hsc_tta.episodes import build_sleep_episode, build_mi_episode, validate_episode


def test_clock_time_sleep_episode_and_mi_runs():
    starts=np.arange(400)*30.; valid=np.ones(400,dtype=bool)
    e=build_sleep_episode(starts,valid,90,100)
    assert e["n_context"] == 180 and min(e["future_indices"]) == 180
    mi=build_mi_episode(np.repeat([4,6,8,10,12,14],5))
    assert mi["n_context"]==10 and mi["n_future"]==20


def test_overlap_rejected():
    with pytest.raises(ValueError): validate_episode([1,2],[2,3])

