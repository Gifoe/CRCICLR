import numpy as np
import pytest

import pandas as pd

from hsc_tta.v3.episodes import EpisodeProtocol, build_v3_episodes, split_context, validate_three_way
from hsc_tta.v3.pseudo_episodes import assert_grouped_assignment, rolling_pseudo_episodes


def test_sleep_split_is_disjoint_chronological_and_preserves_future():
    context=np.arange(18); future=np.arange(18,42)
    adapt,probe,metadata=split_context(context,future,EpisodeProtocol(min_adapt=3,min_probe=3))
    validate_three_way(adapt,probe,future)
    assert np.array_equal(future,np.arange(18,42)) and metadata["split_unit"]=="window"


def test_mi_split_keeps_runs_whole():
    runs=np.repeat([4,6,8,10],5); context=np.arange(10); future=np.arange(10,20)
    adapt,probe,metadata=split_context(context,future,EpisodeProtocol(min_adapt=5,min_probe=5),run_ids=runs)
    assert metadata["adapt_runs"]==[4] and metadata["probe_runs"]==[6]
    assert not set(runs[adapt]) & set(runs[probe]); validate_three_way(adapt,probe,future)


def test_overlap_and_short_context_fail():
    with pytest.raises(ValueError,match="overlap"): validate_three_way(np.array([0,1]),np.array([1,2]),np.array([3]))
    with pytest.raises(ValueError,match="insufficient"): split_context(np.arange(5),np.arange(5,8),EpisodeProtocol(min_adapt=3,min_probe=3))


def test_pseudo_episodes_do_not_cross_subject_folds():
    episodes=rolling_pseudo_episodes("s",np.arange(30),n_adapt=5,n_probe=5,n_future=5,stride=5)
    assert len(episodes)==4 and all(not(set(x.adapt_indices)&set(x.future_indices)) for x in episodes)
    assignment={x.pseudo_id:0 for x in episodes}; assert_grouped_assignment(episodes,assignment)
    assignment[episodes[-1].pseudo_id]=1
    with pytest.raises(RuntimeError,match="split across"): assert_grouped_assignment(episodes,assignment)


def test_episode_protocol_validation_errors():
    with pytest.raises(ValueError, match="positive"): EpisodeProtocol(adapt_probe_ratio="0:1").fraction
    with pytest.raises(ValueError, match="nonempty"): validate_three_way(np.array([]), np.array([1]), np.array([2]))
    with pytest.raises(ValueError, match="temporal order"): validate_three_way(np.array([1,0]), np.array([2]), np.array([3]))
    with pytest.raises(ValueError, match="strictly chronological"): validate_three_way(np.array([2]), np.array([1]), np.array([3]))
    with pytest.raises(ValueError, match="at least two"): split_context(np.arange(5), np.arange(5,10), EpisodeProtocol(min_adapt=1,min_probe=1), run_ids=np.ones(10))


def test_build_v3_episode_artifacts(tmp_path):
    source = tmp_path / "data/episodes_main120/hmc"; source.mkdir(parents=True)
    frame = pd.DataFrame([{"subject_id":"hmc:s1", "split_role":"meta_risk_train", "episode_id":"old",
                           "context_indices":np.arange(12), "future_indices":np.arange(12,24)}])
    for seed in range(5): frame.to_parquet(source / f"seed_{seed}.parquet", index=False)
    result = build_v3_episodes(tmp_path, EpisodeProtocol(min_adapt=3,min_probe=3), ("hmc",))
    assert len(result) == 5 and (tmp_path / "outputs/v3_probecert/episodes/EPISODE_MANIFEST.parquet").exists()
    converted = pd.read_parquet(tmp_path / "data/episodes_v3/hmc/seed_0.parquet").iloc[0]
    assert np.array_equal(np.r_[converted.adapt_indices, converted.probe_indices], np.arange(12))
