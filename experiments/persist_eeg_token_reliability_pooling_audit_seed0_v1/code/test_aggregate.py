import numpy as np

from aggregate import C_GRID, paired_subject_bootstrap, pool, rank_subset, subject_metrics


def test_pool_shape_and_baselines():
    z = np.arange(2 * 8 * 3, dtype=np.float32).reshape(2, 8, 3)
    assert pool(z, "MEAN_ALL").shape == (2, 3)
    assert pool(z, "FLATTEN_LINEAR").shape == (2, 24)
    subset = rank_subset(np.arange(8, dtype=float), .25)
    assert subset.tolist() == [7, 6]
    assert pool(z, "RELIABLE_UTILITY_TOPK", subset=subset).shape == (2, 3)
    assert pool(z, "RU_TWO_STREAM", subset=subset).shape == (2, 6)
    assert pool(z, "RU_WEIGHTED", score=np.arange(8, dtype=float), tau=.25).shape == (2, 3)
    assert C_GRID == (.001, .01, .1, 1., 10.)


def test_subject_resampling_unit():
    result = paired_subject_bootstrap({"s1": .8, "s2": .6}, {"s1": .7, "s2": .5}, draws=20_000)
    assert result["subjects"] == 2
    assert abs(result["mean_difference"] - .1) < 1e-12
    y = np.array([0, 1, 0, 1])
    probability = np.array([[.8, .2], [.2, .8], [.7, .3], [.4, .6]])
    rows = subject_metrics(y, probability, np.array(["s1", "s1", "s2", "s2"]))
    assert len(rows) == 2 and all(row["BA"] == 1 for row in rows)
