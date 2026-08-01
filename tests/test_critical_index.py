import numpy as np
import pandas as pd
import pytest

from hsc_tta.certification import critical_index_from_curve, critical_index_table


def test_nested_risk_curve_and_sentinel_critical_index():
    grid = [0.5, 0.7, 0.9, 1.0]
    assert critical_index_from_curve([0.4, 0.19, 0.08, 0.0], grid, 0.2) == 1
    assert critical_index_from_curve([0.4, 0.3, 0.25, 0.0], grid, 0.2) == 3
    with pytest.raises(ValueError, match="nonincreasing"):
        critical_index_from_curve([0.2, 0.3, 0.1, 0.0], grid, 0.2)


def test_critical_index_table_preserves_complete_keys():
    rows = []
    for dataset, seed, episode, alpha, risks in (
        ("hmc", 0, "e0", 0.1, [0.2, 0.05, 0.0]),
        ("cap", 1, "e1", 0.2, [0.3, 0.1, 0.0]),
    ):
        for index, (lam, risk) in enumerate(zip([0.5, 0.8, 1.0], risks)):
            rows.append({"dataset": dataset, "seed": seed, "episode_id": episode, "subject_id": "same", "action": "no_tta", "alpha": alpha, "lambda": lam, "lambda_index": index, "future_risk": risk})
    result = critical_index_table(pd.DataFrame(rows))
    assert result[["dataset", "seed", "episode_id", "alpha"]].drop_duplicates().shape[0] == 2
    assert result.critical_index.tolist() == [1, 1]
