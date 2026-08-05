import numpy as np
import pandas as pd
import pytest

from hsc_tta.online_blockwise.core import (
    BlockProtocol,
    ScientificStop,
    TechnicalBlock,
    build_canonical_blocks,
    causal_raw_indices,
    certified_policy_indices,
    fold_roles,
    higher_quantile,
    inclusion_indices,
    maximum_true_run,
    role_for_screening_fold,
    rolling_weighted_risk,
    smallest_correction,
    subject_conformal_correction,
    tps_sets,
)


def test_fold_rotation_is_2_2_1_partition():
    for fold in range(5):
        roles = fold_roles(fold)
        assert len(roles["development"]) == 2
        assert len(roles["calibration"]) == 2
        assert roles["evaluation"] == {fold}
        assert set.union(*roles.values()) == set(range(5))
        assert [role_for_screening_fold(value, fold) for value in range(5)].count("evaluation") == 1


def test_higher_quantile_and_max_flag():
    assert higher_quantile(range(9), .9) == (8.0, 9, True)
    assert higher_quantile(range(10), .9) == (8.0, 9, False)


def test_conformal_full_fallback_and_minimal_subject_correction():
    assert subject_conformal_correction([0, 1, 2], .1, 20) == (20, 3, 4, True)
    kappa = np.array([0] * 8 + [4, 5])
    assert smallest_correction(kappa, 3, .1, 20) == 1


def test_tps_nested_full_and_inclusion_indices():
    p = np.array([[.7, .2, .1], [.2, .3, .5]])
    sets = tps_sets(p)
    assert np.all(sets[:, 1:] | ~sets[:, :-1])
    assert sets[:, -1].all()
    kappa = inclusion_indices(sets, np.array([2, 0]))
    assert np.all((kappa >= 0) & (kappa <= sets.shape[1] - 1))


def _metadata(dataset="hmc"):
    n = 150 if dataset == "hmc" else 24
    run = np.zeros(n, dtype=int) if dataset == "hmc" else np.repeat([4, 6, 8], 8)
    return pd.DataFrame({
        "dataset": dataset, "subject_id": f"{dataset}:001", "screening_fold": 0,
        "recording_id": "rec", "run_id": run, "chronological_index": np.arange(n),
        "sample_id": [f"s{i}" for i in range(n)], "window_start": np.arange(n),
        "window_end": np.arange(n) + 1,
    })


def test_hmc_blocks_contiguous_nonoverlap_and_tail_rule():
    blocks, mapping = build_canonical_blocks(_metadata("hmc"), BlockProtocol())
    assert blocks[blocks.retained].number_of_valid_samples.tolist() == [60, 60, 30]
    assert mapping.sample_id.nunique() == 150
    assert mapping.groupby("sample_id").size().max() == 1


def test_eeg_blocks_preserve_runs_without_seed_duplication():
    blocks, mapping = build_canonical_blocks(_metadata("eegmmidb"), BlockProtocol())
    assert len(blocks[blocks.retained]) == 3
    assert "source_seed" not in blocks.columns
    assert mapping.groupby("block_id").size().tolist() == [8, 8, 8]


def test_causal_prefix_does_not_use_current_block():
    blocks = [np.array([0, 0]), np.array([20, 20]), np.array([0, 0])]
    raw = causal_raw_indices("EXPANDING_PREFIX_QUANTILE", blocks, raw_global=3, K=20)
    assert raw.tolist() == [3, 0, 20]


def test_sliding_and_two_timescale_are_history_only():
    original = [np.array([0, 1]), np.array([4, 5]), np.array([2, 3])]
    changed = [original[0], np.array([19, 20]), original[2]]
    for method, parameter in [("SLIDING_WINDOW_QUANTILE", 1), ("TWO_TIMESCALE_FIXED", .5)]:
        left = causal_raw_indices(method, original, 2, 20, parameter)
        right = causal_raw_indices(method, changed, 2, 20, parameter)
        assert left[1] == right[1]


def test_risk_feedback_direction_and_delay():
    blocks = [np.array([20] * 10), np.array([0] * 10)]
    raw = causal_raw_indices("RISK_FEEDBACK_INDEX_CONTROL", blocks, 2, 20, 4.0)
    assert raw[0] == 2 and raw[1] >= raw[0]


def test_first_block_certified_global_fallback():
    result = certified_policy_indices(np.array([1, 2, 3]), correction=2, certified_global=9, K=20)
    assert result.tolist() == [9, 4, 5]


def test_rolling_risk_is_sample_weighted():
    assert rolling_weighted_risk(np.array([1, 9]), np.array([10, 90]), 2).tolist() == pytest.approx([.10])


def test_consecutive_high_risk_run_and_stop_types():
    assert maximum_true_run([False, True, True, False, True]) == 2
    assert not issubclass(ScientificStop, TechnicalBlock)
