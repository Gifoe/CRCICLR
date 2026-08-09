from hsc_tta.v3.simulation import run_simulations


def test_exchangeable_simulation_is_reproducible_and_small_calibration_is_sentinel():
    config={"repetitions":5000,"calibration_sizes":[10],"deltas":[.05],"action_counts":[1],
            "probe_future_correlations":[0.0],"harm_rate_sensitivity":[.15],"site_shift_sensitivity":[0.0],"seed":9}
    first=run_simulations(config); second=run_simulations(config)
    assert first.equals(second)
    row=first[first.scenario=="exchangeable_grid"].iloc[0]
    assert row.calibration_insufficient_rate==1 and row.sentinel_probability==1 and row.joint_validity==1
