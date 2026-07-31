from hsc_tta.simulation import run_simulations


def test_end_to_end_mock_outputs_are_computed(tmp_path):
    out=run_simulations(tmp_path,seed=3,n_subjects=40)
    assert set(out)=={"simulation_summary","certificate_coverage","pseudo_sample_size","post_selection_validity","safety_utility","risk_predictor_misspecification"}
    assert all((tmp_path/f"{name}.csv").exists() for name in out)

