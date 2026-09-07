from hsc_tta.simulation import run_simulations


def test_critical_index_simulations_write_all_families(tmp_path):
    outputs = run_simulations(tmp_path, seed=3, repetitions=8, n_test=10)
    assert set(outputs) == {
        "simulation_summary",
        "simulation_a_subject_vs_window",
        "simulation_b_to_f_repetitions",
        "simulation_e_misspecification",
        "simulation_g_adversarial_selection",
    }
    assert outputs["simulation_b_to_f_repetitions"].csr_nonfull.mean() > 0
