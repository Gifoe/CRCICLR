from hsc_tta.v2.simulation import simulate_joint


def test_simulation_joint_validity_and_u_only_selection():
    result=simulate_joint(repetitions=3000,calibration_size=50,action_count=3,scenario="one_beneficial",seed=9)
    assert result["joint_simultaneous_validity"]>=.88
    adversarial=simulate_joint(repetitions=3000,calibration_size=50,action_count=5,scenario="one_beneficial",selector="adversarial_u_only",seed=10)
    assert adversarial["joint_simultaneous_validity"]>=.88


def test_all_harmful_rarely_selects_tta():
    result=simulate_joint(repetitions=2000,calibration_size=50,action_count=3,scenario="all_harmful",seed=11)
    assert result["tta_selection_rate"]<.1
