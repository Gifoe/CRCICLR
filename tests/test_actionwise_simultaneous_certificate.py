import numpy as np
import pandas as pd

from hsc_tta.certification import apply_critical_index_certificate, fit_actionwise_simultaneous_quantile


def _calibration():
    rows = []
    for subject in range(20):
        for action, residual in zip(("no_tta", "t3a", "entropy_adapter"), (0.2, 0.8, 1.2)):
            prediction = 5.0 + subject % 2
            rows.append({"dataset": "hmc", "seed": 0, "episode_id": f"e{subject}", "subject_id": f"s{subject}", "alpha": 0.2, "action": action, "critical_index": prediction + residual, "predicted_critical_index": prediction, "lambda": 0.5 + subject / 100})
    return pd.DataFrame(rows)


def test_score_maximizes_over_three_actions_not_lambda():
    quantile = fit_actionwise_simultaneous_quantile(_calibration(), delta=0.1, n_nontrivial_lambdas=20)
    assert np.isclose(quantile.raw_quantile, 1.2)
    assert quantile.n_calibration_subjects == 20
    certified = apply_critical_index_certificate([5.1, 19.8], quantile)
    assert certified.tolist() == [7, 20]


def test_post_selection_valid_when_all_action_indices_are_covered():
    truth = np.array([[4, 6, 7], [10, 8, 9]])
    bound = np.array([[5, 6, 8], [11, 9, 10]])
    selected = np.array([1, 2])
    assert np.all(truth[np.arange(2), selected] <= bound[np.arange(2), selected])
