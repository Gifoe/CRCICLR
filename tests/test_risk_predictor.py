import numpy as np
import pandas as pd

from hsc_tta.risk_prediction import CriticalIndexPredictor, enforce_lambda_monotonicity


def test_alpha_specific_critical_index_predictor_round_trip(tmp_path):
    rows = []
    for subject in range(15):
        for action_index, action in enumerate(("no_tta", "t3a", "entropy_adapter")):
            rows.append(
                {
                    "dataset": "hmc",
                    "seed": 0,
                    "episode_id": f"e{subject}",
                    "subject_id": f"s{subject}",
                    "action": action,
                    "alpha": 0.2,
                    "x": subject / 15,
                    "action_feature": action_index,
                    "critical_index": 3 + action_index + subject % 4,
                }
            )
    frame = pd.DataFrame(rows)
    predictor = CriticalIndexPredictor(
        ["x", "action_feature"], alpha=0.2, n_nontrivial_lambdas=20
    ).fit(frame)
    prediction = predictor.predict(frame)
    assert np.all((prediction >= 0) & (prediction <= 20))
    path = tmp_path / "critical.joblib"
    predictor.save(path)
    loaded = CriticalIndexPredictor.load(path)
    assert np.allclose(prediction, loaded.predict(frame))
    assert predictor.model_id == loaded.model_id


def test_monotonicity_uses_complete_group_key():
    rows = []
    for dataset, seed, episode, alpha, values in (
        ("hmc", 0, "e0", 0.1, (0.4, 0.6)),
        ("cap", 1, "e1", 0.2, (0.9, 0.1)),
    ):
        for lam, value in zip((0.5, 0.7), values):
            rows.append({"dataset": dataset, "seed": seed, "episode_id": episode, "subject_id": "same", "action": "no_tta", "alpha": alpha, "lambda": lam, "future_risk": value})
    fixed = enforce_lambda_monotonicity(pd.DataFrame(rows))
    hmc = fixed[fixed.dataset == "hmc"].future_risk.tolist()
    assert hmc == [0.4, 0.4]
    assert fixed[fixed.dataset == "cap"].future_risk.tolist() == [0.9, 0.1]
