import pytest
from pydantic import ValidationError

from hsc_tta.schemas import SubjectContextFeatureRow


def test_context_schema_forbids_future_features():
    row = {
        "dataset": "hmc", "seed": 0, "subject_id": "s", "split_role": "final_test",
        "episode_id": "e", "backbone": "mock", "n_context": 180,
        "embedding_mean": [0.0], "embedding_std": [1.0],
        "entropy_q10": 0.1, "entropy_q50": 0.2, "entropy_q90": 0.3,
        "max_probability_q10": 0.5, "max_probability_q50": 0.6, "max_probability_q90": 0.7,
        "class_proportions": [0.2] * 5, "signal_quality": [1.0], "channel_mask": [True],
        "prediction_instability": 0.1,
    }
    SubjectContextFeatureRow.model_validate(row)
    with pytest.raises(ValidationError):
        SubjectContextFeatureRow.model_validate({**row, "future_entropy": 0.1})
