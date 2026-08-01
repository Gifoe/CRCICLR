import pandas as pd

from hsc_tta.risk_prediction import subject_group_ids


def test_dataset_seed_episode_subject_keys_never_collide():
    frame = pd.DataFrame(
        [
            {"dataset": "hmc", "seed": 0, "episode_id": "e0", "subject_id": "same"},
            {"dataset": "cap", "seed": 0, "episode_id": "e0", "subject_id": "same"},
            {"dataset": "hmc", "seed": 1, "episode_id": "e0", "subject_id": "same"},
            {"dataset": "hmc", "seed": 0, "episode_id": "e1", "subject_id": "same"},
        ]
    )
    assert len(set(subject_group_ids(frame))) == 4
