from pathlib import Path

import pandas as pd

from hsc_tta.v3.external_replication import merge_cap_parts


def test_merge_cap_parts_requires_and_preserves_all_seeds(tmp_path: Path) -> None:
    for seed in range(5):
        part = tmp_path / "outputs/v3_probecert/external_site/parts" / f"cap_seed_{seed}"
        part.mkdir(parents=True)
        pd.DataFrame([{
            "dataset": "cap", "seed": seed, "alpha": 0.1, "subject_id": f"cap:{seed}",
            "policy": "probecert_v3", "joint_violation": False, "average_set_size": 2.0,
            "singleton_rate": 0.2, "argmax_error": 0.1, "macro_f1": 0.8,
            "intervention": seed == 0, "sentinel": False,
        }]).to_parquet(part / "CAP_EXTERNAL_SUBJECT_RESULTS.parquet", index=False)
        pd.DataFrame([{
            "seed": seed, "alpha": 0.1, "subject_id": f"cap:{seed}",
            "joint_index": 2, "no_tta_joint_index": 3, "selected_action": "no_tta",
        }]).to_parquet(part / "CAP_CALIBRATION_JOINT_INDICES.parquet", index=False)

    merged = merge_cap_parts(tmp_path, list(range(5)))

    assert sorted(merged.seed.unique()) == list(range(5))
    assert len(merged) == 5
    summary = pd.read_csv(tmp_path / "outputs/v3_probecert/external_site/CAP_EXTERNAL_SUMMARY.csv")
    assert len(summary) == 5
