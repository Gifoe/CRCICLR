from pathlib import Path

import pandas as pd

from hsc_tta.moa_stage0.pipeline import Stage0Pipeline


def test_gate_a_uses_preregistered_at_least_one_operator_rule(tmp_path: Path):
    pipeline = Stage0Pipeline.__new__(Stage0Pipeline)
    pipeline.output = tmp_path
    pipeline.config = {"gate": {"A_drop_pp": 2.0, "A_rei": 0.08}}
    pd.DataFrame([
        {"method": "B4", "evaluation": "unseen", "operator_id": "weak", "operator_family": "sparse", "operator_drop": 0.0, "rei": 0.0, "balanced_accuracy": .30},
        {"method": "B4", "evaluation": "unseen", "operator_id": "shift", "operator_family": "polarity", "operator_drop": .025, "rei": .04, "balanced_accuracy": .275},
    ]).to_csv(tmp_path / "matched_unseen_all.csv", index=False)
    aggregate = pd.DataFrame([{"method": "B4", "matched_ba": .30, "unseen_ba": .2875, "operator_drop": .0125}])
    result = pipeline._gate_a(aggregate)
    assert result["status"] == "PASS"
    assert result["trigger_operator"] == "shift"
