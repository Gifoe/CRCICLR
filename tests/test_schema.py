import pytest
from pydantic import ValidationError

from hsc_tta.schemas import CertifiedActionCandidateRow, PreOutcomeDecisionRow


def test_candidate_requires_consistent_full_set_definition():
    row = CertifiedActionCandidateRow.model_validate(
        {
            "dataset": "hmc", "seed": 0, "subject_id": "s", "episode_id": "e",
            "alpha": 0.2, "action": "no_tta", "predicted_critical_index": 3.2,
            "q_alpha": 1.1, "certified_critical_index": 5,
            "n_nontrivial_lambdas": 20, "selected_lambda": 0.63,
            "nontrivial_candidate": True, "context_average_set_size": 2.0,
            "context_singleton_rate": 0.4, "adaptation_cost": 0, "n_classes": 5,
        }
    )
    assert row.certified_critical_index == 5
    with pytest.raises(ValidationError):
        CertifiedActionCandidateRow.model_validate({**row.model_dump(by_alias=True), "n_classes": 2, "context_average_set_size": 2.0})


def test_pre_outcome_schema_forbids_future_fields():
    base = {
        "dataset": "hmc", "seed": 0, "subject_id": "s", "episode_id": "e",
        "alpha": 0.2, "selected_action": "no_tta", "selected_lambda": 0.8,
        "certified_critical_index": 5, "nontrivial_certified": True,
        "status": "certified", "selection_reason": "context utility", "freeze_hash": "abc",
    }
    PreOutcomeDecisionRow.model_validate(base)
    with pytest.raises(ValidationError):
        PreOutcomeDecisionRow.model_validate({**base, "future_risk": 0.1})
