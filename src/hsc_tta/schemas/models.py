from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Action = Literal["no_tta", "t3a", "entropy_adapter"]
Role = Literal[
    "task_head_train",
    "meta_risk_train",
    "conformal_calibration",
    "final_test",
    "target_site_calibration",
    "external_final_test",
]


class StrictRow(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class SubjectContextFeatureRow(StrictRow):
    dataset: str
    seed: int
    subject_id: str
    split_role: Role
    episode_id: str
    backbone: str
    n_context: int = Field(ge=1)
    embedding_mean: list[float]
    embedding_std: list[float]
    entropy_q10: float = Field(ge=0)
    entropy_q50: float = Field(ge=0)
    entropy_q90: float = Field(ge=0)
    max_probability_q10: float = Field(ge=0, le=1)
    max_probability_q50: float = Field(ge=0, le=1)
    max_probability_q90: float = Field(ge=0, le=1)
    class_proportions: list[float]
    signal_quality: list[float]
    channel_mask: list[bool]
    prediction_instability: float = Field(ge=0)


class ActionContextDiagnosticRow(StrictRow):
    dataset: str
    seed: int
    subject_id: str
    episode_id: str
    action: Action
    context_average_set_size_by_lambda: list[float]
    context_singleton_rate_by_lambda: list[float]
    entropy_before: float = Field(ge=0)
    entropy_after: float = Field(ge=0)
    prediction_kl: float = Field(ge=0)
    prototype_shift: float = Field(ge=0)
    adapter_update_norm: float = Field(ge=0)
    pseudo_label_balance: list[float]
    adaptation_status: str


class HistoricalActionOutcomeRow(StrictRow):
    dataset: str
    seed: int
    subject_id: str
    split_role: Literal["meta_risk_train", "conformal_calibration", "target_site_calibration"]
    episode_id: str
    action: Action
    lambda_: float = Field(alias="lambda", ge=0, le=1)
    lambda_index: int = Field(ge=0)
    alpha: float = Field(gt=0, lt=1)
    future_risk: float = Field(ge=0, le=1)
    critical_index: int = Field(ge=0)
    future_average_set_size: float = Field(ge=1)
    future_singleton_rate: float = Field(ge=0, le=1)
    argmax_error: float = Field(ge=0, le=1)
    macro_f1: float = Field(ge=0, le=1)
    balanced_accuracy: float = Field(ge=0, le=1)
    n_classes: int = Field(ge=2)


class CriticalIndexPredictionRow(StrictRow):
    dataset: str
    seed: int
    subject_id: str
    split_role: Role
    episode_id: str
    action: Action
    alpha: float = Field(gt=0, lt=1)
    predicted_critical_index: float = Field(ge=0)
    model_id: str
    model_hash: str


class CertifiedActionCandidateRow(StrictRow):
    dataset: str
    seed: int
    subject_id: str
    episode_id: str
    alpha: float = Field(gt=0, lt=1)
    action: Action
    predicted_critical_index: float = Field(ge=0)
    q_alpha: float = Field(ge=0)
    certified_critical_index: int = Field(ge=0)
    n_nontrivial_lambdas: int = Field(ge=1)
    selected_lambda: float = Field(ge=0, le=1)
    nontrivial_candidate: bool
    context_average_set_size: float = Field(ge=1)
    context_singleton_rate: float = Field(ge=0, le=1)
    adaptation_cost: int = Field(ge=0)
    n_classes: int = Field(ge=2)

    @model_validator(mode="after")
    def validate_nontrivial(self) -> "CertifiedActionCandidateRow":
        expected = (
            self.certified_critical_index < self.n_nontrivial_lambdas
            and self.selected_lambda < 1.0
            and self.context_average_set_size < self.n_classes
        )
        if self.nontrivial_candidate != expected:
            raise ValueError("nontrivial_candidate is inconsistent with the sentinel/full-set definition")
        return self


class PreOutcomeDecisionRow(StrictRow):
    dataset: str
    seed: int
    subject_id: str
    episode_id: str
    alpha: float = Field(gt=0, lt=1)
    selected_action: Action
    selected_lambda: float = Field(ge=0, le=1)
    certified_critical_index: int = Field(ge=0)
    nontrivial_certified: bool
    status: Literal["certified", "uncertified"]
    selection_reason: str
    freeze_hash: str


class FinalTestOutcomeRow(StrictRow):
    dataset: str
    seed: int
    subject_id: str
    episode_id: str
    alpha: float = Field(gt=0, lt=1)
    selected_action: Action
    selected_lambda: float = Field(ge=0, le=1)
    true_future_risk: float = Field(ge=0, le=1)
    future_average_set_size: float = Field(ge=1)
    future_singleton_rate: float = Field(ge=0, le=1)
    macro_f1: float = Field(ge=0, le=1)
    balanced_accuracy: float = Field(ge=0, le=1)
    no_tta_error: float = Field(ge=0, le=1)
    selected_error: float = Field(ge=0, le=1)
    harmful_adaptation: bool


class SubjectDecisionRow(PreOutcomeDecisionRow):
    true_future_risk: float = Field(ge=0, le=1)
    future_average_set_size: float = Field(ge=1)
    future_singleton_rate: float = Field(ge=0, le=1)
    macro_f1: float = Field(ge=0, le=1)
    balanced_accuracy: float = Field(ge=0, le=1)
    no_tta_error: float = Field(ge=0, le=1)
    selected_error: float = Field(ge=0, le=1)
    harmful_adaptation: bool


ContextFeatureRow = SubjectContextFeatureRow
