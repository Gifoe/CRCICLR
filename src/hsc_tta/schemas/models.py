from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class StrictRow(BaseModel):
    model_config = ConfigDict(extra="allow")


class ContextFeatureRow(StrictRow):
    dataset: str; seed: int; subject_id: str; split_role: str; backbone: str; episode_id: str
    n_context: int = Field(ge=1)


class ActionSurfaceRow(StrictRow):
    dataset: str; seed: int; subject_id: str; split_role: str; episode_id: str
    action: Literal["no_tta", "t3a", "entropy_adapter"]
    lambda_: float = Field(alias="lambda", ge=0, le=1)
    predicted_risk: float = Field(ge=0, le=1)
    within_subject_empirical_risk: float = Field(ge=0, le=1)
    within_subject_margin: float = Field(ge=0)
    within_subject_upper_risk: float = Field(ge=0, le=1)
    certified_upper_bound: float = Field(ge=0, le=1)
    future_risk: float = Field(ge=0, le=1)
    argmax_error: float = Field(ge=0, le=1)
    macro_f1: float = Field(ge=0, le=1)
    balanced_accuracy: float = Field(ge=0, le=1)
    average_set_size: float = Field(ge=1)
    singleton_rate: float = Field(ge=0, le=1)
    n_context: int = Field(ge=1); n_future: int = Field(ge=1); n_future_blocks: int = Field(ge=0); status: str


class SubjectDecisionRow(StrictRow):
    dataset: str; seed: int; subject_id: str; alpha: float
    selected_action: str | None; selected_lambda: float | None
    predicted_risk: float; certified_upper_bound: float; true_future_risk: float
    certified: bool; nontrivial_certified: bool; average_set_size: float; singleton_rate: float
    no_tta_error: float; selected_error: float; harmful_adaptation: bool; status: str; selection_reason: str

