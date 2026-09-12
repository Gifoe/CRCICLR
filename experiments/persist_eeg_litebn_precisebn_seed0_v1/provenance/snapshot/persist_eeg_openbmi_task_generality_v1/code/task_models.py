"""Frozen task-carrier constructors.

The only task-dependent parameter is the classifier output dimension (and the
EEGNet embedding input implied by the already-frozen canonical epoch length).
"""
from __future__ import annotations

from task_datasets import TASKS, build_model, verify_model

MODEL_FAMILIES = ("EEGNet", "LiteBN")
FUSION = {"name": "FROZEN_LOGIT50", "EEGNet_weight": 0.5, "LiteBN_weight": 0.5, "trainable_parameters": 0}


def build_frozen_carrier(model: str, task: str):
    if model not in MODEL_FAMILIES or task not in TASKS:
        raise ValueError(f"invalid task/model: {task}/{model}")
    return build_model(model, task)


def architecture_audit() -> dict:
    return {task: verify_model(task) for task in TASKS}