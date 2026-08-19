from __future__ import annotations

import numpy as np
import pandas as pd


def build_legal_residual_features(*args, **kwargs) -> pd.DataFrame:
    """Phase 9 entry point; called only after the Phase 7 gate authorizes it."""
    raise RuntimeError("Residual features are not authorized before STRUCTURAL_ACTION_RESIDUAL_EXISTS")


def assert_no_outcome_features(columns: list[str]) -> None:
    forbidden = ("label", "outcome", "correct", "rescue", "harm", "effect")
    offenders = [column for column in columns if any(token in column.lower() for token in forbidden)]
    if offenders:
        raise RuntimeError(f"Outcome-dependent model features are forbidden: {offenders}")
