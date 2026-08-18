from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "experiments" / "persist_eeg_prospective_action_policy_v1" / "code"
sys.path.insert(0, str(CODE))

import build_data  # noqa: E402
import common  # noqa: E402
import modeling  # noqa: E402


def test_outer_guard_does_not_confuse_router_with_outer() -> None:
    payload = {"router_subject_folds_complete": True, "outer_test_used": False}
    assert common.recursive_outer_true(payload) == []
    assert common.recursive_outer_true({"outer_test_used": True}) == ["outer_test_used"]


def test_provenance_audit_is_fail_closed_and_development_only() -> None:
    payload = build_data.provenance_audit()
    assert payload["OUTER_TEST_USED"] is False
    assert all(item["eligible"] for item in payload["pilots"])
    assert len(payload["pilots"]) == 8


def test_dda_uses_crossfit_utility_not_same_cell_outcome() -> None:
    frame, meta = build_data.dda_dataset()
    raw = pd.read_csv(
        ROOT / "experiments" / "persist_eeg_dda_v1" / "outputs" / "results" / "DDA_BLOCK_CROSSFIT.csv"
    )
    assert len(frame) == 215
    assert meta["same_cell_u_excluded"] is True
    assert not np.allclose(frame.f_u_crossrun, raw.signed_u_spec, equal_nan=True)
    assert not np.allclose(frame.f_u_crossouterfold, raw.signed_u_spec, equal_nan=True)
    assert not frame.outer_test_used.any()


def test_wbcic_meta_dataset_excludes_incompetent_fbcnet_and_outer() -> None:
    frame, meta = build_data.wbcic_dataset()
    assert len(frame) == 80
    assert set(frame.backbone_id) == {"EEGNet", "EEGConformer", "DeepConvNet", "TeCh"}
    assert all("FBCNet" not in value for value in frame.backbone_id)
    assert meta["OUTER_TEST_USED"] is False
    assert not frame.outer_test_used.any()


def test_feature_inventory_never_contains_outcomes() -> None:
    frame, _ = build_data.wbcic_dataset()
    for scheme in ("leave_one_fold_out", "leave_one_backbone_out"):
        sets = modeling.feature_sets(frame, "wbcic_development_block", scheme)
        for features in sets.values():
            if features:
                assert not any(column.startswith("effect_") for column in features)


def test_subject_group_splits_are_disjoint() -> None:
    rows = []
    for subject in range(10):
        for trial in range(2):
            rows.append(
                {
                    "family_id": "openbmi_sample_router",
                    "subject_id": str(subject),
                    "fold_id": 0,
                    "seed_id": 0,
                    "unit_weight": 1.0,
                }
            )
    frame = pd.DataFrame(rows)
    _, splits = modeling._splits(frame, "openbmi_sample_router", "leave_one_subject_group_out")
    for train, test in splits:
        assert set(frame.iloc[train].subject_id).isdisjoint(set(frame.iloc[test].subject_id))
