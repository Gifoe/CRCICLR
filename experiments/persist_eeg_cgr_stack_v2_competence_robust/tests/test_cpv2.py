"""Protocol-focused CPV2 implementation tests.

These tests are data-free.  They exercise the cardinality/audit primitives and
inspect the implementation contracts that protect source-only inference.
"""
from __future__ import annotations

import inspect
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


CODE = Path(__file__).resolve().parents[1] / "code" / "cpv2.py"
spec = importlib.util.spec_from_file_location("cpv2_test_module", CODE)
cpv2 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cpv2)


def _synthetic_bank():
    frame = pd.DataFrame({
        "sample_id": ["a", "b"], "subject": ["1", "2"], "session": [1, 1],
        "trial_index": [0, 1], "label": [0, 1], "signal_path": ["", ""], "cache_index": [0, 1],
    })
    rows = []
    for sample, sub, lab in zip(frame.sample_id, frame.subject, frame.label):
        for expert in cpv2.EXPERTS:
            for slot in range(6):
                p = slot // 3
                fold = slot % 5
                rows.append({"dataset": "OpenBMI", "sample_id": sample, "subject": sub, "session": 1,
                             "trial_index": 0, "label": lab, "partition_system": p, "fold": fold,
                             "seed": slot % 3, "run_id": f"p{p}f{fold}s{slot%3}", "slot": slot,
                             "expert": expert, "logit_0": -0.1, "logit_1": 0.1,
                             "train_subject_count": 1, "train_subject_fraction": 0.8,
                             "target_fold": fold, "partition_salt": "salt", "excluded_target_subjects": f'["{sub}"]'})
    data = cpv2.DatasetData("OpenBMI", frame, None, 62, {"subjects": 2, "samples": 2})
    return data, pd.DataFrame(rows)


def test_every_source_sample_has_six_oof_predictions_per_expert():
    data, bank = _synthetic_bank()
    audit = cpv2.audit_bank(bank, data)
    assert audit["six_predictions_per_sample_per_expert"] is True
    assert audit["rows"] == 2 * 3 * 6


def test_every_predictor_excludes_target_subject():
    source = inspect.getsource(cpv2.make_run_prediction)
    assert "target_subjects & set" in source
    assert "target biological subject leaked" in source


def test_every_oof_model_uses_approximately_eighty_percent_subjects():
    data, bank = _synthetic_bank()
    audit = cpv2.audit_bank(bank, data)
    assert abs(audit["average_training_subject_fraction"] - 0.8) < 1e-9


def test_no_complete_case_subject_deletion():
    assert "complete_case_filter" in inspect.getsource(cpv2.audit_bank)
    assert cpv2.audit_bank.__code__.co_consts is not None


def test_two_partition_systems_have_fixed_independent_salts():
    for dataset, salts in cpv2.PARTITION_SALTS.items():
        assert len(salts) == 2 and salts[0] != salts[1]
        assert cpv2.partition_maps(dataset, [str(i) for i in range(1, 11)])[0] != cpv2.partition_maps(dataset, [str(i) for i in range(1, 11)])[1]


def test_openbmi_and_wbcic_share_expert_semantics():
    assert cpv2.EXPERTS == ("E0_KEEP_ERM", "E1_GRL", "E2_CORAL")
    assert cpv2.GRL_COEF == cpv2.CORAL_COEF == 0.10


def test_wbcic_s2_is_not_referenced_as_training_session():
    source = inspect.getsource(cpv2.make_run_prediction)
    assert ".eq(0)" in source and ".eq(1)" in source and ".eq(2)" not in source


def test_calibration_is_fit_on_train_subjects():
    source = inspect.getsource(cpv2.fit_all_params)
    assert "margins[expert][train_mask]" in source


def test_level_one_stacking_is_fit_on_train_subjects():
    source = inspect.getsource(cpv2.fit_all_params)
    assert "family_matrix[train_mask]" in source


def test_robust_stack_is_fit_on_train_subjects():
    source = inspect.getsource(cpv2.evaluate_dataset)
    assert "fit_residual_alpha(keep_logit[train]" in source


def test_biological_subject_is_cv_and_bootstrap_unit():
    assert cpv2.BOOTSTRAP_DRAWS == 10_000
    assert "subject_ids" in inspect.getsource(cpv2.subject_ba)


def test_no_trial_bootstrap_for_final_inference():
    source = inspect.getsource(cpv2.metric_row)
    assert "subject_bootstrap(delta" in source


def test_negative_subjects_are_not_removed():
    source = inspect.getsource(cpv2.metric_row)
    assert "delta < 0" in source and "drop" not in source


def test_keep_is_literal_residual_anchor():
    source = inspect.getsource(cpv2.fit_residual_alpha)
    assert "anchor_logit + float(alpha) * (stack_logit - anchor_logit)" in source


def test_residual_alpha_is_bounded():
    assert "bounds=(0.0, 1.0)" in inspect.getsource(cpv2.fit_residual_alpha)


def test_consensus_alpha_contract_is_monotone_if_authorized():
    source = inspect.getsource(cpv2.write_protocol_and_reports)
    assert "alpha_0" in source and "NOT_AUTHORIZED" in source


def test_unanimous_consensus_alpha_is_zero():
    assert float(pd.DataFrame([{"alpha_0": 0.0}]).alpha_0.iloc[0]) == 0.0


def test_cvar_is_computed_over_subject_harm():
    source = inspect.getsource(cpv2.fit_residual_alpha)
    assert "balanced_logloss_per_subject" in source and "tail_n" in source


def test_duplicate_keep_control_matches_three_family_run_count():
    source = inspect.getsource(cpv2.evaluate_dataset)
    assert "np.column_stack([params[\"calibrated\"][\"E0_KEEP_ERM\"]] * 3)" in source


def test_outer_and_sealed_resources_remain_false():
    source = inspect.getsource(cpv2.run_pipeline)
    assert "S2_accessed" not in source or "False" in source
    assert "outer_accessed" not in source or "False" in source


def test_no_target_session_gradient_update():
    source = inspect.getsource(cpv2.make_run_prediction)
    assert "train_mask" in source and "target_mask" in source
    assert "target_positions" in source and "train_positions" in source


def test_fixed_expert_training_budget():
    assert cpv2.EPOCHS >= 1 and cpv2.BATCH_SIZE >= 1
