from __future__ import annotations

import pandas as pd

from .calibration_schemes import S1, S2, S3, S4


def raw_pass(row: pd.Series) -> bool:
    return bool(row.raw_spearman >= .30 and row.raw_mae_improvement >= .10 and
                row.raw_gain > 0 and row.raw_gain_ci_low >= 0 and row.raw_oracle_recovery >= .10)


def exact_pass(row: pd.Series) -> bool:
    return bool(row.calibrated_violation_mean <= .10 and row.worst_seed_violation <= .20 and
                row.max_seed_cp_upper <= .20 and row.calibrated_gain >= .05 and
                row.calibrated_gain_ci_low > 0 and row.calibrated_oracle_recovery >= .20 and
                row.sentinel_delta <= .05 and row.sentinel_transition_rate <= .10 and
                row.outlier_driven_fold_rate <= .20 and bool(row.loo_gain_sign_stable))


def decide(summary: pd.DataFrame) -> tuple[str, dict[str, str]]:
    primary = summary[(summary.strategy == "temporal") & summary.requested_budget.isin([5, 10, 20])]
    dataset_verdict = {}
    raw_primary = primary[primary.calibration_scheme == S1]
    raw_ok = {d: bool(g.raw_gate_pass.any()) for d, g in raw_primary.groupby("dataset")}
    if not all(raw_ok.get(d, False) for d in ("hmc", "eegmmidb")):
        moderate = summary[(summary.strategy == "temporal") & (summary.requested_budget == 50) & (summary.calibration_scheme == S1)]
        if all(moderate[moderate.dataset == d].raw_gate_pass.any() for d in ("hmc", "eegmmidb")):
            return "V51_MODERATE_BUDGET_ONLY", {d: "MODERATE_ONLY" for d in ("hmc", "eegmmidb")}
        return "V51_STOP_RAW_PREDICTOR_FAILURE", {d: ("RAW_PASS" if raw_ok.get(d) else "RAW_FAIL") for d in ("hmc", "eegmmidb")}
    for scheme in (S2, S3):
        selected = primary[primary.calibration_scheme == scheme]
        if all(selected[selected.dataset == d].exact_gate_pass.any() for d in ("hmc", "eegmmidb")):
            return "V51_CONTINUE_TO_FULL_METHOD", {d: f"EXACT_PASS:{scheme}" for d in ("hmc", "eegmmidb")}
    cross = primary[primary.calibration_scheme == S4]
    if all(cross[cross.dataset == d].exact_gate_pass.any() for d in ("hmc", "eegmmidb")):
        return "V51_HINT_ONLY_CROSSFIT", {d: "CROSSFIT_HINT" for d in ("hmc", "eegmmidb")}
    return "V51_STOP_CALIBRATION_FAILURE", {d: "CALIBRATION_FAIL" for d in ("hmc", "eegmmidb")}
