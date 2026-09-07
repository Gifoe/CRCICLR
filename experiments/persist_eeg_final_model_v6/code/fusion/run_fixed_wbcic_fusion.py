"""Label-free fixed fusion of V5 and the PERSIST-protected encoder update."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from common import ABLATIONS, DIAGNOSTICS, LEADERBOARD, PROTOCOL, V6_SEED, logit, sigmoid, v5_output_root, write_csv, write_json
from evaluation.metrics import summarize


def _v5() -> pd.DataFrame:
    frame = pd.read_csv(v5_output_root() / "diagnostics" / "WBCIC_MULTI_SEED_OOF_PREDICTIONS.csv")
    frame = frame.loc[frame.seed.astype(int).eq(V6_SEED)].copy()
    frame = frame.rename(columns={"dataset": "benchmark"})
    frame["benchmark"] = "WBCIC_S1S2_to_S3_authorized_development"
    frame["method_id"] = "V5_CS_LGS_ANCHOR"
    frame["target_future_labels_used_for_fit"] = False
    if len(frame) != 8_195 or frame.trial_uid.duplicated().any():
        raise RuntimeError("Malformed V5 anchor")
    return frame


def _frame(anchor: pd.DataFrame, method_id: str, probability: np.ndarray) -> pd.DataFrame:
    result = anchor[["benchmark", "trial_uid", "subject_id", "outer_fold", "label"]].copy()
    result["method_id"] = method_id
    result["probability"] = probability
    result["prediction"] = (probability >= 0.5).astype(int)
    result["target_history_labels_used"] = True
    result["target_future_labels_used_for_fit"] = False
    result["exploratory"] = True
    result["OUTER_TEST_USED"] = False
    return result


def run() -> None:
    anchor = _v5()
    encoder = pd.read_csv(DIAGNOSTICS / "WBCIC_ENCODER_FINETUNING_PREDICTIONS.csv")
    encoder = encoder.loc[encoder.method_id.eq("PERSIST_SA_FISHER_PROTECTED")].copy()
    if len(encoder) != len(anchor) or encoder.trial_uid.duplicated().any() or set(encoder.trial_uid) != set(anchor.trial_uid):
        raise RuntimeError("PERSIST/V5 coverage mismatch")
    encoder = encoder.set_index("trial_uid").loc[anchor.trial_uid].reset_index()
    if not np.array_equal(anchor.label.to_numpy(int), encoder.label.to_numpy(int)):
        raise RuntimeError("PERSIST/V5 label alignment mismatch")
    p_anchor = np.clip(anchor.probability.to_numpy(float), 1e-7, 1 - 1e-7)
    p_persist = np.clip(encoder.probability.to_numpy(float), 1e-7, 1 - 1e-7)
    z_anchor = logit(p_anchor)
    z_persist = logit(p_persist)
    candidates = {
        # The primary fusion is fixed before evaluation: equal evidence in
        # log-odds space, with no learned weight or outcome-label selection.
        "V6_FIXED_EQUAL_LOGIT_V5_PERSIST": sigmoid(0.5 * (z_anchor + z_persist)),
        "CONTROL_FIXED_EQUAL_PROB_V5_PERSIST": 0.5 * (p_anchor + p_persist),
        "CONTROL_MAX_CONFIDENCE_V5_PERSIST": np.where(np.abs(z_persist) > np.abs(z_anchor), p_persist, p_anchor),
        # A conservative fail-closed control: PERSIST is permitted to act only
        # where the V5 anchor is intrinsically uncertain.  The 0.10 probability
        # margin is fixed and not selected on outcome labels.
        "V6_V5_UNCERTAINTY_GATE_010": np.where(np.abs(p_anchor - 0.5) <= 0.10, p_persist, p_anchor),
    }
    prediction_parts = [anchor]
    rows, subject_parts, fold_parts = [], [], []
    anchor_row, anchor_subjects, anchor_folds = summarize(anchor)
    rows.append(anchor_row); subject_parts.append(anchor_subjects); fold_parts.append(anchor_folds)
    for method_id, probability in candidates.items():
        frame = _frame(anchor, method_id, probability)
        prediction_parts.append(frame)
        row, subjects, folds = summarize(frame, reference=anchor)
        rows.append(row); subject_parts.append(subjects); fold_parts.append(folds)
    table = pd.DataFrame(rows).sort_values("mean_subject_BA", ascending=False)
    write_csv(LEADERBOARD / "WBCIC_FIXED_PERSIST_FUSION.csv", table)
    write_csv(DIAGNOSTICS / "WBCIC_FIXED_PERSIST_FUSION_PREDICTIONS.csv", pd.concat(prediction_parts, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_FIXED_PERSIST_FUSION_SUBJECT_RESULTS.csv", pd.concat(subject_parts, ignore_index=True))
    write_csv(DIAGNOSTICS / "WBCIC_FIXED_PERSIST_FUSION_FOLD_RESULTS.csv", pd.concat(fold_parts, ignore_index=True))
    write_csv(ABLATIONS / "WBCIC_FIXED_PERSIST_FUSION_ABLATION.csv", table)
    write_json(
        PROTOCOL / "WBCIC_FIXED_PERSIST_FUSION_AUDIT.json",
        {
            "anchor": "V5 M13_CSP_AUGMENTED_REFIT4 seed 20260820",
            "personalized": "PERSIST_SA_FISHER_PROTECTED",
            "primary_rule": "equal logit average",
            "weight": 0.5,
            "weight_fitted_or_selected_on_outcome": False,
            "gate_threshold_fitted_or_selected_on_outcome": False,
            "target_future_labels_used_for_fit": False,
            "exploratory_posthoc_family_comparison": True,
            "OUTER_TEST_USED": False,
        },
    )
    print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    run()
