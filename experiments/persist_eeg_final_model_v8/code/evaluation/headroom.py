from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, log_loss


def subject_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, subject), group in predictions.groupby(["method_id", "subject_id"], sort=False):
        rows.append({
            "benchmark": str(group.benchmark.iloc[0]),
            "family_id": str(group.family_id.iloc[0]),
            "method_id": str(method),
            "subject_id": str(subject),
            "source_fold": int(group.source_fold.iloc[0]),
            "BA": float(balanced_accuracy_score(group.label, group.prediction)),
            "NLL": float(log_loss(group.label, np.clip(group.probability, 1e-7, 1 - 1e-7), labels=[0, 1])),
            "OUTER_TEST_USED": False,
        })
    return pd.DataFrame(rows)


def summarize_headroom(
    predictions: pd.DataFrame,
    baseline_method: str,
    primary_experts: list[str],
) -> dict[str, object]:
    subjects = subject_metrics(predictions)
    baseline = subjects.loc[subjects.method_id.eq(baseline_method)].set_index("subject_id")
    expected_subjects = set(baseline.index.astype(str))
    if not expected_subjects:
        raise RuntimeError("Headroom baseline has no subjects")
    competence_rows = []
    for method in primary_experts:
        group = subjects.loc[subjects.method_id.eq(method)].set_index("subject_id")
        if set(group.index.astype(str)) != expected_subjects:
            raise RuntimeError(f"Incomplete expert {method}")
        delta = group.loc[baseline.index, "BA"].to_numpy(float) - baseline.BA.to_numpy(float)
        competence_rows.append({
            "benchmark": str(predictions.benchmark.iloc[0]),
            "family_id": str(predictions.family_id.iloc[0]),
            "method_id": method,
            "subjects": len(group),
            "mean_subject_BA": float(group.BA.mean()),
            "mean_subject_NLL": float(group.NLL.mean()),
            "Delta_BA": float(delta.mean()),
            "positive_subject_fraction": float(np.mean(delta > 0.0)),
            "nonnegative_subject_fraction": float(np.mean(delta >= 0.0)),
            "harmful_subject_fraction": float(np.mean(delta < 0.0)),
            "worst_subject_delta": float(delta.min()),
            "p10_subject_delta": float(np.quantile(delta, 0.10)),
            "OUTER_TEST_USED": False,
        })
    competence = pd.DataFrame(competence_rows).sort_values("mean_subject_BA", ascending=False)

    oracle_rows = []
    usage = {method: 0 for method in [baseline_method, *primary_experts]}
    for subject in baseline.index.astype(str):
        candidates = subjects.loc[
            subjects.subject_id.astype(str).eq(subject)
            & subjects.method_id.isin([baseline_method, *primary_experts])
        ].copy()
        chosen = candidates.sort_values(["BA", "NLL", "method_id"], ascending=[False, True, True]).iloc[0]
        base = baseline.loc[subject]
        usage[str(chosen.method_id)] += 1
        oracle_rows.append({
            "benchmark": str(chosen.benchmark),
            "family_id": str(chosen.family_id),
            "subject_id": subject,
            "source_fold": int(chosen.source_fold),
            "baseline_BA": float(base.BA),
            "oracle_method": str(chosen.method_id),
            "oracle_BA": float(chosen.BA),
            "oracle_NLL": float(chosen.NLL),
            "oracle_delta_BA": float(chosen.BA - base.BA),
            "OUTER_TEST_USED": False,
        })
    oracle = pd.DataFrame(oracle_rows)
    fold_rows = []
    for fold, group in oracle.groupby("source_fold"):
        fold_rows.append({
            "benchmark": str(group.benchmark.iloc[0]),
            "family_id": str(group.family_id.iloc[0]),
            "source_fold": int(fold),
            "subjects": len(group),
            "baseline_BA": float(group.baseline_BA.mean()),
            "subject_oracle_BA": float(group.oracle_BA.mean()),
            "oracle_headroom_pp": float(100.0 * group.oracle_delta_BA.mean()),
            "OUTER_TEST_USED": False,
        })
    folds = pd.DataFrame(fold_rows)

    correctness: dict[str, np.ndarray] = {}
    ordered = predictions.sort_values(["subject_id", "trial_uid"])
    for method in primary_experts:
        part = ordered.loc[ordered.method_id.eq(method)]
        correctness[method] = (part.prediction.to_numpy(int) == part.label.to_numpy(int)).astype(float)
    diversity_rows = []
    for left, right in combinations(primary_experts, 2):
        a, b = correctness[left], correctness[right]
        correlation = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else 1.0
        disagreement = float(np.mean(a != b))
        diversity_rows.append({
            "benchmark": str(predictions.benchmark.iloc[0]),
            "family_id": str(predictions.family_id.iloc[0]),
            "expert_left": left,
            "expert_right": right,
            "correctness_correlation": correlation,
            "correctness_disagreement": disagreement,
            "OUTER_TEST_USED": False,
        })
    diversity = pd.DataFrame(diversity_rows)
    headroom = float(oracle.oracle_delta_BA.mean())
    if headroom < 0.04:
        state = "V8_HEADROOM_WEAK"
    elif headroom < 0.06:
        state = "V8_HEADROOM_PROMISING"
    elif headroom < 0.08:
        state = "V8_HEADROOM_STRONG"
    else:
        state = "V8_HEADROOM_GATE_PASS"
    summary = {
        "benchmark": str(predictions.benchmark.iloc[0]),
        "family_id": str(predictions.family_id.iloc[0]),
        "subjects": int(len(oracle)),
        "baseline_method": baseline_method,
        "baseline_BA": float(baseline.BA.mean()),
        "strongest_single_candidate": str(competence.iloc[0].method_id),
        "strongest_single_candidate_BA": float(competence.iloc[0].mean_subject_BA),
        "mean_expert_BA": float(competence.mean_subject_BA.mean()),
        "subject_oracle_BA": float(oracle.oracle_BA.mean()),
        "oracle_headroom_pp": float(100.0 * headroom),
        "subjects_rescued_ge_2pp_fraction": float(np.mean(oracle.oracle_delta_BA >= 0.02 - 1e-12)),
        "subjects_rescued_ge_5pp_fraction": float(np.mean(oracle.oracle_delta_BA >= 0.05 - 1e-12)),
        "positive_fold_fraction": float(np.mean(folds.oracle_headroom_pp > 0.0)),
        "mean_pairwise_correctness_correlation": (
            float(diversity.correctness_correlation.mean()) if len(diversity) else 1.0
        ),
        "mean_pairwise_correctness_disagreement": (
            float(diversity.correctness_disagreement.mean()) if len(diversity) else 0.0
        ),
        "oracle_usage": usage,
        "oracle_assignment_entropy": float(
            -sum((count / len(oracle)) * np.log(max(count / len(oracle), 1e-12)) for count in usage.values() if count)
        ),
        "headroom_state": state,
        "outcome_labels_used_for_headroom_only": True,
        "used_to_train_selector": False,
        "internal_holdout_used": False,
        "OUTER_TEST_USED": False,
    }
    return {
        "summary": summary,
        "subjects": subjects,
        "competence": competence,
        "oracle": oracle,
        "folds": folds,
        "diversity": diversity,
    }
