import numpy as np
import pandas as pd

from hsc_tta.v2.ablations import _evaluate, _select


def _candidates():
    rows = []
    for action, index, lower, predicted in (("no_tta", 8, -1, 0), ("t3a", 9, .02, .04), ("robust", 20, .05, .07)):
        row = {"action": action, "available": True, "predicted_critical_index": index - 1,
               "predicted_benefit": predicted, "ablation_index": index, "ablation_lower": lower,
               "separate_certified_critical_index": index, "separate_benefit_lower": lower}
        row.update({f"context_set_size_j{i}": 1 + i / 20 for i in range(21)})
        rows.append(row)
    return pd.DataFrame(rows)


def test_ablation_gates_have_expected_directions():
    candidates = _candidates()
    assert _select(candidates, "A1_risk_only")[0] == "t3a"
    assert _select(candidates, "A2_utility_only_uncertified")[0] == "robust"
    assert _select(candidates, "A3_joint_without_benefit_certificate")[0] == "t3a"
    assert _select(candidates, "A4_separate_risk_gain_calibration")[0] == "t3a"
    assert _select(candidates, "A7_context_set_size_selector")[0] == "t3a"
    assert _select(candidates, "A9_no_positive_gain_gate")[0] == "t3a"


def test_ablation_metrics_do_not_count_full_set_as_csr():
    rows = []
    for subject, action, index, gain in (("s1", "t3a", 3, .1), ("s2", "no_tta", 20, 0.0)):
        row = {"subject_id": subject, "action": action, "selected_index": index, "true_benefit": gain,
               "argmax_error": .2, "macro_f1": .7, "balanced_accuracy": .7, "cohen_kappa": .5}
        for j in range(21):
            row[f"risk_j{j}"] = 0 if j == 20 else .05
            row[f"set_size_j{j}"] = 4 if j == 20 else 1.5
            row[f"singleton_j{j}"] = 0 if j == 20 else .5
        rows.append(row)
    metrics = _evaluate(pd.DataFrame(rows), .1)
    assert metrics["csr"] == .5
    assert metrics["full_set_fallback"] == .5
    assert metrics["tta_selection_rate"] == .5
    assert np.isclose(metrics["selected_vs_no_tta_gain"], .05)
