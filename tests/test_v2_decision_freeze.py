import hashlib
from pathlib import Path

import pandas as pd

from hsc_tta.v2.nested_evaluation import _metrics


def test_decision_hash_changes_before_future_gate(tmp_path):
    path=tmp_path/"decision"; path.write_bytes(b"U-only")
    frozen=hashlib.sha256(path.read_bytes()).hexdigest()
    assert frozen==hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_bytes(b"changed")
    assert frozen!=hashlib.sha256(path.read_bytes()).hexdigest()


def test_metrics_use_frozen_certified_index_not_oracle_index():
    selected = pd.DataFrame([{"subject_id": "s", "selected_action": "no_tta", "certified_critical_index": 0,
                              "true_critical_index": 20, "true_benefit": 0.0, "future_risk": 0.0,
                              "future_average_set_size": 5.0, "future_singleton_rate": 0.0,
                              "risk_j0": .5, "set_size_j0": 1.0, "singleton_j0": 1.0,
                              "argmax_error": .4, "macro_f1": .5, "balanced_accuracy": .5, "cohen_kappa": .2}])
    counter = pd.DataFrame([{"subject_id": "s", "action": "no_tta", "true_critical_index": 20,
                             "true_benefit": 0.0}])
    result = _metrics(selected, counter, .1)
    assert result["marginal_violation"] == 1.0
    assert result["joint_validity"] == 0.0
    assert result["average_set_size"] == 1.0
