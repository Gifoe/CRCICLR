import pandas as pd
import pytest

from hsc_tta.certification import fit_actionwise_simultaneous_quantile


def test_alpha_rows_cannot_be_mixed_in_one_quantile():
    rows = []
    for subject in range(20):
        for action in ("no_tta", "t3a", "entropy_adapter"):
            rows.append({"dataset": "hmc", "seed": 0, "episode_id": f"e{subject}", "subject_id": f"s{subject}", "alpha": 0.1 if subject == 0 else 0.2, "action": action, "critical_index": 5, "predicted_critical_index": 4})
    with pytest.raises(ValueError, match="one alpha"):
        fit_actionwise_simultaneous_quantile(pd.DataFrame(rows), n_nontrivial_lambdas=20)
