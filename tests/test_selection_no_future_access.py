import pandas as pd
import pytest

from test_action_selection import _candidates
from hsc_tta.selection import select_safe_action


def test_selector_rejects_any_future_outcome_column():
    frame = _candidates()
    frame["future_risk"] = 0.0
    with pytest.raises(ValueError, match="future outcome"):
        select_safe_action(frame)


def test_selector_api_requires_candidate_table_not_outcome_table():
    with pytest.raises(ValueError):
        select_safe_action(pd.DataFrame({"future_risk": [0.1]}))
