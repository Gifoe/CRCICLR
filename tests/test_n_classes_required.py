import pytest

from test_action_selection import _candidates
from hsc_tta.selection import select_safe_action


def test_missing_n_classes_fails_without_infinity_fallback():
    with pytest.raises(ValueError, match="n_classes"):
        select_safe_action(_candidates().drop(columns="n_classes"))
