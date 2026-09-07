from test_action_selection import _candidates
from hsc_tta.selection import select_safe_action


def test_sentinel_is_uncertified_and_not_nontrivial_csr():
    frame = _candidates(indices=(20, 20, 20))
    frame["selected_lambda"] = 1.0
    frame["context_average_set_size"] = 5.0
    frame["context_singleton_rate"] = 0.0
    frame["nontrivial_candidate"] = False
    result = select_safe_action(frame)
    assert result["status"] == "uncertified"
    assert not result["certified"] and not result["nontrivial_certified"]
    assert result["selected_lambda"] == 1.0
