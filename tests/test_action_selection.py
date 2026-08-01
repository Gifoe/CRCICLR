import pandas as pd

from hsc_tta.selection import select_safe_action


def _candidates(indices=(5, 4, 4)):
    rows = []
    for action, index, size, singleton, cost in zip(
        ("no_tta", "t3a", "entropy_adapter"),
        indices,
        (2.0, 1.8, 1.8),
        (0.4, 0.5, 0.5),
        (0, 1, 2),
    ):
        rows.append(
            {
                "dataset": "hmc",
                "seed": 0,
                "episode_id": "hmc:0:s:main120",
                "subject_id": "s",
                "alpha": 0.2,
                "action": action,
                "predicted_critical_index": float(index - 1),
                "q_alpha": 1.0,
                "certified_critical_index": index,
                "n_nontrivial_lambdas": 20,
                "selected_lambda": 0.5 + 0.02 * index,
                "nontrivial_candidate": True,
                "context_average_set_size": size,
                "context_singleton_rate": singleton,
                "adaptation_cost": cost,
                "n_classes": 5,
            }
        )
    return pd.DataFrame(rows)


def test_lexicographic_selection_uses_context_only():
    result = select_safe_action(_candidates(), alpha=0.2)
    assert result["selected_action"] == "t3a"
    assert result["status"] == "certified"


def test_selector_rejects_mixed_identity_groups():
    frame = _candidates()
    frame.loc[0, "seed"] = 1
    try:
        select_safe_action(frame)
    except ValueError as error:
        assert "exactly one" in str(error)
    else:
        raise AssertionError("mixed seed groups were accepted")
