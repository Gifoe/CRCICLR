import numpy as np
import pytest

from evaluate_outer import method_features


def test_outer_features_are_fixed_by_train_scores_only():
    rng = np.random.default_rng(11)
    z = rng.normal(size=(3, 248, 4))
    r = np.linspace(-1, 1, 248)
    u = np.sin(np.linspace(0, 5, 248))
    energy = np.linspace(1, 2, 248)
    original = method_features(z, "RELIABLE_UTILITY_TOPK", "0.25", r, u, energy)
    changed_tokens = z.copy()
    changed_tokens[:, 0, :] += 100
    changed = method_features(changed_tokens, "RELIABLE_UTILITY_TOPK", "0.25", r, u, energy)
    # Evaluation tokens can change pooled values, but not their TRAIN-defined
    # membership. The exact pooled difference is determined by token 0's rank.
    selected_score = (r-r.mean())/r.std() + (u-u.mean())/u.std()
    selected = set(np.argsort(-selected_score, kind="stable")[:62])
    expected = 100/62 if 0 in selected else 0
    np.testing.assert_allclose(changed-original, expected, atol=1e-12)
    with pytest.raises(RuntimeError):
        method_features(z, "RELIABLE_UTILITY_TOPK", "0.3", r, u, energy)
