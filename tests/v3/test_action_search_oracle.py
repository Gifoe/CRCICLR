import numpy as np

from hsc_tta.v3.action_search import action_grid, config_id
from hsc_tta.v3.evaluation import safe_point


def test_required_action_grid_is_finite_and_complete():
    config = {"t3a": {"filter_k": [5, 10, 20, 50], "confidence_threshold": [None, .7, .8, .9],
                       "prototype_interpolation": [.25, .5, .75, 1.]},
              "adapter": {"steps": [1, 3, 5], "learning_rate": [1e-5, 5e-5, 1e-4],
                          "source_preservation_weight": [.25, .5, 1.], "reliability_quantile": [.1, .2, .3]},
              "fixed": {"consistency_weight": .1, "parameter_weight": .001, "collapse_weight": .5, "collapse_rho": .8}}
    grid = action_grid(config)
    assert len(grid["official_t3a"]) == 64
    assert len(grid["robust_residual_adapter"]) == 81
    assert len({config_id("official_t3a", x) for x in grid["official_t3a"]}) == 64


def test_safe_point_uses_smallest_valid_set():
    probabilities = np.array([[.8, .2], [.55, .45], [.1, .9]])
    labels = np.array([0, 1, 1])
    result = safe_point(probabilities, labels, .20, np.array([.5, .75, 1.]))
    assert result["future_risk"] <= .20
    assert result["lambda_index"] == 1
