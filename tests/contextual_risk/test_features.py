import numpy as np

from hsc_tta.contextual_risk.features import SIGNATURE_COLUMNS, context_features


def test_signature_is_exactly_12d_and_unlabeled():
    rng=np.random.default_rng(4);p=rng.dirichlet(np.ones(5),size=40)
    features=context_features(p)
    assert len(SIGNATURE_COLUMNS)==12
    assert all(name in features for name in SIGNATURE_COLUMNS)
    assert all(np.isfinite(features[name]) for name in SIGNATURE_COLUMNS)
    assert not any("label" in name or "future" in name for name in features)
