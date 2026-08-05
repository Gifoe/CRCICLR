from pathlib import Path

import numpy as np
import pandas as pd

from hsc_tta.contextual_risk.screening import build_shared_tables


def test_nondevelopment_cache_never_reads_future_members(tmp_path: Path, monkeypatch):
    cache = tmp_path / "repo/outputs/contextual_risk/source_cache/hmc/seed_0"
    cache.mkdir(parents=True)
    path = cache / "hmc_001.npz"
    probabilities = np.tile(np.array([[.7, .2, .1]]), (24, 1))
    np.savez(
        path,
        context_probabilities=probabilities,
        future_probabilities=probabilities,
        future_labels=np.zeros(24, dtype=int),
        episode_hash=np.asarray("episode"),
        source_model_hash=np.asarray("model"),
    )
    cohorts = pd.DataFrame(
        [{"dataset": "hmc", "subject_id": "hmc:001", "master_cohort": "internal_final_evaluation", "screening_fold": -1}]
    )
    real_load = np.load

    class Guarded:
        def __init__(self, loaded): self.loaded = loaded
        def __enter__(self): return self
        def __exit__(self, *args): self.loaded.close()
        def __getitem__(self, key):
            if key in {"future_probabilities", "future_labels"}:
                raise AssertionError("non-development Future cache member was opened")
            return self.loaded[key]

    def guarded_load(current, *args, **kwargs):
        loaded = real_load(current, *args, **kwargs)
        return Guarded(loaded) if Path(current) == path else loaded

    monkeypatch.setattr("hsc_tta.contextual_risk.screening.np.load", guarded_load)
    features, surfaces = build_shared_tables(tmp_path, cohorts)
    assert len(features) == 1
    assert surfaces.empty
