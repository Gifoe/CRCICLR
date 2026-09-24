from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

from directional_streaming_fit_v1 import fit_directions_streamed

CORE = Path(__file__).with_name("directional_core_v3.py")
if not CORE.exists():
    CORE = Path(__file__).resolve().parents[5] / "_transfer" / "directional_review" / "directional_core_v3.py"
spec = importlib.util.spec_from_file_location("directional_core_v3_test", CORE.resolve())
core = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = core
spec.loader.exec_module(core)


def compare(a, q, mean, final_projector):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {"final_projector": final_projector, "device": device}
    expected = core.fit_directions(a, q, mean, **kwargs)
    actual = fit_directions_streamed(a, q, mean, **kwargs,
                                     direction_set_type=core.DirectionSet,
                                     chunk_features=37)
    assert actual.p_rank == expected.p_rank
    assert actual.c_rank == expected.c_rank
    assert actual.projector_mode == expected.projector_mode
    for name in ("p_axes", "c_axes", "p_sd", "c_sd"):
        err = float(np.max(np.abs(getattr(actual, name) - getattr(expected, name))))
        assert err < 2e-5, f"{name} max abs error {err}"


def main():
    rng = np.random.default_rng(20260924)
    a = rng.normal(size=(48, 256)).astype(np.float32)
    raw = rng.normal(size=(256, 6)).astype(np.float32)
    q, _ = np.linalg.qr(raw)
    q = q.astype(np.float32)
    mean = a.mean(axis=0, dtype=np.float64).astype(np.float32)
    compare(a, q, mean, None)
    right = q.T + np.float32(0.01) * rng.normal(size=(6, 256)).astype(np.float32)
    compare(a, q, mean, (q, right))
    print(f"STREAMED_FIT_EQUIVALENCE_PASS orthogonal=1 oblique=1 feature_chunk=37 device={'cuda' if torch.cuda.is_available() else 'cpu'}")


if __name__ == "__main__":
    main()
