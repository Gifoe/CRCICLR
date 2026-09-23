"""Check streaming norm selection equals full activation selection."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from cell_data import materialize_stage, stage_norms
from sampling_core import TrialView, select_pairs


class DummyRunner:
    def __init__(self) -> None:
        self.m = torch.nn.Identity().eval()
        self.head = torch.nn.Identity().eval()

    def tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(x, np.float32))

    def all(self, x: torch.Tensor):
        a = x * torch.tensor([2., -1.])
        return {"dummy": a}, a, torch.stack((a[:, 0], a[:, 1]), dim=1)


class CellDataTest(unittest.TestCase):
    def test_streaming_selection_matches_full_array(self) -> None:
        x = np.arange(60, dtype=np.float32).reshape((30, 2)) / 10
        y = np.asarray([0] * 10 + [1] * 10 + [0] * 5 + [1] * 5, np.int64)
        s = np.asarray(["1"] * 20 + ["2"] * 10)
        se = np.asarray([1] * 20 + [2] * 10, np.int64)
        view = TrialView(x, y, s, se, "TRAIN")
        runner = DummyRunner()
        norms = stage_norms(runner, view, ("dummy",), batch_size=7)
        selected = materialize_stage(runner, view, model="EEGNet", task="OpenBMI_MI",
                                     fold=0, stage="dummy", batch_size=7, precomputed_norms=norms["dummy"])
        full = select_pairs((x * [2, -1]).astype(np.float32), y, s, se,
                            model="EEGNet", task="OpenBMI_MI", fold=0, stage="dummy", split="TRAIN")
        np.testing.assert_array_equal(selected.recipient_global, full.recipient)
        np.testing.assert_array_equal(selected.donor_global, full.donor)
        np.testing.assert_allclose(selected.activations[selected.recipient_local],
                                   (x * [2, -1])[full.recipient])


if __name__ == "__main__":
    unittest.main()
