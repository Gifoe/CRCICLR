"""Algebra and freeze-rule checks for directional interaction primitives."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from directional_core import fit_directions, frozen_top_pairs, mixed_difference, mixed_differences_grouped


class BilinearRunner:
    def __init__(self, bilinear: bool) -> None:
        self.m = torch.nn.Identity().eval()
        self.head = torch.nn.Identity().eval()
        self.device = torch.device("cpu")
        self.bilinear = bilinear

    def tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(x, np.float32))

    def from_stage(self, stage: str, a: torch.Tensor):
        if stage != "test":
            raise KeyError(stage)
        x = a.reshape((len(a), 2))
        score = x[:, 0] * x[:, 1] if self.bilinear else x[:, 0] + x[:, 1]
        return x, torch.stack((score, -score), dim=1)


class DirectionalCoreTest(unittest.TestCase):
    def test_train_axes_and_lexicographic_ties(self) -> None:
        a = np.asarray([[1., 2.], [-1., -2.], [2., -1.], [-2., 1.]], np.float32)
        q = np.asarray([[1.], [0.]], np.float32)
        d = fit_directions(a, q, np.zeros(2, np.float32))
        self.assertEqual((d.p_rank, d.c_rank), (1, 1))
        np.testing.assert_allclose(d.c_axes[:, 0], [0, 1], atol=1e-5)
        self.assertEqual(frozen_top_pairs(np.ones((2, 2)), 3), ((0, 0), (0, 1), (1, 0)))

    def test_mixed_difference_detects_bilinearity_not_linear_effect(self) -> None:
        a = np.asarray([[1., 2.], [2., -1.]], np.float32)
        y = np.asarray([0, 1], np.int64)
        q = np.asarray([[1.], [0.]], np.float32)
        train = np.asarray([[1., 2.], [-1., -2.], [2., -1.], [-2., 1.]], np.float32)
        d = fit_directions(train, q, np.zeros(2, np.float32))
        bilinear = BilinearRunner(True)
        _, native = bilinear.from_stage("test", bilinear.tensor(a))
        got = mixed_difference(bilinear, "test", (2,), a, y, native.numpy(), d, (0, 0))
        expected = (0.5 * d.p_sd[0]) * (0.5 * d.c_sd[0])
        np.testing.assert_allclose(got["centered_logit"][:, 0], expected, atol=1e-6)
        np.testing.assert_allclose(got["centered_logit"][:, 1], -expected, atol=1e-6)
        linear = BilinearRunner(False)
        _, native_linear = linear.from_stage("test", linear.tensor(a))
        zero = mixed_difference(linear, "test", (2,), a, y, native_linear.numpy(), d, (0, 0))
        np.testing.assert_allclose(zero["norm"], 0, atol=1e-6)

    def test_oblique_final_axes_preserve_raw_axis_and_sd(self) -> None:
        a = np.asarray([[1., 2.], [-1., -2.], [2., -1.], [-2., 1.]], np.float32)
        q = np.asarray([[1.], [0.]], np.float32)
        left = np.asarray([[1.], [0.5]], np.float32)
        right = np.asarray([[1., 0.]], np.float32)
        d = fit_directions(a, q, np.zeros(2, np.float32), final_projector=(left, right))
        self.assertEqual(d.projector_mode, "FROZEN_CANONICAL_OBLIQUE")
        np.testing.assert_allclose(d.p_axes[:, 0], [1, 0])
        self.assertGreater(d.p_sd[0], 0)

    def test_mixed_difference_is_independent_of_batch_partition(self) -> None:
        train = np.asarray([[1., 2.], [-1., -2.], [2., -1.], [-2., 1.]], np.float32)
        a = np.tile(train, (18, 1))
        y = np.tile(np.asarray([0, 1, 0, 1], np.int64), 18)
        q = np.asarray([[1.], [0.]], np.float32)
        d = fit_directions(train, q, np.zeros(2, np.float32))
        runner = BilinearRunner(True)
        _, logits = runner.from_stage("test", runner.tensor(a))
        small = mixed_difference(runner, "test", (2,), a, y, logits.numpy(), d,
                                 (0, 0), batch_size=16)
        large = mixed_difference(runner, "test", (2,), a, y, logits.numpy(), d,
                                 (0, 0), batch_size=64)
        for key in small:
            np.testing.assert_allclose(small[key], large[key], rtol=0, atol=1e-6)

    def test_grouped_pairs_preserve_each_four_point_result(self) -> None:
        train = np.asarray([[1., 2.], [-1., -2.], [2., -1.], [-2., 1.]], np.float32)
        a = np.tile(train, (18, 1))
        y = np.tile(np.asarray([0, 1, 0, 1], np.int64), 18)
        q = np.eye(2, dtype=np.float32)
        # Use two protected axes with one complement axis by constructing a
        # valid DirectionSet directly; only batching/evaluation is under test.
        from directional_core import DirectionSet
        d = DirectionSet(q, np.asarray([[0.], [1.]], np.float32),
                         np.ones(2, np.float32), np.ones(1, np.float32), 2, 1,
                         "TEST_ONLY")
        runner = BilinearRunner(True)
        _, logits = runner.from_stage("test", runner.tensor(a))
        pairs = ((0, 0), (1, 0))
        grouped = mixed_differences_grouped(runner, "test", (2,), a, y, logits.numpy(),
                                             d, pairs, trial_batch=16)
        for pair in pairs:
            single = mixed_difference(runner, "test", (2,), a, y, logits.numpy(),
                                      d, pair, batch_size=16)
            for key in single:
                np.testing.assert_allclose(grouped[pair][key], single[key], rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
