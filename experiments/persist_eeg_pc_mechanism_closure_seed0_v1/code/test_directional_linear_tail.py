import unittest
from types import SimpleNamespace

import numpy as np
import torch

from directional_core import mixed_difference
from directional_linear_tail import affine_tail_mixed_difference, eligible


class Runner:
    def __init__(self, classes):
        self.name = "EEGNet"
        self.m = torch.nn.Identity().eval()
        self.head = torch.nn.Linear(3, classes).eval()

    @staticmethod
    def tensor(a):
        return torch.from_numpy(np.ascontiguousarray(a, np.float32))

    def from_stage(self, stage, a):
        return a, self.head(a)


class AffineTailTest(unittest.TestCase):
    def test_exact_shortcut_matches_four_forward_variants(self):
        rng = np.random.default_rng(21)
        for classes in (2, 4):
            runner = Runner(classes)
            with torch.no_grad():
                runner.head.weight.copy_(torch.from_numpy(
                    rng.normal(size=(classes, 3)).astype(np.float32)))
                runner.head.bias.copy_(torch.from_numpy(
                    rng.normal(size=classes).astype(np.float32)))
            a = rng.normal(size=(48, 3)).astype(np.float32)
            native = runner.head(torch.from_numpy(a)).detach().numpy()
            labels = rng.integers(0, classes, len(a))
            axes = SimpleNamespace(
                p_rank=1, c_rank=1,
                p_sd=np.array([0.7], np.float32),
                c_sd=np.array([1.3], np.float32),
                p_axes=np.array([[1], [0], [0]], np.float32),
                c_axes=np.array([[0], [1], [0]], np.float32),
            )
            self.assertTrue(eligible(runner, "classifier_input"))
            self.assertTrue(eligible(runner, "embedding_64d"))
            actual = affine_tail_mixed_difference(runner, "classifier_input", (3,),
                                                   a, labels, native, axes, (0, 0))
            reference = mixed_difference(runner, "classifier_input", (3,),
                                         a, labels, native, axes, (0, 0))
            for key in actual:
                self.assertLess(float(np.max(np.abs(actual[key] - reference[key]))), 1e-5)


if __name__ == "__main__":
    unittest.main()
