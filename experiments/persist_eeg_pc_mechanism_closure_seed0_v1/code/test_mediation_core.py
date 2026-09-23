"""CPU algebra regression for the frozen adjacent-stage mediation primitive."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from mediation_core import _one_direction, adjacent_mediation, long_range_mediation, verify_adjacent_forward


class IdentityStageRunner:
    name = "EEGNet"
    names = ("embedding_64d", "classifier_input")
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.m = torch.nn.Identity().eval()
        self.head = torch.nn.Linear(3, 3, bias=False).eval()
        with torch.no_grad():
            self.head.weight.copy_(torch.tensor([[1.0, 0.5, 0.0], [0.0, 1.0, 0.25], [0.5, 0.0, 1.0]]))

    def tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(x, np.float32))

    def all(self, x: torch.Tensor):
        return {"embedding_64d": x, "classifier_input": x}, x, self.head(x)

    def from_stage(self, stage: str, a: torch.Tensor):
        if stage not in self.names:
            raise KeyError(stage)
        return a, self.head(a)


class MediationCoreTest(unittest.TestCase):
    def test_long_range_linear_identity_has_no_cross_conversion(self) -> None:
        class LongRangeRunner(IdentityStageRunner):
            def from_stage(self, stage: str, a: torch.Tensor):
                if stage not in ("temporal_bn", "classifier_input"):
                    raise KeyError(stage)
                return a, self.head(a)

        runner = LongRangeRunner()
        recipient = np.asarray([[1.0, 2.0, -1.0]], np.float32)
        donor = np.asarray([[-1.0, 1.0, 2.0]], np.float32)
        q = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], np.float32)
        result = long_range_mediation(runner, "temporal_bn", recipient, donor, (3,),
                                      q, np.zeros(3, np.float32), q, np.zeros(3, np.float32),
                                      np.asarray([0]), np.asarray([2]), (q, q.T))
        np.testing.assert_allclose(result["P_to_C_norm"], 0, atol=1e-6)
        np.testing.assert_allclose(result["C_to_P_norm"], 0, atol=1e-6)
        np.testing.assert_allclose(result["P_nonlinear_residual_norm"], 0, atol=1e-6)

    def test_hybrid_restores_nonflat_successor_shape(self) -> None:
        class ConvShapedRunner(IdentityStageRunner):
            def from_stage(self, stage: str, a: torch.Tensor):
                if a.ndim != 4 or a.shape[1:] != (1, 1, 3):
                    raise RuntimeError("hybrid was not reshaped to frozen successor stage")
                return a.flatten(1), self.head(a.flatten(1))

        runner = ConvShapedRunner()
        base = torch.tensor([[1.0, 2.0, 3.0]])
        changed = torch.tensor([[2.0, 2.0, 4.0]])
        q = torch.tensor([[1.0], [0.0], [0.0]])
        zb = runner.head(base)
        zi = runner.head(changed)
        result = _one_direction(runner, "early", "middle", (1, 1, 3), base, changed,
                                zb, zi, q, torch.zeros(3), torch.tensor([0]),
                                torch.tensor([1]), "P")
        self.assertEqual(len(result["P_total_norm"]), 1)

    def test_linear_identity_transition_has_no_cross_conversion(self) -> None:
        runner = IdentityStageRunner()
        self.assertEqual(verify_adjacent_forward(runner, np.ones((2, 3), np.float32)),
                         {"embedding_64d->classifier_input": 0.0})
        recipient = np.asarray([[1.0, 2.0, -1.0], [2.0, -1.0, 0.5]], np.float32)
        donor = np.asarray([[-1.0, 1.0, 2.0], [1.0, 2.0, -0.5]], np.float32)
        q = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], np.float32)
        result = adjacent_mediation(runner, "embedding_64d", recipient, donor, (3,),
                                    q, np.zeros(3, np.float32), q, np.zeros(3, np.float32),
                                    np.asarray([0, 1]), np.asarray([2, 0]), (q, q.T))
        np.testing.assert_allclose(result["P_total_norm"], result["P_to_P_norm"], atol=1e-6)
        np.testing.assert_allclose(result["C_total_norm"], result["C_to_C_norm"], atol=1e-6)
        for key in ("P_to_C_norm", "C_to_P_norm", "P_nonlinear_residual_norm",
                    "C_nonlinear_residual_norm", "P_to_C_rep", "C_to_P_rep"):
            np.testing.assert_allclose(result[key], 0, atol=1e-6)

    def test_canonical_oblique_final_is_not_silently_orthogonalized(self) -> None:
        runner = IdentityStageRunner()
        recipient = np.asarray([[1.0, 2.0, -1.0]], np.float32)
        donor = np.asarray([[-1.0, 1.0, 2.0]], np.float32)
        q_current = np.asarray([[1.0], [0.0], [0.0]], np.float32)
        left = q_current
        right = np.asarray([[1.0, 0.5, 0.0]], np.float32)
        q_final = right.T / np.linalg.norm(right)
        result = adjacent_mediation(runner, "embedding_64d", recipient, donor, (3,),
                                    q_current, np.zeros(3, np.float32),
                                    q_final, np.zeros(3, np.float32),
                                    np.asarray([0]), np.asarray([2]), (left, right))
        self.assertGreater(float(result["P_to_C_rep"][0]), 0.1)
        self.assertGreater(float(result["P_to_C_norm"][0]), 0.01)

    def test_large_float32_activation_reassembles_in_double_before_native_forward(self) -> None:
        runner = IdentityStageRunner()
        recipient = np.asarray([[266.76624, -147.125, 91.375]], np.float32)
        donor = np.asarray([[-193.5, 182.25, -77.0]], np.float32)
        q = np.asarray([[1.0], [0.0], [0.0]], np.float32)
        mu = np.asarray([0.251, -1.031, 0.125], np.float32)
        result = adjacent_mediation(runner, "embedding_64d", recipient, donor, (3,),
                                    q, mu, q, mu, np.asarray([0]), np.asarray([2]),
                                    (q, q.T))
        self.assertTrue(np.isfinite(result["P_total_norm"]).all())


if __name__ == "__main__":
    unittest.main()
