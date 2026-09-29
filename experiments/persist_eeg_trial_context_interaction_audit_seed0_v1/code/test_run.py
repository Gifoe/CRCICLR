"""Small structural tests; scientific outputs still require server provenance checks."""
from __future__ import annotations

import unittest

import numpy as np

import run


class AuditStructuralTests(unittest.TestCase):
    def test_disjoint_first_b_and_acquisition_order(self):
        h = np.arange(3*10*64, dtype=float).reshape(30, 64)
        y = np.tile([0,1], 15)
        a = {"h": h, "y": y, "subject": np.repeat(["1","2","3"], 10)}
        p = run.parts(a, 4)
        self.assertEqual(p["1"]["indices"].tolist(), list(range(4,10)))
        np.testing.assert_array_equal(p["1"]["r"], h[:4].mean(0))
        self.assertEqual(len(p["1"]["y"]), 6)

    def test_arm_dimensions_and_translation(self):
        rng = np.random.default_rng(9)
        h, r = rng.normal(size=(7,64)), rng.normal(size=(7,64))
        self.assertEqual(run.features(h,r,"TRIAL_ONLY").shape, (7,64))
        self.assertEqual(run.features(h,r,"RELATIVE_SUBTRACTION").shape, (7,64))
        self.assertEqual(run.features(h,r,"ADDITIVE_CONTEXT").shape, (7,128))
        self.assertEqual(run.features(h,r,"ADDITIVE_RELATIVE").shape, (7,192))
        self.assertEqual(run.features(h,r,"ELEMENTWISE_INTERACTION").shape, (7,192))
        self.assertEqual(run.features(h,r,"FULL_SIMPLE_INTERACTION").shape, (7,256))
        t = rng.normal(size=(1,64))
        np.testing.assert_allclose(run.features(h+t,r+t,"RELATIVE_SUBTRACTION"), h-r)

    def test_group_derangement_is_deterministic(self):
        people = [str(i) for i in range(1,9)]
        x = run.derangement(people,"unit",3)
        self.assertEqual(x, run.derangement(people,"unit",3))
        self.assertEqual(set(x.values()), set(people))
        self.assertTrue(all(x[s] != s for s in people))

    def test_lowrank_product_shape(self):
        rng = np.random.default_rng(5)
        h, r = rng.normal(size=(12,64)), rng.normal(size=(12,64))
        y = np.tile([0,1], 6)
        basis = run.moment_basis(h,r,y,4)
        self.assertEqual(run.features(h,r,"LOWRANK_BILINEAR",basis).shape, (12,132))


if __name__ == "__main__": unittest.main()
