import unittest

import numpy as np

import aggregate as a
import geometry as g


class ContextAuditTests(unittest.TestCase):
    def test_first_b_disjoint_and_label_free(self):
        h = np.arange(40, dtype=np.float64).reshape(20, 2)
        data = {"h": h, "y": np.tile([0, 1], 10), "subject": np.array(["1"] * 20)}
        got = a.split_context(data, 8)["1"]
        np.testing.assert_allclose(got["reference"], h[:8].mean(0))
        np.testing.assert_array_equal(got["h"], h[8:])
        changed = {**data, "y": 1 - data["y"]}
        np.testing.assert_allclose(a.split_context(changed, 8)["1"]["reference"], got["reference"])

    def test_class_relation_is_translation_invariant(self):
        rng = np.random.default_rng(3)
        h = rng.normal(size=(40, 64))
        y = np.tile([0, 1], 20)
        r = rng.normal(size=64)
        original = h[y == 1].mean(0) - h[y == 0].mean(0)
        relative = (h[y == 1] - r).mean(0) - (h[y == 0] - r).mean(0)
        np.testing.assert_allclose(original, relative, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(g.cosine(original, relative), 1.0)

    def test_global_center_is_common_translation(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(30, 8))
        y = np.tile([0, 1], 15)
        p = {"1": {"h": x, "y": y, "reference": x[:8].mean(0)}}
        mu = x.mean(0)
        absolute = a.transform(p, "ABSOLUTE", mu)[0]
        centered = a.transform(p, "GLOBAL_SOURCE_CENTER", mu)[0]
        np.testing.assert_allclose(absolute - centered, np.broadcast_to(mu, absolute.shape))

    def test_train_pseudotarget_subject_cv(self):
        rng = np.random.default_rng(10)
        owner = np.repeat(np.arange(1, 11).astype(str), 20)
        y = np.tile(np.tile([0, 1], 10), 10)
        h1 = rng.normal(size=(200, 5)) + y[:, None] * .2
        h2 = rng.normal(size=(200, 5)) + y[:, None] * .2
        source = {"h": h1, "y": y, "subject": owner}
        future = {"h": h2, "y": y, "subject": owner}
        absolute = a.cv_curve(source, future, 8, "ABSOLUTE", 1.0)
        relative = a.cv_curve(source, future, 8, "REL_MEAN", 1.0)
        self.assertEqual(len(absolute), 10)
        self.assertEqual(len(relative), 10)
        self.assertEqual({r["subject"] for r in absolute}, set(owner))


if __name__ == "__main__":
    unittest.main()
