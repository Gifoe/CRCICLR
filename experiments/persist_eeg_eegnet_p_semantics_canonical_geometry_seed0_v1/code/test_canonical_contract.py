"""Small synthetic checks for the frozen linear transform implementation."""
from __future__ import annotations

import unittest

import numpy as np

import canonical
import data_geometry as dg
import permutations_v2
import semantics
import transform_runner as tv1
import transform_runner_v2 as tv2
import transform_runner_v3 as tv3


class CanonicalContract(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(7)
        self.source = rng.normal(size=(80, 12)).astype(np.float32)

    def test_identity_exact(self) -> None:
        target = self.source + 3
        fitted = canonical.fit_unsupervised(self.source, target, "identity")
        np.testing.assert_array_equal(fitted.apply(target), target)

    def test_translation_uses_only_moments(self) -> None:
        shift = np.arange(12, dtype=np.float32) / 10
        target = self.source + shift
        fitted = canonical.fit_unsupervised(self.source, target, "translation")
        np.testing.assert_allclose(fitted.apply(target), self.source, atol=2e-6)
        self.assertFalse(fitted.paired_class_centroids_used)

    def test_orthogonal_is_orthogonal_and_train_supported(self) -> None:
        rng = np.random.default_rng(8)
        q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
        target = (self.source @ q).astype(np.float32)
        fitted = canonical.fit_unsupervised(self.source, target, "orthogonal_translation")
        self.assertLessEqual(fitted.basis.shape[1], 12)
        np.testing.assert_allclose(fitted.support_map.T @ fitted.support_map,
                                   np.eye(fitted.support_map.shape[0]), atol=1e-5)

    def test_lowrank_residual_is_rank_capped(self) -> None:
        target = self.source.copy()
        target[:, 0] *= 1.3
        fitted = canonical.fit_unsupervised(self.source, target, "lowrank_residual_affine_r2")
        delta = fitted.support_map - np.eye(fitted.support_map.shape[0])
        self.assertLessEqual(int(np.linalg.matrix_rank(delta, tol=1e-5)), 2)

    def test_gram_label_null_observed_matches_direct_centroids(self) -> None:
        rng = np.random.default_rng(13)
        values, labels, owners, sessions = [], [], [], []
        for subject in ("a", "b", "c"):
            for session in (1, 2):
                for label in (0, 1):
                    for _ in range(8):
                        values.append(rng.normal(size=5) + label * np.array([1, 0, 0, 0, 0]))
                        labels.append(label)
                        owners.append(subject)
                        sessions.append(session)
        x = np.asarray(values, dtype=np.float32)
        pop = dg.Population(x, np.asarray(labels), np.asarray(owners), np.asarray(sessions), "OUTER_DEVELOPMENT")
        reference = {(subject, session): np.array([1, 0, 0, 0, 0], dtype=np.float32)
                     for subject in ("r", "s") for session in (1, 2)}
        observed, _, _, _ = permutations_v2.label_null_gram(reference, x, pop, (1, 2), 200, ("test",))
        grid = semantics.centroid_grid(x, pop)
        expected = np.mean([semantics.sim(grid[(subject, session, 1)] - grid[(subject, session, 0)],
                                          np.array([1, 0, 0, 0, 0]))["cosine"]
                            for subject in ("a", "b", "c") for session in (1, 2)])
        self.assertAlmostEqual(observed, expected, places=6)

    def test_cached_transform_matches_v1_on_synthetic_data(self) -> None:
        rng = np.random.default_rng(91)
        populations = {}
        for role, members in (("TRAIN_GEOMETRY", [f"t{i}" for i in range(6)]),
                              ("CHECKPOINT_VALIDATION", ["v0", "v1"]),
                              ("OUTER_DEVELOPMENT", ["o0", "o1"])):
            features, labels, subjects, sessions = [], [], [], []
            for subject in members:
                baseline = rng.normal(size=8) * .2
                for session in (1, 2):
                    for label in (0, 1):
                        for _ in range(4):
                            features.append(rng.normal(size=8) * .3 + baseline +
                                            (session == 2) * .15 + label * np.array([1, .3, 0, 0, 0, 0, 0, 0]))
                            labels.append(label)
                            subjects.append(subject)
                            sessions.append(session)
            x = np.asarray(features, dtype=np.float32)
            pop = dg.Population(x, np.asarray(labels), np.asarray(subjects), np.asarray(sessions), role)
            populations[role] = (pop, x)
        q, _ = np.linalg.qr(rng.normal(size=(8, 2)))
        geometry = {"mu": np.zeros(8, dtype=np.float32), "q": q.astype(np.float32)}
        a = tv1.analyze_one(0, "embedding_64d", geometry, populations, (1, 2), "P", -1)
        b = tv2.analyze_one(0, "embedding_64d", geometry, populations, (1, 2), "P", -1)
        self.assertEqual(set(a), set(b))
        for name in a:
            self.assertEqual(len(a[name]), len(b[name]), name)
            for old, new in zip(a[name], b[name]):
                self.assertEqual(set(old), set(new), name)
                for key, value in old.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        self.assertAlmostEqual(float(value), float(new[key]), delta=5e-4,
                                               msg=f"{name} {key}")

    def test_unsupervised_rank_cv_is_class_label_invariant(self) -> None:
        rng = np.random.default_rng(104)
        means, centroids = {}, {}
        for subject in (f"t{i}" for i in range(10)):
            for session in (1, 2):
                center = rng.normal(size=8).astype(np.float32)
                means[(subject, session)] = center
                for label in (0, 1):
                    centroids[(subject, session, label)] = center + rng.normal(size=8).astype(np.float32)
        replaced = {key: rng.normal(size=8).astype(np.float32) for key in centroids}
        original = tv3.rank_cv_from_tables(means, centroids, (1, 2), "UNSUPERVISED")
        shuffled = tv3.rank_cv_from_tables(means, replaced, (1, 2), "UNSUPERVISED")
        self.assertEqual(original, shuffled)


if __name__ == "__main__":
    unittest.main()
