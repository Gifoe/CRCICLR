from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import pda_core as c


def synthetic_rep() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    rows = []
    for subject in ("1", "2", "3"):
        for session in (0, 1):
            for i in range(12):
                rows.append((subject, session, i + session * 12, int(i % 2)))
    subjects = np.array([r[0] for r in rows], dtype="U")
    sessions = np.array([r[1] for r in rows], dtype=np.int64)
    indices = np.array([r[2] for r in rows], dtype=np.int64)
    labels = np.array([r[3] for r in rows], dtype=np.int64)
    features = rng.normal(size=(len(rows), 6))
    logits = np.column_stack([features[:, 0], -features[:, 0]])
    return {"subjects": subjects, "sessions": sessions, "indices": indices, "labels": labels, "features": features, "logits": logits}


class PDAIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rep = synthetic_rep()
        cls.basis = c.fit_shared_basis(cls.rep, rank=2, ridge=1e-2)

    def test_temporal_blocks_are_deterministic(self):
        a = c.make_transitions(self.rep)
        b = c.make_transitions(self.rep)
        self.assertEqual([[x.history_blocks[0]["indices"].tolist(), x.future["indices"].tolist()] for x in a], [[x.history_blocks[0]["indices"].tolist(), x.future["indices"].tolist()] for x in b])
        self.assertTrue(all(len(x.history_blocks) == 2 for x in a))

    def test_future_labels_never_enter_adapter_fit(self):
        tr = c.make_transitions(self.rep)[0]
        m1 = c.fit_subject_methods(tr, self.basis, 1e-2, .5, 1e-2)
        changed = c.SubjectTransition(tr.subject, tr.history_blocks, {**tr.future, "labels": 1 - tr.future["labels"]})
        m2 = c.fit_subject_methods(changed, self.basis, 1e-2, .5, 1e-2)
        np.testing.assert_allclose(m1["full_pda"]["a"], m2["full_pda"]["a"])
        np.testing.assert_allclose(m1["fisher_a"], m2["fisher_a"])

    def test_future_data_never_enters_fisher(self):
        tr = c.make_transitions(self.rep)[0]
        m1 = c.fit_subject_methods(tr, self.basis, 1e-2, .5, 1e-2)
        future = {**tr.future, "features": tr.future["features"] * 1000, "logits": tr.future["logits"] * 1000}
        m2 = c.fit_subject_methods(c.SubjectTransition(tr.subject, tr.history_blocks, future), self.basis, 1e-2, .5, 1e-2)
        np.testing.assert_allclose(m1["fisher_a"], m2["fisher_a"])
        np.testing.assert_allclose(m1["fisher_c"], m2["fisher_c"])

    def test_held_block_excluded_from_loo(self):
        tr = c.make_transitions(self.rep)[0]
        parts = [c.fit_block_adapter(x, self.basis, 1e-2) for x in tr.history_blocks]
        expected_a, expected_c, _, _ = c.precision_pool([parts[1]], 1e-2)
        fitted = c.fit_subject_methods(tr, self.basis, 1e-2, .5, 1e-2)
        np.testing.assert_allclose(fitted["loo"][0][0], expected_a)
        np.testing.assert_allclose(fitted["loo"][0][1], expected_c)

    def test_subject_ids_match_across_sessions(self):
        trs = c.make_transitions(self.rep)
        self.assertEqual([x.subject for x in trs], ["1", "2", "3"])
        self.assertTrue(all(set(x.history_blocks[0]["subjects"]) == {x.subject} and set(x.future["subjects"]) == {x.subject} for x in trs))

    def test_wrong_and_shuffled_controls_are_mismatched(self):
        wrong, shuffled = c.control_assignments(["1", "2", "3"], {"1": 1., "2": 1.1, "3": 3.})
        self.assertTrue(all(k != v for k, v in wrong.items()))
        self.assertTrue(all(k != v for k, v in shuffled.items()))

    def test_population_fallback_is_zero_and_frozen(self):
        p = c.unknown_subject_params(self.basis)
        self.assertTrue(np.all(p["a"] == 0) and np.all(p["c"] == 0))
        before = self.basis.U.copy()
        _ = c.fit_subject_methods(c.make_transitions(self.rep)[0], self.basis, 1e-2, .5, 1e-2)
        np.testing.assert_array_equal(before, self.basis.U)

    def test_components_sum_and_transients_center(self):
        session = np.array([[1., 2.], [3., 4.], [5., 8.]])
        persistent = np.array([2., 3.])
        centered, transient = c.center_transient_components(session, persistent)
        np.testing.assert_allclose(centered, persistent[None, :] + transient)
        np.testing.assert_allclose(transient.mean(axis=0), 0., atol=1e-12)

    def test_precision_pooling_is_deterministic_and_positive(self):
        tr = c.make_transitions(self.rep)[0]
        parts = [c.fit_block_adapter(x, self.basis, 1e-2) for x in tr.history_blocks]
        x = c.precision_pool(parts, 1e-2); y = c.precision_pool(parts, 1e-2)
        for a, b in zip(x, y): np.testing.assert_allclose(a, b)
        self.assertTrue(np.isfinite(x[2]).all() and (x[2] > 0).all())

    def test_adapter_codes_finite_noncollapsed(self):
        m = c.fit_subject_methods(c.make_transitions(self.rep)[0], self.basis, 1e-2, .5, 1e-2)
        self.assertTrue(np.isfinite(m["full_pda"]["a"]).all())
        self.assertGreater(m["persistent_norm"] + m["transient_norm"], 0.)

    def test_biological_subject_bootstrap_shape(self):
        values = np.arange(5, dtype=float)
        mean, lo, hi = c.bootstrap_ci(values, 3, draws=100)
        self.assertEqual(mean, 2.)
        self.assertLessEqual(lo, mean); self.assertGreaterEqual(hi, mean)

    def test_matched_population_checkpoint_identifier(self):
        self.assertEqual(c.population_checkpoint_id(self.rep), c.population_checkpoint_id(self.rep))

    def test_future_lock_fails_closed(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "lock.json"
            p.write_text(json.dumps({"status": "SEALED", "source_gate_pass": False}))
            with self.assertRaises(PermissionError): c.assert_future_resource_locked(p)

    def test_primary_code_contains_no_forbidden_method(self):
        source = (Path(__file__).resolve().parents[1] / "code" / "pda_core.py").read_text(encoding="utf-8").lower()
        for token in ("gradientreversal", "grl", "dann", "mmd", "coral", "bures transport", "identity suppression", "feature transport"):
            self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
