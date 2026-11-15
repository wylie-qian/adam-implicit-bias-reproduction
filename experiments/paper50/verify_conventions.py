"""Regression checks for the convention audit, not tests that require agreement with a paper figure."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from conventions import (conditions, epsilon_units, literal_step, margin_metrics,
                         run_study, verify, wrong_scaled_epsilon_step)
from run import array_sha, references, scaled_adam_step, sha


class ConventionChecks(unittest.TestCase):
    def test_fixed_matrix_and_invalid_control_label(self):
        configs = conditions(50)
        self.assertEqual(len(configs), 10)
        self.assertEqual(len({c["name"] for c in configs}), 10)
        self.assertEqual([c["name"] for c in configs if c["role"] == "INVALID_RECURRENCE"],
                         ["wrong_scaled_epsilon"])
        for cfg in configs:
            self.assertEqual(cfg["a"], .5)
            self.assertEqual(cfg["c"], 1.)
            self.assertEqual(cfg["time_offset"], 1)

    def test_correct_epsilon_scaling_and_invalid_control(self):
        z, w = np.array([[1., 2.], [2., .5]]), np.array([10., 10.])
        m, v = np.array([-.2, -.3]), np.array([.08, .12])
        old_scale, shift, eta, eps = 0., 17., .1, .2
        base = scaled_adam_step(z, w, m, v, old_scale, .9, .99, eta, eps)
        moved = scaled_adam_step(z, w, m * np.exp(shift), v * np.exp(2 * shift),
                                 old_scale + shift, .9, .99, eta, eps)
        np.testing.assert_allclose(base[0], moved[0], atol=2e-15, rtol=2e-15)
        cfg = conditions(2)[0]
        zeros = np.zeros_like(w)
        direct, _, _ = literal_step(z, w, zeros, zeros, cfg, 0)
        correct = scaled_adam_step(z, w, zeros, zeros, 0., .9, .99, 1., cfg["epsilon"])
        wrong = wrong_scaled_epsilon_step(z, w, zeros, zeros, 0., 1., epsilon=cfg["epsilon"])
        np.testing.assert_allclose(correct[0], direct, atol=1e-13, rtol=1e-14)
        self.assertGreater(np.max(np.abs(wrong[0] - direct)), .5)
        # Leaving epsilon untransformed changes the original-scale regularizer.
        scale, M, V = 50., np.array([.2, -.8]), np.array([.3, .1])
        untransformed = M / (np.sqrt(V) + 1e-8)
        equivalent = (M * np.exp(-scale)) / (np.sqrt(V * np.exp(-2 * scale)) + 1e-8 * np.exp(-scale))
        np.testing.assert_allclose(untransformed, equivalent, atol=1e-15, rtol=1e-14)

    def test_sum_mean_epsilon_equivalence(self):
        rng = np.random.default_rng(17)
        z = rng.normal(size=(7, 4)); z[:, 0] = np.abs(z[:, 0]) + .5
        initial = rng.normal(size=4)
        mean_cfg = dict(conditions(7)[0], epsilon=1e-8 / 7)
        sum_cfg = next(c for c in conditions(7) if c["name"] == "sum_loss")
        w1, w2 = initial.copy(), initial.copy()
        m1, m2, v1, v2 = [np.zeros(4) for _ in range(4)]
        for t in range(3000):
            w1, m1, v1 = literal_step(z, w1, m1, v1, mean_cfg, t)
            w2, m2, v2 = literal_step(z, w2, m2, v2, sum_cfg, t)
        np.testing.assert_allclose(w1, w2, atol=2e-11, rtol=2e-12)
        np.testing.assert_allclose(m1, m2 / 7, atol=1e-20, rtol=2e-10)

    def test_float32_state_and_original_data_evaluation(self):
        cfg = next(c for c in conditions(2) if c["name"] == "float32")
        original = np.array([[1.00000003, 2.00000004], [2.00000003, .50000001]])
        z = original.astype(np.float32)
        w = np.array([.2, -.1], dtype=np.float32); m = np.zeros_like(w); v = np.zeros_like(w)
        for t in range(2000):
            w, m, v = literal_step(z, w, m, v, cfg, t)
            self.assertEqual(w.dtype, np.float32)
            self.assertEqual(m.dtype, np.float32)
            self.assertEqual(v.dtype, np.float32)
        ref = references(original)
        metric = margin_metrics(original, w, ref)
        expected = float(np.min(original @ w.astype(np.float64)) / np.max(np.abs(w.astype(np.float64))))
        self.assertEqual(metric["margin_linf"], expected)

    def test_epsilon_units_are_not_conflated(self):
        cfg = {c["name"]: c for c in conditions(50)}
        summed = epsilon_units(cfg["sum_loss"], 0.)
        self.assertAlmostEqual(summed["log10_outside_epsilon_in_mean_gradient_units"], np.log10(2e-10))
        inside = epsilon_units(cfg["epsilon_inside_sqrt"], 0.)
        self.assertIsNone(inside["log10_outside_epsilon_in_mean_gradient_units"])
        self.assertEqual(inside["log10_inside_epsilon_in_mean_squared_gradient_units"], -8.)
        self.assertEqual(inside["log10_denominator_floor_in_mean_gradient_units"], -4.)
        wrong = epsilon_units(cfg["wrong_scaled_epsilon"], 50.)
        self.assertAlmostEqual(wrong["log10_outside_epsilon_in_mean_gradient_units"], -8 - 50 / np.log(10))
        self.assertEqual(epsilon_units(cfg["correct_rescaled_eps1e-8"], 50.)["log10_outside_epsilon_in_mean_gradient_units"], -8.)

    def test_mismatched_reference_rejected_before_output_and_in_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            x, y, w0 = np.array([[1., 2.], [-2., -.5]]), np.array([1, -1]), np.array([.2, -.1])
            wrapper = {"reference": references(y[:, None] * x), "X_array_sha256": array_sha(x),
                       "y_array_sha256": array_sha(y), "w0_array_sha256": array_sha(w0)}
            reference = root / "reference.json"
            reference.write_text(json.dumps(wrapper))
            np.savez(root / "data.npz", X=x, y=y, w0=w0)
            changed = x.copy(); changed[0, 0] += .5
            np.savez(root / "wrong-data.npz", X=changed, y=y, w0=w0)
            with self.assertRaisesRegex(ValueError, "Reference/data identity mismatch"):
                run_study(root / "wrong-data.npz", reference, root / "invalid", 20, 1)
            self.assertFalse((root / "invalid").exists())
            with contextlib.redirect_stdout(io.StringIO()):
                run_study(root / "data.npz", reference, root / "result", 20, 1)
            # Even if a file hash is refreshed, a reference from other data must
            # not pass the semantic dataset-identity check.
            wrapper["X_array_sha256"] = array_sha(changed)
            saved_ref = root / "result/reference.json"
            saved_ref.write_text(json.dumps(wrapper))
            manifest_path = root / "result/manifest.json"
            manifest = json.loads(manifest_path.read_text())
            next(f for f in manifest["files"] if f["path"] == "reference.json")["sha256"] = sha(saved_ref)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Reference/data identity mismatch"):
                verify(root / "result")

    def test_evidence_integrity_exclusive_output_and_incomplete_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            x, y, w0 = np.array([[1., 2.], [-2., -.5]]), np.array([1, -1]), np.array([.2, -.1])
            np.savez(root / "data.npz", X=x, y=y, w0=w0)
            (root / "reference.json").write_text(json.dumps(references(y[:, None] * x)))
            with contextlib.redirect_stdout(io.StringIO()):
                report = run_study(root / "data.npz", root / "reference.json", root / "result", 20, 1)
            self.assertTrue(report["passed"])
            self.assertEqual(report["conditions_verified"], 10)
            self.assertLess(report["max_correct_scaled_vs_literal_endpoint_weight_difference"], 1e-12)
            with self.assertRaises(FileExistsError):
                run_study(root / "data.npz", root / "reference.json", root / "result", 20, 1)
            manifest_path = root / "result/manifest.json"
            original_manifest = manifest_path.read_bytes()
            manifest = json.loads(original_manifest)
            manifest["status"] = "incomplete"
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Incomplete"):
                verify(root / "result")
            manifest_path.write_bytes(original_manifest)
            path = root / "result/literal_baseline.json"
            path.write_bytes(path.read_bytes() + b"corruption")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify(root / "result")
            with self.assertRaises(ValueError):
                run_study(root / "data.npz", root / "reference.json", root / "invalid", 20, 4)
            self.assertFalse((root / "invalid").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
