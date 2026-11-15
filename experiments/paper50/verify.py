"""Independent numerical checks for the scientific experiment; standard unittest."""
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.special import expit

from run import DEFAULT_PROTOCOL, diagnostics, load_protocol, logistic, references, scaled_adam_step, simulate


class NumericalChecks(unittest.TestCase):
    def test_gradient_by_finite_differences_and_extreme_loss(self):
        rng = np.random.default_rng(742)
        z, w = rng.normal(size=(9, 5)), rng.normal(size=5)
        loss, gradient = logistic(z, w)
        h = 1e-5
        finite = np.array([(logistic(z, w + h * e)[0] - logistic(z, w - h * e)[0]) / (2 * h)
                           for e in np.eye(5)])
        np.testing.assert_allclose(gradient, finite, atol=2e-10, rtol=2e-9)
        self.assertTrue(loss > 0)
        extreme, g = logistic(np.array([[1., 0.], [0., 1.]]), np.array([-1000., 1000.]))
        self.assertEqual(extreme, 500.)
        self.assertTrue(np.all(np.isfinite(g)))

    def test_first_step_no_bias_correction(self):
        z = np.array([[1., 2.], [2., 0.5]])
        w = np.zeros(2)
        result = scaled_adam_step(z, w, w.copy(), w.copy(), 0., .9, .99, .4, 0.)
        np.testing.assert_allclose(result[0], [.4, .4], atol=1e-15)
        np.testing.assert_allclose(result[1], [-.075, -.0625], atol=1e-15)
        np.testing.assert_allclose(result[2], [.005625, .00390625], atol=1e-15)

    def test_rescaled_matches_literal_adam_with_and_without_epsilon(self):
        rng = np.random.default_rng(17)
        z = rng.normal(size=(7, 4))
        z[:, 0] = np.abs(z[:, 0]) + 0.5
        initial = rng.normal(size=4)
        for eps in [0., 1e-8, 1e-4]:
            w, wr = initial.copy(), initial.copy()
            m, v, mr, vr = [np.zeros(4) for _ in range(4)]
            scale = 0.
            for t in range(2000):
                eta = (t + 2) ** (-.7)
                # Literal recurrence, independently evaluated with expit.
                g = -(z * expit(-(z @ wr))[:, None]).mean(axis=0)
                mr = .9 * mr + .1 * g
                vr = .99 * vr + .01 * g ** 2
                wr -= eta * mr / (np.sqrt(vr) + eps)
                w, m, v, scale = scaled_adam_step(z, w, m, v, scale, .9, .99, eta, eps)
            np.testing.assert_allclose(w, wr, atol=2e-10, rtol=2e-11)
            np.testing.assert_allclose(m * np.exp(-scale), mr, atol=1e-20, rtol=2e-10)
            np.testing.assert_allclose(v * np.exp(-2 * scale), vr, atol=1e-28, rtol=2e-10)

    def test_rescaled_preserves_step_after_literal_gradient_underflow(self):
        z = np.eye(2)
        w = np.array([1000., 1000.])
        self.assertTrue(np.all(logistic(z, w)[1] == 0.))
        result = scaled_adam_step(z, w, np.zeros(2), np.zeros(2), 0., .9, .99, .2, 0.)
        np.testing.assert_allclose(result[0], [1000.2, 1000.2], atol=1e-12)

    def test_references_match_analytic_2d_solution_and_duality(self):
        z = np.array([[1., 2.], [2., .5]])
        ref = references(z)
        self.assertAlmostEqual(ref["gamma_linf"], 2.5, places=12)
        self.assertAlmostEqual(ref["gamma_l2"], 7 / math.sqrt(13), places=10)
        np.testing.assert_allclose(ref["w_l2_svm"], [3 / 7, 2 / 7], atol=1e-10)
        np.testing.assert_allclose(ref["w_linf"], [1., 1.], atol=1e-12)
        for kind in ("lp", "qp"):
            self.assertLess(max(ref[kind]["residuals"].values()), 1e-9)

    def test_exclusive_output_and_checkpoint_recomputability(self):
        p = load_protocol(DEFAULT_PROTOCOL)
        x, y, w0 = np.array([[1., 2.], [-2., -.5]]), np.array([1, -1]), np.array([.2, -.1])
        z = y[:, None] * x
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data.npz"
            np.savez(data, X=x, y=y, w0=w0)
            config = {"method": "adam", "epsilon": 1e-8, "steps": 4, "c": 1., "a": .7}
            ref = references(z)
            result = simulate(data, ref, config, p, root / "run")
            self.assertEqual(result["updates"], 4)
            with self.assertRaises(FileExistsError):
                simulate(data, ref, config, p, root / "run")
            with np.load(root / "run" / "states.npz") as saved:
                w = saved["weights"][-1]
                observed = np.min(z @ w) / np.max(np.abs(w))
                self.assertAlmostEqual(observed, result["endpoint"]["margin_linf"], places=14)
                self.assertEqual(len(saved["steps"]), 5)
            self.assertEqual(json.loads((root / "run" / "summary.json").read_text())["stop_reason"], "completed_fixed_steps")


if __name__ == "__main__":
    unittest.main(verbosity=2)
