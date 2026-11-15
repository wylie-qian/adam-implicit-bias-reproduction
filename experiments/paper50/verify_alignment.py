"""Regression checks for Figure 1 alignment and epsilon diagnostics."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from alignment import conditions, epsilon_metrics, learning_rate, run, verify
from run import DEFAULT_PROTOCOL, prepare


class AlignmentChecks(unittest.TestCase):
    def test_explicit_figure1_rates_and_legacy_index(self):
        cfg = {c["run_id"]: c for c in conditions()}
        self.assertEqual(len(cfg), 9)
        for method in ["gd", "gdm"]:
            self.assertEqual(learning_rate(cfg[f"figure1_{method}"], 500), .1)
        for eps in ["0", "1e-08"]:
            c = cfg[f"figure1_adam_eps-{eps}"]
            self.assertEqual(learning_rate(c, 0), 1.)
            self.assertEqual(learning_rate(c, 3), .5)
            self.assertAlmostEqual(learning_rate(cfg[f"legacy_adam_eps-{eps}"], 0), 2 ** (-.7))
            self.assertAlmostEqual(learning_rate(cfg[f"index_control_adam_eps-{eps}"], 0), 2 ** (-.5))

    def test_epsilon_influence_is_quantitative_and_scale_invariant(self):
        epsilon = 1e-8
        # sqrt(v_raw)/eps = [10, 1, .05, .01]. Threshold 1 alone is not dominance.
        v = (epsilon * np.array([10., 1., .05, .01])) ** 2
        a = epsilon_metrics(v, 0., epsilon, 1)
        b = epsilon_metrics(v * np.exp(40.), 20., epsilon, 1)
        self.assertAlmostEqual(a["epsilon_update_suppression_min"], 1 / 11)
        self.assertAlmostEqual(a["epsilon_update_suppression_max"], 100 / 101)
        self.assertEqual(a["epsilon_strong_coordinates"], 2)
        for key in a:
            self.assertAlmostEqual(a[key], b[key], places=12)
        self.assertIsNone(epsilon_metrics(v, 0., 0., 1)["epsilon_update_suppression_max"])
        self.assertIsNone(epsilon_metrics(v, 0., epsilon, 0)["epsilon_update_suppression_max"])

    def test_saved_evidence_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # One deterministic dataset keeps this integration check small.
            p = json.loads(DEFAULT_PROTOCOL.read_text())
            p["data_seeds"] = [2]
            protocol = root / "protocol.json"
            protocol.write_text(json.dumps(p))
            prepare(protocol, root / "data")
            run(root / "data", root / "result", 30, 1)
            result = verify(root / "result")
            self.assertTrue(result["passed"])
            self.assertEqual(result["runs"], 9)
            self.assertLess(result["max_literal_weight_difference"], 1e-11)
            with self.assertRaises(FileExistsError):
                run(root / "data", root / "result", 30, 1)
            path = root / "result/figure1_adam_eps-0/trajectory.csv"
            path.write_bytes(path.read_bytes() + b"corruption")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify(root / "result")


if __name__ == "__main__":
    unittest.main(verbosity=2)
