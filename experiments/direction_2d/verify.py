"""Independent gradient, recurrence, reference and failure-path checks."""
import copy
import csv
import importlib.util
import json
import math
from pathlib import Path
import random
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("learning_run", ROOT / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


class FormulaChecks(unittest.TestCase):
    def test_gradient_finite_difference(self):
        rng = random.Random(7)
        for w in [[0., 0.]] + [[rng.uniform(-5, 5), rng.uniform(-5, 5)] for _ in range(30)]:
            _, g = run.loss_grad(w, run.Z)
            def loss(v):
                return sum(math.log1p(math.exp(-sum(a*b for a,b in zip(z,v)))) for z in run.Z)/2
            for j in range(2):
                high, low = list(w), list(w)
                high[j] += 1e-5
                low[j] -= 1e-5
                self.assertAlmostEqual(g[j], (loss(high)-loss(low))/2e-5, delta=3e-9)

    def test_first_and_history_updates(self):
        w, g = [0., 0.], [-.75, -.625]
        self.assertEqual(run.loss_grad(w, run.Z), (math.log(2), g))
        self.assertEqual(run.gd_step(w, g, .25), [3/16, 5/32])
        eta = 2**(-2/3)
        actual, m, v = run.adam_step(w, [0.,0.], [0.,0.], g, eta, .9, .99)
        for x in actual:
            self.assertAlmostEqual(x, eta, delta=1e-14)
        actual2, _, _ = run.adam_step(actual, m, v, g, 3**(-2/3), .9, .99)
        for x in actual2:
            self.assertAlmostEqual(x, eta+3**(-2/3)*.19/math.sqrt(.0199), delta=1e-14)
        self.assertEqual(w, [0.,0.])

    def test_stability_and_reference_geometry(self):
        for w in ([1000.,1000.], [-1000.,-1000.]):
            loss, g = run.loss_grad(w, run.Z)
            self.assertTrue(all(math.isfinite(v) for v in [loss]+g))
        self.assertIsNone(run.metrics([0.,0.])["margin_linf"])
        m = run.metrics([3/7,2/7])
        self.assertAlmostEqual(m["gap_l2"], 0., delta=1e-14)
        self.assertAlmostEqual(m["gap_linf"], 1/6, delta=1e-14)
        self.assertEqual(run.metrics([1.,1.])["gap_linf"], 0.)
        self.assertEqual(run.metrics([1.,1.])["margin_raw"], 2.5)
        self.assertEqual(run.metrics([0.,0.])["margin_raw"], 0.)
        with self.assertRaises(run.NumericalStop):
            run.adam_step([0.,0.], [0.,0.], [0.,0.], [0.,0.], 1., .9, .99)


class BoundaryChecks(unittest.TestCase):
    def execute(self, p):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        td = Path(self.tmp.name)
        config, output = td/"protocol.json", td/"result"
        config.write_text(json.dumps(p))
        status = run.main(["--protocol",str(config),"--output",str(output)])
        return status, config, output

    def protocol(self):
        return json.loads((ROOT/"protocol.json").read_text())

    def test_zero_budget_and_output_protection(self):
        p = self.protocol()
        p["process_cpu_seconds"] = 0
        status, config, output = self.execute(p)
        self.assertEqual(status, 0)
        summary = json.loads((output/"summary.json").read_text())
        for name in ("gd","adam"):
            self.assertEqual(summary["results"][name]["updates"], 0)
            self.assertEqual(summary["results"][name]["stop_reason"], "cpu_budget_exhausted")
            with (output/(name+".csv")).open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 1)
        before = (output/"summary.json").read_bytes()
        self.assertEqual(run.main(["--protocol",str(config),"--output",str(output)]), 2)
        self.assertEqual(before, (output/"summary.json").read_bytes())

    def test_invalid_protocol_creates_no_output(self):
        p = self.protocol()
        p["adam"]["epsilon"] = 1e-8
        status, _, output = self.execute(p)
        self.assertEqual(status, 2)
        self.assertFalse(output.exists())

    def test_three_step_trace_independent_recurrence(self):
        p = self.protocol()
        p["steps_per_method"] = 3
        status, _, output = self.execute(p)
        self.assertEqual(status, 0)
        summary = json.loads((output/"summary.json").read_text())
        for name in ("gd","adam"):
            w, m, v = [0.,0.], [0.,0.], [0.,0.]
            for t in range(3):
                z = [[1.,2.],[2.,.5]]
                g = [-sum(row[j]/(1+math.exp(sum(a*b for a,b in zip(row,w)))) for row in z)/2
                     for j in range(2)]
                if name == "gd":
                    w = [w[j]-.25*g[j] for j in range(2)]
                else:
                    m = [.9*m[j]+.1*g[j] for j in range(2)]
                    v = [.99*v[j]+.01*g[j]**2 for j in range(2)]
                    w = [w[j]-(t+2)**(-2/3)*m[j]/math.sqrt(v[j]) for j in range(2)]
            actual = summary["results"][name]
            for k, value in zip(("w1","w2"),w):
                self.assertAlmostEqual(actual["endpoint"][k],value,delta=1e-13)
            self.assertEqual(actual["gradient_evaluations"],3)
            with (output/(name+".csv")).open() as f:
                rows=list(csv.DictReader(f))
            self.assertEqual([int(r["step"]) for r in rows],[0,1,2,3])
            self.assertEqual(float(rows[-1]["loss"]),actual["endpoint"]["loss"])
            for row in rows:
                w1, w2 = float(row["w1"]), float(row["w2"])
                self.assertEqual(float(row["margin_raw"]), min(w1+2*w2, 2*w1+.5*w2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
