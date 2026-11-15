"""Numerical and evidence-integrity tests for the fixed twelve-condition audit."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.special import expit

from multirate_audit import (checked_step, configurations, epsilon_metrics, load_inputs,
                             observed_events, run_study, verify)
from run import array_sha, references, sha


class MultirateChecks(unittest.TestCase):
    def test_matrix_roles_and_rates(self):
        cfg = configurations()
        self.assertEqual(len(cfg), 12)
        self.assertEqual({c['a'] for c in cfg}, {.3, .5, .7, 1.})
        self.assertEqual(sum(c['role'] == 'INVALID_RECURRENCE' for c in cfg), 4)
        self.assertEqual(len({c['run_id'] for c in cfg}), 12)

    def test_frozen_updates_against_independent_direct_recurrence(self):
        z, initial = np.array([[1., 2.], [2., .5]]), np.array([.2, -.1])
        for a in [.3, .5, .7, 1.]:
            for mode, eps in [('fixed_epsilon', 1e-8), ('zero_epsilon', 0.)]:
                w, direct = initial.copy(), initial.copy()
                m, v, md, vd = [np.zeros(2) for _ in range(4)]
                scale = 0.
                for t in range(300):
                    eta = (t+1)**(-a)
                    g = -(z * expit(-(z @ direct))[:, None]).mean(axis=0)
                    md = .9*md + (1-.9)*g
                    vd = .99*vd + (1-.99)*g*g
                    direct -= eta*md/(np.sqrt(vd)+eps)
                    w, m, v, scale = checked_step(z, w, m, v, scale, eta, mode)
                np.testing.assert_allclose(w, direct, atol=2e-11, rtol=2e-12)

    def test_large_scale_log_safe_fixed_epsilon_and_invalid_distinction(self):
        z = np.eye(2); w = np.array([1000., 1000.]); zeros = np.zeros(2)
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            fixed = checked_step(z, w, zeros, zeros, 0., .2, 'fixed_epsilon')
            invalid = checked_step(z, w, zeros, zeros, 0., .2, 'INVALID_wrong_scale')
        np.testing.assert_array_equal(fixed[0], w)
        self.assertGreater(np.max(np.abs(invalid[0]-w)), .19)
        metrics = epsilon_metrics(invalid[2], invalid[3], 'INVALID_wrong_scale', 1)
        self.assertAlmostEqual(metrics['log10_effective_epsilon'], -8-1000/np.log(10))
        self.assertLess(metrics['maximum_fractional_step_reduction'], 1e-5)
        true = epsilon_metrics(fixed[2], fixed[3], 'fixed_epsilon', 1)
        self.assertEqual(true['minimum_fractional_step_reduction'], 1.)

    def test_nonfinite_and_invalid_states_fail_without_floor(self):
        z, w, zeros = np.eye(2), np.array([1.,1.]), np.zeros(2)
        with self.assertRaises(FloatingPointError):
            checked_step(z, np.array([np.inf,1.]), zeros, zeros, 0., .1, 'fixed_epsilon')
        with self.assertRaises(FloatingPointError):
            checked_step(z, w, zeros, np.array([-1.,0.]), 0., .1, 'fixed_epsilon')
        with self.assertRaises(ValueError):
            checked_step(z, w, zeros, zeros, 0., .1, 'unknown')
        with self.assertRaises(FloatingPointError):
            checked_step(np.zeros((2,2)), w, zeros, zeros, 0., .1, 'zero_epsilon')

    def test_event_units_and_timestamps(self):
        names = ['first_any_actual_epsilon_exceeds_sqrt_v','first_all_actual_epsilon_exceeds_sqrt_v',
                 'first_all_sqrt_v_at_most_tenth_actual_epsilon']
        events = dict.fromkeys(names)
        v = np.array([1e-18,1e-14])
        observed_events(events,v,0.,'fixed_epsilon',7)
        self.assertEqual(events[names[0]],7); self.assertIsNone(events[names[1]])
        observed_events(events,np.array([1e-20,1e-20]),0.,'fixed_epsilon',8)
        self.assertEqual(events[names[1]],8); self.assertEqual(events[names[2]],8)
        observed_events(events,np.array([1e-22,1e-22]),0.,'fixed_epsilon',9)
        self.assertEqual(events[names[0]],7)

    def test_exclusive_outputs_all_states_and_hash_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); x=np.array([[1.,2.],[-2.,-.5]]); y=np.array([1,-1]); w0=np.array([.2,-.1])
            data=root/'data.json'; ref=root/'reference.json'
            data.write_text(json.dumps(dict(X=x.tolist(),y=y.tolist(),y_dtype='int64',w0=w0.tolist())))
            wrapper=dict(reference=references(y[:,None]*x),X_array_sha256=array_sha(x),y_array_sha256=array_sha(y),w0_array_sha256=array_sha(w0))
            ref.write_text(json.dumps(wrapper))
            with contextlib.redirect_stdout(io.StringIO()):
                result=run_study(data,ref,root/'out',20,1)
            self.assertTrue(result['passed']); self.assertEqual(result['conditions_verified'],12)
            self.assertGreater(result['checkpoints_recomputed'],12)
            with self.assertRaises(FileExistsError): run_study(data,ref,root/'out',20,1)
            rowpath=root/'out/a-0.5_fixed_epsilon.json'
            rowpath.write_bytes(rowpath.read_bytes()+b'corrupted')
            with self.assertRaisesRegex(ValueError,'hash mismatch'): verify(root/'out')
            wrapper['X_array_sha256']='bad'; ref.write_text(json.dumps(wrapper))
            with self.assertRaisesRegex(ValueError,'identity mismatch'): run_study(data,ref,root/'invalid',20,1)
            self.assertFalse((root/'invalid').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
