#!/usr/bin/env python3
'''Private synthetic learning experiment: GD vs full-batch Adam on a fixed 2D logistic problem.
Standard library only. Importable without running. Finite runs are descriptive, not proofs.'''
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time

X = ((1.0, 2.0), (-2.0, -0.5))
Y = (1, -1)
Z = tuple((y * a, y * b) for (a, b), y in zip(X, Y))
REF_L2 = (0.42857142857142855, 0.2857142857142857)
REF_LINF = (1.0, 1.0)
GAMMA_L2 = 7.0 / math.sqrt(13.0)
GAMMA_INF = 2.5
FIRST_STEPS = 10
FIELDS = ['method', 'step', 'loss', 'w1', 'w2', 'norm_l2', 'norm_linf', 'unit_l2_1', 'unit_l2_2',
          'unit_linf_1', 'unit_linf_2', 'margin_raw', 'margin_l2', 'margin_linf', 'gap_l2', 'gap_linf',
          'direction_error_l2_reference_l2', 'direction_error_l2_reference_linf', 'min_sqrt_v']


class NumericalStop(Exception):
    pass


def _finite(*xs):
    return all(math.isfinite(x) for x in xs)


def softplus_neg(s):
    '''Stable softplus(-s) = log(1 + exp(-s)).'''
    if s > 0:
        return math.log1p(math.exp(-s))
    return -s + math.log1p(math.exp(s))


def sigmoid_neg(s):
    '''Stable sigmoid(-s) = 1 / (1 + exp(s)).'''
    if s >= 0:
        e = math.exp(-s)
        return e / (1.0 + e)
    return 1.0 / (1.0 + math.exp(s))


def loss_grad(w, z):
    '''Mean logistic loss and gradient; pure, does not mutate inputs.'''
    n = len(z)
    loss, g0, g1 = 0.0, 0.0, 0.0
    for zi in z:
        s = zi[0] * w[0] + zi[1] * w[1]
        loss += softplus_neg(s)
        p = sigmoid_neg(s)
        g0 -= zi[0] * p
        g1 -= zi[1] * p
    return loss / n, [g0 / n, g1 / n]


def gd_step(w, g, eta):
    nw = [w[0] - eta * g[0], w[1] - eta * g[1]]
    if not _finite(*nw):
        raise NumericalStop('nonfinite_weights')
    return nw


def adam_step(w, m, v, g, eta, beta1, beta2):
    '''Adam without bias correction or epsilon; raises on zero sqrt(v) or nonfinite values.'''
    nm = [beta1 * m[i] + (1.0 - beta1) * g[i] for i in range(2)]
    nv = [beta2 * v[i] + (1.0 - beta2) * g[i] * g[i] for i in range(2)]
    if not _finite(*nm, *nv):
        raise NumericalStop('nonfinite_moments')
    r = [math.sqrt(x) for x in nv]
    if min(r) == 0.0:
        raise NumericalStop('zero_sqrt_v')
    nw = [w[i] - eta * nm[i] / r[i] for i in range(2)]
    if not _finite(*nw):
        raise NumericalStop('nonfinite_weights')
    return nw, nm, nv


def adam_lr(t, a):
    '''lr(t) = (t+2)^(-a), t = 0 for the first update.'''
    return (t + 2.0) ** (-a)


def direction_error(u, ref):
    rn = math.hypot(ref[0], ref[1])
    return math.hypot(u[0] - ref[0] / rn, u[1] - ref[1] / rn)


def metrics(w):
    loss = sum(softplus_neg(z[0] * w[0] + z[1] * w[1]) for z in Z) / len(Z)
    n2 = math.hypot(w[0], w[1])
    ni = max(abs(w[0]), abs(w[1]))
    raw = min(z[0] * w[0] + z[1] * w[1] for z in Z)
    d = {'loss': loss, 'w1': w[0], 'w2': w[1], 'norm_l2': n2, 'norm_linf': ni, 'margin_raw': raw}
    for k in FIELDS[7:18]:
        if k not in d:
            d[k] = None
    if n2 > 0.0:
        u = (w[0] / n2, w[1] / n2)
        d.update(unit_l2_1=u[0], unit_l2_2=u[1], margin_l2=raw / n2, gap_l2=GAMMA_L2 - raw / n2,
                 direction_error_l2_reference_l2=direction_error(u, REF_L2),
                 direction_error_l2_reference_linf=direction_error(u, REF_LINF))
    if ni > 0.0:
        d.update(unit_linf_1=w[0] / ni, unit_linf_2=w[1] / ni, margin_linf=raw / ni,
                 gap_linf=GAMMA_INF - raw / ni)
    return d


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def validate(p):
    def req(c, msg):
        if not c:
            raise ValueError('protocol invalid: ' + msg)
    req(isinstance(p, dict), 'not an object')
    req(p.get('kind') == 'synthetic_math_learning_experiment_not_course_submission', 'kind')
    req(_int(p.get('schema_version')) and p['schema_version'] == 1, 'schema_version')
    d = p.get('data')
    req(isinstance(d, dict) and isinstance(d.get('x'), list) and len(d['x']) == 2, 'data shape')
    req(all(isinstance(r, list) and len(r) == 2 and all(_num(c) for c in r) for r in d['x']), 'x 2D finite')
    req(d['x'] == [[1.0, 2.0], [-2.0, -0.5]] and d.get('y') == [1, -1]
        and all(_int(c) for c in d['y']), 'data must equal reference problem')
    iw = p.get('initial_weights')
    req(isinstance(iw, list) and len(iw) == 2 and all(_num(c) for c in iw) and iw == [0.0, 0.0], 'initial_weights')
    req(p.get('loss') == 'mean_logistic' and p.get('intercept') is False, 'loss/intercept')
    req(_int(p.get('seed')) and p['seed'] == 7, 'seed')
    for k in ('steps_per_method', 'trace_every'):
        req(_int(p.get(k)) and p[k] > 0, k + ' positive integer')
    for k in ('process_cpu_seconds', 'wall_seconds'):
        req(_num(p.get(k)) and p[k] >= 0, k + ' finite nonnegative')
    req(_int(p.get('also_record_first_steps')) and p['also_record_first_steps'] == FIRST_STEPS, 'first steps')
    req(p.get('gd') == {'learning_rate': 0.25} and _num(p['gd']['learning_rate']), 'gd')
    a = p.get('adam')
    req(isinstance(a, dict) and a == {'beta1': 0.9, 'beta2': 0.99, 'epsilon': 0.0, 'bias_correction': False,
        'weight_decay': 0.0, 'learning_rate_formula': '(t+2)^(-2/3)', 'a': 0.6666666666666666}
        and a['bias_correction'] is False and all(_num(a[k]) for k in ('beta1', 'beta2', 'epsilon', 'a')), 'adam')
    req(p.get('references') == {'linfinity_unit': [1.0, 1.0], 'gamma_infinity': 2.5, 'l2_svm': list(REF_L2)}, 'refs')
    req(p.get('endpoint_criteria') == {'l2_direction_distance_at_most': 0.001, 'own_margin_gap_at_most': 0.001}, 'crit')
    return p


def run_method(name, p, clock):
    w, m, v = [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]
    rows, stop, steps, gevals = [], 'completed_fixed_steps', 0, 0
    n, te, ad = p['steps_per_method'], p['trace_every'], p['adam']
    c0, w0 = time.process_time(), time.monotonic()

    def record(k):
        r = {'method': name, 'step': k}
        r.update(metrics(w))
        r['min_sqrt_v'] = math.sqrt(min(v)) if name == 'adam' else None
        rows.append(r)
    record(0)
    try:
        for t in range(n):
            if time.process_time() - clock[0] >= p['process_cpu_seconds']:
                stop = 'cpu_budget_exhausted'
                break
            if time.monotonic() - clock[1] >= p['wall_seconds']:
                stop = 'wall_budget_exhausted'
                break
            loss, g = loss_grad(w, Z)
            gevals += 1
            if not _finite(loss, *g):
                raise NumericalStop('nonfinite_loss_or_gradient')
            if g[0] == 0.0 and g[1] == 0.0:
                raise NumericalStop('underflowed_zero_gradient')
            if name == 'gd':
                w = gd_step(w, g, p['gd']['learning_rate'])
            else:
                w, m, v = adam_step(w, m, v, g, adam_lr(t, ad['a']), ad['beta1'], ad['beta2'])
            steps = t + 1
            if steps <= FIRST_STEPS or steps % te == 0 or steps == n:
                record(steps)
    except NumericalStop as e:
        stop = 'numerical_stop:' + str(e)
    if rows[-1]['step'] != steps:
        record(steps)
    end = dict(rows[-1])
    tol = p['endpoint_criteria']
    de = end['direction_error_l2_reference_l2' if name == 'gd' else 'direction_error_l2_reference_linf']
    gap = end['gap_l2' if name == 'gd' else 'gap_linf']
    crit = {'reference': 'l2_svm' if name == 'gd' else 'linfinity_unit',
            'direction_ok': de is not None and de <= tol['l2_direction_distance_at_most'],
            'own_margin_gap_ok': gap is not None and -1e-12 <= gap <= tol['own_margin_gap_at_most'],
            'note': 'descriptive finite-endpoint check only; no theorem is proved'}
    return rows, {'stop_reason': stop, 'updates': steps, 'gradient_evaluations': gevals,
                  'requested_updates': n, 'endpoint': end, 'endpoint_criteria': crit,
                  'cpu_seconds': time.process_time() - c0, 'wall_seconds': time.monotonic() - w0}


def assumptions(p):
    ad, lr = p['adam'], p['gd']['learning_rate']
    L = sum(z[0] ** 2 + z[1] ** 2 for z in Z) / (4.0 * len(Z))
    _, g0 = loss_grad([0.0, 0.0], Z)
    rng, fd = random.Random(p['seed']), 0.0
    for _ in range(3):
        w, h = [rng.uniform(-2, 2), rng.uniform(-2, 2)], 1e-6
        _, g = loss_grad(w, Z)
        for i in range(2):
            a, b = list(w), list(w)
            a[i] += h
            b[i] -= h
            fd = max(fd, abs((loss_grad(a, Z)[0] - loss_grad(b, Z)[0]) / (2 * h) - g[i]))
    wit = [z[0] + z[1] for z in Z]
    return {'separability_witness': [1.0, 1.0], 'witness_margins': wit, 'separable': min(wit) > 0,
            'g0': g0, 'g0_nonzero': all(g != 0.0 for g in g0),
            'beta_ordering_0_lt_b1_lt_b2_lt_1': 0 < ad['beta1'] < ad['beta2'] < 1,
            'beta1_squared_lt_beta2': ad['beta1'] ** 2 < ad['beta2'],
            'lr_family_diminishing_divergent_sum': 0 < ad['a'] <= 1,
            'hessian_trace_bound_L': L, 'L_equals_37_over_32': L == 37 / 32,
            'gd_step_le_1_over_L': lr <= 1.0 / L,
            'finite_difference_max_abs_error_seed_points': fd,
            'note': 'exact mathematical assumptions are not exhaustively proved by a finite run'}


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(FIELDS)
        for r in rows:
            wr.writerow(['' if r.get(k) is None else r[k] for k in FIELDS])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args(argv)
    if os.path.exists(args.output):
        print('error: output already exists', file=sys.stderr)
        return 2
    try:
        with open(args.protocol, 'rb') as f:
            raw = f.read()
        proto = validate(json.loads(raw.decode('utf-8')))
    except (OSError, ValueError, KeyError, TypeError) as e:
        print('error: ' + str(e), file=sys.stderr)
        return 2
    with open(os.path.abspath(__file__), 'rb') as f:
        src_sha = hashlib.sha256(f.read()).hexdigest()
    check = assumptions(proto)
    required = ('separable', 'g0_nonzero', 'beta_ordering_0_lt_b1_lt_b2_lt_1',
                'beta1_squared_lt_beta2', 'lr_family_diminishing_divergent_sum',
                'L_equals_37_over_32', 'gd_step_le_1_over_L')
    if not all(check[key] for key in required) or check['finite_difference_max_abs_error_seed_points'] > 1e-7:
        print('error: mathematical preflight failed', file=sys.stderr)
        return 2
    os.makedirs(args.output)
    clock = (time.process_time(), time.monotonic())
    gd_rows, gd_sum = run_method('gd', proto, clock)
    ad_rows, ad_sum = run_method('adam', proto, clock)
    write_csv(os.path.join(args.output, 'gd.csv'), gd_rows)
    write_csv(os.path.join(args.output, 'adam.csv'), ad_rows)
    summary = {'protocol': proto, 'protocolSHA256': hashlib.sha256(raw).hexdigest(),
               'sourceSHA256': src_sha, 'python_version': platform.python_version(),
               'seed': proto['seed'], 'seed_used_in_training': False,
               'timings': {'total_cpu_seconds': time.process_time() - clock[0],
                           'total_wall_seconds': time.monotonic() - clock[1]},
               'results': {'gd': gd_sum, 'adam': ad_sum}, 'assumptions_check': check,
               'notes': ['No external data or sources used.', 'Plots handled separately.',
                         'Criteria are descriptive; a false value means unmet at the finite endpoint.']}
    with open(os.path.join(args.output, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    return 0


if __name__ == '__main__':
    sys.exit(main())
