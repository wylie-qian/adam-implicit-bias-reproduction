"""Fixed four-rate audit: fixed epsilon, zero epsilon, and an INVALID scaling control.

Only the declared fixed matrix is run. Matching a plot with INVALID_wrong_scale
is not a reproduction fix: it changes epsilon in original gradient units.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import scipy
from scipy.special import expit, logsumexp

from conventions import validate_reference_data, wrong_scaled_epsilon_step
from run import array_sha, scaled_adam_step, sha, write_json

HERE = Path(__file__).resolve().parent
PRIOR = HERE / "results/2026-11-12-diagnosis/conventions"
EPS = 1e-8
LN10 = math.log(10.)
MODES = ("zero_epsilon", "fixed_epsilon", "INVALID_wrong_scale")
BETA1, BETA2 = .9, .99


def configurations():
    return [dict(run_id=f"a-{a:g}_{mode}", a=a, mode=mode, epsilon=0. if mode == "zero_epsilon" else EPS,
                 beta1=BETA1, beta2=BETA2, c=1., time_offset=1, bias_correction=False,
                 role="INVALID_RECURRENCE" if mode.startswith("INVALID") else "PUBLISHED_RECURRENCE")
            for a in [.3, .5, .7, 1.] for mode in MODES]


def checked_step(z, w, m, v, old_scale, eta, mode):
    """Use the frozen kernels; guard exceptional large-scale epsilon safely.

    The log-domain fallback is algebraically the same fixed-epsilon recurrence.
    It avoids constructing epsilon*exp(scale) when that number exceeds float64.
    Historical-moment overflow is detected rather than clipped or renormalized
    with an undocumented floor. Normal selected-data trajectories use the exact
    frozen kernels, including their floating-point operation order.
    """
    if mode not in MODES or eta <= 0 or not math.isfinite(eta):
        raise ValueError("Invalid mode or learning rate")
    if not all(np.isfinite(a).all() for a in (z, w, m, v)) or not math.isfinite(old_scale):
        raise FloatingPointError("Nonfinite input state")
    if np.any(v < 0):
        raise FloatingPointError("Negative second moment")
    if mode == "INVALID_wrong_scale":
        result = wrong_scaled_epsilon_step(z, w, m, v, old_scale, eta)
    elif mode == "zero_epsilon":
        result = scaled_adam_step(z, w, m, v, old_scale, BETA1, BETA2, eta, 0.)
    else:
        margins = z @ w
        scale = max(0., float(np.min(margins)))
        log_eps_scaled = math.log(EPS) + scale
        if log_eps_scaled < 700.:
            result = scaled_adam_step(z, w, m, v, old_scale, BETA1, BETA2, eta, EPS)
        else:
            g = -z.T @ np.exp(scale - np.logaddexp(0., margins)) / len(z)
            factor = 1. if not np.any(m) and not np.any(v) else math.exp(scale - old_scale)
            new_m = BETA1 * factor * m + (1 - BETA1) * g
            new_v = BETA2 * (factor * factor) * v + (1 - BETA2) * g * g
            if np.any(new_v <= 0) or not np.isfinite(new_v).all() or not np.isfinite(new_m).all():
                raise FloatingPointError("Unrepresentable rescaled moments")
            log_denom = np.logaddexp(.5 * np.log(new_v), log_eps_scaled)
            update = np.zeros_like(new_m)
            nonzero = new_m != 0
            update[nonzero] = np.sign(new_m[nonzero]) * np.exp(np.log(np.abs(new_m[nonzero])) - log_denom[nonzero])
            result = w - eta * update, new_m, new_v, scale
    if not all(np.isfinite(a).all() for a in result[:3]) or not math.isfinite(result[3]):
        raise FloatingPointError("Nonfinite updated state")
    if np.any(result[2] <= 0):
        raise FloatingPointError("Nonpositive second moment; no denominator floor applied")
    return result


def margin_metrics(z, w, ref):
    margins = z @ w
    raw = float(np.min(margins))
    loss_terms = np.logaddexp(0., -margins)
    logs = -margins.copy()
    mask = margins < 36.
    logs[mask] = np.log(loss_terms[mask])
    log_loss = float(logsumexp(logs) - math.log(len(z)))
    n2, ni = float(np.linalg.norm(w)), float(np.max(np.abs(w)))
    if not n2 or not ni:
        raise FloatingPointError("Undefined normalized margin at zero weights")
    q2, qi = raw / n2, raw / ni
    return dict(loss_float64=float(loss_terms.mean()), log_loss=log_loss, log10_loss=log_loss / LN10,
                margin_raw=raw, norm_l2=n2, norm_linf=ni, margin_l2=q2, margin_linf=qi,
                gap_l2=ref["gamma_l2"] - q2, gap_linf=ref["gamma_linf"] - qi,
                relative_gap_l2=1 - q2 / ref["gamma_l2"], relative_gap_linf=1 - qi / ref["gamma_linf"])


def epsilon_metrics(v, scale, mode, step):
    keys = ("log10_effective_epsilon", "log10_min_sqrt_v_over_actual_epsilon", "log10_max_sqrt_v_over_actual_epsilon",
            "log10_min_sqrt_v_over_nominal_epsilon", "log10_max_sqrt_v_over_nominal_epsilon",
            "minimum_fractional_step_reduction", "maximum_fractional_step_reduction")
    if not step or mode == "zero_epsilon":
        return dict.fromkeys(keys)
    log_actual = math.log(EPS) - (scale if mode == "INVALID_wrong_scale" else 0.)
    log_sqrt_raw = .5 * np.log(v) - scale
    actual_ratio = log_sqrt_raw - log_actual
    nominal_ratio = log_sqrt_raw - math.log(EPS)
    # Attenuation is epsilon/(sqrt(v_raw)+epsilon), comparing the same moments.
    attenuation = expit(-actual_ratio)
    return dict(log10_effective_epsilon=log_actual / LN10,
                log10_min_sqrt_v_over_actual_epsilon=float(actual_ratio.min() / LN10),
                log10_max_sqrt_v_over_actual_epsilon=float(actual_ratio.max() / LN10),
                log10_min_sqrt_v_over_nominal_epsilon=float(nominal_ratio.min() / LN10),
                log10_max_sqrt_v_over_nominal_epsilon=float(nominal_ratio.max() / LN10),
                minimum_fractional_step_reduction=float(attenuation.min()), maximum_fractional_step_reduction=float(attenuation.max()))


def observed_events(events, v, scale, mode, step):
    if mode == "zero_epsilon":
        return
    log_eps = math.log(EPS) - (scale if mode == "INVALID_wrong_scale" else 0.)
    ratio = .5 * np.log(v) - scale - log_eps
    for name, hit in [("first_any_actual_epsilon_exceeds_sqrt_v", ratio.min() < 0),
                      ("first_all_actual_epsilon_exceeds_sqrt_v", ratio.max() < 0),
                      ("first_all_sqrt_v_at_most_tenth_actual_epsilon", ratio.max() <= math.log(.1))]:
        if hit and events[name] is None:
            events[name] = step


def load_inputs(data_path, reference_path):
    data = json.loads(data_path.read_text())
    x, y, w0 = np.asarray(data["X"], dtype=np.float64), np.asarray(data["y"], dtype=data.get("y_dtype", "int64")), np.asarray(data["w0"], dtype=np.float64)
    if x.ndim != 2 or not len(x) or y.shape != (len(x),) or w0.shape != (x.shape[1],):
        raise ValueError("Input shapes do not match")
    if not np.isin(y, [-1, 1]).all() or not np.isfinite(x).all() or not np.isfinite(w0).all():
        raise ValueError("Inputs must be finite with binary labels")
    reference = json.loads(reference_path.read_text())
    validate_reference_data(reference, x, y, w0)
    return data, reference, y[:, None] * x, w0


def simulate(task):
    root, cfg, steps = task
    root = Path(root)
    _, wrapped, z, w = load_inputs(root / "data.json", root / "reference.json")
    ref = wrapped.get("reference", wrapped)
    m, v, scale = np.zeros_like(w), np.zeros_like(w), 0.
    checkpoints = {0, steps} | {int(i) for i in np.geomspace(1, steps, 64)}
    trace = []
    events = dict(first_any_actual_epsilon_exceeds_sqrt_v=None,
                  first_all_actual_epsilon_exceeds_sqrt_v=None,
                  first_all_sqrt_v_at_most_tenth_actual_epsilon=None)
    start, cpu, updates, status, error = time.perf_counter(), time.process_time(), 0, "completed", None

    def record():
        row = dict(step=updates, **margin_metrics(z, w, ref), **epsilon_metrics(v, scale, cfg["mode"], updates))
        row.update(weights=w.tolist(), first_moment_scaled=m.tolist(), second_moment_scaled=v.tolist(), log_scale_applied_on_last_update=scale)
        trace.append(row)

    record()
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            for t in range(steps):
                eta = (t + 1) ** (-cfg["a"])
                w, m, v, scale = checked_step(z, w, m, v, scale, eta, cfg["mode"])
                updates = t + 1
                observed_events(events, v, scale, cfg["mode"], updates)
                if updates in checkpoints:
                    record()
    except (ArithmeticError, FloatingPointError, OverflowError, ValueError) as exc:
        status, error = "numerical_failure", f"{type(exc).__name__}: {exc}"
    result = dict(config=cfg, status=status, error=error, requested_updates=steps, completed_updates=updates,
                  trace=trace, endpoint=trace[-1],
                  endpoint_state=None if status != "completed" else dict(weights=w.tolist(), first_moment_scaled=m.tolist(),
                  second_moment_scaled=v.tolist(), log_scale_applied_on_last_update=scale),
                  events=events, event_scope="Each update checked; step identifies the denominator used to produce that completed update, not a sampled crossing.",
                  wall_seconds=time.perf_counter() - start, cpu_seconds=time.process_time() - cpu)
    write_json(root / (cfg["run_id"] + ".json"), result)
    print(json.dumps(dict(run_id=cfg["run_id"], status=status, steps=updates,
                         gap_linf=result["endpoint"]["relative_gap_linf"])), flush=True)
    return result


def sources():
    return {p.name: sha(p) for p in [Path(__file__), HERE / "run.py", HERE / "conventions.py"]}


def run_study(data_path, reference_path, output, steps=1000000, jobs=3):
    if steps < 1 or not 1 <= jobs <= 3:
        raise ValueError("Positive steps and 1..3 jobs are required")
    data, reference, _, _ = load_inputs(data_path, reference_path)
    output.mkdir(parents=True, exist_ok=False)
    # Preserve input bytes and their original metadata, rather than regenerating samples.
    (output / "data.json").write_bytes(data_path.read_bytes())
    (output / "reference.json").write_bytes(reference_path.read_bytes())
    manifest = dict(schema_version=1, status="running", planned_configs=configurations(), steps_per_condition=steps,
                    fixed_before_this_run=True, study_type="Diagnostic extension informed by earlier a=.5 controls; not preregistration",
                    sources=sources(), input_sha256={"data.json": sha(data_path), "reference.json": sha(reference_path)},
                    environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, platform=platform.platform(),
                                     threads={k: os.environ.get(k) for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]}),
                    jobs=jobs, stopping_rule="Fixed requested updates or explicit numerical failure; no silent runtime truncation",
                    attribution="AI-assisted independent experiments; no author code. INVALID_wrong_scale is not fixed-epsilon Adam.")
    write_json(output / "manifest.json", manifest)
    tasks = [(str(output), cfg, steps) for cfg in configurations()]
    if jobs == 1:
        results = [simulate(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(simulate, tasks))
    manifest["status"] = "completed" if all(r["status"] == "completed" for r in results) else "incomplete"
    manifest["results"] = [dict(run_id=r["config"]["run_id"], status=r["status"], relative_gap_linf=r["endpoint"]["relative_gap_linf"]) for r in results]
    manifest["files"] = [{"path": p.name, "sha256": sha(p)} for p in sorted(output.glob("*.json")) if p.name != "manifest.json"]
    write_json(output / "manifest.json", manifest)
    if manifest["status"] != "completed":
        raise RuntimeError("Numerical failure: all planned outcomes retained")
    report = verify(output)
    write_json(output / "verification.json", report)
    return report


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "completed" or manifest["planned_configs"] != configurations():
        raise ValueError("Incomplete or altered fixed matrix")
    if manifest["sources"] != sources():
        raise ValueError("Source hash mismatch")
    expected_files = {"data.json", "reference.json"} | {c["run_id"] + ".json" for c in configurations()}
    if {f["path"] for f in manifest["files"]} != expected_files or len(manifest["files"]) != len(expected_files):
        raise ValueError("Evidence files differ from fixed matrix")
    for f in manifest["files"]:
        if sha(root / f["path"]) != f["sha256"]:
            raise ValueError("Evidence hash mismatch: " + f["path"])
    for name, digest in manifest["input_sha256"].items():
        if sha(root / name) != digest:
            raise ValueError("Input hash mismatch")
    _, wrapped, z, _ = load_inputs(root / "data.json", root / "reference.json")
    ref = wrapped.get("reference", wrapped)
    expected_steps = sorted({0, manifest["steps_per_condition"]} | {int(i) for i in np.geomspace(1, manifest["steps_per_condition"], 64)})
    count, max_error, runs = 0, 0., {}
    for cfg, summary in zip(configurations(), manifest["results"]):
        r = json.loads((root / (cfg["run_id"] + ".json")).read_text())
        if (r["config"] != cfg or r["status"] != "completed" or r["completed_updates"] != manifest["steps_per_condition"]
                or r["requested_updates"] != manifest["steps_per_condition"] or r["endpoint"] != r["trace"][-1]
                or summary != dict(run_id=cfg["run_id"], status=r["status"], relative_gap_linf=r["endpoint"]["relative_gap_linf"])
                or [row["step"] for row in r["trace"]] != expected_steps):
            raise ValueError("Run/config/checkpoint consistency failed")
        for row in r["trace"]:
            w, v, scale = np.asarray(row["weights"]), np.asarray(row["second_moment_scaled"]), row["log_scale_applied_on_last_update"]
            if any(np.asarray(row[key]).shape != (z.shape[1],) or not np.isfinite(row[key]).all()
                   for key in ["weights", "first_moment_scaled", "second_moment_scaled"]):
                raise ValueError("Invalid checkpoint state")
            recomputed = margin_metrics(z, w, ref)
            recomputed.update(epsilon_metrics(v, scale, cfg["mode"], row["step"]))
            for key, value in recomputed.items():
                if value is None:
                    if row[key] is not None:
                        raise ValueError("Epsilon diagnostic units differ")
                else:
                    error = abs(value - row[key]); max_error = max(max_error, error)
                    if not math.isfinite(value) or error > 2e-10:
                        raise ValueError("Checkpoint recomputation differs")
            count += 1
        state = r["endpoint_state"]
        if (state["weights"] != r["endpoint"]["weights"] or state["second_moment_scaled"] != r["endpoint"]["second_moment_scaled"]
                or state["first_moment_scaled"] != r["endpoint"]["first_moment_scaled"]):
            raise ValueError("Endpoint state differs from final checkpoint")
        runs[cfg["run_id"]] = r
    if len(runs) != 12 or len(manifest["results"]) != 12:
        raise ValueError("A planned outcome is missing")
    prior_checks = []
    # Cross-check all three a=.5 million-step outcomes against the existing study
    # when the same original data/reference bytes are used. Short smoke runs are
    # explicitly outside this cross-validation scope.
    if manifest["steps_per_condition"] == 1000000 and (PRIOR / "data.json").exists() and sha(root / "data.json") == sha(PRIOR / "data.json"):
        for mode, filename in [("zero_epsilon", "scaled_zero_epsilon.json"), ("fixed_epsilon", "correct_rescaled_eps1e-8.json"), ("INVALID_wrong_scale", "wrong_scaled_epsilon.json")]:
            old_path = PRIOR / filename
            old = json.loads(old_path.read_text())
            new = runs[f"a-0.5_{mode}"]
            difference = float(np.max(np.abs(np.asarray(old["endpoint_state"]["weights"]) - np.asarray(new["endpoint_state"]["weights"]))))
            if difference > 2e-9 or abs(old["endpoint"]["relative_gap_linf"] - new["endpoint"]["relative_gap_linf"]) > 2e-10:
                raise ValueError("Prior a=.5 endpoint cross-validation failed")
            prior_checks.append(dict(mode=mode, prior_file=filename, prior_sha256=sha(old_path), max_weight_difference=difference))
    return dict(passed=True, conditions_verified=12, updates_per_condition=manifest["steps_per_condition"],
                checkpoints_recomputed=count, maximum_metric_recomputation_error=max_error, prior_a05_checks=prior_checks,
                scope="Margins/loss/epsilon diagnostics recomputed from saved weights and second moments. First moments checked for shape/finiteness and endpoint consistency, not recomputed by replay. Exact event times recorded during execution; unsaved updates are not replayed.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["run", "verify"])
    ap.add_argument("--data", type=Path, default=PRIOR / "data.json")
    ap.add_argument("--reference", type=Path, default=PRIOR / "reference.json")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=1000000)
    ap.add_argument("--jobs", type=int, default=3)
    args = ap.parse_args()
    report = run_study(args.data, args.reference, args.output, args.steps, args.jobs) if args.command == "run" else verify(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
