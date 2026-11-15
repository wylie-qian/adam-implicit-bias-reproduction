"""Fixed implementation-convention diagnostics for the Adam reproduction.

The intentionally wrong scaling control is labeled INVALID_RECURRENCE. It is
included to detect misleading curve agreement, never as a proposed algorithm.
All metrics are evaluated on the original float64 data, including float32 runs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import scipy
from scipy.special import expit

from run import array_sha, scaled_adam_step, sha, write_json

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "results/2026-11-12-alignment/data.npz"
FROZEN_SOURCE_SHA = "cff71403539c260aa910a48a7332900c25d69b47be74ab464aac43d40c6fdc67"
LN10 = math.log(10.)


def conditions(n):
    """Prespecified diagnostic matrix; no outcome-dependent selection."""
    common = dict(beta1=.9, beta2=.99, epsilon=1e-8, bias_correction=False,
                  loss_multiplier=1., epsilon_placement="outside_sqrt", dtype="float64",
                  gradient="stable", arithmetic="literal", a=.5, c=1., time_offset=1,
                  role="CONVENTION_CONTROL")
    changes = [
        dict(name="literal_baseline", role="PAPER_BASELINE"),
        dict(name="bias_correction", bias_correction=True),
        dict(name="beta2_0999", beta2=.999),
        dict(name="sum_loss", loss_multiplier=float(n)),
        dict(name="epsilon_inside_sqrt", epsilon_placement="inside_sqrt"),
        dict(name="naive_1_minus_sigmoid", gradient="naive"),
        dict(name="wrong_scaled_epsilon", arithmetic="scaled_wrong_epsilon",
             role="INVALID_RECURRENCE"),
        dict(name="float32", dtype="float32"),
        dict(name="correct_rescaled_eps1e-8", arithmetic="scaled_correct", role="PAPER_BASELINE"),
        dict(name="scaled_zero_epsilon", arithmetic="scaled_correct", epsilon=0., role="THEORY_BASELINE"),
    ]
    return [dict(common, **change) for change in changes]


def wrong_scaled_epsilon_step(z, w, m, v, old_scale, eta, beta1=.9, beta2=.99, epsilon=1e-8):
    """INVALID_RECURRENCE: intentionally leave epsilon untransformed.

    This changes original-scale epsilon to epsilon*exp(-scale). It is a
    diagnostic counterexample, not an implementation of fixed-epsilon Adam.
    """
    margins = z @ w
    scale = max(0., float(np.min(margins)))
    g = -z.T @ np.exp(scale - np.logaddexp(0., margins)) / len(z)
    factor = math.exp(scale - old_scale) if np.any(m) or np.any(v) else 1.
    m = beta1 * factor * m + (1 - beta1) * g
    v = beta2 * factor ** 2 * v + (1 - beta2) * g * g
    return w - eta * m / (np.sqrt(v) + epsilon), m, v, scale


def literal_step(z, w, m, v, config, t):
    """Independent direct recurrence, with only the named convention changed."""
    margins = z @ w
    g = (z.T @ (expit(margins) - 1.) if config["gradient"] == "naive"
         else -z.T @ expit(-margins)) / len(z)
    g = g * config["loss_multiplier"]
    b1, b2 = config["beta1"], config["beta2"]
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    mm = m / (1 - b1 ** (t + 1)) if config["bias_correction"] else m
    vv = v / (1 - b2 ** (t + 1)) if config["bias_correction"] else v
    denominator = (np.sqrt(vv + config["epsilon"]) if config["epsilon_placement"] == "inside_sqrt"
                   else np.sqrt(vv) + config["epsilon"])
    eta = config["c"] * (t + config["time_offset"]) ** (-config["a"])
    return w - eta * mm / denominator, m, v


def margin_metrics(z_original, w, ref):
    # Conversion is intentional: all runs use the same original evaluation data.
    w = np.asarray(w, dtype=np.float64)
    raw = float(np.min(z_original @ w))
    n2, ni = float(np.linalg.norm(w)), float(np.max(np.abs(w)))
    if not n2 or not ni:
        raise FloatingPointError("Zero weight norm makes normalized margins undefined")
    q2, qi = raw / n2, raw / ni
    return dict(margin_raw=raw, norm_l2=n2, norm_linf=ni, margin_l2=q2, margin_linf=qi,
                gap_l2=ref["gamma_l2"] - q2, gap_linf=ref["gamma_linf"] - qi,
                relative_gap_l2=1 - q2 / ref["gamma_l2"], relative_gap_linf=1 - qi / ref["gamma_linf"])


def epsilon_units(config, scale):
    """Keep additive epsilon and squared-gradient epsilon in different fields."""
    eps, multiplier = config["epsilon"], config["loss_multiplier"]
    outside, inside = None, None
    if eps:
        if config["epsilon_placement"] == "inside_sqrt":
            inside = math.log10(eps) - 2 * math.log10(multiplier)
        else:
            outside = math.log10(eps) - math.log10(multiplier)
            if config["arithmetic"] == "scaled_wrong_epsilon":
                outside -= scale / LN10
    return dict(log10_outside_epsilon_in_mean_gradient_units=outside,
                log10_inside_epsilon_in_mean_squared_gradient_units=inside,
                log10_denominator_floor_in_mean_gradient_units=inside / 2 if inside is not None else outside)



def validate_reference_data(reference, x, y, w0):
    """Reject accidental mixing of a wrapped reference and another dataset.

    Bare analytic references remain supported for small formula tests. Dataset
    wrappers must retain all three original array hashes, including initialization.
    """
    arrays = [("X", x), ("y", y), ("w0", w0)]
    keys = [name + "_array_sha256" for name, _ in arrays]
    if "reference" not in reference and not any(k in reference for k in keys):
        return
    if not all(k in reference for k in keys):
        raise ValueError("Wrapped reference lacks complete dataset identity hashes")
    for name, array in arrays:
        if array_sha(array) != reference[name + "_array_sha256"]:
            raise ValueError("Reference/data identity mismatch: " + name)


def simulate(task):
    output, config, steps, wall_limit = task
    output = Path(output)
    data = json.loads((output / "data.json").read_text())
    reference = json.loads((output / "reference.json").read_text())
    ref = reference.get("reference", reference)
    x = np.asarray(data["X"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=np.int64)
    z_original = y[:, None] * x
    dtype = np.dtype(config["dtype"])
    z = z_original.astype(dtype)
    w = np.asarray(data["w0"], dtype=dtype)
    m, v, scale = np.zeros_like(w), np.zeros_like(w), 0.
    checkpoints = {0, steps} | {int(i) for i in np.geomspace(1, steps, 64)}
    records = []
    start, cpu = time.perf_counter(), time.process_time()
    status, error, updates = "completed", None, 0

    def record():
        if not all(a.dtype == dtype for a in (w, m, v)):
            raise TypeError("Unexpected optimizer state dtype")
        if not all(np.isfinite(a).all() for a in (w, m, v)):
            raise FloatingPointError("Nonfinite optimizer state")
        metrics = margin_metrics(z_original, w, ref)
        metrics.update(epsilon_units(config, scale))
        records.append(dict(step=updates, **metrics))

    record()
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            for t in range(steps):
                eta = config["c"] * (t + config["time_offset"]) ** (-config["a"])
                if config["arithmetic"] == "scaled_correct":
                    w, m, v, scale = scaled_adam_step(z, w, m, v, scale,
                                                     config["beta1"], config["beta2"], eta, config["epsilon"])
                elif config["arithmetic"] == "scaled_wrong_epsilon":
                    w, m, v, scale = wrong_scaled_epsilon_step(z, w, m, v, scale, eta,
                                                              config["beta1"], config["beta2"], config["epsilon"])
                else:
                    w, m, v = literal_step(z, w, m, v, config, t)
                updates = t + 1
                if updates in checkpoints:
                    record()
                    if time.perf_counter() - start > wall_limit or time.process_time() - cpu > wall_limit:
                        status = "budget_exceeded"
                        break
    except (ArithmeticError, FloatingPointError, OverflowError, TypeError) as exc:
        status, error = "numerical_failure", f"{type(exc).__name__}: {exc}"
    # On failure, retain the last successfully checked observation. Do not turn
    # an invalid terminal state into a purported completed endpoint.
    if status == "completed" and records[-1]["step"] != updates:
        record()
    state = None if status == "numerical_failure" else dict(
        weights=w.tolist(), first_moment=m.tolist(), second_moment=v.tolist(),
        log_scale_applied_on_last_update=scale, state_dtype=str(dtype),
        moment_representation="exp(scale)*m_raw, exp(2*scale)*v_raw" if config["arithmetic"].startswith("scaled") else "raw",
    )
    result = dict(config=config, requested_updates=steps, completed_updates=updates,
                  status=status, error=error, state_dtype=str(dtype), endpoint=records[-1],
                  endpoint_state=state, trace=records, wall_seconds=time.perf_counter() - start,
                  cpu_seconds=time.process_time() - cpu)
    write_json(output / (config["name"] + ".json"), result)
    print(json.dumps(dict(condition=config["name"], role=config["role"], status=status,
                          updates=updates, relative_gap_linf=records[-1]["relative_gap_linf"])), flush=True)
    return result


def run_study(data_path, reference_path, output, steps=1000000, jobs=3):
    if steps < 1 or not 1 <= jobs <= 3:
        raise ValueError("steps must be positive and jobs must be between 1 and 3")
    if sha(HERE / "run.py") != FROZEN_SOURCE_SHA:
        raise ValueError("Frozen Adam implementation changed; re-audit before running")
    with np.load(data_path, allow_pickle=False) as data:
        x, y, w0 = data["X"].copy(), data["y"].copy(), data["w0"].copy()
    if (x.ndim != 2 or y.shape != (len(x),) or w0.shape != (x.shape[1],)
            or not np.all(np.isin(y, [-1, 1])) or not np.isfinite(x).all()
            or not np.isfinite(w0).all() or x.dtype != np.float64 or w0.dtype != np.float64):
        raise ValueError("Expected finite float64 X/w0 and binary labels with matching dimensions")
    reference = json.loads(reference_path.read_text())
    validate_reference_data(reference, x, y, w0)
    configs = conditions(len(x))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "data.json", dict(X=x.tolist(), y=y.tolist(), y_dtype=str(y.dtype), w0=w0.tolist()))
    write_json(output / "reference.json", reference)
    try:
        base_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        base_commit = None
    manifest = dict(schema_version=1, purpose="Diagnostic convention audit, including an explicitly invalid recurrence control",
                    status="running", steps_per_condition=steps, jobs=jobs, wall_cpu_limit_per_condition=600.,
                    planned_conditions=configs, evaluation_data="original float64 X and y for every condition",
                    sources={"conventions.py": sha(__file__), "run.py": FROZEN_SOURCE_SHA},
                    inputs=dict(data_sha256=sha(data_path), reference_sha256=sha(reference_path),
                                X_array_sha256=array_sha(x), y_array_sha256=array_sha(y), w0_array_sha256=array_sha(w0)),
                    environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                                     platform=platform.platform(), base_git_commit=base_commit),
                    provenance="Independent AI-assisted diagnostics of the published formulas; no author code. INVALID_RECURRENCE is not a reproduction fix.")
    write_json(output / "manifest.json", manifest)
    tasks = [(str(output), cfg, steps, 600.) for cfg in configs]
    if jobs == 1:
        results = [simulate(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(simulate, tasks))
    manifest["status"] = "completed" if all(r["status"] == "completed" for r in results) else "incomplete"
    manifest["results"] = [dict(name=r["config"]["name"], status=r["status"], endpoint=r["endpoint"], state_dtype=r["state_dtype"]) for r in results]
    manifest["files"] = [{"path": p.name, "sha256": sha(p)} for p in sorted(output.glob("*.json")) if p.name != "manifest.json"]
    write_json(output / "manifest.json", manifest)
    if manifest["status"] != "completed":
        raise RuntimeError("Incomplete diagnostics; partial records preserved")
    report = verify(output)
    write_json(output / "verification.json", report)
    return report


def verify(output):
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Incomplete study")
    if manifest["sources"] != {"conventions.py": sha(__file__), "run.py": sha(HERE / "run.py")}:
        raise ValueError("Scientific source hash mismatch")
    expected_names = {"data.json", "reference.json"} | {c["name"] + ".json" for c in manifest["planned_conditions"]}
    if {f["path"] for f in manifest["files"]} != expected_names or len(manifest["files"]) != len(expected_names):
        raise ValueError("Evidence file matrix differs")
    for f in manifest["files"]:
        if sha(output / f["path"]) != f["sha256"]:
            raise ValueError("Evidence hash mismatch: " + f["path"])
    data = json.loads((output / "data.json").read_text())
    x, y, w0 = np.asarray(data["X"], dtype=np.float64), np.asarray(data["y"], dtype=data.get("y_dtype", "int64")), np.asarray(data["w0"], dtype=np.float64)
    for a, k in [(x, "X"), (y, "y"), (w0, "w0")]:
        if array_sha(a) != manifest["inputs"][k + "_array_sha256"]:
            raise ValueError("Original evaluation data changed")
    z = y[:, None] * x
    reference = json.loads((output / "reference.json").read_text())
    validate_reference_data(reference, x, y, w0)
    ref = reference.get("reference", reference)
    configs = conditions(len(x))
    if manifest["planned_conditions"] != configs or [r["name"] for r in manifest["results"]] != [c["name"] for c in configs]:
        raise ValueError("Fixed condition matrix differs")
    results, max_error = {}, 0.
    for cfg, summary in zip(configs, manifest["results"]):
        r = json.loads((output / (cfg["name"] + ".json")).read_text())
        if (r["config"] != cfg or r["status"] != "completed" or r["completed_updates"] != manifest["steps_per_condition"]
                or r["requested_updates"] != manifest["steps_per_condition"] or r["endpoint"]["step"] != r["completed_updates"]
                or r["endpoint"] != r["trace"][-1] or summary["endpoint"] != r["endpoint"] or summary["status"] != r["status"]):
            raise ValueError("Incomplete or inconsistent condition: " + cfg["name"])
        state = r["endpoint_state"]
        if state["state_dtype"] != cfg["dtype"] or r["state_dtype"] != cfg["dtype"] or summary["state_dtype"] != cfg["dtype"]:
            raise ValueError("State dtype metadata differs")
        for k in ["weights", "first_moment", "second_moment"]:
            a = np.asarray(state[k], dtype=np.float64)
            if a.shape != w0.shape or not np.isfinite(a).all():
                raise ValueError("Invalid endpoint state")
            if cfg["dtype"] == "float32" and not np.array_equal(a, a.astype(np.float32).astype(np.float64)):
                raise ValueError("Float32 state is not float32-representable")
        metrics = margin_metrics(z, state["weights"], ref)
        metrics.update(epsilon_units(cfg, state["log_scale_applied_on_last_update"]))
        for k, value in metrics.items():
            reported = r["endpoint"][k]
            if value is None:
                if reported is not None:
                    raise ValueError("Epsilon units differ")
            else:
                error = abs(value - reported)
                max_error = max(error, max_error)
                if error > 2e-11:
                    raise ValueError("Endpoint metric differs")
        results[cfg["name"]] = r
    direct, stable = results["literal_baseline"], results["correct_rescaled_eps1e-8"]
    delta = float(np.max(np.abs(np.asarray(direct["endpoint_state"]["weights"]) - np.asarray(stable["endpoint_state"]["weights"]))))
    if delta > 2e-9:
        raise ValueError("Correct rescaling disagrees with literal baseline")
    if [r["step"] for r in direct["trace"]] != [r["step"] for r in stable["trace"]]:
        raise ValueError("Comparison checkpoints differ")
    for a, b in zip(direct["trace"], stable["trace"]):
        if abs(a["margin_linf"] - b["margin_linf"]) > 2e-9:
            raise ValueError("Correct rescaling trace disagrees with literal baseline")
    return dict(passed=True, conditions_verified=len(results), updates_per_condition=manifest["steps_per_condition"],
                endpoint_metrics_recomputed=len(results), max_endpoint_recomputation_error=max_error,
                max_correct_scaled_vs_literal_endpoint_weight_difference=delta,
                trace_metric_comparisons=len(direct["trace"]),
                scope="Endpoint states independently recomputed; scalar traces hashed and paired baseline metrics compared. Intermediate states are not retained/recomputed.",
                warning="INVALID_RECURRENCE is an intentionally wrong control, not fixed-epsilon Adam.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["run", "verify"])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--reference", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=1000000)
    ap.add_argument("--jobs", type=int, default=3)
    args = ap.parse_args()
    if args.command == "run":
        if args.steps < 1 or not 1 <= args.jobs <= 3:
            ap.error("steps must be positive and jobs must be between 1 and 3")
        report = run_study(args.data, args.reference or args.data.with_name("reference.json"), args.output, args.steps, args.jobs)
    else:
        report = verify(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
