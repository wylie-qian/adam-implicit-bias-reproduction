"""Source-based synthetic Adam replication; no author code or private data.

Commands: prepare --output NEWDIR; run --data DIR --output NEWDIR
All outputs are exclusive creates. See README for the provenance and limitations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog, minimize
from scipy.special import expit, logsumexp

HERE = Path(__file__).resolve().parent
DEFAULT_PROTOCOL = HERE / "protocol.json"
LN10 = math.log(10.0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(json.dumps(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def load_protocol(path):
    p = json.loads(Path(path).read_text())
    if (p["n"] <= 0 or p["d"] <= 0 or len(set(p["data_seeds"])) != len(p["data_seeds"])
            or not 0 < p["beta1"] <= p["beta2"] < 1
            or p["loss"] != "full_batch_mean_logistic" or p["intercept"]
            or p["bias_correction"] or p["weight_decay"] != 0
            or p["dtype"] != "float64" or p["adam_c"] <= 0
            or not 0 < p["primary_adam_a"] <= 1
            or p["checkpoint_every"] <= 0):
        raise ValueError("Unsupported or invalid protocol")
    return p


def environment():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=HERE, text=True).strip()
    return {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "cpu_count": os.cpu_count(),
            "git_sha": git("rev-parse", "HEAD"), "git_branch": git("branch", "--show-current"),
            "git_status": git("status", "--porcelain"), "source_sha256": sha(__file__),
            "thread_settings": {k: os.environ.get(k) for k in
                                ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}}


def logistic(z, w):
    margins = z @ w
    return float(np.mean(np.logaddexp(0.0, -margins))), -z.T @ expit(-margins) / len(z)


def references(z, tolerance=1e-7):
    """LP plus strongly convex hard-margin QP; certify with dual residuals."""
    n, d = z.shape
    objective = np.r_[np.zeros(d), -1.0]
    constraints = np.c_[-z, np.ones(n)]
    lp = linprog(objective, A_ub=constraints, b_ub=np.zeros(n),
                 bounds=[(-1.0, 1.0)] * d + [(None, None)], method="highs",
                 options={"dual_feasibility_tolerance": 1e-9,
                          "primal_feasibility_tolerance": 1e-9})
    if not lp.success or lp.x[-1] <= 0:
        raise ArithmeticError("Cannot certify positive Linfinity margin: " + lp.message)
    u = lp.x[:d]
    gamma = float(lp.x[-1])
    # SciPy marginals have the objective-derivative sign convention.
    lp_dual = float(-lp.lower.marginals[:d].sum() + lp.upper.marginals[:d].sum())
    stationarity = (objective - constraints.T @ lp.ineqlin.marginals
                    - lp.lower.marginals - lp.upper.marginals)
    lp_res = {
        "primal_violation": float(max(0.0, -np.min(z @ u - gamma), np.max(np.abs(u)) - 1)),
        "primal_dual_objective_gap": float(abs(lp.fun - lp_dual)),
        "stationarity_linf": float(np.max(np.abs(stationarity))),
        "dual_sign_violation": float(max(0.0, np.max(lp.ineqlin.marginals),
                                         -np.min(lp.lower.marginals), np.max(lp.upper.marginals))),
        "complementarity_linf": float(max(np.max(np.abs(lp.ineqlin.marginals * lp.ineqlin.residual)),
                                            np.max(np.abs(lp.lower.marginals[:d] * (u + 1))),
                                            np.max(np.abs(lp.upper.marginals[:d] * (1 - u))))),
    }
    qp = minimize(lambda w: 0.5 * np.dot(w, w), u / gamma,
                  jac=lambda w: w,
                  constraints=[{"type": "ineq", "fun": lambda w: z @ w - 1.0,
                                "jac": lambda w: z}], method="SLSQP",
                  options={"ftol": 1e-12, "maxiter": 2000})
    if not qp.success or not hasattr(qp, "multipliers"):
        raise ArithmeticError("L2 QP not certified: " + qp.message)
    w2 = qp.x
    alpha = qp.multipliers[:n]
    dual_value = float(alpha.sum() - 0.5 * np.linalg.norm(z.T @ alpha) ** 2)
    qp_res = {
        "primal_violation": float(max(0.0, -np.min(z @ w2 - 1))),
        "primal_dual_objective_gap": float(abs(qp.fun - dual_value)),
        "stationarity_linf": float(np.max(np.abs(w2 - z.T @ alpha))),
        "dual_sign_violation": float(max(0.0, -np.min(alpha))),
        "complementarity_linf": float(np.max(np.abs(alpha * (z @ w2 - 1)))),
    }
    if max(*lp_res.values(), *qp_res.values()) > tolerance:
        raise ArithmeticError(f"Reference residual exceeds {tolerance}: {lp_res}, {qp_res}")
    unit2 = w2 / np.linalg.norm(w2)
    unit_inf_l2 = u / np.linalg.norm(u)
    return {
        "gamma_linf": gamma, "gamma_l2": float(1 / np.linalg.norm(w2)),
        "w_linf": u.tolist(), "w_l2_svm": w2.tolist(),
        "direction_separation_l2": float(np.linalg.norm(unit2 - unit_inf_l2)),
        "q_linf_at_l2_reference": float(np.min(z @ unit2) / np.max(np.abs(unit2))),
        "q_l2_at_linf_reference": float(np.min(z @ unit_inf_l2)),
        "lp": {"method": "scipy.optimize.linprog(method=highs)", "status": int(lp.status),
               "message": lp.message, "primal_objective": float(lp.fun),
               "dual_objective": lp_dual, "residuals": lp_res,
               "inequality_marginals": lp.ineqlin.marginals.tolist(),
               "lower_marginals": lp.lower.marginals[:d].tolist(),
               "upper_marginals": lp.upper.marginals[:d].tolist()},
        "qp": {"method": "scipy.optimize.minimize(method=SLSQP)", "status": int(qp.status),
               "message": qp.message, "primal_objective": float(qp.fun),
               "dual_objective": dual_value, "residuals": qp_res, "alpha": alpha.tolist()},
        "direction_note": "distance is diagnostic to one LP optimum; uniqueness is not asserted",
    }


def prepare(protocol_path, output):
    p = load_protocol(protocol_path)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", p)
    catalog = []
    started = time.perf_counter()
    for seed in p["data_seeds"]:
        rng = np.random.Generator(np.random.PCG64(seed))
        x = rng.standard_normal((p["n"], p["d"]))
        y = rng.choice([-1, 1], p["n"])
        init_seed = seed + p["initialization_seed_offset"]
        w0 = np.random.Generator(np.random.PCG64(init_seed)).standard_normal(p["d"])
        z = y[:, None] * x
        ref = references(z, p["reference_tolerance"])
        g0 = logistic(z, w0)[1]
        if np.any(g0 == 0):
            raise ArithmeticError("Initial gradient has zero coordinate")
        folder = output / f"seed-{seed:02d}"
        folder.mkdir()
        np.savez_compressed(folder / "data.npz", X=x, y=y, w0=w0)
        np.savetxt(folder / "X.csv", x, delimiter=",", fmt="%.17g")
        np.savetxt(folder / "y.csv", y, delimiter=",", fmt="%d")
        np.savetxt(folder / "w0.csv", w0, delimiter=",", fmt="%.17g")
        spectral_l = float(np.linalg.norm(x, 2) ** 2 / (4 * p["n"]))
        entry = {"seed": seed, "initialization_seed": init_seed, "shape": list(x.shape),
                 "X_array_sha256": array_sha(x), "y_array_sha256": array_sha(y),
                 "w0_array_sha256": array_sha(w0), "data_npz_sha256": sha(folder / "data.npz"),
                 "spectral_hessian_bound_L": spectral_l, "gd_gdm_eta": 0.5 / spectral_l,
                 "min_initial_gradient_squared": float(np.min(g0 * g0)),
                 "reference": ref}
        write_json(folder / "reference.json", entry)
        catalog.append(entry)
    selected = max(catalog, key=lambda c: (c["reference"]["direction_separation_l2"], -c["seed"]))["seed"]
    manifest = {"protocol_sha256": sha(protocol_path), "saved_protocol_sha256": sha(output / "protocol.json"),
                "environment": environment(), "selected_seed": selected,
                "selection": p["selection"], "datasets": catalog,
                "wall_seconds": time.perf_counter() - started}
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"event": "prepared", "selected_seed": selected,
                      "geometry": [{"seed": c["seed"], "gamma_linf": c["reference"]["gamma_linf"],
                                    "gamma_l2": c["reference"]["gamma_l2"],
                                    "separation": c["reference"]["direction_separation_l2"]} for c in catalog]}), flush=True)


def scaled_adam_step(z, w, m, v, old_scale, beta1, beta2, eta, epsilon):
    """m,v are e^scale*m_raw and e^(2*scale)*v_raw; no denominator floor.

    The common scalar rescaling cancels exactly in m_raw/sqrt(v_raw).
    Positive epsilon becomes epsilon*e^scale, preserving its actual size.
    """
    margins = z @ w
    scale = max(0.0, float(np.min(margins)))
    h_scaled = np.exp(scale - np.logaddexp(0.0, margins))
    g_scaled = -z.T @ h_scaled / len(z)
    delta = scale - old_scale
    # Detect rather than silently clip an unrepresentable historical rescale.
    # The initial moments are exactly zero, so changing their representation
    # needs no exponential at all (important for a very large initial margin).
    factor = 1.0 if not np.any(v) and not np.any(m) else math.exp(delta)
    m = beta1 * factor * m + (1 - beta1) * g_scaled
    v = beta2 * (factor * factor) * v + (1 - beta2) * g_scaled * g_scaled
    denominator = np.sqrt(v)
    if epsilon:
        denominator = denominator + math.exp(math.log(epsilon) + scale)
    if (np.any(denominator <= 0) or not np.all(np.isfinite(denominator))
            or not np.all(np.isfinite(m))):
        raise FloatingPointError("Adam denominator/moments nonfinite or nonpositive; no clamp applied")
    new_w = w - eta * m / denominator
    if not np.all(np.isfinite(new_w)):
        raise FloatingPointError("Nonfinite updated weights")
    return new_w, m, v, scale


def diagnostics(z, w, ref, step, method, epsilon, m, v, scale, elapsed_wall, elapsed_cpu):
    margins = z @ w
    loss_terms = np.logaddexp(0.0, -margins)
    # log(softplus(-margin)): exp(-margin) asymptote is rounded correctly here;
    # using the asymptote beyond 36 avoids taking log(0) after raw loss underflow.
    logs = -margins.copy()
    mask = margins < 36.0
    logs[mask] = np.log(loss_terms[mask])
    log_loss = float(logsumexp(logs) - math.log(len(z)))
    norm2 = float(np.linalg.norm(w))
    norm_inf = float(np.max(np.abs(w)))
    raw = float(np.min(margins))
    q2, qinf = raw / norm2, raw / norm_inf
    u2 = np.array(ref["w_l2_svm"])
    uinf = np.array(ref["w_linf"])
    row = {"step": step, "method": method, "epsilon": epsilon,
           "loss_float64": float(np.mean(loss_terms)), "log_loss": log_loss,
           "margin_raw": raw, "norm_l2": norm2, "norm_linf": norm_inf,
           "margin_l2": q2, "margin_linf": qinf,
           "gap_l2": ref["gamma_l2"] - q2, "gap_linf": ref["gamma_linf"] - qinf,
           "relative_gap_l2": 1 - q2 / ref["gamma_l2"],
           "relative_gap_linf": 1 - qinf / ref["gamma_linf"],
           "distance_l2_unit_to_l2_ref": float(np.linalg.norm(w / norm2 - u2 / np.linalg.norm(u2))),
           "distance_l2_unit_to_linf_ref": float(np.linalg.norm(w / norm2 - uinf / np.linalg.norm(uinf))),
           "adam_moment_log_scale": scale if method == "adam" else None,
           "log10_min_sqrt_v": None, "log10_max_sqrt_v": None,
           "log10_min_sqrt_v_over_epsilon": None,
           "epsilon_dominates_coordinates": None,
           "elapsed_wall_seconds": elapsed_wall, "elapsed_cpu_seconds": elapsed_cpu}
    if method == "adam" and step:
        logsv = 0.5 * np.log(v) - scale
        row["log10_min_sqrt_v"] = float(np.min(logsv) / LN10)
        row["log10_max_sqrt_v"] = float(np.max(logsv) / LN10)
        if epsilon:
            ratio = logsv - math.log(epsilon)
            row["log10_min_sqrt_v_over_epsilon"] = float(np.min(ratio) / LN10)
            row["epsilon_dominates_coordinates"] = int(np.sum(ratio < 0))
    if not all(v is None or not isinstance(v, float) or math.isfinite(v) for v in row.values()):
        raise FloatingPointError("Nonfinite diagnostic")
    return row


def checkpoint_steps(steps, every):
    # The log-spaced portion permits log-log plots without outputting every update.
    return set(range(0, min(11, steps + 1))) | set(range(every, steps + 1, every)) | {
        int(x) for x in np.geomspace(1, max(1, steps), 250)} | {steps}


def simulate(data_path, ref, config, protocol, out):
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "config.json", config)
    with np.load(data_path) as data:
        z = data["y"][:, None] * data["X"]
        w = data["w0"].copy()
    m, v, scale = np.zeros_like(w), np.zeros_like(w), 0.0
    checkpoints = checkpoint_steps(config["steps"], protocol["checkpoint_every"])
    weights, moments_m, moments_v, scales, recorded_steps = [], [], [], [], []
    started, cpu = time.perf_counter(), time.process_time()
    stop_reason, error, updates = "completed_fixed_steps", None, 0
    method, epsilon = config["method"], config["epsilon"]
    first_epsilon_dominant_checkpoint = None
    first_all_epsilon_dominant_checkpoint = None
    rows = []
    with (out / "trajectory.csv").open("x", newline="") as handle:
        writer = None

        def record(step):
            nonlocal writer, first_epsilon_dominant_checkpoint, first_all_epsilon_dominant_checkpoint
            row = diagnostics(z, w, ref, step, method, epsilon, m, v, scale,
                              time.perf_counter() - started, time.process_time() - cpu)
            if writer is None:
                writer = csv.DictWriter(handle, list(row))
                writer.writeheader()
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            recorded_steps.append(step)
            weights.append(w.copy())
            moments_m.append(m.copy())
            moments_v.append(v.copy())
            scales.append(scale)
            count = row["epsilon_dominates_coordinates"]
            if count and first_epsilon_dominant_checkpoint is None:
                first_epsilon_dominant_checkpoint = step
            if count == len(w) and first_all_epsilon_dominant_checkpoint is None:
                first_all_epsilon_dominant_checkpoint = step

        record(0)
        try:
            for t in range(config["steps"]):
                eta = config["eta"] if method != "adam" else config["c"] * (t + 2) ** (-config["a"])
                if method == "adam":
                    w, m, v, scale = scaled_adam_step(z, w, m, v, scale, protocol["beta1"],
                                                     protocol["beta2"], eta, epsilon)
                else:
                    g = -z.T @ expit(-(z @ w)) / len(z)
                    if method == "gdm":
                        m = protocol["beta1"] * m + (1 - protocol["beta1"]) * g
                        g = m
                    w = w - eta * g
                updates = t + 1
                if updates in checkpoints:
                    record(updates)
                    if time.perf_counter() - started > protocol["wall_seconds_per_run"]:
                        stop_reason = "wall_budget"
                        break
                    if time.process_time() - cpu > protocol["cpu_seconds_per_run"]:
                        stop_reason = "cpu_budget"
                        break
        except (ArithmeticError, FloatingPointError, OverflowError) as exc:
            stop_reason, error = "numerical_failure", f"{type(exc).__name__}: {exc}"
        if recorded_steps[-1] != updates:
            record(updates)
    np.savez_compressed(out / "states.npz", steps=np.array(recorded_steps),
                        weights=np.stack(weights), m_scaled=np.stack(moments_m),
                        v_scaled=np.stack(moments_v), log_scales=np.array(scales))
    own = "l2" if method != "adam" else "linf"
    # Positive-epsilon Adam is checked against the finite Linfinity target only;
    # its asymptotic target is not asserted to be Linfinity.
    threshold = protocol["endpoint_relative_own_margin_gap_threshold"]
    endpoint = rows[-1]
    summary = {"config": config, "config_sha256": sha(out / "config.json"),
               "data_sha256": sha(data_path), "stop_reason": stop_reason, "error": error,
               "updates": updates, "requested_updates": config["steps"],
               "wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu,
               "recorded_checkpoints": len(rows), "endpoint": endpoint,
               "finite_criterion": {"norm": own, "relative_gap_threshold": threshold,
                                    "met": bool(0 <= endpoint[f"relative_gap_{own}"] <= threshold)},
               "first_epsilon_dominant_recorded_checkpoint": first_epsilon_dominant_checkpoint,
               "first_all_epsilon_dominant_recorded_checkpoint": first_all_epsilon_dominant_checkpoint,
               "trajectory_sha256": sha(out / "trajectory.csv"), "states_sha256": sha(out / "states.npz")}
    write_json(out / "summary.json", summary)
    return summary


def run_one(task):
    data_path, ref, config, protocol, out = task
    return simulate(Path(data_path), ref, config, protocol, Path(out))


def run_suite(data_dir, output, suite, steps, jobs):
    p = load_protocol(data_dir / "protocol.json")
    manifest = json.loads((data_dir / "manifest.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    configs = []
    selected = manifest["selected_seed"]
    for entry in manifest["datasets"]:
        seed = entry["seed"]
        if suite in {"pilot", "primary"}:
            for method, eps in [("gd", 0.0), ("gdm", 0.0), ("adam", 0.0), ("adam", 1e-8)]:
                configs.append({"seed": seed, "method": method, "epsilon": eps,
                                "eta": entry["gd_gdm_eta"], "a": p["primary_adam_a"],
                                "c": p["adam_c"], "steps": steps})
        elif seed == selected:
            values = [(a, eps) for a in p["rate_sweep_a"] for eps in p["epsilon_values"]] if suite == "rates" else [
                (p["primary_adam_a"], eps) for eps in p["epsilon_sweep"]]
            for a, eps in values:
                configs.append({"seed": seed, "method": "adam", "epsilon": eps,
                                "eta": None, "a": a, "c": p["adam_c"], "steps": steps})
    by_seed = {e["seed"]: e for e in manifest["datasets"]}
    tasks = []
    for i, cfg in enumerate(configs):
        entry = by_seed[cfg["seed"]]
        path = data_dir / f"seed-{cfg['seed']:02d}" / "data.npz"
        if sha(path) != entry["data_npz_sha256"]:
            raise ValueError("Dataset changed since preparation")
        cfg["run_id"] = f"seed-{cfg['seed']:02d}_{cfg['method']}_eps-{cfg['epsilon']:g}_a-{cfg['a']:g}"
        tasks.append((str(path), entry["reference"], cfg, p, str(output / cfg["run_id"])))
    header = {"suite": suite, "steps_per_run": steps, "runs": len(tasks), "selected_seed": selected,
              "dataset_manifest_sha256": sha(data_dir / "manifest.json"),
              "protocol_sha256": sha(data_dir / "protocol.json"), "environment": environment(),
              "jobs": jobs, "planned_configs": configs, "status": "running"}
    write_json(output / "manifest.json", header)
    started, completed = time.perf_counter(), []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(run_one, task) for task in tasks]
        for f in as_completed(futures):
            result = f.result()
            completed.append(result)
            e = result["endpoint"]
            print(json.dumps({"event": "run_completed", "run_id": result["config"]["run_id"],
                              "updates": result["updates"], "stop_reason": result["stop_reason"],
                              "qinf": e["margin_linf"], "q2": e["margin_l2"],
                              "relative_gap_linf": e["relative_gap_linf"],
                              "relative_gap_l2": e["relative_gap_l2"],
                              "wall_seconds": result["wall_seconds"]}), flush=True)
    completed.sort(key=lambda r: r["config"]["run_id"])
    header.update({"status": "completed", "wall_seconds": time.perf_counter() - started,
                   "sum_worker_cpu_seconds": sum(r["cpu_seconds"] for r in completed),
                   "results": completed})
    write_json(output / "manifest.json", header)
    print(json.dumps({"event": "suite_completed", "suite": suite,
                      "wall_seconds": header["wall_seconds"], "runs": len(completed)}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    prep.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--suite", choices=["pilot", "primary", "rates", "epsilon"], required=True)
    run.add_argument("--steps", type=int)
    run.add_argument("--jobs", type=int, default=3)
    args = ap.parse_args()
    if args.command == "prepare":
        prepare(args.protocol, args.output)
    else:
        p = load_protocol(args.data / "protocol.json")
        steps = args.steps if args.steps is not None else p["pilot_steps" if args.suite == "pilot" else "primary_steps"]
        if steps < 0 or args.jobs < 1:
            ap.error("steps must be nonnegative and jobs must be positive")
        run_suite(args.data, args.output, args.suite, steps, args.jobs)


if __name__ == "__main__":
    main()
