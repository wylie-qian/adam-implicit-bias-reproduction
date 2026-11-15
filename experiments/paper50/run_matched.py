"""Reruns of the section 5 experiment with the paper's figure-legend settings.

Settings come from a protocol file (--protocol):
  * protocol_legend.json (2026-11-13, presented on the slides): averaged loss
    as in eq. 3.1, Adam learning rate t^-a with t = 1, 2, ... (c = 1, primary
    a = 0.5), GD / GDM learning rate 0.1, exactly as the legends write;
  * protocol_matched.json (2026-11-12, kept for backup B7): summed loss and
    c = 0.1, both chosen after the first run to follow the paper's curves.
Everything else (data, initial vectors, rescaled Adam arithmetic, diagnostics,
checkpoints, references) is reused from run.py unchanged; run.py itself is not
modified, so the frozen 2026-11-05 snapshot still verifies.

Usage, from the repository root:
    python3 experiments/paper50/run_matched.py --protocol experiments/paper50/protocol_legend.json \
        --output experiments/paper50/results/2026-11-13-legend --jobs 9
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.special import expit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as base  # noqa: E402  (frozen 2026-11-05 runner, reused read-only)

PROTOCOL = HERE / "protocol_matched.json"
DATA = HERE / "results/2026-11-05/datasets"
KEEP_TRAJECTORY_SEED = 2   # full checkpoint CSVs are kept for the selected seed only


def loss_divisor(p, z):
    """Gradient of the averaged loss divides by n; the summed loss does not."""
    return 1.0 if p["loss"] == "full_batch_summed_logistic" else float(len(z))


def adam_step(z, w, m, v, old_scale, beta1, beta2, eta, epsilon, divisor):
    """run.scaled_adam_step, with the loss divisor taken from the protocol."""
    margins = z @ w
    scale = max(0.0, float(np.min(margins)))
    h_scaled = np.exp(scale - np.logaddexp(0.0, margins))
    g_scaled = -z.T @ h_scaled / divisor
    delta = scale - old_scale
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


def simulate(task):
    data_path, ref, config, p, out = task
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    base.write_json(out / "config.json", config)
    with np.load(data_path) as data:
        z = data["y"][:, None] * data["X"]
        w = data["w0"].copy()
    m, v, scale = np.zeros_like(w), np.zeros_like(w), 0.0
    checkpoints = base.checkpoint_steps(config["steps"], p["checkpoint_every"])
    method, epsilon = config["method"], config["epsilon"]
    divisor = loss_divisor(p, z)
    keep = config["seed"] == KEEP_TRAJECTORY_SEED
    rows, first_dom, first_all = [], None, None
    started, cpu = time.perf_counter(), time.process_time()
    stop_reason, error, updates = "completed_fixed_steps", None, 0

    def record(step):
        nonlocal first_dom, first_all
        row = base.diagnostics(z, w, ref, step, method, epsilon, m, v, scale,
                               time.perf_counter() - started, time.process_time() - cpu)
        rows.append(row)
        count = row["epsilon_dominates_coordinates"]
        if count and first_dom is None:
            first_dom = step
        if count == len(w) and first_all is None:
            first_all = step

    record(0)
    try:
        for t in range(config["steps"]):
            if method == "adam":
                eta = config["c"] * (t + 1) ** (-config["a"])
                w, m, v, scale = adam_step(z, w, m, v, scale, p["beta1"], p["beta2"], eta, epsilon, divisor)
            else:
                g = -z.T @ expit(-(z @ w)) / divisor
                if method == "gdm":
                    m = p["beta1"] * m + (1 - p["beta1"]) * g
                    g = m
                w = w - config["eta"] * g
            updates = t + 1
            if updates in checkpoints:
                record(updates)
    except (ArithmeticError, FloatingPointError, OverflowError) as exc:
        stop_reason, error = "numerical_failure", f"{type(exc).__name__}: {exc}"
    if rows[-1]["step"] != updates:
        record(updates)
    if keep:
        with (out / "trajectory.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    base.write_json(out / "endpoint_state.json", {"step": updates, "w": w.tolist(), "m_scaled": m.tolist(),
                                                  "v_scaled": v.tolist(), "log_scale": scale})
    own = "l2" if method != "adam" else "linf"
    endpoint = rows[-1]
    summary = {"config": config, "data_sha256": base.sha(data_path), "stop_reason": stop_reason, "error": error,
               "updates": updates, "wall_seconds": time.perf_counter() - started,
               "cpu_seconds": time.process_time() - cpu, "recorded_checkpoints": len(rows),
               "trajectory_kept": keep, "endpoint": endpoint,
               "finite_criterion": {"norm": own, "relative_gap_threshold": p["endpoint_relative_own_margin_gap_threshold"],
                                    "met": bool(0 <= endpoint[f"relative_gap_{own}"] <= p["endpoint_relative_own_margin_gap_threshold"])},
               "first_epsilon_dominant_recorded_checkpoint": first_dom,
               "first_all_epsilon_dominant_recorded_checkpoint": first_all}
    if keep:
        summary["trajectory_sha256"] = base.sha(out / "trajectory.csv")
    base.write_json(out / "summary.json", summary)
    return summary


def configs(p, manifest):
    selected = manifest["selected_seed"]
    out = []
    for entry in manifest["datasets"]:
        seed = entry["seed"]
        for method, eps in [("gd", 0.0), ("gdm", 0.0), ("adam", 0.0), ("adam", 1e-8)]:
            out.append(("primary", {"seed": seed, "method": method, "epsilon": eps,
                                    "eta": p["gd_gdm_learning_rate"] if method != "adam" else None,
                                    "a": p["primary_adam_a"], "c": p["adam_c"]}))
        for a in p["rate_sweep_a"]:
            out.append(("order", {"seed": seed, "method": "adam", "epsilon": 0.0, "eta": None, "a": a, "c": p["adam_c"]}))
        if seed == selected:
            for a in p["rate_sweep_a"]:
                for eps in p["epsilon_values"]:
                    out.append(("rates", {"seed": seed, "method": "adam", "epsilon": eps, "eta": None, "a": a, "c": p["adam_c"]}))
            for eps in p["epsilon_sweep"]:
                out.append(("epsilon", {"seed": seed, "method": "adam", "epsilon": eps, "eta": None,
                                        "a": p["primary_adam_a"], "c": p["adam_c"]}))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", type=Path, default=PROTOCOL)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--steps", type=int)
    args = ap.parse_args()
    p = json.loads(args.protocol.read_text())
    steps = args.steps or p["primary_steps"]
    manifest = json.loads((DATA / "manifest.json").read_text())
    by_seed = {e["seed"]: e for e in manifest["datasets"]}
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = []
    for suite, cfg in configs(p, manifest):
        cfg["steps"] = steps
        cfg["run_id"] = f"seed-{cfg['seed']:02d}_{cfg['method']}_eps-{cfg['epsilon']:g}_a-{cfg['a']:g}"
        entry = by_seed[cfg["seed"]]
        path = DATA / f"seed-{cfg['seed']:02d}" / "data.npz"
        if base.sha(path) != entry["data_npz_sha256"]:
            raise ValueError("Dataset changed since preparation")
        tasks.append((str(path), entry["reference"], cfg, p, str(args.output / suite / cfg["run_id"])))
    header = {"protocol": str(args.protocol.name), "protocol_sha256": base.sha(args.protocol),
              "runner_sha256": base.sha(Path(__file__)), "base_runner_sha256": base.sha(HERE / "run.py"),
              "dataset_manifest_sha256": base.sha(DATA / "manifest.json"), "selected_seed": manifest["selected_seed"],
              "steps_per_run": steps, "runs": len(tasks), "python": sys.version.split()[0],
              "numpy": np.__version__}
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(simulate, t) for t in tasks]
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            print(json.dumps({"run": r["config"]["run_id"], "stop": r["stop_reason"],
                              "rel_linf": r["endpoint"]["relative_gap_linf"],
                              "rel_l2": r["endpoint"]["relative_gap_l2"]}), flush=True)
    endpoints = []
    for t in tasks:
        s = json.loads((Path(t[4]) / "summary.json").read_text())
        e, c = s["endpoint"], s["config"]
        endpoints.append({"suite": Path(t[4]).parent.name, "run_id": c["run_id"], "seed": c["seed"],
                          "method": c["method"], "epsilon": c["epsilon"], "a": c["a"], "c": c["c"], "eta": c["eta"],
                          "updates": s["updates"], "stop_reason": s["stop_reason"],
                          "margin_linf": e["margin_linf"], "margin_l2": e["margin_l2"],
                          "relative_gap_linf": e["relative_gap_linf"], "relative_gap_l2": e["relative_gap_l2"],
                          "first_epsilon_dominant_recorded_checkpoint": s["first_epsilon_dominant_recorded_checkpoint"],
                          "first_all_epsilon_dominant_recorded_checkpoint": s["first_all_epsilon_dominant_recorded_checkpoint"]})
    base.write_json(args.output / "endpoints.json", endpoints)
    header.update({"wall_seconds": time.perf_counter() - started,
                   "sum_worker_cpu_seconds": sum(r["cpu_seconds"] for r in results),
                   "numerical_failures": [r["config"]["run_id"] for r in results if r["stop_reason"] != "completed_fixed_steps"]})
    base.write_json(args.output / "manifest.json", header)
    print(json.dumps({"event": "done", "runs": len(results), "wall_seconds": header["wall_seconds"]}))


if __name__ == "__main__":
    main()
