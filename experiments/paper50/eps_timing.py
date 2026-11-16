"""When does epsilon take over in Adam? Follow-up study, 2026-11-13.

Not part of any earlier protocol: designed after the figure-legend rerun
(results/2026-11-13-legend) showed that Adam with epsilon = 1e-8 stops about
31.5% short of the L-infinity optimum. Settings are in protocol_eps_timing.json.

After every Adam update the script compares sqrt(v) at its real size with
epsilon and records the first update at which one coordinate, and then all 50
coordinates, fall below it ("takeover"), together with the margins at that
update. Runs with epsilon = 0 record the same for a list of thresholds; there
epsilon does not change the run, so they show where on Adam's own path a given
epsilon would start to matter.

The Adam update is run_matched.adam_step, unchanged (same rescaled arithmetic,
epsilon at its real size). run.py and run_matched.py are only imported.

Usage, from the repository root:
    python3 experiments/paper50/eps_timing.py --output experiments/paper50/results/2026-11-13-eps-timing --jobs 10
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as base  # noqa: E402  (frozen 2026-11-05 runner, read-only)
import run_matched as rm  # noqa: E402  (adam_step and loss divisor, read-only)

PROTOCOL = HERE / "protocol_eps_timing.json"
DATA = HERE / "results/2026-11-05/datasets"


def simulate(task):
    data_path, ref, config, p, out = task
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    base.write_json(out / "config.json", config)
    with np.load(data_path) as data:
        z = data["y"][:, None] * data["X"]
        w = data["w0"].copy()
    m, v, scale = np.zeros_like(w), np.zeros_like(w), 0.0
    epsilon, steps = config["epsilon"], config["steps"]
    divisor = rm.loss_divisor(p, z)
    thresholds = sorted([epsilon] if epsilon else p["tracked_thresholds"], reverse=True)
    log_thr = [math.log(x) for x in thresholds]
    next_first = next_all = 0
    events = []
    checkpoints = base.checkpoint_steps(steps, p["checkpoint_every"])
    rows = []
    started, cpu = time.perf_counter(), time.process_time()
    stop_reason, error, updates = "completed_fixed_steps", None, 0

    def diag(step):
        return base.diagnostics(z, w, ref, step, "adam", epsilon, m, v, scale,
                                time.perf_counter() - started, time.process_time() - cpu)

    rows.append(diag(0))
    try:
        for t in range(steps):
            eta = config["c"] * (t + 1) ** (-config["a"])
            w, m, v, scale = rm.adam_step(z, w, m, v, scale, p["beta1"], p["beta2"], eta, epsilon, divisor)
            updates = t + 1
            if next_all < len(thresholds):
                with np.errstate(divide="ignore"):
                    log_sqrt_v = 0.5 * np.log(v) - scale  # natural log of sqrt(v) at real size
                low, high = float(np.min(log_sqrt_v)), float(np.max(log_sqrt_v))
                while next_first < len(thresholds) and low < log_thr[next_first]:
                    events.append({"threshold": thresholds[next_first], "kind": "first", "row": diag(updates)})
                    next_first += 1
                while next_all < len(thresholds) and high < log_thr[next_all]:
                    events.append({"threshold": thresholds[next_all], "kind": "all", "row": diag(updates)})
                    next_all += 1
            if updates in checkpoints:
                rows.append(diag(updates))
    except (ArithmeticError, FloatingPointError, OverflowError) as exc:
        stop_reason, error = "numerical_failure", f"{type(exc).__name__}: {exc}"
    if rows[-1]["step"] != updates:
        rows.append(diag(updates))
    keep = config["keep_trajectory"]
    if keep:
        with gzip.open(out / "trajectory.csv.gz", "wt", newline="") as handle:
            writer = csv.DictWriter(handle, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    closest = min(rows[1:], key=lambda r: r["relative_gap_linf"])
    summary = {"config": config, "data_sha256": base.sha(data_path), "stop_reason": stop_reason, "error": error,
               "updates": updates, "wall_seconds": time.perf_counter() - started,
               "cpu_seconds": time.process_time() - cpu, "recorded_checkpoints": len(rows),
               "trajectory_kept": keep, "endpoint": rows[-1],
               "closest_recorded_linf": {"step": closest["step"], "relative_gap_linf": closest["relative_gap_linf"]},
               "thresholds": thresholds, "events": events}
    base.write_json(out / "summary.json", summary)
    return summary


def configs(p, manifest):
    selected = manifest["selected_seed"]
    a0, eps0 = p["primary_adam_a"], p["epsilon_main"]
    out = []
    for entry in manifest["datasets"]:
        seed = entry["seed"]
        for c in p["c_values"]:
            for eps in (0.0, eps0):
                out.append(("c", {"seed": seed, "a": a0, "c": c, "epsilon": eps}))
        for eps in p["epsilon_sweep"]:
            out.append(("epsilon", {"seed": seed, "a": a0, "c": 1.0, "epsilon": eps}))
        if seed == selected:
            for a in p["a_values_selected_seed"]:
                for c in p["c_values"]:
                    for eps in (0.0, eps0):
                        out.append(("a", {"seed": seed, "a": a, "c": c, "epsilon": eps}))
    for suite, cfg in out:
        cfg["keep_trajectory"] = suite == "c" and cfg["seed"] == selected
    return out


def flatten(suite, s):
    c, e = s["config"], s["endpoint"]
    row = {"suite": suite, "run_id": c["run_id"], "seed": c["seed"], "a": c["a"], "c": c["c"],
           "epsilon": c["epsilon"], "updates": s["updates"], "stop_reason": s["stop_reason"],
           "relative_gap_linf": e["relative_gap_linf"], "relative_gap_l2": e["relative_gap_l2"],
           "norm_linf": e["norm_linf"], "log_loss": e["log_loss"],
           "closest_recorded_linf": s["closest_recorded_linf"], "crossings": {}}
    for ev in s["events"]:
        r = ev["row"]
        row["crossings"].setdefault(f"{ev['threshold']:g}", {})[ev["kind"]] = {
            "step": r["step"], "relative_gap_linf": r["relative_gap_linf"], "relative_gap_l2": r["relative_gap_l2"],
            "norm_linf": r["norm_linf"], "log_loss": r["log_loss"]}
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", type=Path, default=PROTOCOL)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--steps", type=int, help="shorter runs for a quick test")
    args = ap.parse_args()
    p = json.loads(args.protocol.read_text())
    steps = args.steps or p["steps"]
    manifest = json.loads((DATA / "manifest.json").read_text())
    by_seed = {e["seed"]: e for e in manifest["datasets"]}
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = []
    for suite, cfg in configs(p, manifest):
        cfg["steps"] = steps
        cfg["run_id"] = f"seed-{cfg['seed']:02d}_a-{cfg['a']:g}_c-{cfg['c']:g}_eps-{cfg['epsilon']:g}"
        entry = by_seed[cfg["seed"]]
        path = DATA / f"seed-{cfg['seed']:02d}" / "data.npz"
        if base.sha(path) != entry["data_npz_sha256"]:
            raise ValueError("Dataset changed since preparation")
        tasks.append((suite, (str(path), entry["reference"], cfg, p, str(args.output / suite / cfg["run_id"]))))
    header = {"protocol": args.protocol.name, "protocol_sha256": base.sha(args.protocol),
              "runner_sha256": base.sha(Path(__file__)), "adam_step_source_sha256": base.sha(HERE / "run_matched.py"),
              "base_runner_sha256": base.sha(HERE / "run.py"),
              "dataset_manifest_sha256": base.sha(DATA / "manifest.json"), "selected_seed": manifest["selected_seed"],
              "steps_per_run": steps, "runs": len(tasks), "python": sys.version.split()[0], "numpy": np.__version__}
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(simulate, t): suite for suite, t in tasks}
        for f in as_completed(futs):
            s = f.result()
            results.append(s)
            print(json.dumps({"run": s["config"]["run_id"], "stop": s["stop_reason"],
                              "rel_linf": s["endpoint"]["relative_gap_linf"],
                              "events": len(s["events"])}), flush=True)
    endpoints = []
    for suite, t in tasks:
        endpoints.append(flatten(suite, json.loads((Path(t[4]) / "summary.json").read_text())))
    base.write_json(args.output / "endpoints.json", endpoints)
    header.update({"wall_seconds": time.perf_counter() - started,
                   "sum_worker_cpu_seconds": sum(s["cpu_seconds"] for s in results),
                   "numerical_failures": [s["config"]["run_id"] for s in results if s["stop_reason"] != "completed_fixed_steps"]})
    base.write_json(args.output / "manifest.json", header)
    print(json.dumps({"event": "done", "runs": len(results), "wall_seconds": header["wall_seconds"]}))


if __name__ == "__main__":
    main()
