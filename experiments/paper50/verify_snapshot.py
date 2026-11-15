"""Verify the compact scientific snapshot without rerunning a million updates."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.special import expit

from run import array_sha, sha

HERE = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(root):
    snapshot = json.loads((root / "snapshot.json").read_text())
    for entry in snapshot["files"]:
        path = root / entry["path"]
        require(path.is_file(), f"Missing snapshot file: {entry['path']}")
        require(path.stat().st_size == entry["bytes"], f"Size changed: {entry['path']}")
        require(sha(path) == entry["sha256"], f"Hash changed: {entry['path']}")
    require(sha(HERE / "run.py") == snapshot["experiment_source_sha256"], "Training source changed")
    require(sha(HERE / "protocol.json") == snapshot["original_protocol_sha256"], "Frozen protocol changed")
    dm = json.loads((root / "datasets/manifest.json").read_text())
    protocol = json.loads((root / "datasets/protocol.json").read_text())
    require(protocol == json.loads((HERE / "protocol.json").read_text()), "Protocol copies differ")
    datasets = {}
    residuals = []
    for entry in dm["datasets"]:
        seed, ref = entry["seed"], entry["reference"]
        path = root / f"datasets/seed-{seed:02d}/data.npz"
        require(sha(path) == entry["data_npz_sha256"], "Dataset hash mismatch")
        with np.load(path) as data:
            x, y, w0 = data["X"], data["y"], data["w0"]
        rng = np.random.Generator(np.random.PCG64(seed))
        require(np.array_equal(x, rng.standard_normal((50, 50))), "Feature generation differs")
        require(np.array_equal(y, rng.choice([-1, 1], 50)), "Label generation differs")
        require(np.array_equal(w0, np.random.Generator(np.random.PCG64(seed + 10000)).standard_normal(50)),
                "Initialization differs")
        for array, key in [(x, "X"), (y, "y"), (w0, "w0")]:
            require(array_sha(array) == entry[f"{key}_array_sha256"], "Array hash mismatch")
        z = y[:, None] * x
        u, w = np.array(ref["w_linf"]), np.array(ref["w_l2_svm"])
        gamma = ref["gamma_linf"]
        lp = ref["lp"]
        ineq = np.array(lp["inequality_marginals"])
        lower, upper = np.array(lp["lower_marginals"]), np.array(lp["upper_marginals"])
        objective = np.r_[np.zeros(50), -1.]
        constraint = np.c_[-z, np.ones(50)]
        stationarity = objective - constraint.T @ ineq - np.r_[lower + upper, 0.]
        alpha = np.array(ref["qp"]["alpha"])
        checked = [max(0., gamma - (z @ u).min(), np.abs(u).max() - 1),
                   abs(-gamma - (-lower.sum() + upper.sum())),
                   np.abs(stationarity).max(), max(0., ineq.max(), -lower.min(), upper.max()),
                   np.abs(ineq * (z @ u - gamma)).max(),
                   np.abs(lower * (u + 1)).max(), np.abs(upper * (1 - u)).max(),
                   max(0., 1 - (z @ w).min()), np.abs(w - z.T @ alpha).max(),
                   max(0., -alpha.min()), np.abs(alpha * (z @ w - 1)).max(),
                   abs(.5 * np.dot(w, w) - (alpha.sum() - .5 * np.linalg.norm(z.T @ alpha) ** 2))]
        require(max(checked) < protocol["reference_tolerance"], "Reference certificate failed")
        require(abs(ref["gamma_l2"] - 1 / np.linalg.norm(w)) < 1e-12, "L2 reference differs")
        separation = np.linalg.norm(w / np.linalg.norm(w) - u / np.linalg.norm(u))
        require(abs(separation - ref["direction_separation_l2"]) < 1e-12, "Selection geometry differs")
        require(np.all((-z.T @ expit(-(z @ w0)) / 50) != 0), "Zero initial gradient coordinate")
        residuals.extend(float(v) for v in checked)
        datasets[seed] = (z, entry)
    selected = max(datasets, key=lambda seed:(datasets[seed][1]["reference"]["direction_separation_l2"], -seed))
    require(selected == snapshot["selected_seed"] == dm["selected_seed"] == 2, "Selected seed differs")
    endpoint_count, trace_rows, recomputed_checkpoints, max_difference = 0, 0, 0, 0.
    traces = {}
    for run in snapshot["runs"]:
        suite, runid = run["suite"], run["run_id"]
        folder = root / suite / runid
        result = json.loads((folder / "summary.json").read_text())
        cfg, endpoint = result["config"], result["endpoint"]
        require(sha(folder / "config.json") == result["config_sha256"], "Config hash differs")
        require(result["updates"] == 1000000 and result["stop_reason"] == "completed_fixed_steps",
                "Unexpected run completion")
        z, entry = datasets[cfg["seed"]]
        ref = entry["reference"]
        state = json.loads((folder / "endpoint_state.json").read_text())
        weights = np.array(state["weights"])
        margin = float((z @ weights).min())
        q2, qi = margin / np.linalg.norm(weights), margin / np.abs(weights).max()
        for actual, key in [(margin, "margin_raw"), (q2, "margin_l2"), (qi, "margin_linf"),
                            (1 - q2 / ref["gamma_l2"], "relative_gap_l2"),
                            (1 - qi / ref["gamma_linf"], "relative_gap_linf")]:
            require(abs(actual - endpoint[key]) < 2e-10, "Endpoint metric differs")
        require(state["steps"] == result["updates"], "Endpoint step differs")
        if cfg["method"] == "adam":
            logsv = .5 * np.log(np.array(state["v_scaled"])) - state["log_scales"]
            require(abs(logsv.min() / math.log(10) - endpoint["log10_min_sqrt_v"]) < 1e-10,
                    "Moment scale differs")
            if cfg["epsilon"]:
                require(int(np.sum(logsv < math.log(cfg["epsilon"]))) == endpoint["epsilon_dominates_coordinates"],
                        "Epsilon dominance differs")
        endpoint_count += 1
        if not run["full_trajectory_retained"]:
            continue
        require(sha(folder / "trajectory.csv") == result["trajectory_sha256"], "Trace hash differs")
        with (folder / "trajectory.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        require(len(rows) == result["recorded_checkpoints"], "Checkpoint count differs")
        require(int(rows[-1]["step"]) == result["updates"], "Trace endpoint differs")
        traces[(suite, runid)] = rows
        trace_rows += len(rows)
        if not run["full_checkpoint_states_retained"]:
            continue
        require(sha(folder / "states.npz") == result["states_sha256"], "States hash differs")
        with np.load(folder / "states.npz") as states:
            require(np.array_equal(states["steps"], [int(r["step"]) for r in rows]), "State steps differ")
            w = states["weights"]
            raw = (w @ z.T).min(axis=1)
            n2, ni = np.linalg.norm(w, axis=1), np.abs(w).max(axis=1)
            for actual, key in [(raw, "margin_raw"), (raw / n2, "margin_l2"), (raw / ni, "margin_linf"),
                                (1 - raw / n2 / ref["gamma_l2"], "relative_gap_l2"),
                                (1 - raw / ni / ref["gamma_linf"], "relative_gap_linf")]:
                reported = np.array([float(r[key]) for r in rows])
                difference = float(np.max(np.abs(actual - reported)))
                max_difference = max(max_difference, difference)
                require(np.allclose(actual, reported, atol=2e-10, rtol=5e-12), "Checkpoint metric differs")
            recomputed_checkpoints += len(rows)
    for eps in (0., 1e-8):
        runid = f"seed-02_adam_eps-{eps:g}_a-0.7"
        one, two = traces[("primary", runid)], traces[("rates", runid)]
        require(len(one) == len(two), "Scheduled repeat lengths differ")
        for first, second in zip(one, two):
            for key in first:
                if key not in {"elapsed_cpu_seconds", "elapsed_wall_seconds"}:
                    require(first[key] == second[key], "Scheduled repeat measurements differ")
    require(endpoint_count == 50, "Formal run count differs")
    return {"snapshot_integrity_passed": True, "datasets_regenerated_exactly": len(datasets),
            "formal_endpoints_recomputed": endpoint_count, "retained_trace_checkpoints": trace_rows,
            "checkpoints_recomputed_from_full_states": recomputed_checkpoints,
            "max_absolute_recomputed_metric_difference": max_difference,
            "max_independently_checked_certificate_residual": max(residuals),
            "scheduled_repeat_measurements_equal": True,
            "scope": "Compact snapshot only; the earlier full 90-run audit is recorded separately"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot", type=Path, default=HERE / "results/2026-11-05")
    ap.add_argument("--plot-output", type=Path)
    args = ap.parse_args()
    report = verify(args.snapshot)
    if args.plot_output:
        from analyze import plot
        report["regenerated_figures"] = plot(args.snapshot, args.plot_output)
    print(json.dumps(report, indent=2, allow_nan=False))
