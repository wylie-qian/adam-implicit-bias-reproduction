"""Figure 1 setting audit, preserving the frozen 2026-11-05 experiment.

Run: python3 experiments/paper50/alignment.py run --data NEW_DATA --output NEW_DIR
Verify saved checkpoints: ... alignment.py verify --output NEW_DIR
No author code is used. See ALIGNMENT.md for provenance and interpretation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import math
import multiprocessing
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.special import expit

from run import (HERE, array_sha, checkpoint_steps, diagnostics, environment,
                 load_protocol, scaled_adam_step, sha, write_json)

BASE_COMMIT = "cd22acd6d8e6f6fc97266a752a7f6b20112a4251"
FROZEN_SOURCE_SHA = "cff71403539c260aa910a48a7332900c25d69b47be74ab464aac43d40c6fdc67"


def conditions():
    """Fixed comparisons; neither seeds nor configurations depend on outcomes."""
    result = []
    for name, a, offset in [("legacy", .7, 2), ("figure1", .5, 1), ("index_control", .5, 2)]:
        for epsilon in [0., 1e-8]:
            result.append(dict(run_id=f"{name}_adam_eps-{epsilon:g}", method="adam",
                               epsilon=epsilon, a=a, c=1., offset=offset, arithmetic="scaled"))
    for method in ["gd", "gdm"]:
        result.append(dict(run_id=f"figure1_{method}", method=method, epsilon=0., eta=.1,
                           arithmetic="literal"))
    literal = dict(next(c for c in result if c["run_id"] == "figure1_adam_eps-1e-08"))
    literal.update(run_id="literal_adam_eps-1e-08", arithmetic="literal")
    return result + [literal]


def learning_rate(config, t):
    if config["method"] == "adam":
        return config["c"] * (t + config["offset"]) ** (-config["a"])
    return config["eta"]


def epsilon_metrics(v, scale, epsilon, step):
    """Measure the fraction removed from each coordinate's epsilon=0 update.

    epsilon/(sqrt(v_raw)+epsilon) is exact for a shared moment state. It is
    not a comparison between moments on two different optimization trajectories.
    """
    if not epsilon or not step:
        return {"log10_max_sqrt_v_over_epsilon": None,
                "epsilon_update_suppression_min": None,
                "epsilon_update_suppression_max": None,
                "epsilon_strong_coordinates": None}
    log_ratio = .5 * np.log(v) - scale - math.log(epsilon)
    suppression = expit(-log_ratio)
    return {"log10_max_sqrt_v_over_epsilon": float(log_ratio.max() / math.log(10)),
            "epsilon_update_suppression_min": float(suppression.min()),
            "epsilon_update_suppression_max": float(suppression.max()),
            "epsilon_strong_coordinates": int(np.sum(log_ratio <= math.log(.1)))}


def simulate_condition(task):
    root, entry, protocol, config, steps = task
    root = Path(root)
    folder = root / config["run_id"]
    folder.mkdir(exist_ok=False)
    write_json(folder / "config.json", config)
    with np.load(root / "data.npz") as data:
        z, w = data["y"][:, None] * data["X"], data["w0"].copy()
    m, v, scale = np.zeros_like(w), np.zeros_like(w), 0.
    checkpoints = checkpoint_steps(steps, 20000)
    rows, weights, moments_m, moments_v, scales, saved_steps = [], [], [], [], [], []
    start, cpu = time.perf_counter(), time.process_time()
    updates, stop_reason, error = 0, "completed_fixed_steps", None
    beta1, beta2 = protocol["beta1"], protocol["beta2"]

    def record():
        row = diagnostics(z, w, entry["reference"], updates, config["method"],
                          config["epsilon"], m, v, scale,
                          time.perf_counter() - start, time.process_time() - cpu)
        row.update(epsilon_metrics(v, scale, config["epsilon"], updates))
        rows.append(row)
        saved_steps.append(updates)
        weights.append(w.copy())
        moments_m.append(m.copy())
        moments_v.append(v.copy())
        scales.append(scale)

    record()
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            for t in range(steps):
                eta = learning_rate(config, t)
                if config["method"] == "adam" and config["arithmetic"] == "scaled":
                    w, m, v, scale = scaled_adam_step(z, w, m, v, scale, beta1, beta2,
                                                     eta, config["epsilon"])
                else:
                    # Independent literal reference: no common moment/gradient scaling.
                    gradient = -np.mean(z * expit(-(z @ w))[:, None], axis=0)
                    g = gradient
                    if config["method"] in {"adam", "gdm"}:
                        m = beta1 * m + (1 - beta1) * g
                        g = m
                    if config["method"] == "adam":
                        v = beta2 * v + (1 - beta2) * gradient ** 2
                        g = m / (np.sqrt(v) + config["epsilon"])
                    w = w - eta * g
                updates = t + 1
                if updates in checkpoints:
                    record()
                    if time.perf_counter() - start > protocol["wall_seconds_per_run"]:
                        stop_reason = "wall_budget"
                        break
                    if time.process_time() - cpu > protocol["cpu_seconds_per_run"]:
                        stop_reason = "cpu_budget"
                        break
    except (ArithmeticError, FloatingPointError, OverflowError) as exc:
        stop_reason, error = "numerical_failure", f"{type(exc).__name__}: {exc}"
    if saved_steps[-1] != updates:
        record()
    with (folder / "trajectory.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(folder / "states.npz", steps=saved_steps, weights=weights,
                        m_scaled=moments_m, v_scaled=moments_v, log_scales=scales)
    first = lambda key, value: next((r["step"] for r in rows if r[key] == value), None)
    result = {"config": config, "requested_updates": steps, "updates": updates,
              "stop_reason": stop_reason, "error": error, "endpoint": rows[-1],
              "first_all_epsilon_below_recorded_checkpoint": first("epsilon_dominates_coordinates", len(w)),
              "first_all_epsilon_strong_recorded_checkpoint": first("epsilon_strong_coordinates", len(w)),
              "wall_seconds": time.perf_counter() - start,
              "cpu_seconds": time.process_time() - cpu,
              "hashes": {name: sha(folder / name) for name in
                         ["config.json", "trajectory.csv", "states.npz"]}}
    write_json(folder / "summary.json", result)
    print(json.dumps({"run": config["run_id"], "updates": updates, "stop": stop_reason,
                      "relative_gap_linf": rows[-1]["relative_gap_linf"]}), flush=True)
    return result


def run(data, output, steps, jobs):
    if steps < 1 or jobs < 1:
        raise ValueError("steps and jobs must be positive")
    protocol = load_protocol(data / "protocol.json")
    manifest = json.loads((data / "manifest.json").read_text())
    selected = manifest["selected_seed"]
    entry = next(e for e in manifest["datasets"] if e["seed"] == selected)
    source_data = data / f"seed-{selected:02d}" / "data.npz"
    if sha(source_data) != entry["data_npz_sha256"]:
        raise ValueError("Dataset changed since preparation")
    if sha(HERE / "run.py") != FROZEN_SOURCE_SHA:
        raise ValueError("Frozen update implementation changed; re-audit before running")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source_data, output / "data.npz")
    write_json(output / "reference.json", entry)
    write_json(output / "protocol.json", protocol)
    configs = conditions()
    header = {"schema_version": 1, "status": "running", "base_commit": BASE_COMMIT,
              "selected_seed": selected, "selection": manifest["selection"],
              "steps_per_run": steps, "planned_conditions": configs, "jobs": jobs,
              "data_sha256": sha(output / "data.npz"),
              "dataset_manifest_sha256": sha(data / "manifest.json"),
              "alignment_source_sha256": sha(__file__), "environment": environment()}
    write_json(output / "manifest.json", header)
    tasks = [(str(output), entry, protocol, cfg, steps) for cfg in configs]
    with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(simulate_condition, tasks))
    header["status"] = "completed" if all(r["stop_reason"] == "completed_fixed_steps" for r in results) else "incomplete"
    header["results"] = results
    header["files"] = [{"path": str(p.relative_to(output)), "sha256": sha(p)}
                       for p in sorted(output.rglob("*")) if p.is_file() and p.name != "manifest.json"]
    write_json(output / "manifest.json", header)
    if header["status"] != "completed":
        raise RuntimeError("Some runs did not complete; partial evidence preserved")
    write_json(output / "verification.json", verify(output))


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "completed":
        raise ValueError("Incomplete study")
    expected = conditions()
    if manifest["planned_conditions"] != expected or [r["config"] for r in manifest["results"]] != expected:
        raise ValueError("Condition matrix differs")
    if sha(HERE / "run.py") != FROZEN_SOURCE_SHA:
        raise ValueError("Frozen update implementation changed")
    for item in manifest["files"]:
        if sha(root / item["path"]) != item["sha256"]:
            raise ValueError("Snapshot hash mismatch: " + item["path"])
    if manifest["alignment_source_sha256"] != sha(__file__):
        raise ValueError("Alignment source changed")
    entry = json.loads((root / "reference.json").read_text())
    with np.load(root / "data.npz") as data:
        for key in ["X", "y", "w0"]:
            if array_sha(data[key]) != entry[f"{key}_array_sha256"]:
                raise ValueError("Data array hash mismatch")
        z = data["y"][:, None] * data["X"]
    count, max_error = 0, 0.
    for result in manifest["results"]:
        folder = root / result["config"]["run_id"]
        if json.loads((folder / "summary.json").read_text()) != result:
            raise ValueError("Manifest and saved summary differ")
        if json.loads((folder / "config.json").read_text()) != result["config"]:
            raise ValueError("Saved condition differs")
        if result["updates"] != manifest["steps_per_run"]:
            raise ValueError("Run ended early")
        with (folder / "trajectory.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for key, value in result["endpoint"].items():
            reported = rows[-1][key]
            if value is None:
                matches = reported == ""
            elif isinstance(value, (int, float)):
                matches = float(reported) == value
            else:
                matches = reported == str(value)
            if not matches:
                raise ValueError("Endpoint and last checkpoint differ")
        with np.load(folder / "states.npz") as saved:
            if saved["steps"].tolist() != [int(r["step"]) for r in rows]:
                raise ValueError("Checkpoint indices differ")
            for j, row in enumerate(rows):
                w = saved["weights"][j]
                raw = float(np.min(z @ w))
                qi, q2 = raw / np.max(np.abs(w)), raw / np.linalg.norm(w)
                for key, actual in [("margin_linf", qi), ("margin_l2", q2),
                                    ("relative_gap_linf", 1 - qi / entry["reference"]["gamma_linf"]),
                                    ("relative_gap_l2", 1 - q2 / entry["reference"]["gamma_l2"])]:
                    error = abs(actual - float(row[key]))
                    max_error = max(error, max_error)
                    if error > 2e-10:
                        raise ValueError("Saved margin metric differs")
                extra = epsilon_metrics(saved["v_scaled"][j], saved["log_scales"][j],
                                        result["config"]["epsilon"], int(row["step"]))
                for key, value in extra.items():
                    if value is not None and not math.isclose(value, float(row[key]), abs_tol=1e-12):
                        raise ValueError("Saved epsilon diagnostic differs")
                count += 1
    with np.load(root / "figure1_adam_eps-1e-08/states.npz") as stable, np.load(root / "literal_adam_eps-1e-08/states.npz") as literal:
        if not np.array_equal(stable["steps"], literal["steps"]):
            raise ValueError("Literal comparison checkpoint mismatch")
        delta = float(np.max(np.abs(stable["weights"] - literal["weights"])))
        if not np.allclose(stable["weights"], literal["weights"], atol=2e-9, rtol=2e-11):
            raise ValueError("Scaled Adam disagrees with independent literal Adam")
    return {"passed": True, "runs": len(manifest["results"]), "checkpoints_recomputed": count,
            "max_margin_recomputation_error": max_error, "max_literal_weight_difference": delta,
            "note": "Saved checkpoints and literal positive-epsilon comparison; not every intervening update"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "verify"])
    parser.add_argument("--data", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--jobs", type=int, default=3)
    args = parser.parse_args()
    if args.command == "run":
        if args.data is None:
            parser.error("run requires --data")
        run(args.data, args.output, args.steps, args.jobs)
    else:
        print(json.dumps(verify(args.output), indent=2))


if __name__ == "__main__":
    main()
