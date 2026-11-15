"""Fixed legacy NumPy MT19937 probes; no claim these were the authors' seeds.

Each root seed generates ten candidates sequentially. Geometry selects the
candidate before training; both epsilon conditions use the same independent
Gaussian initialization. All candidate metadata and all outcomes are retained.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import multiprocessing
from pathlib import Path
import shutil

import numpy as np

import alignment
from run import (HERE, DEFAULT_PROTOCOL, array_sha, environment, load_protocol,
                 references, sha, write_json)

ROOT_SEEDS = (0, 42, 2024)


def candidates(root_seed):
    rng = np.random.RandomState(root_seed)
    return [(rng.standard_normal((50, 50)), rng.choice([-1, 1], 50)) for _ in range(10)]


def configs():
    return [dict(run_id=f"adam_eps-{eps:g}", method="adam", epsilon=eps,
                 a=.5, c=1., offset=1, arithmetic="scaled") for eps in [0., 1e-8]]


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "completed" or manifest["source_sha256"] != sha(__file__):
        raise ValueError("Incomplete study or changed runner")
    if manifest["alignment_source_sha256"] != sha(HERE / "alignment.py"):
        raise ValueError("Changed simulation implementation")
    count, max_error = 0, 0.
    for case in manifest["cases"]:
        folder = root / case["run_id"]
        generated = candidates(case["root_seed"])
        for i, ((x, y), saved) in enumerate(zip(generated, case["candidates"])):
            if array_sha(x) != saved["X_array_sha256"] or array_sha(y) != saved["y_array_sha256"]:
                raise ValueError("Candidate generation changed")
        selected = max(range(10), key=lambda i: (case["candidates"][i]["reference"]["direction_separation_l2"], -i))
        if selected != case["selected_index"]:
            raise ValueError("Geometry selection changed")
        x, y = generated[selected]
        w0 = np.random.RandomState(case["root_seed"] + 10000).standard_normal(50)
        with np.load(folder / "data.npz") as data:
            if not all(np.array_equal(data[k], v) for k, v in [("X", x), ("y", y), ("w0", w0)]):
                raise ValueError("Selected inputs differ")
        z = y[:, None] * x
        ref = case["candidates"][selected]["reference"]
        for result in manifest["results"]:
            if result["root_seed"] != case["root_seed"]:
                continue
            r = result["result"]
            run_folder = folder / r["config"]["run_id"]
            if r["stop_reason"] != "completed_fixed_steps" or r["updates"] != manifest["steps_per_run"]:
                raise ValueError("Incomplete run")
            for name, digest in r["hashes"].items():
                if sha(run_folder / name) != digest:
                    raise ValueError("Changed raw evidence")
            with (run_folder / "trajectory.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            with np.load(run_folder / "states.npz") as saved:
                if saved["steps"].tolist() != [int(row["step"]) for row in rows]:
                    raise ValueError("Checkpoint indices differ")
                for j, row in enumerate(rows):
                    w = saved["weights"][j]
                    raw = float(np.min(z @ w))
                    qi, q2 = raw / np.max(np.abs(w)), raw / np.linalg.norm(w)
                    for key, value in [("margin_linf", qi), ("margin_l2", q2),
                                       ("relative_gap_linf", 1 - qi / ref["gamma_linf"]),
                                       ("relative_gap_l2", 1 - q2 / ref["gamma_l2"])]:
                        err = abs(value - float(row[key])); max_error = max(max_error, err)
                        if err > 2e-10:
                            raise ValueError("Saved metric differs")
                    count += 1
    return dict(passed=True, runs=len(manifest["results"]), candidates_regenerated=30,
                checkpoints_recomputed=count, max_margin_recomputation_error=max_error)


def run(output, steps, jobs):
    if steps < 1 or jobs < 1:
        raise ValueError("steps/jobs must be positive")
    if sha(HERE / "run.py") != alignment.FROZEN_SOURCE_SHA:
        raise ValueError("Frozen Adam recurrence changed")
    output.mkdir(parents=True, exist_ok=False)
    protocol = load_protocol(DEFAULT_PROTOCOL)
    header = dict(status="running", source_sha256=sha(__file__),
                  alignment_source_sha256=sha(HERE / "alignment.py"), environment=environment(),
                  steps_per_run=steps, root_seeds=list(ROOT_SEEDS), planned_configs=configs(),
                  design="Ten MT19937 candidates per fixed root; select maximum reference direction separation before training; all outcomes retained", cases=[])
    tasks = []
    for root_seed in ROOT_SEEDS:
        folder = output / f"mt19937-root{root_seed}"
        folder.mkdir()
        generated = candidates(root_seed)
        metadata = []
        for i, (x, y) in enumerate(generated):
            ref = references(y[:, None] * x)
            metadata.append(dict(index=i, X_array_sha256=array_sha(x), y_array_sha256=array_sha(y), reference=ref))
        selected = max(range(10), key=lambda i: (metadata[i]["reference"]["direction_separation_l2"], -i))
        x, y = generated[selected]
        w0 = np.random.RandomState(root_seed + 10000).standard_normal(50)
        np.savez_compressed(folder / "data.npz", X=x, y=y, w0=w0)
        case = dict(root_seed=root_seed, run_id=folder.name, selected_index=selected,
                    candidates=metadata, initialization_seed=root_seed + 10000,
                    w0_array_sha256=array_sha(w0), data_sha256=sha(folder / "data.npz"))
        header["cases"].append(case)
        write_json(folder / "candidates.json", case)
        for cfg in configs():
            tasks.append((str(folder), {"reference": metadata[selected]["reference"]}, protocol, cfg, steps))
    write_json(output / "manifest.json", header)
    with ProcessPoolExecutor(max_workers=jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(alignment.simulate_condition, tasks))
    header["results"] = [dict(root_seed=seed, result=result)
                         for seed, result in zip([s for s in ROOT_SEEDS for _ in configs()], results)]
    header["status"] = "completed" if all(r["stop_reason"] == "completed_fixed_steps" for r in results) else "incomplete"
    write_json(output / "manifest.json", header)
    write_json(output / "verification.json", verify(output))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["run", "verify"])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=1000000)
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    if args.command == "run":
        run(args.output, args.steps, args.jobs)
    else:
        print(json.dumps(verify(args.output), indent=2))
