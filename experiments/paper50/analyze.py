"""Recompute checkpoint metrics, audit hashes, and draw measured results."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from run import array_sha, sha, write_json


COLORS = {"GD": "#1877b9", "GDM": "#739c28", "Adam eps=0": "#d45b21", "Adam eps=1e-8": "#883aa3"}


def label(config):
    if config["method"] == "adam":
        return "Adam eps=0" if config["epsilon"] == 0 else f"Adam eps={config['epsilon']:g}".replace("e-08", "e-8")
    return config["method"].upper()


def load_trace(root, summary):
    path = root / summary["config"]["run_id"] / "trajectory.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([float(row[key]) if row[key] else np.nan for row in rows])
            for key in rows[0] if key != "method"}


def validate(root, data_dir):
    datasets = json.loads((data_dir / "manifest.json").read_text())
    p = json.loads((data_dir / "protocol.json").read_text())
    by_seed = {e["seed"]: e for e in datasets["datasets"]}
    runs = []
    maximum_metric_error, max_cert_residual = 0., 0.
    for seed, entry in by_seed.items():
        with np.load(data_dir / f"seed-{seed:02d}" / "data.npz") as data:
            x, y, w0 = data["X"], data["y"], data["w0"]
        assert sha(data_dir / f"seed-{seed:02d}" / "data.npz") == entry["data_npz_sha256"]
        for a, key in [(x, "X"), (y, "y"), (w0, "w0")]:
            assert array_sha(a) == entry[f"{key}_array_sha256"]
        z = x * y[:, None]
        ref = entry["reference"]
        u, w = np.array(ref["w_linf"]), np.array(ref["w_l2_svm"])
        alpha = np.array(ref["qp"]["alpha"])
        independently_checked = [max(0, ref["gamma_linf"] - (z @ u).min()),
                                 max(0, np.abs(u).max() - 1), max(0, 1 - (z @ w).min()),
                                 np.abs(w - z.T @ alpha).max(), max(0, -alpha.min()),
                                 np.abs(alpha * (z @ w - 1)).max(),
                                 abs(.5 * np.dot(w, w) - (alpha.sum() - .5 * np.dot(z.T @ alpha, z.T @ alpha)))]
        max_cert_residual = max(max_cert_residual, *independently_checked)
        assert max(independently_checked) < p["reference_tolerance"]
    for suite in ["pilot", "primary", "rates", "epsilon"]:
        folder = root / suite
        if not folder.exists():
            continue
        manifest = json.loads((folder / "manifest.json").read_text())
        assert manifest["status"] == "completed", f"Suite still running: {suite}"
        assert manifest["dataset_manifest_sha256"] == sha(data_dir / "manifest.json")
        assert manifest["protocol_sha256"] == sha(data_dir / "protocol.json")
        for result in manifest["results"]:
            cfg = result["config"]
            run_dir = folder / cfg["run_id"]
            for name, key in [("config.json", "config_sha256"), ("trajectory.csv", "trajectory_sha256"),
                              ("states.npz", "states_sha256")]:
                assert sha(run_dir / name) == result[key]
            tr = load_trace(folder, result)
            entry = by_seed[cfg["seed"]]
            assert result["data_sha256"] == entry["data_npz_sha256"]
            ref = entry["reference"]
            with np.load(data_dir / f"seed-{cfg['seed']:02d}" / "data.npz") as data:
                z = data["X"] * data["y"][:, None]
            with np.load(run_dir / "states.npz") as states:
                assert np.array_equal(states["steps"], tr["step"])
                weights = states["weights"]
                all_margins = weights @ z.T
                norms2 = np.linalg.norm(weights, axis=1)
                norminf = np.max(np.abs(weights), axis=1)
                raw = all_margins.min(axis=1)
                for actual, key in [(raw, "margin_raw"), (norms2, "norm_l2"), (norminf, "norm_linf"),
                                    (raw / norms2, "margin_l2"), (raw / norminf, "margin_linf"),
                                    (1 - raw / norminf / ref["gamma_linf"], "relative_gap_linf"),
                                    (1 - raw / norms2 / ref["gamma_l2"], "relative_gap_l2")]:
                    difference = np.max(np.abs(actual - tr[key]))
                    maximum_metric_error = max(maximum_metric_error, float(difference))
                    np.testing.assert_allclose(actual, tr[key], atol=2e-10, rtol=5e-12)
                assert len(tr["step"]) == result["recorded_checkpoints"]
                assert tr["step"][-1] == result["updates"]
            runs.append({"suite": suite, "run_id": cfg["run_id"], "updates": result["updates"],
                         "stop_reason": result["stop_reason"], "checkpoints": len(tr["step"])})
    repeats = []
    if (root / "rates" / "manifest.json").exists():
        primary_runs = json.loads((root / "primary" / "manifest.json").read_text())["results"]
        rate_runs = json.loads((root / "rates" / "manifest.json").read_text())["results"]
        for result in rate_runs:
            cfg = result["config"]
            if cfg["a"] != p["primary_adam_a"]:
                continue
            other = next(r for r in primary_runs if r["config"]["run_id"] == cfg["run_id"])
            with np.load(root / "primary" / cfg["run_id"] / "states.npz") as one, np.load(root / "rates" / cfg["run_id"] / "states.npz") as two:
                for key in one.files:
                    assert np.array_equal(one[key], two[key]), (cfg["run_id"], key)
            repeats.append({"run_id": cfg["run_id"], "all_saved_states_bitwise_equal": True})
    return {"all_hash_and_recomputed_metric_checks_passed": True,
            "datasets": len(by_seed), "runs": len(runs),
            "recorded_checkpoints": sum(r["checkpoints"] for r in runs),
            "max_abs_checkpoint_metric_difference": maximum_metric_error,
            "max_reference_certificate_residual_independently_recomputed": max_cert_residual,
            "independent_scheduled_repeat_runs": repeats,
            "runs_with_numerical_or_budget_stop": [r for r in runs if r["stop_reason"] != "completed_fixed_steps"],
            "runs_detail": runs}


def plot(root, output):
    output.mkdir(parents=True, exist_ok=False)
    datasets = json.loads((root / "datasets" / "manifest.json").read_text())
    primary = json.loads((root / "primary" / "manifest.json").read_text())
    rates = json.loads((root / "rates" / "manifest.json").read_text())
    selected = datasets["selected_seed"]
    ref = next(e["reference"] for e in datasets["datasets"] if e["seed"] == selected)
    selected_results = [r for r in primary["results"] if r["config"]["seed"] == selected]
    selected_results.sort(key=lambda r: ["GD", "GDM", "Adam eps=0", "Adam eps=1e-8"].index(label(r["config"])))
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 130, "savefig.dpi": 180, "svg.fonttype": "none"})
    figures = []

    def save(fig, name):
        fig.savefig(output / f"{name}.png", bbox_inches="tight")
        fig.savefig(output / f"{name}.svg", bbox_inches="tight")
        figures.append((name, fig))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for result in selected_results:
        tr = load_trace(root / "primary", result)
        key = label(result["config"])
        mask = tr["step"] >= 1
        for ax, norm in zip(axes, ["linf", "l2"]):
            ax.plot(tr["step"][mask], tr[f"margin_{norm}"][mask], label=key, color=COLORS[key], lw=2)
    for ax, norm, text in zip(axes, ["linf", "l2"], ["L-infinity", "L2"]):
        ax.axhline(ref[f"gamma_{norm}"], ls="--", color="#333333", label="Convex reference optimum")
        ax.set(xscale="log", xlabel="Observed update", ylabel=f"Normalized {text} margin")
        ax.grid(alpha=.18)
    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle(f"Independent replication | n=d=50 | selected seed {selected}")
    fig.text(.5, .005, "Adam: eta=(t+2)^(-0.7), c=1; GD/GDM: eta=0.5/L. Measured checkpoints; original paper figure not copied.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, .94])
    save(fig, "01_selected_margins")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for result in selected_results:
        tr = load_trace(root / "primary", result)
        key = label(result["config"])
        mask = tr["step"] >= 1000
        for ax, norm in zip(axes, ["linf", "l2"]):
            ax.plot(tr["step"][mask], tr[f"margin_{norm}"][mask] / ref[f"gamma_{norm}"],
                    label=key, color=COLORS[key], lw=2.3)
    for ax, text in zip(axes, ["L-infinity", "L2"]):
        ax.axhline(1, ls="--", color="#333333", label="Convex reference optimum")
        ax.set(xscale="log", xlabel="Observed update", ylabel=f"{text} margin / optimum", ylim=(.58, 1.025))
        ax.grid(alpha=.18)
    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle(f"Margin geometry after update 1000 | seed {selected} | n=d=50")
    fig.text(.5, .005, "Focused view. Initial transient is shown in figure 01; every recorded checkpoint is retained. Adam: a=0.7, c=1.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, .94])
    save(fig, "06_selected_margins_detail")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), sharey=True)
    rate_colors = {0.3: "#b54023", 0.5: "#a58b12", 0.7: "#197898", 1.: "#635093"}
    for result in rates["results"]:
        cfg = result["config"]
        tr = load_trace(root / "rates", result)
        ax = axes[int(cfg["epsilon"] > 0)]
        mask = (tr["step"] > 0) & (tr["gap_linf"] > 0)
        ax.loglog(tr["step"][mask], tr["gap_linf"][mask], label=f"a={cfg['a']:g}", color=rate_colors[cfg["a"]], lw=2)
    for ax, title in zip(axes, ["epsilon = 0", "epsilon = 1e-8"]):
        ax.set(title=title, xlabel="Observed update", ylabel="L-infinity margin gap (absolute units)")
        ax.grid(alpha=.18, which="both")
        ax.legend(fontsize=9)
    fig.suptitle(f"Step-schedule sensitivity | selected seed {selected} | c=1")
    fig.text(.5, .005, "All curves are observed data. Straight segments in log-log coordinates do not prove an asymptotic rate.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, .94])
    save(fig, "02_rate_sensitivity")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for result in selected_results:
        tr = load_trace(root / "primary", result)
        key = label(result["config"])
        mask = tr["step"] >= 1
        axes[0].semilogx(tr["step"][mask], tr["log_loss"][mask] / math.log(10), label=key, color=COLORS[key], lw=2)
        if result["config"]["method"] == "adam":
            axes[1].semilogx(tr["step"][mask], tr["log10_min_sqrt_v"][mask], label=key, color=COLORS[key], lw=2)
    axes[0].set(xlabel="Observed update", ylabel="log10 mean logistic loss")
    axes[1].axhline(-8, ls="--", color="#333333", label="epsilon = 1e-8")
    axes[1].set(xlabel="Observed update", ylabel="log10 minimum sqrt(v)")
    for ax in axes:
        ax.grid(alpha=.18)
        ax.legend(fontsize=9)
    fig.suptitle(f"Loss and epsilon scale | selected seed {selected}")
    fig.text(.5, .005, "sqrt(v) is reconstructed in log space from measured rescaled moments. A tiny loss is not a margin-convergence certificate.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, .94])
    save(fig, "03_loss_and_epsilon_scale")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for result_key in ["GD", "GDM", "Adam eps=0", "Adam eps=1e-8"]:
        vals = sorted([r for r in primary["results"] if label(r["config"]) == result_key], key=lambda r:r["config"]["seed"])
        for ax, norm in zip(axes, ["linf", "l2"]):
            ax.plot([r["config"]["seed"] for r in vals], [1 - r["endpoint"][f"relative_gap_{norm}"] for r in vals],
                    "o-", label=result_key, color=COLORS[result_key], alpha=.9)
    for ax, title in zip(axes, ["L-infinity", "L2"]):
        ax.axhline(1, ls="--", color="#333333")
        ax.axvline(selected, ls=":", color="#777777", lw=1)
        ax.set(xlabel="Data seed (vertical line: preselected)", ylabel=f"{title} margin / optimum", xticks=range(10))
        ax.grid(alpha=.18)
    axes[0].legend(fontsize=9)
    fig.suptitle("All ten candidates retained | 1,000,000 updates per method per seed")
    fig.text(.5, .005, "Descriptive seed coverage, not a confidence interval. No seed was discarded based on optimizer performance.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .04, 1, .94])
    save(fig, "04_all_seeds")

    extension = json.loads((root / "epsilon" / "manifest.json").read_text())
    fig, ax = plt.subplots(figsize=(9, 4.7))
    controls = [r for r in selected_results if r["config"]["method"] == "adam"] + extension["results"]
    for result in sorted(controls, key=lambda r:r["config"]["epsilon"]):
        suite = "epsilon" if result in extension["results"] else "primary"
        tr = load_trace(root / suite, result)
        mask = tr["step"] > 0
        ax.semilogx(tr["step"][mask], tr["relative_gap_linf"][mask], label=f"epsilon={result['config']['epsilon']:g}", lw=2)
    ax.axhline(.01, color="#555555", ls="--", label="Team 1% finite-gap criterion")
    ax.set(xlabel="Observed update", ylabel="Relative L-infinity margin gap")
    ax.grid(alpha=.18)
    ax.legend(fontsize=9)
    fig.suptitle(f"Additional epsilon controls | seed {selected} | a=0.7, c=1")
    fig.tight_layout(rect=[0, 0, 1, .94])
    save(fig, "05_epsilon_controls")

    with PdfPages(output / "figures.pdf") as pdf:
        for _, fig in figures:
            pdf.savefig(fig, bbox_inches="tight")
    for _, fig in figures:
        plt.close(fig)
    return [name for name, _ in figures]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    audit = validate(args.evidence, args.evidence / "datasets")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "validation.json", audit)
    figures = plot(args.evidence, args.output / "figures")
    print(json.dumps({"audit": {k:v for k,v in audit.items() if k != "runs_detail"}, "figures": figures}), flush=True)


if __name__ == "__main__":
    main()
