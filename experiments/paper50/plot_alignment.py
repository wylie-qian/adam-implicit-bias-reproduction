"""Plot only verified, measured checkpoints from the additive alignment audit."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from alignment import verify


def plot(root, output):
    verify(root)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((root / "manifest.json").read_text())
    traces = {}
    for result in manifest["results"]:
        runid = result["config"]["run_id"]
        with (root / runid / "trajectory.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        traces[runid] = {key: np.array([float(row[key]) if row[key] else np.nan for row in rows])
                         for key in ["step", "relative_gap_linf", "relative_gap_l2",
                                     "epsilon_update_suppression_min", "epsilon_update_suppression_max",
                                     "log10_max_sqrt_v_over_epsilon"]}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    styles = [("figure1_adam_eps-0", "Adam, epsilon=0, a=0.5", "#0072B2", "-"),
              ("figure1_adam_eps-1e-08", "Adam, epsilon=1e-8, a=0.5", "#D55E00", "-"),
              ("legacy_adam_eps-0", "Old Adam, epsilon=0, a=0.7", "#0072B2", "--"),
              ("legacy_adam_eps-1e-08", "Old Adam, epsilon=1e-8, a=0.7", "#D55E00", "--"),
              ("figure1_gd", "GD, eta=0.1", "#009E73", "-"),
              ("figure1_gdm", "GDM (EMA), eta=0.1", "#CC79A7", "-")]
    for ax, norm in zip(axes, ["linf", "l2"]):
        for runid, label, color, ls in styles:
            trace = traces[runid]
            valid = trace["step"] > 0
            ax.plot(trace["step"][valid], trace[f"relative_gap_{norm}"][valid] * 100,
                    label=label, color=color, ls=ls, lw=1.7)
        ax.set(xscale="log", yscale="log", xlabel="Completed updates",
               ylabel="Relative margin gap (%)", title=f"{'L-infinity' if norm == 'linf' else 'L2'} margin")
        ax.grid(alpha=.18, which="major")
    axes[0].axhline(1, color="gray", ls=":", lw=1, label="Team's finite 1% criterion")
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"Figure 1 settings audit — selected seed {manifest['selected_seed']} (own simulations)")
    for ext in ["png", "svg"]:
        fig.savefig(output / f"margin_comparison.{ext}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for runid, label, color in [("figure1_adam_eps-1e-08", "Figure 1 schedule (a=0.5)", "#D55E00"),
                               ("legacy_adam_eps-1e-08", "Old schedule (a=0.7)", "#0072B2")]:
        trace = traces[runid]
        valid = trace["step"] > 0
        step = trace["step"][valid]
        axes[0].plot(step, trace["log10_max_sqrt_v_over_epsilon"][valid], color=color, label=label)
        low = trace["epsilon_update_suppression_min"][valid]
        high = trace["epsilon_update_suppression_max"][valid]
        axes[1].fill_between(step, low, high, color=color, alpha=.14)
        axes[1].plot(step, low, color=color, label=label)
    axes[0].axhline(0, color="gray", ls=":", label="All sqrt(v) below epsilon")
    axes[0].axhline(-1, color="black", ls="--", lw=1, label="All sqrt(v) below 0.1 epsilon")
    axes[0].set(ylabel="log10 max coordinate sqrt(v) / epsilon", title="Original-scale second moment")
    axes[1].axhline(10 / 11, color="black", ls="--", lw=1)
    axes[1].set(ylabel="Fraction of zero-epsilon update\nremoved (same moments)", ylim=(-.02, 1.02),
                title="Epsilon attenuation (line=min; band=min to max)")
    for ax in axes:
        ax.set(xscale="log", xlabel="Completed updates")
        ax.grid(alpha=.18)
        ax.legend(fontsize=8)
    fig.suptitle("Positive epsilon changes the update well before one million steps (own simulations)")
    for ext in ["png", "svg"]:
        fig.savefig(output / f"epsilon_influence.{ext}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    plot(args.evidence, args.output)
