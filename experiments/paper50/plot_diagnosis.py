"""Regenerate diagnostic figures from verified own measurements only."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from conventions import verify as verify_conventions
from verify_randomness import verify as verify_randomness, summarize


def plot(evidence, output):
    verify_conventions(evidence / "conventions")
    verify_randomness(evidence / "randomness")
    random_manifest = json.loads((evidence / "randomness/manifest.json").read_text())
    random_summary = summarize(random_manifest)
    configs = json.loads((evidence / "conventions/manifest.json").read_text())["planned_conditions"]
    runs = {c["name"]: json.loads((evidence / "conventions" / (c["name"] + ".json")).read_text()) for c in configs}
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    order = ["literal_baseline", "correct_rescaled_eps1e-8", "bias_correction", "beta2_0999", "sum_loss",
             "epsilon_inside_sqrt", "naive_1_minus_sigmoid", "float32", "scaled_zero_epsilon", "wrong_scaled_epsilon"]
    names = ["Fixed epsilon, literal", "Fixed epsilon, correct scaling", "Bias correction",
             "Beta2 = 0.999", "Sum loss", "Epsilon inside square root", "Naive sigmoid subtraction",
             "Float32 training", "Zero-epsilon reference", "Wrong scaling (INVALID)"]
    values = [runs[k]["endpoint"]["relative_gap_linf"] * 100 for k in order]
    colors = ["#0072B2"] * 2 + ["#737373"] * 6 + ["#009E73", "#D55E00"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), layout="constrained", gridspec_kw={"width_ratios": [1.25, 1]})
    bars = axes[0].barh(names, values, color=colors, height=.65)
    bars[-1].set_hatch("///")
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Relative L-infinity margin gap (%)", xlim=(0, 48),
                title="Named implementation controls (same data/start)")
    for bar, value in zip(bars, values):
        axes[0].text(value + .5, bar.get_y() + bar.get_height() / 2, f"{value:.3f}%", va="center", fontsize=9)
    for key, name, color, style in [("literal_baseline", "Correct fixed epsilon = 1e-8", "#0072B2", "-"),
                                   ("scaled_zero_epsilon", "Correct epsilon = 0", "#009E73", "-"),
                                   ("wrong_scaled_epsilon", "Wrong epsilon scaling (INVALID)", "#D55E00", "--")]:
        trace = runs[key]["trace"]
        rows = [r for r in trace if r["step"] > 0]
        axes[1].plot([r["step"] for r in rows], [r["relative_gap_linf"] * 100 for r in rows],
                     label=name, color=color, ls=style, lw=1.8)
    axes[1].set(xscale="log", yscale="log", xlabel="Completed updates", ylabel="Relative L-infinity gap (%)",
                title="A matching curve can still implement the wrong epsilon")
    axes[1].legend(fontsize=8, loc="lower left")
    for ax in axes: ax.grid(axis="x", alpha=.15)
    fig.suptitle("Own diagnostic simulations — one million updates per condition", fontsize=14)
    for ext in ["png", "svg"]: fig.savefig(output / f"convention_controls.{ext}", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
    specs = [("dataset", "Ten original Gaussian datasets", lambda r: int(r["data_key"].split("-")[-1]), "Data seed"),
             ("independent_initialization", "Five disjoint initialization seeds", lambda r: r["initialization_seed"], "Initialization seed"),
             ("legacy_rng", "Legacy MT19937: geometry-selected data", lambda r: int(r["data_key"].split("-")[1]), "Root seed")]
    for ax, (family, title, sort_key, xlabel) in zip(axes, specs):
        family_rows = [r for r in random_summary["results"] if r["family"] == family]
        seeds = sorted({sort_key(r) for r in family_rows})
        for eps, label, color in [(0., "epsilon = 0", "#009E73"), (1e-8, "epsilon = 1e-8", "#0072B2")]:
            rows = sorted([r for r in family_rows if r["epsilon"] == eps], key=sort_key)
            ax.plot(range(len(rows)), [100 * r["relative_gap_linf"] for r in rows], "o-", color=color, label=label, lw=1.3)
        ax.axhline(1., color="gray", ls=":", lw=1)
        ax.set(yscale="log", ylim=(.4, 55), xlabel=xlabel, ylabel="Relative L-infinity gap (%)", title=title,
               xticks=range(len(seeds)), xticklabels=[str(s) for s in seeds])
        ax.grid(alpha=.15)
        ax.legend(fontsize=8)
    fig.suptitle("Own paired randomness controls — all displayed outcomes retained", fontsize=14)
    for ext in ["png", "svg"]: fig.savefig(output / f"randomness_controls.{ext}", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", type=Path, default=Path(__file__).resolve().parent / "results/2026-11-12-diagnosis")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    plot(args.evidence, args.output)
