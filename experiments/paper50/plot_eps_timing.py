"""Tables and figure for the epsilon-timing study (eps_timing.py), 2026-11-13.

Usage, from the repository root:
    python3 experiments/paper50/plot_eps_timing.py experiments/paper50/results/2026-11-13-eps-timing
Writes summary.json and figures/eps_timing.{png,svg} into that folder.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from scipy.special import expit

DATA = Path(__file__).resolve().parent / "results/2026-11-05/datasets"


def crossing(row, threshold, kind="all"):
    return row["crossings"].get(f"{threshold:g}", {}).get(kind)


def span(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"min": min(values), "median": float(np.median(values)), "max": max(values), "n": len(values)}


def first_sqrt_v(seed, p):
    """sqrt(v) after the first update (no bias correction): sqrt(1 - beta2) * |g_1| at w0."""
    with np.load(DATA / f"seed-{seed:02d}" / "data.npz") as data:
        z = data["y"][:, None] * data["X"]
        w = data["w0"]
    g = -z.T @ expit(-(z @ w)) / len(z)
    return math.sqrt(1 - p["beta2"]) * np.abs(g)


def earliest_takeover(seed, eps, p):
    """v_t >= beta2 * v_(t-1), so sqrt(v) shrinks by at most a factor sqrt(beta2) per update.
    sqrt(v_j) < eps for all j therefore needs t > 1 + 2 ln(max_j sqrt(v_1j) / eps) / ln(1 / beta2)."""
    return 1 + 2 * math.log(float(np.max(first_sqrt_v(seed, p))) / eps) / math.log(1 / p["beta2"])


def tables(rows, p, selected):
    eps0, a0 = p["epsilon_main"], p["primary_adam_a"]
    by = {(r["suite"], r["seed"], r["a"], r["c"], r["epsilon"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    bound = {(s, e): earliest_takeover(s, e, p) for s in seeds for e in set(p["epsilon_sweep"]) | {eps0}}
    out = {"earliest_possible_note": "1 + 2 ln(max_j sqrt(v_1j) / eps) / ln(1 / beta2): sqrt(v) shrinks by at most a "
                                     "factor sqrt(beta2) per update, whatever the learning rate",
           "c_sweep": [], "a_sweep": [], "epsilon_sweep": [], "path_gap_at_threshold": []}

    def hit(r, eps, key, kind="all"):
        x = crossing(r, eps, kind)
        return x[key] if x else None

    for c in p["c_values"]:
        runs = {s: by[("c", s, a0, c, eps0)] for s in seeds}
        zero = {s: by[("c", s, a0, c, 0.0)] for s in seeds}
        sel = runs[selected]
        out["c_sweep"].append({
            "c": c,
            "selected_first_step": hit(sel, eps0, "step", "first"), "selected_all_step": hit(sel, eps0, "step"),
            "selected_earliest_possible": bound[(selected, eps0)],
            "all_step": span([hit(r, eps0, "step") for r in runs.values()]),
            "earliest_possible": span([bound[(s, eps0)] for s in seeds]),
            "all_step_over_earliest": span([hit(r, eps0, "step") / bound[(s, eps0)] if hit(r, eps0, "step") else None
                                            for s, r in runs.items()]),
            "taken_over_by_end": sum(hit(r, eps0, "step") is not None for r in runs.values()),
            "selected_gap_linf_at_takeover": hit(sel, eps0, "relative_gap_linf"),
            "gap_linf_at_takeover": span([hit(r, eps0, "relative_gap_linf") for r in runs.values()]),
            "norm_linf_at_takeover": span([hit(r, eps0, "norm_linf") for r in runs.values()]),
            "log_loss_at_takeover": span([hit(r, eps0, "log_loss") for r in runs.values()]),
            "selected_end_gap_linf": sel["relative_gap_linf"],
            "end_gap_linf": span([r["relative_gap_linf"] for r in runs.values()]),
            "selected_end_gap_linf_eps0": zero[selected]["relative_gap_linf"],
            "end_gap_linf_eps0": span([r["relative_gap_linf"] for r in zero.values()]),
            "closest_recorded_gap_linf": span([r["closest_recorded_linf"]["relative_gap_linf"] for r in runs.values()]),
        })
    for a in [a0] + p["a_values_selected_seed"]:
        suite = "c" if a == a0 else "a"
        for c in p["c_values"]:
            r = by[(suite, selected, a, c, eps0)]
            out["a_sweep"].append({"a": a, "c": c, "all_step": hit(r, eps0, "step"),
                                   "earliest_possible": bound[(selected, eps0)],
                                   "gap_linf_at_takeover": hit(r, eps0, "relative_gap_linf"),
                                   "end_gap_linf": r["relative_gap_linf"],
                                   "end_gap_linf_eps0": by[(suite, selected, a, c, 0.0)]["relative_gap_linf"]})
    for eps in sorted(set(p["epsilon_sweep"]) | {eps0}, reverse=True):
        suite = "c" if eps == eps0 else "epsilon"
        runs = {s: by[(suite, s, a0, 1.0, eps)] for s in seeds}
        out["epsilon_sweep"].append({
            "epsilon": eps, "selected_all_step": hit(runs[selected], eps, "step"),
            "selected_earliest_possible": bound[(selected, eps)],
            "all_step": span([hit(r, eps, "step") for r in runs.values()]),
            "earliest_possible": span([bound[(s, eps)] for s in seeds]),
            "taken_over_by_end": sum(hit(r, eps, "step") is not None for r in runs.values()),
            "gap_linf_at_takeover": span([hit(r, eps, "relative_gap_linf") for r in runs.values()]),
            "selected_end_gap_linf": runs[selected]["relative_gap_linf"],
            "end_gap_linf": span([r["relative_gap_linf"] for r in runs.values()])})
    for thr in p["tracked_thresholds"]:
        entry = {"threshold": thr}
        for c in p["c_values"]:
            zero = [by[("c", s, a0, c, 0.0)] for s in seeds]
            entry[f"c={c:g}"] = {"step": span([hit(r, thr, "step") for r in zero]),
                                 "relative_gap_linf": span([hit(r, thr, "relative_gap_linf") for r in zero])}
        out["path_gap_at_threshold"].append(entry)
    return out


def digits(eps):
    return -math.log10(eps)


def eps_axis(ax, eps_list):
    """Log scale in the number of digits of epsilon, so 1e-4, 1e-8, ... 1e-256 are evenly spaced."""
    ax.set_xscale("log")
    # 1e-12 sits too close to 1e-16 for a label of its own
    ax.set_xticks([digits(e) for e in eps_list], ["" if e == 1e-12 else f"1e-{digits(e):.0f}" for e in eps_list])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("ε (smaller to the right)")


def read_trajectory(path):
    with gzip.open(path, "rt", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if int(r["step"]) > 0]
    return np.array([int(r["step"]) for r in rows]), np.array([float(r["relative_gap_linf"]) for r in rows])


def plot(results, rows, p, selected, out):
    eps0, a0 = p["epsilon_main"], p["primary_adam_a"]
    by = {(r["suite"], r["seed"], r["a"], r["c"], r["epsilon"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    colors = dict(zip(p["c_values"], ["#9e9ac8", "#6a51a3", "#3182bd", "#e6550d", "#a63603"]))
    blue, grey = "#3182bd", "#bdbdbd"
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), layout="constrained")

    ax = axes[0, 0]
    for c in p["c_values"]:
        for eps, style in [(0.0, ":"), (eps0, "-")]:
            run_id = f"seed-{selected:02d}_a-{a0:g}_c-{c:g}_eps-{eps:g}"
            steps, gap = read_trajectory(results / "c" / run_id / "trajectory.csv.gz")
            ax.plot(steps, gap * 100, ls=style, color=colors[c], lw=1.6, label=f"c = {c:g}" if eps else None)
        x = crossing(by[("c", selected, a0, c, eps0)], eps0)
        if x:
            ax.plot(x["step"], x["relative_gap_linf"] * 100, "o", color=colors[c], ms=6, mec="black", mew=.6)
    ax.set(xscale="log", yscale="log", ylim=(0.3, 300), xlabel="Update t", ylabel="L∞ gap to the optimum (%)",
           title=f"(a) Seed {selected}, η = c·t^-{a0:g}. Solid: ε = 1e-8; dotted: ε = 0.\n"
                 "Dot: √v < ε in all 50 coordinates")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[0, 1]
    for s in seeds:
        xs = [c for c in p["c_values"] if crossing(by[("c", s, a0, c, eps0)], eps0)]
        ys = [crossing(by[("c", s, a0, c, eps0)], eps0)["step"] for c in xs]
        ax.plot(xs, ys, "o-", color=blue if s == selected else grey, lw=1.8 if s == selected else 1, ms=4,
                zorder=3 if s == selected else 2, label=f"seed {s}" if s == selected else None)
    ax.axhline(earliest_takeover(selected, eps0, p), color="black", ls="--", lw=1,
               label=f"earliest possible, seed {selected}")
    ax.axhline(p["steps"], color="#999999", lw=.8)
    ax.text(min(p["c_values"]), p["steps"] * .7, "end of run", fontsize=8, color="#666666", va="top")
    ax.set(xscale="log", yscale="log", ylim=(1e3, 2e6), xlabel="Learning-rate constant c",
           ylabel="Update when √v < 1e-8 in all coordinates",
           title="(b) When ε = 1e-8 takes over (ten datasets, a = 0.5)\n"
                 "Earliest possible: √v shrinks by at most 0.5% per update")
    ax.legend(fontsize=8)

    eps_list = sorted(set(p["epsilon_sweep"]) | {eps0}, reverse=True)
    ax = axes[1, 0]
    for s in seeds:
        pts = [(e, crossing(by[("c" if e == eps0 else "epsilon", s, a0, 1.0, e)], e)) for e in eps_list]
        pts = [(digits(e), x["step"]) for e, x in pts if x]
        if pts:
            ax.plot(*zip(*pts), "o-", color=blue if s == selected else grey, lw=1.8 if s == selected else 1, ms=4,
                    zorder=3 if s == selected else 2, label=f"seed {s}" if s == selected else None)
    ax.plot([digits(e) for e in eps_list], [earliest_takeover(selected, e, p) for e in eps_list], "k--", lw=1,
            label=f"earliest possible, seed {selected}")
    ax.axhline(p["steps"], color="#999999", lw=.8)
    ax.set(yscale="log", ylabel="Update when √v < ε in all coordinates",
           title="(c) When ε takes over, c = 1 (ten datasets)")
    eps_axis(ax, eps_list)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for key, label, marker, fill in [("end", "after 10⁶ updates", "s", True), ("take", "when ε takes over", "o", False)]:
        med, lo, hi, xs = [], [], [], []
        for e in eps_list:
            runs = [by[("c" if e == eps0 else "epsilon", s, a0, 1.0, e)] for s in seeds]
            vals = ([r["relative_gap_linf"] for r in runs] if key == "end"
                    else [(crossing(r, e) or {}).get("relative_gap_linf") for r in runs])
            if any(v is None for v in vals):
                continue
            xs.append(digits(e))
            vals = np.array(vals) * 100
            med.append(np.median(vals)); lo.append(vals.min()); hi.append(vals.max())
        med, lo, hi = map(np.array, (med, lo, hi))
        ax.errorbar(xs, med, yerr=[med - lo, hi - med], fmt=marker + "-", color="black", ms=5, lw=1, capsize=3,
                    mfc="black" if fill else "white", label=f"ε > 0, {label}")
    zero = np.array([by[("c", s, a0, 1.0, 0.0)]["relative_gap_linf"] for s in seeds]) * 100
    ax.axhspan(zero.min(), zero.max(), color="#31a354", alpha=.25, lw=0, label="ε = 0 after 10⁶ updates")
    ax.set(yscale="log", ylabel="L∞ gap to the optimum (%)",
           title="(d) L∞ gap, c = 1: median and range of ten datasets")
    eps_axis(ax, eps_list)
    ax.legend(fontsize=8)
    (out / "figures").mkdir(exist_ok=True)
    for ext in ["png", "svg"]:
        fig.savefig(out / "figures" / f"eps_timing.{ext}", dpi=150)
    plt.close(fig)


def main():
    results = Path(sys.argv[1])
    p = json.loads((Path(__file__).resolve().parent / "protocol_eps_timing.json").read_text())
    rows = json.loads((results / "endpoints.json").read_text())
    selected = json.loads((results / "manifest.json").read_text())["selected_seed"]
    summary = tables(rows, p, selected)
    (results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(results, rows, p, selected, results)


if __name__ == "__main__":
    main()
