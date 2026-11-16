"""Figures for the Group 7 report. All panels are drawn from our own saved runs.

Run from the repository root (needs Matplotlib; the pinned paper50 environment works):

    python3 docs/report/figures/make_figures.py

Inputs (read only):
  experiments/direction_2d/runs/frozen-20k-fixed-metrics/{gd,adam}.csv
  experiments/paper50/results/2026-11-13-legend/primary/seed-02_*/trajectory.csv
  experiments/paper50/results/2026-11-13-eps-timing/{endpoints.json, c/*/summary.json, c/*/trajectory.csv.gz}
Outputs: docs/report/figures/fig{1,2,3}_*.pdf (vector, for LaTeX) and .png (preview).
"""
import csv
import gzip
import json
import math
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "docs", "report", "figures")
P50 = os.path.join(ROOT, "experiments", "paper50", "results")

# Same colours as the slides: GD blue, Adam (eps = 0) orange, Adam (eps > 0) purple.
C = {"gd": "#2B5797", "gdm": "#7FA7D6", "adam0": "#C0601F", "adameps": "#6C3D91", "grey": "#8A8F99"}
LABEL = {"gd": "GD", "gdm": "GDM", "adam0": r"Adam, $\varepsilon=0$", "adameps": r"Adam, $\varepsilon=10^{-8}$"}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,  # embed TrueType, not Type 3
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})
TEXTWIDTH = 5.5  # inches, NeurIPS text width


def read_csv(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", newline="") as f:
        return list(csv.DictReader(f))


def col(rows, name, skip_empty=True):
    out = []
    for r in rows:
        v = r[name]
        if v == "" and skip_empty:
            out.append(math.nan)
        else:
            out.append(float(v))
    return out


def panel_label(ax, s):
    ax.text(-0.02, 1.04, s, transform=ax.transAxes, ha="right", va="bottom", fontweight="bold")


def save(fig, name):
    # No creation date in the PDF, so a rerun gives the same bytes.
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), bbox_inches="tight", pad_inches=0.02, metadata={"CreationDate": None})
    fig.savefig(os.path.join(OUT, f"{name}.png"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- Figure 1
def fig1():
    """(a) steepest-descent steps in the L2 ball and the L-inf box; (b) our 2D runs."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.65), gridspec_kw={"width_ratios": [1, 1.9]})

    # (a) geometry for g = (1, 0.05), as on slide 3
    g = (1.0, 0.05)
    n = math.hypot(*g)
    t = [i / 200 * 2 * math.pi for i in range(201)]
    a.plot([math.cos(s) for s in t], [math.sin(s) for s in t], color=C["gd"], lw=0.9)
    a.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color=C["adam0"], lw=0.9)
    kw = dict(arrowstyle="-|>", mutation_scale=7, lw=1.0)
    a.annotate("", xy=(0.62 * g[0] / n, 0.62 * g[1] / n), xytext=(0, 0), arrowprops=dict(color="black", **kw))
    a.text(0.66, 0.06, r"$g$", ha="left", va="bottom")
    a.annotate("", xy=(-g[0] / n, -g[1] / n), xytext=(0, 0), arrowprops=dict(color=C["gd"], **kw))
    a.annotate("", xy=(-1, -1), xytext=(0, 0), arrowprops=dict(color=C["adam0"], **kw))
    a.text(-1.08, 0.0, r"GD step" "\n" r"$-g/\|g\|_2$", color=C["gd"], ha="right", va="center", fontsize=7)
    a.text(-1.05, -1.08, r"sign step $-\mathrm{sign}(g)$", color=C["adam0"], ha="left", va="top", fontsize=7)
    a.text(-0.93, 0.93, r"$\|d\|_2\leq 1$", color=C["gd"], ha="left", va="top", fontsize=7)
    a.text(1.0, 1.05, r"$\|d\|_\infty\leq 1$", color=C["adam0"], ha="right", va="bottom", fontsize=7)
    a.set_aspect("equal")
    a.set_xlim(-1.75, 1.1)
    a.set_ylim(-1.35, 1.3)
    a.axis("off")
    panel_label(a, "(a)")

    # (b) direction of w_t in our 2D runs (z_1 = (1, 2), z_2 = (2, 1/2))
    run = os.path.join(ROOT, "experiments", "direction_2d", "runs", "frozen-20k-fixed-metrics")
    l2_dir = math.degrees(math.atan2(2, 3))   # max L2-margin direction (3, 2)/sqrt(13)
    linf_dir = 45.0                             # max L-inf-margin direction (1, 1)
    for key, fname in (("gd", "gd.csv"), ("adam0", "adam.csv")):
        rows = [r for r in read_csv(os.path.join(run, fname)) if int(r["step"]) >= 1]
        steps = col(rows, "step")
        ang = [math.degrees(math.atan2(float(r["w2"]), float(r["w1"]))) for r in rows]
        b.plot(steps, ang, color=C[key], lw=1.1)
    b.axhline(l2_dir, color=C["gd"], ls="--", lw=0.7)
    b.axhline(linf_dir, color=C["adam0"], ls="--", lw=0.7)
    b.text(1.15, l2_dir + 0.35, r"max $L_2$-margin direction ($33.7^\circ$)", color=C["gd"], va="bottom", fontsize=7)
    b.text(1.15, linf_dir + 0.35, r"max $L_\infty$-margin direction ($45^\circ$)", color=C["adam0"], va="bottom", fontsize=7)
    b.text(2e3, 36.9, "GD", color=C["gd"], ha="center", va="top", fontsize=7)
    b.text(30, 43.6, r"Adam, $\varepsilon=0$", color=C["adam0"], ha="center", va="top", fontsize=7)
    b.set_ylim(32.5, 46.5)
    b.set_xscale("log")
    b.set_xlim(1, 2e4)
    b.set_xlabel("update $t$")
    b.set_ylabel(r"angle of $w_t$ (degrees)")
    panel_label(b, "(b)")
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig1_geometry")


# ---------------------------------------------------------------- Figure 2
def legend_traj(name):
    return read_csv(os.path.join(P50, "2026-11-13-legend", "primary", name, "trajectory.csv"))


def fig2():
    runs = {
        "gd": "seed-02_gd_eps-0_a-0.5",
        "gdm": "seed-02_gdm_eps-0_a-0.5",
        "adam0": "seed-02_adam_eps-0_a-0.5",
        "adameps": "seed-02_adam_eps-1e-08_a-0.5",
    }
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 1.7), sharex=True)
    ends = {}
    for key, name in runs.items():
        rows = [r for r in legend_traj(name) if int(r["step"]) >= 1]
        s = col(rows, "step")
        for ax, gap in zip(axes, ("relative_gap_linf", "relative_gap_l2")):
            ratio = [1 - v for v in col(rows, gap)]
            ax.plot(s, ratio, color=C[key], lw=1.1 if key != "gdm" else 0.9,
                    ls="-" if key != "gdm" else (0, (3, 1.5)), label=LABEL[key])
        ends[key] = (1 - float(rows[-1]["relative_gap_linf"]), 1 - float(rows[-1]["relative_gap_l2"]))
    for ax, title in zip(axes, (r"$L_\infty$ margin $q_\infty(w_t)/\gamma_\infty$", r"$L_2$ margin $q_2(w_t)/\gamma_2$")):
        ax.axhline(1.0, color="black", ls="--", lw=0.6)
        ax.set_xscale("log")
        ax.set_xlim(1, 1e6)
        ax.set_ylim(0.4, 1.03)
        ax.set_xlabel("update $t$")
        ax.set_title(title, loc="left")
    axes[0].set_ylabel("margin / optimum")
    axes[0].legend(loc="upper left", handlelength=1.8)
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig2_margins")
    return ends


# ---------------------------------------------------------------- Figure 3
def fig3():
    base = os.path.join(P50, "2026-11-13-eps-timing")
    beta2 = 0.99
    eps = 1e-8
    r0 = [r for r in read_csv(os.path.join(base, "c", "seed-02_a-0.5_c-1_eps-0", "trajectory.csv.gz")) if int(r["step"]) >= 1]
    r1 = [r for r in read_csv(os.path.join(base, "c", "seed-02_a-0.5_c-1_eps-1e-08", "trajectory.csv.gz")) if int(r["step"]) >= 1]
    summ = json.load(open(os.path.join(base, "c", "seed-02_a-0.5_c-1_eps-1e-08", "summary.json")))
    ev = {e["kind"]: e["row"]["step"] for e in summ["events"] if e["threshold"] == eps}
    first, alls = ev["first"], ev["all"]

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(TEXTWIDTH, 1.65), gridspec_kw={"width_ratios": [1, 1, 1.05]})

    # (a) L-inf gap, seed 2
    for rows, key in ((r0, "adam0"), (r1, "adameps")):
        a.plot(col(rows, "step"), [100 * v for v in col(rows, "relative_gap_linf")], color=C[key], lw=1.1, label=LABEL[key])
    a.axvline(alls, color=C["adameps"], ls=":", lw=0.8)
    a.text(alls * 1.4, 150, "$\\varepsilon$ takes over\n($t$ = %s)" % format(alls, ","), color=C["adameps"], fontsize=6.5, va="center")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlim(1, 1e6)
    a.set_ylim(0.3, 700)
    a.set_yticks([1, 10, 100])
    a.set_yticklabels(["1", "10", "100"])
    a.set_xlabel("update $t$")
    a.set_ylabel(r"$L_\infty$ margin gap (%)")
    a.legend(loc="lower left", handlelength=1.4)
    panel_label(a, "(a)")

    # (b) largest sqrt(v) along the eps = 1e-8 run, with the beta2 lower bound
    s1 = col(r1, "step")
    mx = [10 ** v if not math.isnan(v) else math.nan for v in col(r1, "log10_max_sqrt_v")]
    b.plot(s1, mx, color=C["adameps"], lw=1.1)
    b.text(1.3, 0.12, r"largest $\sqrt{v_{t,j}}$", color=C["adameps"], fontsize=6.5, va="bottom")
    v1 = next(v for v in mx if not math.isnan(v))
    lb_s = [10 ** (k / 20) for k in range(0, 121)]
    b.plot(lb_s, [v1 * beta2 ** ((t - 1) / 2) for t in lb_s], color=C["grey"], ls=":", lw=1.0)
    b.text(6e3, 3e-13, "lower bound", color=C["grey"], fontsize=6.5, va="bottom")
    b.axhline(eps, color="black", ls="--", lw=0.6)
    b.text(1.3, eps * 1.8, r"$\varepsilon=10^{-8}$", fontsize=7, va="bottom")
    b.axvline(alls, color=C["adameps"], ls=":", lw=0.8)
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlim(1, 1e6)
    b.set_ylim(1e-13, 1)
    b.set_xlabel("update $t$")
    b.set_ylabel(r"$\sqrt{v}$ (Adam, $\varepsilon=10^{-8}$)")
    panel_label(b, "(b)")

    # (c) final gap against the learning-rate constant c, ten datasets
    e = [r for r in json.load(open(os.path.join(base, "endpoints.json"))) if r["suite"] == "c"]
    cs = sorted({r["c"] for r in e})
    for epsv, key, dx in ((0.0, "adam0", 0.94), (1e-8, "adameps", 1.06)):
        med, lo, hi = [], [], []
        for cv in cs:
            g = [100 * r["relative_gap_linf"] for r in e if r["c"] == cv and r["epsilon"] == epsv]
            assert len(g) == 10, (cv, epsv, len(g))
            med.append(statistics.median(g))
            lo.append(min(g))
            hi.append(max(g))
        xs = [cv * dx for cv in cs]
        c.errorbar(xs, med, yerr=[[m - l for m, l in zip(med, lo)], [h - m for m, h in zip(med, hi)]],
                   fmt="o", ms=3, color=C[key], lw=0.9, capsize=1.5, label=LABEL[key])
    c.set_xscale("log")
    c.set_yscale("log")
    c.set_ylim(0.3, 100)
    c.set_xticks(cs)
    c.set_xticklabels([f"{v:g}" for v in cs])
    c.minorticks_off()
    c.set_yticks([0.5, 1, 2, 5, 10, 20, 50])
    c.set_yticklabels(["0.5", "1", "2", "5", "10", "20", "50"])
    c.set_xlabel(r"constant $c$ in $\eta_t=c\,t^{-1/2}$")
    c.set_ylabel(r"$L_\infty$ gap after $10^6$ updates (%)")
    c.legend(loc="center right", handlelength=1.0, fontsize=6.5)
    panel_label(c, "(c)")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig3_epsilon")
    return first, alls


if __name__ == "__main__":
    fig1()
    ends = fig2()
    print("seed 2 endpoints (q_inf/gamma_inf, q_2/gamma_2):", {k: tuple(round(x, 5) for x in v) for k, v in ends.items()})
    print("eps takeover on seed 2 (first coordinate, all 50):", fig3())
