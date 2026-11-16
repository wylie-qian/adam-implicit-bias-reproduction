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
import matplotlib.ticker  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "docs", "report", "figures")
P50 = os.path.join(ROOT, "experiments", "paper50", "results")

# Slate for the baselines (GD dark, GDM light, 3:1 contrast on white) and two accents for the two Adam runs:
# vermilion for eps = 0, steel blue for eps = 1e-8. The two accents have a similar grey level, so the
# panels also name them with direct labels.
C = {"gd": "#475569", "gdm": "#8796AB", "adam0": "#E4572E", "adameps": "#2E86AB", "grey": "#8A8F99"}
LABEL = {"gd": "GD", "gdm": "GDM", "adam0": r"Adam, $\varepsilon=0$", "adameps": r"Adam, $\varepsilon=10^{-8}$"}

# Sans-serif text with matching sans maths (Arial; DejaVu Sans where Arial is missing).
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.fallback": "stixsans",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 6.3,
    "ytick.labelsize": 6.3,
    "axes.linewidth": 0.55,
    "axes.edgecolor": "#4A4A4A",
    "axes.labelcolor": "#262626",
    "axes.titleweight": "normal",
    "xtick.color": "#4A4A4A",
    "ytick.color": "#4A4A4A",
    "xtick.labelcolor": "#404040",
    "ytick.labelcolor": "#404040",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": "#ECEEF1",
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "lines.solid_capstyle": "round",
    "pdf.fonttype": 42,  # embed TrueType, not Type 3
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})
TEXTWIDTH = 5.5  # inches, NeurIPS text width
LW = 1.3         # data lines
REF = dict(color="#7A7F87", ls=(0, (3.5, 2.5)), lw=0.6)  # reference lines (optimum, epsilon)
SMALL = 6.3      # in-plot labels
MUTED = "#6B7078"  # annotation text


def tint(hex_colour, f):
    """Mix a colour with white; f = 0 keeps it, f = 1 gives white."""
    rgb = [int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(c + (255 - c) * f):02X}" for c in rgb)


def end_dot(ax, x, y, colour):
    ax.plot([x], [y], "o", ms=2.6, color=colour, mec="white", mew=0.5, zorder=5, clip_on=False)


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


def panel_label(ax, s, x=-0.02, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, ha="right", va="bottom", fontweight="bold", fontsize=7.5,
            color="#1A1A1A")


def log_axis(ax, axis="x"):
    """Decades as major ticks, unlabelled minor ticks at 2..9."""
    major = matplotlib.ticker.LogLocator(base=10, numticks=12)
    minor = matplotlib.ticker.LogLocator(base=10, subs=range(2, 10), numticks=12)
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(major)
    target.set_minor_locator(minor)
    target.set_minor_formatter(matplotlib.ticker.NullFormatter())


def new_figure(height, ncols, width_ratios, wspace=0.04, **kw):
    """Figure at the exact text width, so text keeps its nominal size when LaTeX places it at \\linewidth.
    Heights equal the printed heights of the earlier, cropped versions, so the page layout does not move."""
    fig, axes = plt.subplots(1, ncols, figsize=(TEXTWIDTH, height), layout="constrained",
                             gridspec_kw={"width_ratios": width_ratios}, **kw)
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=wspace)
    return fig, axes


def save(fig, name):
    # Page size kept as set (no "tight" crop). No creation date in the PDF, so a rerun gives the same bytes.
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), metadata={"CreationDate": None})
    fig.savefig(os.path.join(OUT, f"{name}.png"))
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- Figure 1
def fig1():
    """(a) steepest-descent steps in the L2 ball and the L-inf box; (b) our 2D runs."""
    fig, (a, b) = new_figure(1.56, 2, [1, 1.9])

    # (a) geometry for g = (1, 0.05), as on slide 3
    g = (1.0, 0.05)
    n = math.hypot(*g)
    t = [i / 200 * 2 * math.pi for i in range(201)]
    sq_x, sq_y = [-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1]
    a.fill(sq_x, sq_y, color=tint(C["adam0"], 0.9), lw=0)                                     # L-inf box
    a.fill([math.cos(s) for s in t], [math.sin(s) for s in t], color=tint(C["gd"], 0.88), lw=0)  # L2 ball
    a.plot(sq_x, sq_y, color=C["adam0"], lw=0.9)
    a.plot([math.cos(s) for s in t], [math.sin(s) for s in t], color=C["gd"], lw=0.9)
    kw = dict(arrowstyle="-|>", mutation_scale=7, lw=1.1, shrinkA=0, shrinkB=0)
    a.annotate("", xy=(0.55 * g[0] / n, 0.55 * g[1] / n), xytext=(0, 0), arrowprops=dict(color="#222222", **kw))
    a.text(0.6, 0.0, r"$g$", ha="left", va="center")
    a.annotate("", xy=(-g[0] / n, -g[1] / n), xytext=(0, 0), arrowprops=dict(color=C["gd"], **kw))
    a.annotate("", xy=(-1, -1), xytext=(0, 0), arrowprops=dict(color=C["adam0"], **kw))
    a.plot([0], [0], "o", ms=2.2, color="#222222")
    a.text(-1.12, 0.0, r"GD step" "\n" r"$-g/\|g\|_2$", color=C["gd"], ha="right", va="center", fontsize=SMALL,
           linespacing=1.15)
    a.text(0.0, -1.12, r"sign step $-\mathrm{sign}(g)$", color=C["adam0"], ha="center", va="top", fontsize=SMALL)
    a.text(0.0, 0.58, r"$\|d\|_2\leq 1$", color=C["gd"], ha="center", va="center", fontsize=SMALL)
    a.text(0.0, 1.08, r"$\|d\|_\infty\leq 1$", color=C["adam0"], ha="center", va="bottom", fontsize=SMALL)
    a.set_aspect("equal")
    a.set_xlim(-2.05, 1.2)
    a.set_ylim(-1.5, 1.45)
    a.axis("off")
    panel_label(a, "(a)", x=0.04, y=0.97)

    # (b) direction of w_t in our 2D runs (z_1 = (1, 2), z_2 = (2, 1/2))
    run = os.path.join(ROOT, "experiments", "direction_2d", "runs", "frozen-20k-fixed-metrics")
    l2_dir = math.degrees(math.atan2(2, 3))   # max L2-margin direction (3, 2)/sqrt(13)
    linf_dir = 45.0                             # max L-inf-margin direction (1, 1)
    b.axhline(l2_dir, color=C["gd"], ls=(0, (4, 2.5)), lw=0.7)
    b.axhline(linf_dir, color=C["adam0"], ls=(0, (4, 2.5)), lw=0.7)
    for key, fname in (("gd", "gd.csv"), ("adam0", "adam.csv")):
        rows = [r for r in read_csv(os.path.join(run, fname)) if int(r["step"]) >= 1]
        steps = col(rows, "step")
        ang = [math.degrees(math.atan2(float(r["w2"]), float(r["w1"]))) for r in rows]
        b.plot(steps, ang, color=C[key], lw=LW)
    b.text(1.2, l2_dir + 0.4, "max $L_2$-margin direction (33.7°)", color=C["gd"], va="bottom", fontsize=SMALL)
    b.text(1.2, linf_dir + 0.4, "max $L_\\infty$-margin direction (45°)", color=C["adam0"], va="bottom",
           fontsize=SMALL)
    b.text(2e3, 36.6, "GD", color=C["gd"], ha="center", va="top", fontsize=SMALL)
    b.text(30, 43.5, r"Adam, $\varepsilon=0$", color=C["adam0"], ha="center", va="top", fontsize=SMALL)
    b.set_ylim(32.5, 46.8)
    b.set_yticks([35, 40, 45])
    b.set_xscale("log")
    log_axis(b, "x")
    b.set_xlim(1, 2e4)
    b.set_xlabel("update $t$")
    b.set_ylabel(r"angle of $w_t$ (degrees)")
    panel_label(b, "(b)")
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
    fig, axes = new_figure(1.51, 2, [1, 1], wspace=0.06, sharex=True, sharey=True)
    ends = {}
    for ax in axes:
        ax.axhline(1.0, **REF)
    for key, name in runs.items():
        rows = [r for r in legend_traj(name) if int(r["step"]) >= 1]
        s = col(rows, "step")
        for ax, gap in zip(axes, ("relative_gap_linf", "relative_gap_l2")):
            ratio = [1 - v for v in col(rows, gap)]
            if key == "gdm":  # GDM lies on GD: draw it as a light dashed line on top
                ax.plot(s, ratio, color=C[key], lw=0.9, ls=(0, (2.5, 1.8)), label=LABEL[key])
            else:
                ax.plot(s, ratio, color=C[key], lw=LW, label=LABEL[key])
                end_dot(ax, s[-1], ratio[-1], C[key])
        ends[key] = (1 - float(rows[-1]["relative_gap_linf"]), 1 - float(rows[-1]["relative_gap_l2"]))
    for ax, title in zip(axes, (r"$L_\infty$ margin $q_\infty(w_t)/\gamma_\infty$", r"$L_2$ margin $q_2(w_t)/\gamma_2$")):
        ax.set_xscale("log")
        log_axis(ax, "x")
        ax.set_xlim(1, 1e6)
        ax.set_ylim(0.4, 1.04)
        ax.set_yticks([0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("update $t$")
        ax.set_title(title, loc="left", pad=4)
        ax.tick_params(axis="y", labelleft=True)
    axes[0].set_ylabel("margin / optimum")
    axes[1].text(1.5, 0.985, "optimum", color=REF["color"], fontsize=SMALL, ha="left", va="top")
    # Direct labels next to the curves in (a), instead of a legend; (b) uses the same colours.
    lab = dict(fontsize=SMALL, ha="center")
    axes[0].text(1.0e3, 0.84, r"Adam, $\varepsilon=0$", color=C["adam0"], va="bottom", **lab)
    axes[0].text(1.5e5, 0.705, r"Adam, $\varepsilon=10^{-8}$", color=C["adameps"], va="bottom", **lab)
    axes[0].text(1.5e5, 0.545, "GD and GDM", color=C["gd"], va="top", **lab)
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
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

    fig, (a, b, c) = new_figure(1.47, 3, [1, 1, 1.05])

    # (a) L-inf gap, seed 2; the pale band marks the updates after eps has taken over
    for ax in (a, b):
        ax.axvspan(alls, 1e6, color=tint(C["adameps"], 0.93), lw=0, zorder=0)
        ax.axvline(alls, color=C["adameps"], ls=(0, (1, 1.5)), lw=0.8)
    for rows, key in ((r0, "adam0"), (r1, "adameps")):
        s = col(rows, "step")
        y = [100 * v for v in col(rows, "relative_gap_linf")]
        a.plot(s, y, color=C[key], lw=LW)
        end_dot(a, s[-1], y[-1], C[key])
    a.text(alls / 1.35, 330, "$\\varepsilon$ takes over\n($t$ = %s)" % format(alls, ","), color=C["adameps"],
           fontsize=SMALL, ha="right", va="center", linespacing=1.15)
    a.text(1.5e5, 52, r"$\varepsilon=10^{-8}$", color=C["adameps"], fontsize=SMALL, ha="center", va="bottom")
    a.text(1.2e4, 2.2, r"$\varepsilon=0$", color=C["adam0"], fontsize=SMALL, ha="center", va="top")
    a.set_xscale("log")
    a.set_yscale("log")
    log_axis(a, "x")
    a.set_xlim(1, 1e6)
    a.set_ylim(0.3, 900)
    a.set_yticks([1, 10, 100])
    a.set_yticklabels(["1", "10", "100"])
    a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_xlabel("update $t$")
    a.set_ylabel(r"$L_\infty$ margin gap (%)")
    panel_label(a, "(a)")

    # (b) largest sqrt(v) along the eps = 1e-8 run, with the beta2 lower bound
    s1 = col(r1, "step")
    mx = [10 ** v if not math.isnan(v) else math.nan for v in col(r1, "log10_max_sqrt_v")]
    v1 = next(v for v in mx if not math.isnan(v))
    lb_s = [10 ** (k / 40) for k in range(0, 241)]
    # The bound is drawn as a wide pale band under the run, so that "the run follows the bound" is visible.
    b.plot(lb_s, [v1 * beta2 ** ((t - 1) / 2) for t in lb_s], color=C["grey"], lw=3.2, alpha=0.35,
           solid_capstyle="butt")
    b.axhline(eps, **REF)
    b.plot(s1, mx, color=C["adameps"], lw=LW)
    b.text(1.5, eps * 2.2, r"$\varepsilon=10^{-8}$", color=REF["color"], fontsize=SMALL, va="bottom")
    b.annotate("lower bound (5)", xy=(4.2e3, 3e-11), xytext=(1.6, 1.5e-12), color=MUTED, fontsize=SMALL,
               ha="left", va="center",
               arrowprops=dict(arrowstyle="-", color="#9DA2AA", lw=0.6, shrinkA=1, shrinkB=1))
    b.set_xscale("log")
    b.set_yscale("log")
    log_axis(b, "x")
    b.set_xlim(1, 1e6)
    b.set_ylim(1e-13, 1)
    b.set_yticks([1e-1, 1e-5, 1e-9, 1e-13])
    b.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    b.set_xlabel("update $t$")
    b.set_ylabel(r"largest $\sqrt{v_{t,j}}$")
    panel_label(b, "(b)")

    # (c) final gap against the learning-rate constant c, ten datasets
    e = [r for r in json.load(open(os.path.join(base, "endpoints.json"))) if r["suite"] == "c"]
    cs = sorted({r["c"] for r in e})
    # Median over the ten datasets as a line with points; the band is the min-max range.
    for epsv, key in ((0.0, "adam0"), (1e-8, "adameps")):
        med, lo, hi = [], [], []
        for cv in cs:
            g = [100 * r["relative_gap_linf"] for r in e if r["c"] == cv and r["epsilon"] == epsv]
            assert len(g) == 10, (cv, epsv, len(g))
            med.append(statistics.median(g))
            lo.append(min(g))
            hi.append(max(g))
        c.fill_between(cs, lo, hi, color=C[key], alpha=0.16, lw=0)
        c.plot(cs, med, "-o", color=C[key], lw=1.0, ms=3.2, mec="white", mew=0.5)
    c.text(0.04, 42, r"$\varepsilon=10^{-8}$", color=C["adameps"], fontsize=SMALL, ha="left", va="bottom")
    c.text(0.04, 5.0, r"$\varepsilon=0$", color=C["adam0"], fontsize=SMALL, ha="left", va="bottom")
    c.set_xscale("log")
    c.set_yscale("log")
    c.set_xlim(0.022, 4.1)
    c.set_ylim(0.3, 100)
    c.set_xticks(cs)
    c.set_xticklabels([f"{v:g}" for v in cs])
    c.minorticks_off()
    c.set_yticks([0.5, 1, 2, 5, 10, 20, 50])
    c.set_yticklabels(["0.5", "1", "2", "5", "10", "20", "50"])
    c.set_xlabel(r"constant $c$ in $\eta_t=c\,t^{-1/2}$")
    c.set_ylabel(r"final $L_\infty$ gap (%)")
    panel_label(c, "(c)")
    save(fig, "fig3_epsilon")
    return first, alls


if __name__ == "__main__":
    fig1()
    ends = fig2()
    print("seed 2 endpoints (q_inf/gamma_inf, q_2/gamma_2):", {k: tuple(round(x, 5) for x in v) for k, v in ends.items()})
    print("eps takeover on seed 2 (first coordinate, all 50):", fig3())
