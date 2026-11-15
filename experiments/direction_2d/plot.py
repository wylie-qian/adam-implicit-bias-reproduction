"""Render the recorded learning experiment with existing Matplotlib."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_rows(path):
    with path.open(newline="") as stream:
        return [{key: (float(value) if value else None)
                 for key, value in row.items() if key != "method"}
                for row in csv.DictReader(stream)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    destination = args.run / "figures"
    if destination.exists():
        raise SystemExit("Figures already exist; choose a fresh run.")
    series = {name: read_rows(args.run / (name + ".csv")) for name in ("gd", "adam")}
    colors = {"gd": "#1261a0", "adam": "#d55e00"}
    labels = {"gd": "GD: fixed 0.25", "adam": "Adam: no epsilon / no bias correction"}
    with plt.rc_context({"font.size": 10, "svg.hashsalt": "direction-2d-learning"}):
        fig, axes = plt.subplots(3, 2, figsize=(12, 11), constrained_layout=True)
        for name, rows in series.items():
            steps = [r["step"] for r in rows]
            color, label = colors[name], labels[name]
            axes[0, 0].plot(steps, [r["loss"] for r in rows], color=color, label=label)
            for key, style, suffix in (("norm_l2", "-", "L2"), ("norm_linf", "--", "L-inf")):
                axes[0, 1].plot(steps, [r[key] for r in rows], style, color=color,
                                label=name.upper() + " " + suffix)
            valid = [r for r in rows if r["unit_l2_1"] is not None]
            axes[1, 0].plot([r["unit_l2_1"] for r in valid],
                            [r["unit_l2_2"] for r in valid], color=color, label=name.upper())
            if valid:
                axes[1, 0].scatter(valid[-1]["unit_l2_1"], valid[-1]["unit_l2_2"],
                                    marker="o", s=50, facecolors="none", edgecolors=color)
            axes[1, 1].plot(steps, [r["margin_l2"] for r in rows], color=color, label=name.upper())
            axes[2, 0].plot(steps, [r["margin_linf"] for r in rows], color=color, label=name.upper())
            key = "direction_error_l2_reference_" + ("l2" if name == "gd" else "linf")
            defined = [r for r in rows if r[key] is not None]
            axes[2, 1].plot([r["step"] for r in defined], [r[key] for r in defined],
                            color=color, label=name.upper() + " to its reference")
        axes[0, 0].set(title="Mean logistic loss", yscale="log", ylabel="Loss")
        axes[0, 1].set(title="Parameter norms (unbounded limits expected)", ylabel="Norm")
        axes[1, 0].scatter([3 / math.sqrt(13)], [2 / math.sqrt(13)], marker="*",
                            s=130, color="#1261a0", label="L2 max-margin reference")
        axes[1, 0].scatter([1 / math.sqrt(2)], [1 / math.sqrt(2)], marker="X",
                            s=80, color="#d55e00", label="L-inf reference, L2 normalized")
        axes[1, 0].set(title="L2-normalized direction (circle = final recorded state)",
                        xlabel="w1 / ||w||2", ylabel="w2 / ||w||2")
        axes[1, 0].set_aspect("equal", adjustable="datalim")
        axes[1, 1].axhline(7 / math.sqrt(13), color="black", ls=":", label="L2 optimum")
        axes[1, 1].set(title="Training L2-normalized margin", ylabel="q2")
        axes[2, 0].axhline(2.5, color="black", ls=":", label="L-inf optimum")
        axes[2, 0].set(title="Training L-inf-normalized margin", ylabel="q-infinity")
        axes[2, 1].set(title="Direction error (each method's theoretical reference)",
                        ylabel="L2 distance between L2 unit vectors (linear)")
        for row in axes:
            for ax in row:
                if ax is not axes[1, 0]:
                    ax.set_xscale("symlog", linthresh=10)
                    ax.set_xlim(0, max(r["step"] for rows in series.values() for r in rows) or 1)
                    ax.set_xlabel("Completed full-batch updates")
                ax.grid(alpha=0.25)
                ax.legend(fontsize=8)
        fig.suptitle("Fixed two-point learning experiment: finite trajectories, not a limit proof\n"
                        "Recorded checkpoints only; no smoothing or parameter tuning",
                        fontsize=12)
        destination.mkdir()
        fig.savefig(destination / "trajectories.png", dpi=150)
        fig.savefig(destination / "trajectories.svg", metadata={"Date": None})
        fig.savefig(destination / "trajectories.pdf", metadata={"CreationDate": None, "ModDate": None})
        plt.close(fig)
    metadata = {
        "python": platform.python_version(),
        "matplotlib": matplotlib.__version__,
        "plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_sha256": {name: hashlib.sha256((args.run / name).read_bytes()).hexdigest()
                         for name in ("gd.csv", "adam.csv", "summary.json")},
        "note": "Recorded data only; linear error axis retains zero and roundoff-sized errors.",
    }
    (destination / "plot-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
