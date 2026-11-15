"""Plot verified own four-rate measurements and summarize retained checkpoints."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from multirate_audit import MODES, verify
from run import sha, write_json


def render(root, output):
    verification = verify(root)
    manifest = json.loads((root / "manifest.json").read_text())
    runs = [json.loads((root / (c["run_id"] + ".json")).read_text())
            for c in manifest["planned_configs"]]
    output.mkdir(parents=True, exist_ok=False)
    styles = {
        "zero_epsilon": ("Correct epsilon = 0", "#009E73", "-"),
        "fixed_epsilon": ("Correct fixed epsilon = 1e-8", "#0072B2", "-"),
        "INVALID_wrong_scale": ("Wrong epsilon scaling (INVALID)", "#D55E00", "--"),
    }
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    summary = []
    for ax, a in zip(axes.flat, [.3, .5, .7, 1.]):
        selected = {r["config"]["mode"]: r for r in runs if r["config"]["a"] == a}
        row = {"a": a, "endpoints": {}, "fixed_epsilon_events": selected["fixed_epsilon"]["events"]}
        for mode in MODES:
            r = selected[mode]
            label, color, line = styles[mode]
            trace = [x for x in r["trace"] if x["step"] > 0]
            gaps = [abs(x["gap_linf"]) for x in trace]
            if any(not math.isfinite(v) or v <= 0 for v in gaps):
                raise ValueError("Log plot requires finite positive measured absolute gaps")
            ax.plot([x["step"] for x in trace], gaps, color=color, ls=line,
                    label=label, lw=1.9)
            ep = r["endpoint"]
            first = next((x for x in trace if x["log_loss"] < math.log(1e-8)), None)
            row["endpoints"][mode] = {
                "relative_gap_linf_percent": 100 * ep["relative_gap_linf"],
                "absolute_gap_linf": abs(ep["gap_linf"]),
                "log_loss": ep["log_loss"],
                "first_retained_checkpoint_loss_below_1e_minus8": None if first is None else {
                    "step": first["step"], "log_loss": first["log_loss"],
                    "relative_gap_linf_percent": 100 * first["relative_gap_linf"],
                    "log10_min_sqrt_v_over_actual_epsilon": first["log10_min_sqrt_v_over_actual_epsilon"],
                    "log10_max_sqrt_v_over_actual_epsilon": first["log10_max_sqrt_v_over_actual_epsilon"],
                },
            }
        row["fixed_vs_zero_gap_difference_percentage_points"] = abs(
            row["endpoints"]["fixed_epsilon"]["relative_gap_linf_percent"] -
            row["endpoints"]["zero_epsilon"]["relative_gap_linf_percent"])
        row["invalid_vs_zero_gap_difference_percentage_points"] = abs(
            row["endpoints"]["INVALID_wrong_scale"]["relative_gap_linf_percent"] -
            row["endpoints"]["zero_epsilon"]["relative_gap_linf_percent"])
        summary.append(row)
        ax.set(xscale="log", yscale="log", title=f"Learning-rate exponent a = {a:g}")
        ax.grid(alpha=.17)
    for ax in axes[-1]: ax.set_xlabel("Completed updates")
    for ax in axes[:, 0]: ax.set_ylabel("Absolute L-infinity margin gap")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Own four-rate diagnostic: one million updates per condition", fontsize=16)
    fig.subplots_adjust(top=.88, bottom=.13, hspace=.26, wspace=.12)
    for ext in ["png", "svg"]:
        fig.savefig(output / f"four_rate_comparison.{ext}", dpi=180, facecolor="white")
        if ext == "svg":
            path = output / f"four_rate_comparison.{ext}"
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5.2), layout="constrained")
    for mode in MODES:
        r = next(r for r in runs if r["config"]["a"] == .5 and r["config"]["mode"] == mode)
        trace = [x for x in r["trace"] if x["step"] > 0]
        label, color, line = styles[mode]
        ax.plot([x["step"] for x in trace], [100*x["relative_gap_linf"] for x in trace],
                label=label, color=color, ls=line, lw=2.5)
    ax.set(xscale="log", yscale="log", xlabel="Completed updates",
           ylabel="Relative L-infinity margin gap (%)",
           title="Own paired experiment: learning rate = 1 / sqrt(t)")
    ax.grid(alpha=.17)
    ax.legend(fontsize=11, loc="lower left")
    for ext in ["png", "svg"]:
        fig.savefig(output / f"slide_epsilon_comparison.{ext}", dpi=180, facecolor="white")
        if ext == "svg":
            path = output / f"slide_epsilon_comparison.{ext}"
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)
    report = {
        "scope": "Own frozen seed-2 dataset and shared initialization; not author arrays or a pointwise figure reproduction",
        "source_sha256": sha(__file__), "input_manifest_sha256": sha(root / "manifest.json"),
        "verification": verification,
        "plot_quantity": "Absolute |gamma_infinity - min_i y_i x_i^T w / ||w||_infinity|, matching Figure 2 quantity",
        "table_quantity": "Relative gap percent = 100 * (gamma_infinity - normalized margin) / gamma_infinity",
        "loss_checkpoint_scope": "First retained observation below 1e-8, not an exact first crossing or a rerun with early stopping",
        "rows": summary,
    }
    write_json(output / "comparison.json", report)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", type=Path, default=Path(__file__).resolve().parent / "results/2026-11-12-multirate")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = render(args.evidence, args.output)
    print(json.dumps({"rows": len(result["rows"]), "output": str(args.output)}))
