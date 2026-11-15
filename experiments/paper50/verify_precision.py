"""Check saved Adam updates with an independent Decimal implementation.

This reconstructs raw moments at selected saved states, including states beyond
float64 underflow. It is NOT a high-precision rerun of the full training history.
Only the Python standard library and the existing NumPy dependency are needed.
"""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import json
import math
from pathlib import Path

import numpy as np

from run import scaled_adam_step, sha

HERE = Path(__file__).resolve().parent
DEFAULT_EVIDENCE = HERE / "results/2026-11-12-alignment"


def decimal_update(z, w, m_scaled, v_scaled, scale, eta, epsilon, precision):
    """Literal unscaled equations, starting from the SAME represented state."""
    with localcontext() as ctx:
        ctx.prec = precision
        d = lambda x: Decimal.from_float(float(x))
        x = [[d(v) for v in row] for row in z]
        weights = [d(v) for v in w]
        raw_factor = (-d(scale)).exp()
        m = [d(v) * raw_factor for v in m_scaled]
        v = [d(a) * raw_factor * raw_factor for a in v_scaled]
        h = []
        for row in x:
            margin = sum(a * b for a, b in zip(row, weights))
            h.append(Decimal(1) / (Decimal(1) + margin.exp()))
        gradient = [-sum(row[j] * loss_weight for row, loss_weight in zip(x, h)) / len(x)
                    for j in range(len(weights))]
        beta1, beta2 = d(.9), d(.99)
        next_m = [beta1 * a + (1 - beta1) * g for a, g in zip(m, gradient)]
        next_v = [beta2 * a + (1 - beta2) * g * g for a, g in zip(v, gradient)]
        step = [d(eta) * a / (b.sqrt() + d(epsilon)) for a, b in zip(next_m, next_v)]
        next_w = [a - b for a, b in zip(weights, step)]
        return np.array([float(a) for a in next_w]), np.array([float(a) for a in step])


def check(evidence):
    with np.load(evidence / "data.npz") as data:
        z = data["y"][:, None] * data["X"]
    targets = [0, 3118, 3683, 97263, 1000000]
    results = []
    inputs = {"data.npz": sha(evidence / "data.npz")}
    min_subnormal_log = math.log(float(np.nextafter(0., 1.)))
    for epsilon in [0., 1e-8]:
        runid = f"figure1_adam_eps-{epsilon:g}"
        state_path = evidence / runid / "states.npz"
        inputs[f"{runid}/states.npz"] = sha(state_path)
        with np.load(state_path) as saved:
            for target in targets:
                # The chosen states must be real saved observations, not interpolated.
                match = np.flatnonzero(saved["steps"] == target)
                if len(match) != 1:
                    raise ValueError(f"Requested saved checkpoint absent: {runid}/{target}")
                i = int(match[0])
                w, m, v, scale = (saved[k][i] for k in
                                  ["weights", "m_scaled", "v_scaled", "log_scales"])
                eta = (target + 1) ** (-.5)
                w80, step80 = decimal_update(z, w, m, v, scale, eta, epsilon, 80)
                w120, step120 = decimal_update(z, w, m, v, scale, eta, epsilon, 120)
                new_w, new_m, new_v, new_scale = scaled_adam_step(
                    z, w, m, v, scale, .9, .99, eta, epsilon)
                denominator = np.sqrt(new_v)
                if epsilon:
                    denominator += math.exp(math.log(epsilon) + new_scale)
                float_step = eta * new_m / denominator
                precision_error = float(np.max(np.abs(step80 - step120)))
                step_error = float(np.max(np.abs(float_step - step120)))
                weight_error = float(np.max(np.abs(new_w - w120)))
                if not np.allclose(step80, step120, atol=1e-25, rtol=1e-18):
                    raise ArithmeticError("Decimal 80/120-digit checks disagree")
                if not np.allclose(float_step, step120, atol=5e-13, rtol=5e-9):
                    raise ArithmeticError("Scaled update disagrees with raw Decimal update")
                if not np.allclose(new_w, w120, atol=1e-12, rtol=2e-14):
                    raise ArithmeticError("Updated weights disagree with Decimal")
                if target:
                    raw_log_v = np.log(v) - 2 * scale
                    min_log10_sqrt_v = float(raw_log_v.min() / (2 * math.log(10)))
                    subnormal_count = int(np.sum(raw_log_v < min_subnormal_log))
                else:
                    min_log10_sqrt_v, subnormal_count = None, 0
                results.append({"run_id": runid, "saved_step": target,
                                "next_learning_rate": eta, "epsilon": epsilon,
                                "decimal_80_vs_120_step_max_abs": precision_error,
                                "float_vs_decimal_step_max_abs": step_error,
                                "float_vs_decimal_weight_max_abs": weight_error,
                                "saved_min_log10_sqrt_v_raw": min_log10_sqrt_v,
                                "saved_v_below_float64_min_subnormal_coordinates": subnormal_count})
    return {"passed": True, "precision_digits": [80, 120], "checked_updates": len(results),
            "scope": "Independent one-step raw-moment Decimal checks at saved float64 states; not full-history high-precision training",
            "source_sha256": sha(__file__), "frozen_update_source_sha256": sha(HERE / "run.py"),
            "input_sha256": inputs, "results": results}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    result = check(args.evidence)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps({"passed": result["passed"], "checked_updates": result["checked_updates"],
                      "max_step_error": max(x["float_vs_decimal_step_max_abs"] for x in result["results"]),
                      "max_weight_error": max(x["float_vs_decimal_weight_max_abs"] for x in result["results"])}))


if __name__ == "__main__":
    main()
