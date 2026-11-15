"""Diagnose fixed-epsilon update geometry without rerunning training.

The probe compares denominators at the SAME stored first/second moments. Its
zero-epsilon vector is a counterfactual, not a separately trained zero-epsilon
trajectory and not a rewind of the moment history. Stored post-update weights
are used only for the separately labeled current-gradient alignment diagnostic.
Thus these finite-state measurements do not establish an asymptotic L2 theorem.

Example (after the multirate run has completed)::

    python experiments/paper50/transition_geometry.py analyze \
        --input-dir experiments/paper50/results/2026-11-12-multirate

This reads either multirate checkpoint JSON or convention-audit endpoint JSON.
All matching outcomes are retained, including INVALID_RECURRENCE controls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

LN10 = math.log(10.)
SCALED_REPRESENTATION = "exp(scale)*m_raw, exp(2*scale)*v_raw"


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(json.dumps(array.shape).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def _unit(vector):
    """Normalize finite vectors without overflowing or underflowing their norm."""
    vector = np.asarray(vector, dtype=np.float64)
    maximum = float(np.max(np.abs(vector)))
    if maximum == 0.:
        return None
    scaled = vector / maximum
    return scaled / np.linalg.norm(scaled)


def _signed_log_unit(signs, log_abs):
    active = np.asarray(signs) != 0
    if not np.any(active):
        return None
    logs = np.asarray(log_abs)[active]
    vector = np.zeros_like(np.asarray(signs), dtype=np.float64)
    vector[active] = np.asarray(signs)[active] * np.exp(logs - float(np.max(logs)))
    return _unit(vector)


def _cosine(unit_a, unit_b):
    if unit_a is None or unit_b is None:
        return None
    return float(np.clip(np.dot(unit_a, unit_b), -1., 1.))


def _summary(values):
    values = np.asarray(values, dtype=np.float64)
    return dict(min=float(values.min()), median=float(np.median(values)), max=float(values.max()))


def gradient_direction(z, weights):
    """Stable full-batch mean-logistic gradient direction and log10 L2 norm.

    A common positive factor is removed before exponentiating. The returned
    direction is unchanged even when every unscaled logistic weight underflows.
    """
    margins = z @ weights
    if not np.isfinite(margins).all():
        raise ValueError("Unrepresentable margins")
    log_weights = -np.logaddexp(0., margins)
    shift = float(np.max(log_weights))
    scaled_gradient = -(z.T @ np.exp(log_weights - shift)) / len(z)
    unit = _unit(scaled_gradient)
    maximum = float(np.max(np.abs(scaled_gradient)))
    log_norm = None if unit is None else (
        shift + math.log(maximum) + math.log(float(np.linalg.norm(scaled_gradient / maximum)))
    ) / LN10
    return unit, log_norm


def state_geometry(first_moment, second_moment, log_scale, probe_epsilon=1e-8,
                   z=None, weights=None, coordinates=False):
    """Measure fixed-epsilon counterfactuals from raw or jointly scaled moments.

    M=exp(b)*m and V=exp(2*b)*v; use b=0 for raw moments. No raw exp(-b)
    reconstruction is needed. Fractions may round to zero, but their logarithms
    remain available. Coordinate ratios at an uninitialized v=0 state are
    explicitly undefined for the zero-epsilon counterfactual.
    """
    v = np.asarray(second_moment, dtype=np.float64)
    m = None if first_moment is None else np.asarray(first_moment, dtype=np.float64)
    b, eps = float(log_scale), float(probe_epsilon)
    if (v.ndim != 1 or not len(v) or not np.isfinite(v).all() or np.any(v < 0)
            or not math.isfinite(b) or not math.isfinite(eps) or eps <= 0
            or (m is not None and (m.shape != v.shape or not np.isfinite(m).all()))):
        raise ValueError("Expected finite matching moments, nonnegative V, and positive probe epsilon")
    if np.any(v == 0):
        if m is not None and np.any(m[v == 0] != 0):
            raise ValueError("Nonzero first moment with zero second moment")
        return dict(status="undefined_zero_second_moment", dimension=len(v),
                    reason="Zero-epsilon counterfactual includes 0/0; no denominator floor is introduced.")

    log_sqrt_raw = .5 * np.log(v) - b
    log_ratio = log_sqrt_raw - math.log(eps)
    # A=sqrt(v_raw)/(sqrt(v_raw)+eps); this is the RETAINED fraction,
    # complementary to the removed fraction eps/(sqrt(v_raw)+eps).
    log_retained = -np.logaddexp(0., -log_ratio)
    retained = np.exp(log_retained)
    result = dict(
        status="defined", dimension=len(v),
        retained_update_fraction=_summary(retained),
        log10_retained_update_fraction=_summary(log_retained / LN10),
        log10_sqrt_v_over_probe_epsilon=_summary(log_ratio / LN10),
        fraction_coordinates_sqrt_v_at_most_epsilon=float(np.mean(log_ratio <= 0)),
        fraction_coordinates_sqrt_v_at_most_tenth_epsilon=float(np.mean(log_ratio <= math.log(.1))),
        cosine_update_probe_vs_minus_first_moment=None,
        cosine_update_zero_vs_minus_first_moment=None,
        cosine_update_probe_vs_update_zero=None,
        cosine_first_moment_vs_gradient_at_stored_weights=None,
        log10_gradient_l2_at_stored_weights=None,
        direction_status="unavailable_first_moment" if m is None else "defined",
    )
    if m is not None:
        active = m != 0
        log_m = np.zeros_like(m)
        log_m[active] = np.log(np.abs(m[active]))
        minus_m_unit = _unit(-m)
        zero_unit = _signed_log_unit(-np.sign(m), log_m - .5 * np.log(v))
        probe_unit = _signed_log_unit(-np.sign(m), log_m - b - np.logaddexp(log_sqrt_raw, math.log(eps)))
        result.update(
            cosine_update_probe_vs_minus_first_moment=_cosine(probe_unit, minus_m_unit),
            cosine_update_zero_vs_minus_first_moment=_cosine(zero_unit, minus_m_unit),
            cosine_update_probe_vs_update_zero=_cosine(probe_unit, zero_unit),
            direction_status="undefined_zero_first_moment" if minus_m_unit is None else "defined",
        )
    if z is not None and weights is not None:
        z, weights = np.asarray(z, dtype=np.float64), np.asarray(weights, dtype=np.float64)
        if (z.ndim != 2 or not len(z) or z.shape[1] != len(v) or weights.shape != v.shape
                or not np.isfinite(z).all() or not np.isfinite(weights).all()):
            raise ValueError("Invalid evaluation data or weights")
        gradient_unit, log_norm = gradient_direction(z, weights)
        result["log10_gradient_l2_at_stored_weights"] = log_norm
        if m is not None:
            result["cosine_first_moment_vs_gradient_at_stored_weights"] = _cosine(_unit(m), gradient_unit)
    if coordinates:
        result["coordinates"] = dict(retained_update_fraction=retained.tolist(),
                                     log10_retained_update_fraction=(log_retained / LN10).tolist(),
                                     log10_sqrt_v_over_probe_epsilon=(log_ratio / LN10).tolist())
    return result


def test_invariants():
    """Scientific invariants, including an analytically checkable 2-D state."""
    m, v, eps = np.array([2., -3.]), np.array([4., 9.]), 2.
    z, w = np.eye(2), np.array([0., math.log(3.)])
    direct_zero, direct_probe = -m / np.sqrt(v), -m / (np.sqrt(v) + eps)
    direct_gradient = np.array([-.25, -.125])
    report = state_geometry(m, v, 0., eps, z, w, coordinates=True)
    np.testing.assert_allclose(report["coordinates"]["retained_update_fraction"], [.5, .6], rtol=2e-15)
    for key, a, b in [
        ("cosine_update_probe_vs_minus_first_moment", direct_probe, -m),
        ("cosine_update_zero_vs_minus_first_moment", direct_zero, -m),
        ("cosine_update_probe_vs_update_zero", direct_probe, direct_zero),
        ("cosine_first_moment_vs_gradient_at_stored_weights", m, direct_gradient),
    ]:
        np.testing.assert_allclose(report[key], _cosine(_unit(a), _unit(b)), atol=3e-15, rtol=0.)
    # Merely changing the moment representation must not change physical results.
    shifted = state_geometry(m * math.exp(300.), v * math.exp(600.), 300., eps, z, w, True)
    for key in report["coordinates"]:
        np.testing.assert_allclose(report["coordinates"][key], shifted["coordinates"][key], atol=3e-13, rtol=0.)
    for key in report:
        if key.startswith("cosine_"):
            np.testing.assert_allclose(report[key], shifted[key], atol=3e-13, rtol=0.)
    # Proportional denominators change magnitude but not direction.
    proportional = state_geometry(m, np.array([4., 4.]), 0., eps)
    np.testing.assert_allclose(proportional["cosine_update_probe_vs_update_zero"], 1., atol=2e-15)
    # An unrepresentably small RAW state remains diagnosable from scaled moments.
    tiny = state_geometry(m, v, 2000., eps, coordinates=True)
    assert tiny["retained_update_fraction"]["max"] == 0.
    np.testing.assert_allclose(tiny["coordinates"]["log10_retained_update_fraction"],
                               (np.log(np.sqrt(v) / eps) - 2000.) / LN10, atol=3e-13)
    np.testing.assert_allclose(tiny["cosine_update_probe_vs_minus_first_moment"], 1., atol=2e-15)
    # Raw logistic gradients underflow at margin 2000; direction should not.
    direction, log_norm = gradient_direction(np.eye(2), np.array([2000., 2001.]))
    np.testing.assert_allclose(direction, _unit(-np.array([1., math.exp(-1.)])), atol=2e-15)
    assert log_norm < -800.
    assert state_geometry(np.zeros(2), np.zeros(2), 0.)["status"] == "undefined_zero_second_moment"
    return dict(passed=True, checks=["known_2d_raw_state", "moment_representation_invariance",
                                    "proportional_denominator_direction", "raw_moment_underflow",
                                    "raw_gradient_underflow", "uninitialized_state_is_undefined"])


def _state_parts(state):
    if "second_moment_scaled" in state:
        return state.get("first_moment_scaled"), state["second_moment_scaled"], state["log_scale_applied_on_last_update"]
    representation = state.get("moment_representation")
    if representation not in ("raw", SCALED_REPRESENTATION):
        raise ValueError("Unknown moment representation")
    scale = 0. if representation == "raw" else state["log_scale_applied_on_last_update"]
    return state["first_moment"], state["second_moment"], scale


def analyze(input_dir, probe_epsilon=1e-8):
    """Consume every completed compatible outcome; do not select by its metric."""
    input_dir = Path(input_dir)
    data_path, reference_path = input_dir / "data.json", input_dir / "reference.json"
    data, reference = json.loads(data_path.read_text()), json.loads(reference_path.read_text())
    x = np.asarray(data["X"], dtype=np.float64)
    y = np.asarray(data["y"], dtype=data.get("y_dtype", "int64"))
    w0 = np.asarray(data["w0"], dtype=np.float64)
    if (x.ndim != 2 or not len(x) or y.shape != (len(x),) or w0.shape != (x.shape[1],)
            or not np.isfinite(x).all() or not np.isfinite(w0).all() or not np.isin(y, [-1, 1]).all()):
        raise ValueError("Invalid original dataset")
    for name, value in [("X", x), ("y", y), ("w0", w0)]:
        if reference.get(name + "_array_sha256") != array_sha(value):
            raise ValueError("Reference/data identity mismatch: " + name)
    z, runs, inputs = y[:, None] * x, [], {}
    inputs.update({p.name: file_sha(p) for p in (data_path, reference_path)})
    for path in sorted(input_dir.glob("*.json")):
        obj = json.loads(path.read_text())
        if "config" not in obj or "endpoint_state" not in obj:
            continue
        cfg = obj["config"]
        if (cfg.get("bias_correction", False) or cfg.get("epsilon_placement", "outside_sqrt") != "outside_sqrt"
                or cfg.get("loss_multiplier", 1.) != 1.):
            raise ValueError("This helper requires uncorrected moments of the mean loss with outside-sqrt epsilon")
        if obj["status"] != "completed" or obj["endpoint_state"] is None:
            raise ValueError("Incomplete outcome: " + path.name)
        state = obj["endpoint_state"]
        m, v, scale = _state_parts(state)
        endpoint = state_geometry(m, v, scale, probe_epsilon, z, state["weights"], coordinates=True)
        trace = []
        for row in obj.get("trace", []):
            if "second_moment_scaled" in row:
                row_m, row_v, row_scale = _state_parts(row)
                measurement = state_geometry(row_m, row_v, row_scale, probe_epsilon, z, row["weights"])
                trace.append(dict(step=row["step"], **measurement))
        inputs[path.name] = file_sha(path)
        runs.append(dict(run_id=cfg.get("run_id", cfg.get("name", path.stem)), config=cfg,
                         source_file=path.name, completed_updates=obj["completed_updates"],
                         endpoint=endpoint, checkpoint_geometry=trace))
    if not runs:
        raise ValueError("No compatible completed outcomes")
    if len({r["run_id"] for r in runs}) != len(runs):
        raise ValueError("Duplicate run identities")
    manifest_path = input_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "completed":
            raise ValueError("Study manifest is not completed")
        planned = manifest.get("planned_configs")
        if planned is not None and {c["run_id"] for c in planned} != {r["run_id"] for r in runs}:
            raise ValueError("Some planned conditions are missing or extra outcomes are present")
        if planned is not None:
            expected_configs = {c["run_id"]: c for c in planned}
            if any(r["config"] != expected_configs[r["run_id"]] for r in runs):
                raise ValueError("Outcome configuration differs from manifest")
        if "files" in manifest:
            expected_hashes = {f["path"]: f["sha256"] for f in manifest["files"]}
            if any(expected_hashes.get(name) != digest for name, digest in inputs.items()):
                raise ValueError("Consumed evidence differs from manifest hashes")
        inputs[manifest_path.name] = file_sha(manifest_path)
    return dict(
        schema_version=1, probe_epsilon=probe_epsilon,
        source_sha256=file_sha(__file__), input_sha256=inputs,
        definitions=dict(
            retained_update_fraction="sqrt(v_raw)/(sqrt(v_raw)+probe_epsilon); this is 1 minus the removed fraction",
            log_scale="M=exp(b)*m_raw and V=exp(2*b)*v_raw; all calculations avoid reconstructing raw tiny moments",
            cosine_update_zero="Counterfactual using the SAME stored moments and epsilon=0; not a separately trained trajectory",
            moment_timing="Stored m/v include the gradient used for the preceding update; the frozen-moment comparison does not advance recurrence",
            current_gradient="Mean-logistic gradient reevaluated at stored post-update weights; scalar rescaling preserves its direction",
            update_direction="Negative preconditioned first moment; the common positive learning-rate scalar is omitted",
            source_invalid_control="A fixed probe epsilon is applied even when the source trajectory used INVALID_wrong_scale; source config remains labeled",
            limitations="Finite-state geometry does not prove convergence or identify author code. Fractions can round to zero; log fractions remain available. No new training is performed.",
        ),
        self_test=test_invariants(), conditions_analyzed=len(runs), runs=runs,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["analyze", "verify", "self-test"])
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--probe-epsilon", type=float, default=1e-8)
    args = parser.parse_args()
    if args.command == "self-test":
        print(json.dumps(test_invariants(), indent=2))
        return
    if args.input_dir is None:
        parser.error("--input-dir is required")
    output = args.output or args.input_dir / "transition_geometry.json"
    result = analyze(args.input_dir, args.probe_epsilon)
    if args.command == "verify":
        if json.loads(output.read_text()) != result:
            raise ValueError("Saved geometry differs from recomputation or input/source hashes changed")
    else:
        with output.open("x") as handle:
            handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(passed=True, conditions_analyzed=result["conditions_analyzed"],
                          saved_checkpoint_geometries=sum(len(r["checkpoint_geometry"]) for r in result["runs"]),
                          output=str(output)), indent=2))


if __name__ == "__main__":
    main()
