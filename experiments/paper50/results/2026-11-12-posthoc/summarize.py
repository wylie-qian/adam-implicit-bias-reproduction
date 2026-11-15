"""Summarize the 2026-11-12 post-hoc runs. Usage: python3 -I summarize.py RUN_DIR_OR_ARCHIVE OUT_JSON
RUN_DIR holds one <name>.json per job (diag.py output); the archive is checkpoints.json.gz."""
import gzip, json, statistics, sys
from collections import Counter
from pathlib import Path

src = Path(sys.argv[1])
if src.suffix == ".gz":
    runs = json.load(gzip.open(src, "rt"))
else:
    runs = {p.stem: json.load(open(p)) for p in sorted(src.glob("*.json"))}
A = [0.3, 0.5, 0.7, 1.0]

def at(name, step):
    return min(runs[name]["rows"], key=lambda r: abs(r["step"] - step))

out = {"note": "Post-hoc diagnostics, not the frozen protocol; eta_t = c * t^(-a), t = 1, 2, ...",
       "runs": len(runs), "errors": [n for n, r in runs.items() if r["error"]]}

# 1. Figure-1 legend settings on seed 2
out["figure1_settings_seed2"] = {
    n: {str(s): {k: at(n, s)[k] for k in ("rel_linf", "rel_l2", "log_loss")}
        for s in (100, 1000, 10000, 100000, 1000000)}
    for n in ("f1_adam_eps0", "f1_adam_eps1e-8", "f1_gd", "f1_gdm", "f1_gdm_torch")}

# 2. Adam eta = t^-0.5, eps = 0 on all ten datasets
rel = [at(f"ord_s{s}_a0.5", 10**6)["rel_linf"] for s in range(10)]
out["figure1_adam_all_seeds"] = {"rel_linf_by_seed": rel, "min": min(rel), "median": statistics.median(rel),
                                 "max": max(rel), "within_1pct": sum(r <= 0.01 for r in rel)}

# 3. Ordering of the exponent a at 1e6 steps, eps = 0, by learning-rate constant c
order = {}
for c in (1.0, 0.3, 0.1, 0.03):
    name = (lambda s, a: f"ord_s{s}_a{a}") if c == 1.0 else (lambda s, a: f"c{c}_s{s}_a{a}")
    fastest, orders, per = Counter(), Counter(), []
    for s in range(10):
        gaps = [at(name(s, a), 10**6)["gap_linf"] for a in A]
        o = [a for _, a in sorted(zip(gaps, A))]
        fastest[o[0]] += 1; orders[" < ".join(map(str, o))] += 1
        per.append({"seed": s, "abs_gap_linf": dict(zip(map(str, A), gaps))})
    out.setdefault("ordering_by_c", {})[str(c)] = {"fastest_count": {str(k): v for k, v in fastest.items()},
                                                   "orders": dict(orders), "per_seed": per,
                                                   "median_abs_gap_at_step10_a0.5": statistics.median(at(name(s, 0.5), 10)["gap_linf"] for s in range(10))}

# 4. Longer runs, seed 2, c = 1
out["long_seed2_c1"] = {str(a): {str(s): at(f"long_s2_a{a}", s)["gap_linf"] for s in (10**4, 10**5, 10**6, 3 * 10**6, 10**7)}
                        for a in (0.3, 0.5)}

# 5. Hypothesis: epsilon added to the rescaled moments (seed 2, c = 1)
out["rescaled_epsilon_hypothesis_seed2"] = {
    str(a): {"max_abs_gap_difference_vs_eps0": max(abs(x["gap_linf"] - y["gap_linf"]) for x, y in
                                                     zip(runs[f"h_resc_eps1e-8_a{a}"]["rows"], runs[f"ord_s2_a{a}"]["rows"])),
             "abs_gap_1e6": at(f"h_resc_eps1e-8_a{a}", 10**6)["gap_linf"]} for a in A}

# 6. Best L-inf gap reached while every coordinate of sqrt(v) is still >= 1e-8 (eps = 0 runs)
best = []
for n, r in runs.items():
    if r["method"] != "adam" or r["eps"] != 0.0:
        continue
    ok = [x["rel_linf"] for x in r["rows"] if x.get("log10_min_sqrt_v_raw") is not None and x["log10_min_sqrt_v_raw"] >= -8]
    if ok:
        best.append((min(ok), n))
best.sort()
out["best_gap_while_sqrt_v_above_1e-8"] = {"eps0_runs": len(best), "smallest_rel_linf": best[0][0], "run": best[0][1],
                                          "median_over_runs": statistics.median(b for b, _ in best)}
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k in ("runs", "errors", "figure1_adam_all_seeds", "long_seed2_c1",
                                                       "best_gap_while_sqrt_v_above_1e-8")}, indent=1))
for c, v in out["ordering_by_c"].items():
    print("c", c, v["fastest_count"], v["orders"], round(v["median_abs_gap_at_step10_a0.5"], 2))
for a, v in out["rescaled_epsilon_hypothesis_seed2"].items():
    print("resc a", a, v)
for n, v in out["figure1_settings_seed2"].items():
    print(n, {s: round(x["rel_linf"] * 100, 2) for s, x in v.items()}, "L2:", round(v["1000000"]["rel_l2"] * 100, 2))
