# Full-history literal Adam audit — 2026-11-13

Five independent million-update recurrences were run in compiled native long double, using raw logistic gradients and raw moments. No common gradient/moment rescaling or epsilon transformation is used. Inputs are the frozen seed-2 binary64 arrays, parsed as double then cast to long double. Each file retains 61 complete weight and moment states; 305 states total.

| Rate a | Epsilon | Relative L-infinity gap (%) |
|---|---:|---:|
| .5 | 0 | 0.657248230 |
| .3 | 1e-8 | 34.336482570 |
| .5 | 1e-8 | 31.511942504 |
| .7 | 1e-8 | 27.429596319 |
| 1 | 1e-8 | 24.072277430 |

These are our own data/results, not recovered author curves. Correct fixed-epsilon discrepancies persist without moment stabilization. Epsilon-zero and a=1 fixed-epsilon trajectories show small floating-arithmetic sensitivity; the full differences are retained, not rounded to zero or omitted.

## Files and provenance

- `input.txt`: exact binary64 inputs represented by round-trippable decimal strings and checkpoint schedule.
- `*.csv.gz`: deterministic gzip files containing full native-extended decimal weights and raw moments. They are strings on input to the NumPy long-double verifier, so tiny moments are not first cast to binary64.
- `*_run.json`: arithmetic capabilities, completion status, timing and underflow counters.
- `comparison.json`: all checkpoint comparison metrics and maximum differences.
- `validation.json`: independent analytic first-update checks and file protection checks.
- `manifest.json`: executed C++ source hash, later packaging/verifier source hash, input/output hashes, compiler and flags. Production runs were launched with direct subprocess calls before the final Python wrapper was assembled; the manifest distinguishes this from its future `run` command.

Compiler: g++ with `-O3 -std=c++17 -fno-fast-math`. Native long double has a 64-bit mantissa and binary exponent limit 16384 here, approximately 19 decimal digits; it is not 80-digit decimal arithmetic.

## Numerical limits

Across a=.3/.5/.7 fixed-epsilon traces, maximum weight differences from the scaled binary64 baseline are at most 1.14e-12. The a=1 fixed-epsilon maximum is 3.47e-7, with relative margin-gap difference at most 5.48e-9. No monitored logistic-weight or nonzero-gradient-square underflow occurred in any fixed-epsilon run.

Epsilon zero has maximum weight difference 0.0168655 (relative infinity norm at most 7.89e-5), first-moment relative difference at most 2.44e-4, and maximum relative margin-gap difference 5.93e-6. Its endpoint relative gap differs by about 6.48e-8 (0.00000648 percentage points). Decimal constants, reduction order, precision and moment representation differ between implementations, so these comparisons are not bitwise-equivalence claims.

In that epsilon-zero run, 237,949 per-example logistic-weight evaluations rounded to zero, beginning at update 826,832, on already easy examples with extreme margins. All second moments remained strictly positive and all nonzero gradient squares remained representable. The smallest observed second moment was about 10^-1706.12, while the smallest native-extended positive subnormal is about 10^-4950.44. Thus the replay avoids second-moment underflow but does not claim every per-example exponential is represented exactly. The positive-epsilon conclusion does not depend on this zero-epsilon caveat.

## Recompute comparisons or rerun

From the repository root:

```sh
python3 experiments/paper50/literal_replay.py verify
python3 experiments/paper50/literal_replay.py run --output artifacts/literal-replay-new
```

The verifier is read-only and checks frozen inputs, source and output identities, then recomputes every retained comparison. It does not rerun training. The `run` command refuses an existing output directory and requires a C++ compiler providing the stated extended range; all five complete histories are run anew.
