# Figure 1 alignment audit

This additive audit corrects a source-reading error in the 2026-11-05 study.
The final [NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/2ac79356a03fe5e9250e5e77ebc76e6e-Paper-Conference.pdf),
physical page 7, Figure 1 **legends**, gives Adam eta_t=t^(-0.5) and GD/GDM
eta_t=0.1. They are visible in the rendered figure but absent from extracted
PDF body text. The old primary study used a=.7 and GD/GDM eta=.5/L. Those
measurements are valid for their declared settings, but were not a Figure 1
parameter match. `SOURCES.md` now explicitly corrects the mistaken attribution.

## Preserved evidence and scope

- `run.py`, `protocol.json` and the entire 2026-11-05 snapshot remain byte-for-byte
  unchanged. The existing snapshot verifier still applies to that study.
- `alignment.py` imports the frozen, validated scaled Adam update and writes a
  separate result directory. It also implements a literal positive-epsilon
  recurrence independently using SciPy's stable `expit`, with no moment rescaling.
- All methods use the same selected data and initialization. Selection remains
  based on the convex reference geometry of all ten candidates, never on the
  observed optimizer results. The selected seed is still 2.
- New results are independent simulations, not digitized author curves. The
  author's random seeds/arrays, precise indexing and GDM momentum convention
  remain unavailable. We interpret Figure 1's t^(-.5) from t=1 and explicitly
  include the old t+2 indexing as a control. Normalized EMA GDM is retained.
- No attempt is made to tune seeds, epsilon or the objective to force agreement.
  A finite normalized-margin observation is not a theorem about the limit.

## Fixed comparison matrix

Each row uses 1,000,000 updates in the full run. Adam uses beta1=.9, beta2=.99,
no bias correction, mean logistic loss, epsilon outside the square root and
float64. Formula indices below use t=0 for the first code update.

| Conditions | Number | Learning rate | Purpose |
|---|---:|---|---|
| Legacy Adam, epsilon=0 / 1e-8 | 2 | (t+2)^(-.7) | Recover the old main endpoints |
| Figure 1 Adam, epsilon=0 / 1e-8 | 2 | (t+1)^(-.5) | Match the figure's displayed schedule |
| Index control Adam, epsilon=0 / 1e-8 | 2 | (t+2)^(-.5) | Isolate the indexing choice and check old rate-sweep endpoints |
| Figure 1 GD / GDM | 2 | 0.1 | Match the figure's displayed constant rate |
| Literal Adam, epsilon=1e-8 | 1 | (t+1)^(-.5) | Independently check scaled arithmetic through the epsilon-dominated regime |

Nine conditions are fixed in `conditions()` before this audit's full runs.
This is a diagnostic study informed by earlier results, not preregistration.
The existing 600-second wall/CPU budgets per condition remain active; any
incomplete run is preserved and prevents a successful study status.

## Quantifying epsilon's influence

For a fixed moment state, the magnitude ratio of a coordinate's update with
positive epsilon to its epsilon=0 update is sqrt(v)/(sqrt(v)+epsilon), whenever
the zero-epsilon update is nonzero. The denominator attenuation factor is still
well-defined when the first moment is zero. Accordingly we record

    removed fraction = epsilon / (sqrt(v_raw) + epsilon).

Crossing sqrt(v)<epsilon only implies a removed fraction greater than 1/2;
it is not a strong approximation of the denominator by epsilon. The new
`epsilon_strong_coordinates` uses sqrt(v)<=0.1*epsilon, which implies removal
of at least 10/11 of that coordinate's zero-epsilon update magnitude.
We record both the minimum and maximum removed fractions across coordinates
and the maximum sqrt(v)/epsilon ratio. These quantities use the original
moment scale, including for the rescaled implementation.

The first crossing reported is the first **saved checkpoint** that meets the
condition, an upper bound on the first crossing. It is not an exact update time
or a guarantee that the condition never reverses. This local comparison of
shared moments does not equate the trajectories of different optimizers.

## Run and verify

Use the existing `requirements.lock.txt`. From the repository root:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 experiments/paper50/verify.py
python3 experiments/paper50/verify_alignment.py
python3 experiments/paper50/run.py prepare --output artifacts/figure1-new/datasets
python3 experiments/paper50/alignment.py run --data artifacts/figure1-new/datasets --output artifacts/figure1-new/results --jobs 3
python3 experiments/paper50/alignment.py verify --output artifacts/figure1-new/results
```

All output directories must be new. `--steps 100` is only a smoke test.
Saved evidence includes every recorded weight and moment state, CSV metrics,
configuration, environment/source hashes, data and reference certificates.
The verifier checks file hashes, the condition matrix, all recorded margins,
epsilon diagnostics, summary consistency, and literal-vs-scaled positive-epsilon
weights at every common checkpoint. It does not verify every intervening update
or claim to reproduce unavailable author samples.

Code, diagnostics and analysis were prepared with AI assistance and reviewed by
the group. No
author code or figures are copied; the paper is cited as the formula source.
