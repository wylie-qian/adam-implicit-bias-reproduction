# Paper-matched rerun — 2026-11-12

**Partial qualitative replication; no longer the presented run.** From
2026-11-13 the slides show the [figure-legend rerun](../2026-11-13-legend/README.md),
which uses the legends' learning rates and the averaged loss literally. This
variant is cited only on backup slide B7, for the order of the exponents. Its
exponents and GD rate follow the legends; its constant c = 0.1 and summed loss
were matched to the paper's Figure 1 curves; see
[protocol_matched.json](../../protocol_matched.json). The first run
([2026-11-05](../2026-11-05/README.md)) is kept unchanged.
Code and analysis were prepared with AI assistance.

## Settings

| Setting | Value | Source |
|---|---|---|
| Data, initial vectors, selected seed | the ten 2026-11-05 datasets, seed 2 | same as the first run |
| Adam learning rate | η_t = 0.1 · t^-a, t = 1, 2, …; main a = 0.5 | exponents from the Figure 1/2 legends; c = 0.1 chosen to match Figure 1 |
| GD / GDM learning rate | η = 0.1, GDM m = 0.9m + 0.1g | Figure 1 legend |
| Loss | logistic loss summed over the 50 points | chosen to match Figure 1(b); eq. 3.1 averages |
| Adam | β1 = 0.9, β2 = 0.99, no bias correction, ε ∈ {0, 1e-8}; ε controls 1e-12, 1e-4 | §5 |
| Updates | 10⁶ per run, 90 runs, no numerical stops | |

The constant c = 0.1 and the summed loss were chosen after the first run, by
comparison with the paper's published curves (diagnostics in
[../2026-11-12-posthoc](../2026-11-12-posthoc/README.md)). They are disclosed as
such, not presented as the paper's settings.

## Results (relative gap to the reference optimum after 10⁶ updates)

| Seed 2 | L∞ gap | L2 gap |
|---|---:|---:|
| GD | 31.684% | 6.786% |
| GDM | 32.809% | 7.674% |
| Adam ε = 0 | 1.310% | 17.880% |
| Adam ε = 1e-8 | 16.403% | 8.912% |

| Ten datasets | min–max |
|---|---|
| Adam ε = 0, L∞ | 0.82–1.31% (5 of 10 within 1%) |
| Adam ε = 1e-8, L∞ | 16.40–29.76% |
| GD / GDM, L2 | 6.56–8.02% / 7.36–10.23% |

- Exponent sweep, seed 2, ε = 0: a = 0.3 / 0.5 / 0.7 / 1 → 0.264% / 1.310% /
  12.259% / 99.193%. With ε = 1e-8: 22.442% / 16.403% / 18.166% / 99.192%.
  With c = 0.1 and a = 1 the learning rates add up to about 1.44 in 10⁶
  updates, so that run barely moves from its initial vector.
- Order of the exponents (ε = 0): 0.3 < 0.5 < 0.7 < 1 on all ten datasets,
  the order in the paper's Figure 2.
- ε dominance on seed 2 (a = 0.5): first coordinate by recorded checkpoint
  4,115, all 50 by 10,570. On the ten datasets all 50 coordinates are
  dominated by checkpoints 5,741–13,000. These are checkpoint bounds.
- Other ε at a = 0.5: 1e-12 → 12.548%, 1e-4 → 23.511%.

Compared with the paper's Figure 1 (read off by eye: L∞ optimum about 1.68,
Adam without ε about 1% short, GD about 37% short on L∞ and 5% on L2), the
ε = 0 and GD curves are close. The paper's similar ε = 0 / 1e-8 curves are
still not recovered; see the hypothesis in the post-hoc README.

## Files and rerun

`endpoints.json` lists every run's endpoint; `manifest.json` records the runner
and protocol hashes. Each run folder has `config.json`, `summary.json` and
`endpoint_state.json`; seed-2 runs also keep `trajectory.csv` (all recorded
checkpoints), which the slide animations read. From the repository root:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 experiments/paper50/run_matched.py --output artifacts/paper50-matched-new --jobs 9
```

The rerun takes about 4 minutes on 10 workers. On 2026-11-13 `run_matched.py`
gained a `--protocol` option (default `protocol_matched.json`) and reads the
loss scale from the protocol; with the default it gives the same arithmetic
(identical weights in a 3,000-update regression check), but its hash differs
from the `runner_sha256` in this folder's `manifest.json`, which records the
2026-11-12 version.
