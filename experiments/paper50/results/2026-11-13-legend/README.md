# Figure-legend rerun — 2026-11-13

**Partial qualitative reproduction.** This is the main run reported in the
[root README](../../../../README.md). It uses the learning rates that are printed in the
paper's Figure 1 and Figure 2 legends and the averaged loss of eq. 3.1; see
[protocol_legend.json](../../protocol_legend.json). The first run
([2026-11-05](../2026-11-05/README.md)) and the c = 0.1 / summed-loss variant
([2026-11-12-matched](../2026-11-12-matched/README.md), cited only for the order
of the exponents) are kept unchanged. Code and analysis were prepared with AI
assistance.

## Settings

| Setting | Value | Source |
|---|---|---|
| Data, initial vectors, selected seed | the ten 2026-11-05 datasets, seed 2 | same as the first run |
| Loss | logistic loss averaged over the 50 points | eq. 3.1 |
| Adam learning rate | η_t = t^-a, t = 1, 2, …; main a = 0.5; sweep a ∈ {0.3, 0.5, 0.7, 1} | Figure 1 and 2 legends; t = 1 as the first step is our choice |
| GD / GDM learning rate | η = 0.1, GDM m = 0.9m + 0.1g | Figure 1 legend |
| Adam | β1 = 0.9, β2 = 0.99, no bias correction, ε ∈ {0, 1e-8}; ε controls 1e-12, 1e-4 | §5 |
| Updates | 10⁶ per run, 90 runs, no numerical stops | |

The only setting the paper does not state is where the step count starts.

## Results (relative gap to the reference optimum after 10⁶ updates)

| Seed 2 | L∞ gap | L2 gap |
|---|---:|---:|
| GD | 35.369% | 9.978% |
| GDM | 35.459% | 10.011% |
| Adam ε = 0 | 0.657% | 17.669% |
| Adam ε = 1e-8 | 31.512% | 5.716% |

| Ten datasets | min–max |
|---|---|
| Adam ε = 0, L∞ | 0.50–1.29% (9 of 10 within 1%) |
| Adam ε = 1e-8, L∞ | 24.19–42.87% |
| GD / GDM, L2 | 9.65–11.79% / 9.65–11.80% |
| GD / GDM, L∞ | 35.37–49.93% / 35.44–50.03% |

- Exponent sweep, seed 2, ε = 0: a = 0.3 / 0.5 / 0.7 / 1 → 1.523% / 0.657% /
  2.435% / 22.738%. With ε = 1e-8: 34.336% / 31.512% / 27.430% / 24.072%.
- Order of the exponents (ε = 0): a = 0.5 is fastest on all ten datasets and
  a = 1 slowest. The paper's Figure 2 shows a = 0.3 fastest. With c = 0.1 and a
  summed loss the paper's order appears
  ([2026-11-12-matched](../2026-11-12-matched/README.md)), so the order depends
  on the learning-rate constant.
- ε dominance on seed 2 (a = 0.5): first coordinate by recorded checkpoint
  2,791, all 50 coordinates by checkpoint 3,118. On the ten datasets all 50
  coordinates are dominated by checkpoints 3,000–3,118. These are checkpoint
  bounds. The exact values from checks after every update are 2,705 and 3,075
  on seed 2 and 2,963–3,075 for all 50 coordinates on the ten datasets; see
  [2026-11-13-eps-timing](../2026-11-13-eps-timing/README.md).
- Other ε at a = 0.5: 1e-12 → 28.021%, 1e-4 → 39.877%.
- Adam ε = 0 on seed 2 ends at an L∞ margin of 1.43401 against the optimum
  1.44349 (ratio 0.993).

Compared with the paper's Figure 1 (read off by eye: L∞ optimum about 1.68,
Adam without ε about 1% short, GD about 37% short on L∞ and 5% on L2), the
ε = 0 and GD L∞ curves are close; GD's L2 gap is about twice the paper's. The
paper's similar ε = 0 / 1e-8 curves are not recovered.

A group member's independent implementation, whose results are kept in
[2026-11-12-alignment](../2026-11-12-alignment/README.md) and
[2026-11-12-diagnosis](../2026-11-12-diagnosis/README.md), reports the same seed 2
values at these settings (0.657242% and 31.511943%; GD 35.368657% and
9.977666%). It also changes one convention at a time on seed 2 with ε = 1e-8
(bias correction 28.591%, β2 = 0.999 42.377%, ε inside the square root 39.753%,
summed loss 29.755%, float32 31.491%) and compares the update without
rescaling over all 10⁶ steps (weights within 1.78e-14).

## Files and rerun

`endpoints.json` lists every run's endpoint; `manifest.json` records the
runner, protocol and dataset hashes. Each run folder has `config.json`,
`summary.json` and `endpoint_state.json`; seed-2 runs also keep
`trajectory.csv` (all recorded checkpoints). From the repository root:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 experiments/paper50/run_matched.py --protocol experiments/paper50/protocol_legend.json \
    --output artifacts/paper50-legend-new --jobs 9
```

The rerun took about 4 minutes on 9 workers.
