# Post-hoc diagnostics — 2026-11-12

**These runs are not part of the frozen 2026-11-05 protocol.** They were designed
after the formal results were known, to explain three differences from the paper.
They do not replace the formal endpoints in [../2026-11-05](../2026-11-05/README.md).
The full runs that followed are in [../2026-11-12-matched](../2026-11-12-matched/README.md)
(c = 0.1 and summed loss, backup B7 only) and
[../2026-11-13-legend](../2026-11-13-legend/README.md) (the legends' settings, shown
on the slides).
Code and analysis were prepared with AI assistance, like the rest of this experiment.

## Why these runs

The protocol said that figure 1's learning rates were not supplied. That was wrong.
The text of §5 only writes η_t = Θ(t^-a), but the legends inside the figure images
(arXiv v1 and the NeurIPS 2024 final PDF, physical page 7) state them:

- Figure 1: Adam with and without ε uses η_t = t^-0.5; GD and GDM use η_t = 0.1.
- Figure 2: Adam uses η_t = t^-a for a ∈ {0.3, 0.5, 0.7, 1}.

The legends are raster images, so a text search of the PDF does not find them.
The frozen primary condition used Adam η_t = (t+2)^-0.7 and GD/GDM η = 0.5/L
(0.552 on seed 2), so it differs from the figure-1 setting. Its c = 1 does match.

## What was run

171 runs on the retained 2026-11-05 datasets (`jobs.json`), with `diag.py`, which
reuses `run.py`'s rescaled Adam step. Learning rates are η_t = c·t^-a with t = 1, 2, …
as in the legends. All runs completed without numerical errors.

1. Figure-1 settings on seed 2: Adam ε = 0 and ε = 1e-8 (η_t = t^-0.5), GD and
   GDM (η = 0.1; GDM both as m = 0.9m + 0.1g and as m = 0.9m + g).
2. Adam ε = 0, η_t = c·t^-a, a ∈ {0.3, 0.5, 0.7, 1}, c ∈ {1, 0.3, 0.1, 0.03},
   all ten datasets, 10⁶ updates.
3. Seed 2, c = 1, a ∈ {0.3, 0.5}, 10⁷ updates.
4. A hypothesis check (below): ε added to the rescaled moments, seed 2, c = 1.

`checkpoints.json.gz` keeps every recorded checkpoint (about 400 log-spaced per run).
`summary.json` is produced by:

```sh
python3 -I summarize.py checkpoints.json.gz summary.json
```

To regenerate the runs (about 10–15 minutes on 9 workers; the two 10⁷-update runs
take about 5 minutes each), from this folder:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir new-out && python3 -I diag.py new-out jobs.json 9
```

## Results

| Question | Result |
|---|---|
| Adam ε = 0 with the figure-1 learning rate | Seed 2: 0.66% short of the L∞ optimum after 10⁶ updates (frozen a = 0.7: 2.61%). Ten datasets: 0.50–1.29%, nine within 1%. |
| GD with the figure-1 learning rate | Seed 2: L∞ margin 35.4% short (paper figure 1 ≈ 36%); L2 margin 10.0% short. |
| Adam ε = 1e-8 with the figure-1 learning rate | Seed 2: stalls at 31.5% short from about 10⁴ updates; min √v about 1e-13 at 10⁶. |
| Order of the exponents, c = 1 | a = 0.5 fastest on 10 of 10 datasets; a = 0.3 never fastest. |
| Order of the exponents, c = 0.1 or 0.03 | 0.3 < 0.5 < 0.7 < 1 on 10 of 10 datasets, the order in the paper's figure 2. c = 0.3: a = 0.3 fastest on 6 of 10. |
| Longer run, seed 2, c = 1 | a = 0.3 catches up with a = 0.5 by 10⁷ updates (both 0.0024 absolute gap). |
| ε while Adam approaches the L∞ optimum | Over 163 ε = 0 runs, the smallest gap reached while every coordinate of √v was still ≥ 1e-8 was 18.8% (median 36%). |
| ε added to the rescaled moments (hypothesis) | Curves differ from ε = 0 by at most 3e-4 absolute gap for all four exponents. |

## Interpretation and limits

- The ε = 0 result moves closer to the paper when its figure-1 learning rate is used.
  This is a post-hoc rerun, not a pre-planned endpoint.
- The order of the exponents depends on the learning-rate constant and on the
  horizon. The paper itself says the best schedule can differ between data sets.
  With c = 1, as the legends state, our order differs from figure 2; c = 0.1 is
  a value we found afterwards, not one the paper reports. The absolute gaps and
  curve shapes still differ from figure 2 for every c tried. Our datasets also
  differ: the paper's figure 1 shows γ∞ ≈ 1.68 and γ2 ≈ 0.475, while our seed 2
  has 1.443 and 0.330.
- With ε = 1e-8 at its real size, ε dominated before Adam came close to the L∞
  optimum in every condition tried. The paper notes in §3 that ε will eventually
  affect the result, but §5 reports similar curves with and without ε.
- With ε = 0 and η_t = t^-0.5, the loss after 10⁶ updates is about e^-1954, far
  below the smallest positive float64 (about e^-745). A plain float64
  implementation cannot carry these runs to 10⁶ updates, so the authors' code
  presumably avoided underflow in some way. If ε was added after a rescaling of
  the gradient, ε stays negligible. Our check of that variant reproduces the
  near-identical panels of figure 2. This is only a possible explanation: no code
  was released, and it does not explain the small early differences in figure 1.
