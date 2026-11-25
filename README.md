# Reproduction of the Section 5 experiment in "The Implicit Bias of Adam on Separable Data"

This repository holds COMP3314 Group 7's independent reproduction of the
Section 5 experiment in Zhang, Zou and Cao, "The Implicit Bias of Adam on
Separable Data", NeurIPS 2024 ([arXiv:2406.10650](https://arxiv.org/abs/2406.10650)).
The authors did not release code or data with the paper. The paper's NeurIPS
checklist answers "No" to question 5, which asks about open access to data and
code. We therefore wrote the experiment from the formulas and settings in the
paper and chose the settings it does not state. [SOURCES.md](experiments/paper50/SOURCES.md)
lists the sources we checked and marks each setting as taken from the paper or
chosen by us.

## Status

The result is a partial qualitative reproduction. Adam without ε approaches the
L∞ max-margin direction, as the paper reports. Adam with the default ε = 1e-8
stays far from it, while the paper's Figure 1 shows the curves with and without
ε close together. This does not contradict the paper's theorem, which assumes
ε = 0. We do not recreate the authors' random samples or their exact curves.

The main run, which the group presented, is [results/2026-11-13-legend](experiments/paper50/results/2026-11-13-legend/README.md).
It uses the learning rates that are printed in the legends of Figures 1 and 2 (Adam
η_t = t^-0.5, GD and GDM η = 0.1) and the averaged logistic loss of
equation 3.1. Each run has 10⁶ full-batch updates on 50×50 Gaussian data. For
the learning rates, the only setting we chose is that the step count t starts
at 1. Our other choices, such as the random draws and the GDM update
m = 0.9m + 0.1g, are listed in SOURCES.md. Seed 2 was selected from ten
datasets by reference geometry, as Section 5 describes, before any optimizer
results were seen.

The relative gap is (γ_p − q_p(w))/γ_p, where q_p(w) = min_i y_i x_iᵀw / ‖w‖_p
and γ_p is the p-norm max-margin value of the dataset. It measures the training
margin and is not a classification error. The values after 10⁶ updates on the
selected seed-2 data are as follows.

| Method, seed 2 | Relative L∞ margin gap | Relative L2 margin gap |
|---|---:|---:|
| GD | 35.369% | 9.978% |
| GDM | 35.459% | 10.011% |
| Adam, ε = 0 | 0.657% | 17.669% |
| Adam, ε = 1e-8 | 31.512% | 5.716% |

On all ten datasets, the L∞ gap of Adam with ε = 0 is 0.50–1.29%, and nine of
the ten gaps are below 1%. With ε = 1e-8, the gap is 24.19–42.87%.
In the exponent sweep with ε = 0, a = 0.5 ends with the smallest gap on all ten
datasets, while the paper's Figure 2 shows a = 0.3 as the fastest. All numbers
come from [endpoints.json](experiments/paper50/results/2026-11-13-legend/endpoints.json).

A follow-up study, [results/2026-11-13-eps-timing](experiments/paper50/results/2026-11-13-eps-timing/README.md),
measures when ε takes over, that is, when √v falls below ε in every
coordinate. Because v_t ≥ β2·v_(t−1) without bias correction, √v shrinks by at
most about 0.5% per update, so ε = 1e-8 cannot take over before about 3,000
updates, whatever the learning rate. With the legend learning rate it takes
over at that time on all ten datasets. Over learning-rate constants from 0.03
to 3 and exponents from 0.3 to 1, every run with ε = 1e-8 ended with an L∞ gap
of at least 15%.

The diagnostic studies did not find the cause of the ε = 1e-8 difference.
Common implementation conventions and other random draws did not bring the two
Adam curves together. One-step checks at 80 and 120 digits support our
numerical method. A deliberately wrong rescaling of ε makes the curves match,
but we have no evidence that the authors used it.

## What is in experiments/paper50

The audit notes are Markdown files next to the code.

- [README.md](experiments/paper50/README.md) describes the experiment, the fixed settings and the commands for each study.
- [SOURCES.md](experiments/paper50/SOURCES.md) records the paper sources we checked and where each setting comes from.
- [ALIGNMENT.md](experiments/paper50/ALIGNMENT.md) corrects the learning rates that the first run missed in the Figure 1 legends.
- [DIAGNOSIS.md](experiments/paper50/DIAGNOSIS.md) tests whether conventions, random draws or precision explain the ε = 1e-8 gap.
- [MULTIRATE.md](experiments/paper50/MULTIRATE.md) repeats the ε comparison at four learning-rate exponents and studies the late update direction.
- [FOLLOWUP_SOURCES.md](experiments/paper50/FOLLOWUP_SOURCES.md) separates the paper's claims from our own algebra and from related work.
- [reproduction_update.md](experiments/paper50/reproduction_update.md) gives a short summary of the four-rate results and answers to likely questions.
- [FINAL_REPRODUCTION.md](experiments/paper50/FINAL_REPRODUCTION.md) summarizes the fixed-ε evidence, including the full long-double replays.

Each results folder keeps its own README.

- [2026-11-05](experiments/paper50/results/2026-11-05/README.md) is the first run under the frozen protocol, with a = 0.7 as its main Adam setting. The 2026-11-13-legend run supersedes it.
- [2026-11-12-alignment](experiments/paper50/results/2026-11-12-alignment/README.md) holds nine seed-2 runs with the Figure 1 learning rates and an independent unscaled Adam check.
- [2026-11-12-diagnosis](experiments/paper50/results/2026-11-12-diagnosis/README.md) holds 56 convention and randomness controls and high-precision one-step checks.
- [2026-11-12-multirate](experiments/paper50/results/2026-11-12-multirate/README.md) holds 12 runs at four exponents with their saved moment states.
- [2026-11-12-posthoc](experiments/paper50/results/2026-11-12-posthoc/README.md) holds post-hoc runs that explain three differences between the first run and the paper.
- [2026-11-12-matched](experiments/paper50/results/2026-11-12-matched/README.md) holds a rerun with c = 0.1 and a summed loss, chosen after the first run by comparison with the paper's curves. We cite it only for the order of the exponents.
- [2026-11-13-legend](experiments/paper50/results/2026-11-13-legend/README.md) holds the main run with the legend settings.
- [2026-11-13-eps-timing](experiments/paper50/results/2026-11-13-eps-timing/README.md) holds 200 runs that measure when ε takes over as the learning rate, ε and the dataset change.
- [2026-11-13-literal-replay](experiments/paper50/results/2026-11-13-literal-replay/README.md) holds five million-step replays of the raw Adam recurrence in C++ long double, without rescaling, written independently by a second group member.

## Two-dimensional example

[experiments/direction_2d](experiments/direction_2d/README.md) holds a small
two-point example that illustrates the claim. GD and Adam without ε each take
20,000 steps, and Adam turns towards the L∞ max-margin direction after a
detour. The example is not part of the reproduction. Its check,
`python3 experiments/direction_2d/verify.py`, needs only the Python standard
library.

## Setup

The 2026-11-05 study used Python 3.12.14. The package versions are pinned in
[requirements.lock.txt](experiments/paper50/requirements.lock.txt) (NumPy 2.3.5,
SciPy 1.17.0 and Matplotlib 3.10.8). The manifest of the main run records
Python 3.13.2 and NumPy 1.26.4. Run every command from the repository root.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r experiments/paper50/requirements.lock.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## Verify the saved results

These checks read the saved results and do not train again. Each takes a few
seconds.

```sh
python3 experiments/paper50/verify.py
python3 experiments/paper50/verify_snapshot.py
python3 experiments/paper50/verify_alignment.py
python3 experiments/paper50/alignment.py verify --output experiments/paper50/results/2026-11-12-alignment
python3 experiments/paper50/verify_conventions.py
python3 experiments/paper50/conventions.py verify --output experiments/paper50/results/2026-11-12-diagnosis/conventions
python3 experiments/paper50/verify_randomness.py test
python3 experiments/paper50/verify_randomness.py verify
python3 experiments/paper50/verify_precision.py
python3 experiments/paper50/verify_multirate.py
python3 experiments/paper50/multirate_audit.py verify --output experiments/paper50/results/2026-11-12-multirate
python3 experiments/paper50/transition_geometry.py self-test
python3 experiments/paper50/transition_geometry.py verify --input-dir experiments/paper50/results/2026-11-12-multirate
```

The checks cover less than a full rerun.

- `verify.py` runs six numerical tests of the gradient, the rescaled Adam update and the reference solvers.
- `verify_snapshot.py` checks the file hashes of the 2026-11-05 snapshot, regenerates the ten datasets, checks the reference certificates and recomputes the 50 formal endpoints. The snapshot is compact, so it does not hold every history of the earlier 90-run audit.
- The alignment check recomputes the metrics at 2,376 saved checkpoints and compares the direct and rescaled positive-ε weights. It does not recompute all nine million updates.
- The diagnosis checks run seven convention tests and three randomness tests. They recompute the retained states and check ten single updates at 80 and 120 decimal digits. Only retained weights can be recomputed, and the one-step checks are not high-precision training.
- The four-rate checks recompute the margins, losses and ε diagnostics at all 732 saved states. First moments are checked for shape and finite values but are not replayed. The ε crossing times were logged during the runs and are not replayed.
- The `INVALID_RECURRENCE` control in the diagnosis and four-rate studies is deliberately wrong. It is not a fix for the optimizer.

The long-double replays have their own read-only check,
`python3 experiments/paper50/literal_replay.py verify`. It needs x86-64 native
extended precision, so it stops with "Verifier requires native extended
precision/range" on Apple Silicon, where NumPy's long double is the same as a
double.

The main run has no separate verifier. Its seed-2 values agree with the
verified alignment run at the reported precision (0.657% and 31.512%). The next
section shows how to rerun it in full.

Two checks compare exact bytes or exact floating-point values. The saved results
were produced on Linux x86_64. On macOS with Python 3.13.2 and the pinned
packages, `verify_randomness.py verify` reports "Input generation changed" for
the 30 legacy MT19937 datasets, while the PCG64 datasets match.
`transition_geometry.py verify` also fails there, with relative differences
below 5e-14. The other checks pass.

## Rerun the main run

The runner reads the ten saved datasets in `results/2026-11-05/datasets` and
checks their hashes before training. The output directory must not exist yet.

```sh
python3 experiments/paper50/run_matched.py --protocol experiments/paper50/protocol_legend.json \
    --output artifacts/paper50-legend-new --jobs 9
```

Pass `--protocol` explicitly, because the default is `protocol_matched.json`.
The 90 runs of 10⁶ updates took about 4 minutes on 9 workers. Compare the new
`endpoints.json` with the saved one in `results/2026-11-13-legend`. A rerun on
macOS on 2026-11-15, with Python 3.13.2 and the pinned packages, reproduced all
90 saved endpoints exactly. The option
`--steps` shortens every run for a quick test, but a shorter run does not
reproduce the study. The commands for the other studies are in
[experiments/paper50/README.md](experiments/paper50/README.md) and in each audit
file. The `artifacts/` folder is ignored by git.

## Figures in the report

The three figures of the group's report are drawn only from results saved in
this repository: the two-point example, the seed-2 runs of the main run and the
ε-takeover study. Matplotlib is needed.

```sh
python3 docs/report/figures/make_figures.py
```

The script writes `docs/report/figures/fig1_geometry`, `fig2_margins` and
`fig3_epsilon` as PDF and PNG. A rerun with the pinned packages gives the same
bytes as the saved files.

## AI assistance

Code and analysis were prepared with AI assistance and reviewed by the group.
