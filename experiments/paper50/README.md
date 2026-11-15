# Independent replication of the paper-scale experiment

**Result: partial qualitative replication.** The implementation follows the
published formulas, with explicitly declared choices for unavailable parameters.
It does not recreate the authors' original samples, exact figure curves or code.
Code and analysis were prepared with AI assistance and reviewed by the group.

Read the [result snapshot](results/2026-11-05/README.md) for measured outcomes,
figures, retained data and verification scope, and [SOURCES.md](SOURCES.md) for
paper settings, team choices, numerical representation and source availability.

**Correction and post-hoc check (2026-11-12).** The paper's figure legends do give
learning rates (figure 1: Adam η_t = t^-0.5, GD/GDM η = 0.1), which the protocol
missed; see the correction in [SOURCES.md](SOURCES.md). Post-hoc reruns, outside the
frozen protocol, are in [results/2026-11-12-posthoc](results/2026-11-12-posthoc/README.md):
with the figure-1 rate, Adam without epsilon ends 0.50–1.29% short on the ten
datasets (nine within 1%); epsilon=1e-8 remains far from the optimum; the order
of the exponents a depends on the learning-rate constant c.

**Main results (2026-11-13).** The main full run uses the learning rates
printed in the figure legends and the averaged loss of equation 3.1:
[results/2026-11-13-legend](results/2026-11-13-legend/README.md)
([protocol_legend.json](protocol_legend.json), [run_matched.py](run_matched.py)).
Only the start of the step count (t = 1) is our choice. On seed 2 Adam without
epsilon ends 0.657% short of the L-infinity optimum (0.50–1.29% on the ten
datasets) and epsilon=1e-8 ends 31.512% short (24.19–42.87%). An earlier variant
with c = 0.1 and a summed loss, chosen after this first run by comparison with
the paper's curves, is in [results/2026-11-12-matched](results/2026-11-12-matched/README.md)
([protocol_matched.json](protocol_matched.json)) and is cited only for the order
of the exponents. The same seed 2 values were obtained independently in
[results/2026-11-12-alignment](results/2026-11-12-alignment/README.md) (see the
section below). `run.py` and the 2026-11-05 snapshot are unchanged and still
verify.

The primary study retains ten 50×50 synthetic candidates and runs GD, GDM and
Adam with epsilon=0/1e-8 for one million updates each. Seed 2 was selected by
reference geometry before observing optimizer results. The main a=.7 Adam
condition remains 2.611% short of the L-infinity optimum without epsilon and
27.990% short with epsilon=1e-8. A prespecified a=.5, epsilon=0 rate condition
reaches a 0.769355% relative gap, satisfying the team's descriptive 1% criterion.
The paper's similar epsilon=0/1e-8 trajectories are not recovered under these
declared step-size constants.

## 2026-11-12: corrected Figure 1 comparison

The original source audit missed the rates in Figure 1's rendered legends.
[The additive alignment audit](ALIGNMENT.md) now uses Adam eta_t=t^(-.5) and
GD/GDM eta=.1, with explicit first-step indexing and controls. In
[nine new million-step runs](results/2026-11-12-alignment/README.md), the
zero-epsilon relative L-infinity gap improves from 2.611% to 0.657%; the
positive-epsilon gap remains 31.512%. Independent literal Adam agrees with the
scaled implementation to 1.78e-14 at saved checkpoints. Positive epsilon becomes
non-negligible early in these runs, so changing the learning rate does not recover
the paper's epsilon-similarity observation. The frozen study below is preserved.

## Further fixed-epsilon diagnostics

[DIAGNOSIS.md](DIAGNOSIS.md) records 56 additional million-step controls and
80/120-digit one-step checks. Ordinary conventions, seeds, independent starts
and legacy RNG choices did not recover positive-epsilon overlap in the tested
matrix. An isolated `INVALID_RECURRENCE` control demonstrates that incorrectly
leaving epsilon unchanged during moment rescaling can make the curves match;
it changes the algorithm and is not used as a repair. The audited implementation
is preserved, and data/reference identity checking was added to the new runner.

## Verify the retained results

Python 3.12.14 was used, with versions in [requirements.lock.txt](requirements.lock.txt).
From the repository root:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/artifacts/mpl-cache"
python3 experiments/paper50/verify.py
python3 experiments/paper50/verify_snapshot.py
```

The first command performs six numerical checks. The second checks retained
hashes, regenerates all ten data sets, checks reference certificates, recomputes
all 50 formal endpoints and the checkpoint metrics for five complete histories.
It checks the measurements in the two scheduled repeated curves. The earlier
full 90-run audit is a separate recorded result; not all its underlying histories
are included in this compact snapshot.

To regenerate the six figure pairs from the retained measurements, use a new
output directory:

```sh
python3 experiments/paper50/verify_snapshot.py --plot-output artifacts/paper50-review/figures
```

## Rerun the frozen experiment

Every output directory must be new. These commands regenerate all observations,
including histories omitted from the compact snapshot:

```sh
python3 experiments/paper50/run.py prepare --output artifacts/paper50-new/datasets
python3 experiments/paper50/run.py run --data artifacts/paper50-new/datasets --suite pilot --output artifacts/paper50-new/pilot --jobs 3
python3 experiments/paper50/run.py run --data artifacts/paper50-new/datasets --suite primary --output artifacts/paper50-new/primary --jobs 3
python3 experiments/paper50/run.py run --data artifacts/paper50-new/datasets --suite rates --output artifacts/paper50-new/rates --jobs 3
python3 experiments/paper50/run.py run --data artifacts/paper50-new/datasets --suite epsilon --output artifacts/paper50-new/epsilon --jobs 2
python3 experiments/paper50/analyze.py --evidence artifacts/paper50-new --output artifacts/paper50-new/analysis
```

`analyze.py` validates complete newly generated experiment outputs. For the
compact retained snapshot use `verify_snapshot.py` instead. `--steps` can shorten
a diagnostic run, but changes the saved configuration and does not reproduce the
million-step study. Budgets are 600 wall/CPU seconds per run; no run in the saved
study hit either limit.

## Fixed settings and finite criterion

- Full-batch mean logistic loss, no intercept, no weight decay, float64.
- PCG64 data seeds 0–9; independent initialization seed+10000, shared by methods.
- Select the maximum L2 distance between L2-unit-normalized L2 and L-infinity
  reference directions; ties use the smallest seed. No seed is discarded based
  on optimizer performance.
- GD and normalized-momentum GDM both use eta=.5/L with
  L=||X||_spectral²/(4n), independently fixed per candidate; GDM uses m=.9m+.1g.
- Adam uses beta1=.9, beta2=.99, no bias correction and epsilon outside sqrt.
  eta_t=c(t+2)^(-a), t starts at zero; c=1 for every condition. Primary a=.7;
  rate sweep a∈{.3,.5,.7,1}, epsilon∈{0,1e-8}; extra epsilon controls 1e-12/1e-4
  at a=.7.
- The internal finite criterion is 0≤(gamma_p-q_p(w_T))/gamma_p≤.01. GD/GDM
  are checked against L2 and Adam against L-infinity for this descriptive
  comparison. This does not assert the asymptotic target of positive-epsilon
  Adam. The criterion was in the protocol before formal pilot and primary runs;
  there was no external preregistration.

Adam moments use a documented common exponential rescaling, algebraically
equivalent in real arithmetic, to avoid underflow. No hidden epsilon or
denominator floor is inserted. Six numerical checks include direct-recurrence
agreement, underflow handling and analytic reference geometry. Loss may round to
zero while its logged scale remains finite. Saved states are recorded
checkpoints, not every individual update.

The unchanged training source SHA-256 is
`cff71403539c260aa910a48a7332900c25d69b47be74ab464aac43d40c6fdc67`;
the original protocol SHA-256 is
`19008c8ba366af91cb9e1d94b42e6ff1ee610a34832033feabcd1e140115604a`.
The source hash records the implementation used for training, before this
result snapshot was committed.

## Four-rate follow-up

[MULTIRATE.md](MULTIRATE.md) adds 12 million-update runs (including three exact a=.5 cross-checks), 732 saved moment states and local update-geometry diagnostics. The deliberately wrong scaling remains close to zero epsilon across all four rates; correct epsilon materially affects the late update for a=.3/.5/.7. The a=1 run is only partially epsilon-dominated and even its zero-epsilon comparator is not close to optimum. The [summary and Q&A](reproduction_update.md) distinguish these observations from unverified hypotheses about the authors' implementation.
