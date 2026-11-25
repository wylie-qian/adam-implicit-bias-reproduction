# Reproduction summary, questions and answers

Prepared on 2026-11-12, before the main run. The final result is in
[results/2026-11-13-legend](results/2026-11-13-legend/README.md). This note summarizes the corrected Figure 1 comparison
and the four-rate diagnostics in short form. All results below are **our own
independent experiments**, prepared with AI assistance and reviewed by the
group. They are not the authors' code or data.

## What was reproduced, and what remains different

- The experiment uses paired 50×50 separable data, full-batch logistic loss and 10⁶ updates.
- With η_t=1/√t, the relative L∞ margin gap is **0.657% for ε=0** and **31.512% for ε=10⁻⁸**.
- Correct raw and stabilized implementations agree. The paper's positive-ε overlap remains unreproduced in our setting.
- The status is **partial reproduction, with an audited discrepancy**.

The comparison figure is available as [PNG](results/2026-11-12-multirate/figures/slide_epsilon_comparison.png)
and [SVG](results/2026-11-12-multirate/figures/slide_epsilon_comparison.svg). It
shows our own experiments on fixed seed-2 data with a shared initialization, not
author code or data. Its vertical axis is the relative margin gap, not accuracy.
The orange curve, labeled **INVALID**, is an intentionally incorrect scaling
control, not a successful reproduction.

We first corrected our learning-rate interpretation using the paper's figure
legends. On our fixed dataset, the zero-epsilon gap then decreased to 0.657
percent, while positive epsilon still gives a 31.512 percent gap. Independent raw
and stabilized recurrences agree, so the difference is not explained by our
numerical rescaling when it is applied correctly. We use our own declared random
data because the authors' original arrays and a verified implementation were
unavailable.

## Why the discrepancy is informative

\[
\frac{m}{\sqrt v+\epsilon}
=\frac{M}{\sqrt V+e^b\epsilon},\qquad M=e^b m,\;V=e^{2b}v.
\]

- Omitting ε's scale factor changes the algorithm and mimics the ε=0 curve at all four tested rates.
- At a=.5, fixed ε dominates the late denominator, with cosine(update, −momentum) ≈ **0.99999999**.
- A small loss does not imply directional convergence. At the first saved checkpoint with loss below 10⁻⁸, the correct ε run still has a **34.82%** margin gap.
- There is **no verified evidence of an author implementation error**.

Epsilon is small in absolute terms, but the relevant comparison is with the
square root of the second moment. When the moments are stabilized, epsilon must
be rescaled as well. Leaving it unchanged makes its effective original-scale
value shrink. This deliberately wrong control closely follows the zero-epsilon
behavior across all four rates, but it does not tell us what the authors
implemented. In our correct positive-epsilon run, the late update is almost
parallel to the negative first moment, which supports an epsilon-induced change
in update geometry. These observations concern a finite-epsilon experiment, not
a counterexample to the paper's zero-epsilon theorem (Zhang, Zou and Cao,
NeurIPS 2024, Sections 3 and 5).

## All four rates

All conditions run one million updates with shared seed-2 data and initialization. Values are relative L∞ training margin gaps.

| a in η_t=t^(−a), t≥1 | Correct ε=0 | Correct ε=10⁻⁸ | Incorrect scaling control |
|---|---:|---:|---:|
| .3 | 1.523% | 34.336% | 1.524% |
| .5 | 0.657% | 31.512% | 0.657% |
| .7 | 2.435% | 27.430% | 2.435% |
| 1 | 22.738% | 24.072% | 22.738% |

The four-panel figure is available as [PNG](results/2026-11-12-multirate/figures/four_rate_comparison.png) and [SVG](results/2026-11-12-multirate/figures/four_rate_comparison.svg). It uses the **absolute** L∞ margin gap, unlike the relative-percent comparison figure above, so the two vertical scales are not directly comparable.

For a=.3/.5/.7, all coordinates enter ε>√v by updates 3036/3075/3964 and are strongly epsilon-dominated at the endpoint. For a=1, this does not hold for all coordinates by one million updates; even ε=0 remains far from optimum. Similar curves at a=1 therefore do not establish successful convergence. The three a=.5 runs repeat earlier conditions exactly; this round has 12 completed runs, not 12 novel settings.

## Questions and answers

**Does this disprove the theorem?** No. The main theorem concerns epsilon=0 under its assumptions. We investigate a separate finite-positive-epsilon numerical observation. Finite training trajectories cannot establish or refute an asymptotic statement merely from apparent plateaus.

**Did the authors omit the epsilon transformation?** We have no verified evidence of their implementation. We show that this mistake can create a similar-looking curve. This is a diagnostic hypothesis, not an attribution.

**Why is 10⁻⁸ not negligible?** Because it is added to √v, which can become smaller than 10⁻⁸ as gradients decrease. At the same stored moments, the coordinatewise retained update factor is √v/(√v+ε). At a=.5's endpoint it ranges from about 5.08×10⁻⁶ to 5.54×10⁻⁴. This counterfactual changes the denominator at that state; it is not a separately trained zero-epsilon trajectory.

**Does this prove convergence to the L2 max-margin solution?** No. Our measured update is locally momentum/gradient-like. Establishing an asymptotic L2 direction requires a theorem with matching assumptions or further evidence. Related work uses different conventions and settings.

**Could randomness explain the difference?** It is not eliminated in every possible dataset. Earlier controlled data/initialization/RNG checks did not recover the overlap in the tested matrix. Here all four rates share one dataset selected earlier using reference geometry, not optimizer performance. This round strengthens a mechanism diagnosis, not a population-level success estimate.

**Could numerical underflow explain it?** We now also replayed the raw recurrence for one million updates in native extended precision, with no moment scaling. All four fixed-positive-epsilon runs retain the same reported margin results and have no monitored logistic-weight or squared-gradient underflows. The zero-epsilon a=.5 replay preserves the approximately 0.65725% endpoint margin gap, but has small arithmetic-dependent trajectory differences and some late underflows of extremely small easy-example contributions. This is about 19-digit native arithmetic with wider exponent range, not full arbitrary-precision training. See the [literal replay report](../../experiments/paper50/results/2026-11-13-literal-replay/README.md) for complete errors and scope.

**What did verification cover?** The four-rate study completed all 12 conditions and checked scalar diagnostics at 732 saved states. The final independent raw recurrence replays five valid conditions from initialization, reconstructing both moment histories and comparing all 305 retained weight/moment states. The deliberately wrong recurrence is not part of that replay. Exact epsilon-crossing times from the original study were logged during execution rather than independently revalidated by the replay.

**Does low loss justify stopping?** It may be a practical prediction-oriented stopping rule, but it does not certify max-margin directional convergence. In this dataset the first retained loss<10⁻⁸ checkpoint still has a 34.82% relative gap for the correct a=.5 positive-epsilon run. “First retained” is not the exact first crossing.

**Why could you not simply run the authors' code?** The final paper's NeurIPS checklist item 5 explicitly answers No to open access to the data and code sufficient for reproduction. Our bounded follow-up search did not locate a verified later implementation. We do not claim code can never exist.

**What does this work contribute?** A formula-aligned independent implementation, declared missing-setting choices, preserved controls and negative results, reproducible diagnostics, and a careful distinction between theorem assumptions, empirical observations and possible implementation pitfalls.

## Sources and evidence

- Target: [Zhang, Zou & Cao, The Implicit Bias of Adam on Separable Data, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/2ac79356a03fe5e9250e5e77ebc76e6e-Paper-Conference.pdf), Sections 3 and 5 and Figure 2.
- Our full [multirate report](MULTIRATE.md), [previous diagnostic audit](DIAGNOSIS.md), [source distinctions](FOLLOWUP_SOURCES.md), [machine-readable comparison](results/2026-11-12-multirate/figures/comparison.json).
- Related mechanism evidence, not an exact replication: [Fan, Schmidt & Thrampoulidis, Appendix G, arXiv:2502.04664v4](https://arxiv.org/html/2502.04664v4).
