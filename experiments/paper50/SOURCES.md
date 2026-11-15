# Source and implementation audit — 2026-11-05

This is an independent implementation prepared with AI assistance and reviewed
by the group. Earlier results are preserved. No author code, external dataset,
course material or paper figure is redistributed.

## Primary sources checked

1. Zhang, Zou and Cao, [NeurIPS 2024 final conference paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/2ac79356a03fe5e9250e5e77ebc76e6e-Paper-Conference.pdf),
   34 physical pages; §3 equations 3.1–3.4, §4, §5 (physical pp. 6–7),
   the structure of technical appendices A–C, a search for experimental settings,
   and checklist Q5/Q8 (physical pp. 30–31). This is not a full proof audit.
2. [Official proceedings entry](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2ac79356a03fe5e9250e5e77ebc76e6e-Abstract-Conference.html):
   authors, conference and paper link. The inspected entry exposes no supplemental
   code/data link. The PDF includes the technical appendices; no separate
   experiment appendix supplies missing seeds or learning-rate constants.
3. [Author's publication entry](https://difanzou.github.io/publications/): this
   paper has Paper and Arxiv links, no Code link in the inspected entry.
4. [arXiv v1](https://arxiv.org/html/2406.10650v1), 2024-06-15: version cross-check;
   the formal source for this experiment remains the final conference paper.
5. [OpenReview](https://openreview.net/forum?id=xRQxan3WkM): a browser-verification
   page prevented inspecting the discussion or attachment history. No bypass or
   login was attempted. Later unlinked author code therefore cannot be ruled out.

The final PDF's checklist Q5 answers No for open data/code. Its Q8 does not
provide resource measurements. This supports reimplementing from the published
method, not assuming an official implementation or identical original samples.
An exact-title web search did not establish an author-controlled code release;
third-party projects returned by search were not used or treated as author code.

## 2026-11-12 correction: Figure 1 legends

The original audit missed learning rates rendered inside the figure image.
Physical page 7 of the final PDF (and of arXiv v1) explicitly labels GD and GDM with eta=0.1,
and both Adam curves with eta_t=t^(-0.5). Figure 2 also labels t^(-a), without
an extra multiplicative constant; the §5 text itself only writes Θ(t^-a). The claims below that Figure 1 supplied no
schedule/constants were therefore incorrect. The historical protocol and
measurements remain unchanged; see [ALIGNMENT.md](ALIGNMENT.md) for the additive
Figure 1 comparison and the remaining unspecified conventions.
Post-hoc diagnostics with the legend settings are in
[results/2026-11-12-posthoc](results/2026-11-12-posthoc/README.md). The main full
rerun is [results/2026-11-13-legend](results/2026-11-13-legend/README.md);
a variant with c = 0.1 and a summed loss, chosen afterwards by comparison with the
paper's curves, is [results/2026-11-12-matched](results/2026-11-12-matched/README.md).

## Provenance of each implemented setting

| Setting | Classification | Implemented value and source / reason |
|---|---|---|
| Data dimensions, feature/label distributions | paper_setting | n=d=50, independent standard-normal features and uniform independent ±1 labels (§5) |
| Ten candidates and geometry-based selection | paper_setting | Generate ten data sets and choose the greatest difference between reference directions (§5) |
| Actual random draws | team_choice | NumPy PCG64 seeds 0–9; author seeds and arrays unavailable |
| Exact selection metric | team_choice | L2 distance of L2-unit directions; smallest seed wins a tie; fixed before training |
| Initial distribution | paper_setting | Standard normal (§5); no zero initialization inherited from the 2D demo |
| Exact initialization and pairing | team_choice | seed+10000, independent from data; shared between methods within a data set |
| Objective and model | paper_setting | Mean logistic loss, full training data, linear model without intercept (§3/§5) |
| Adam moment equations | paper_setting | m=.9m+.1g, v=.99v+.01g²; epsilon outside sqrt; no bias correction in equations 3.2–3.4 |
| Adam beta values and epsilon comparison | paper_setting | beta1=.9, beta2=.99; epsilon=0 or 1e-8 (§5) |
| GDM momentum | paper_setting / team_choice | .9 is from §5; normalized EMA convention m=.9m+.1g is our declared choice, since §5 does not specify the recurrence |
| Primary Adam step schedule | team_choice | eta=(t+2)^(-.7), c=1, t begins at zero. Exponent .7 appears in Figure 2; this differs from Figure 1's explicit t^(-.5) legend |
| GD/GDM step sizes | team_choice | Historical choice eta=.5/L, L=||X||_spectral²/(4n), fixed independently for each candidate; Figure 1 explicitly uses eta=.1 |
| Iterations | paper_setting | 1,000,000 updates per primary run (§5) |
| All-ten training | deviation / extension | Keep and train all ten, while paper shows its selected data set only |
| Rate sweep exponents | paper_setting / team_choice | .3,.5,.7,1 and c=1 from Figure 2 legends; the t+2 convention is ours |
| Additional epsilon levels | deviation / extension | 1e-12 and 1e-4 on selected data, planned in protocol before primary results |
| Arithmetic | team_choice | float64, algebraically rescaled Adam moments to avoid numerical underflow; equivalent real recurrence, no hidden epsilon or denominator clamp |
| Reference solvers | team_choice | SciPy HiGHS LP and SLSQP QP, independently checked primal/dual/KKT residuals |
| Finite success threshold | team_choice | At most 1% relative gap to the relevant reference; descriptive, unrelated to a theorem's limit or the old 2D tolerance |
| Runtime budget and checkpoints | team_choice | 600 s wall/CPU per run; first steps, log-spaced steps and every 1000th step; all weights/moments retained at checkpoints |

## References and numerical representation

Let Z_i=y_i X_i. Solve the infinity reference as the LP
minimize -gamma, with -Z u + gamma <= 0 and -1 <= u_j <= 1.
Solve the L2 reference as the strictly convex QP
minimize ||w||²/2 subject to Z w >= 1, then gamma2=1/||w||2.
For every candidate record primal feasibility, dual feasibility, stationarity,
complementarity, and primal–dual objective differences; reject residuals >1e-7.
Direction distances refer to one returned LP optimum; uniqueness is not asserted.
The primary outcomes use margin gaps and do not require direction uniqueness.

The Adam implementation stores M_t=exp(b_t)m_t, V_t=exp(2b_t)v_t,
where b_t=max(0,min_i Z_i w_t). Before each update multiply historical M,V by
exp(b_t-b_previous), exp(2(b_t-b_previous)), respectively, then add the gradient
scaled by exp(b_t). For epsilon=0 the common scale cancels. For positive epsilon,
the denominator is sqrt(V_t)+epsilon*exp(b_t). This is a change of numerical
representation, not a change to the recurrence. Initial zero moments are handled
exactly; no division by a floor is allowed. The independent check compares 2000
literal updates for epsilon=0,1e-8,1e-4 and verifies a case where direct gradients
underflow but the correct rescaled update remains nonzero. It is not claimed that
this floating-point path is identical to an unavailable author's implementation.

Saved CSVs contain measured checkpoints. `states.npz` contains their actual
weights and scaled moments; it is not a log of every one of the million updates.
`loss_float64` may become zero after underflow; `log_loss` preserves its scale.
Logs of sqrt(v) are computed from the scaled states. An epsilon-dominance time
is the first *recorded checkpoint* meeting the condition, not the exact crossing.

## Code, data and AI attribution

The mathematical implementation, verification utilities and plots were prepared
with AI assistance from the cited formulas and reviewed by the group. They are
not the authors' released code. No author or third-party implementation is
copied.

The inputs are newly generated Gaussian samples and independent binary labels;
no third-party dataset, downloaded model, paper figure or course material is
redistributed. NumPy/SciPy are BSD-licensed and Matplotlib uses its PSF-based
license. The cited article and its figures are linked rather than repackaged.
The repository has no license file. This experiment does not assign a license
or make a claim about permission to redistribute existing materials.

The compact result snapshot documents its retention scope in its own README.
Omitted checkpoint histories are not represented as included or reverified by
the compact-snapshot checker.
