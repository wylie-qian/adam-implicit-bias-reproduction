# Friday reproduction decision — 2026-11-13

**Presentation status: partial reproduction with a controlled fixed-epsilon discrepancy.**
The zero-epsilon margin trend is supported. The positive-epsilon overlap in the
paper's Figures 1–2 remains unreproduced on our declared inputs. No verified
author implementation has been obtained, so no author code error has been
established. The main theorem concerns zero epsilon and is not refuted here.

## Results to use in the main presentation

Use the frozen seed-2 dataset and shared initialization, full-batch mean logistic
loss, beta1=.9, beta2=.99, no bias correction, and one million updates with
eta_t=(t+1)^(-.5), t starting at zero. These are our AI-assisted experiments.

| Recurrence | Relative training L-infinity margin gap |
|---|---:|
| Correct epsilon=0 | 0.657242% |
| Correct fixed epsilon=1e-8 | 31.511943% |
| Intentionally wrong epsilon scaling | 0.657301% |

The third row is an **INVALID_RECURRENCE** control. It tests an explanation for
similar-looking curves and must not be described as a repaired implementation
or a successful fixed-epsilon reproduction. Percentages measure distance to the
optimal normalized training margin, not classification error.

The four-rate controls and their saved moment geometry are in [MULTIRATE.md](MULTIRATE.md).
The wrong scaling follows zero epsilon at all four tested rates. Correct fixed
epsilon produces larger gaps at a=.3/.5/.7. At a=1, both valid recurrences remain
far from the maximum margin at the finite endpoint.

## What the final independent replay adds

The [literal replay evidence](results/2026-11-13-literal-replay/README.md) executes
the raw Adam recurrence in C++ long double, with no gradient or moment rescaling.
It replays all four fixed-positive-epsilon rates and the zero-epsilon a=.5
condition from initialization for one million updates each. Inputs are the same
binary64 arrays, cast to the wider arithmetic format.

This extends independent validation from selected next updates to full-history
replay, using native extended precision rather than the earlier 80/120-digit
arithmetic. The fixed-positive-epsilon replays compare every retained
weight and both moment vectors with the scaled histories. Their agreement
supports the existing positive-epsilon implementation and preserves the large
margin discrepancy.

The zero-epsilon result requires a narrower statement. Its final margin remains
approximately 0.65725%, but its weight and moment history is sensitive to
arithmetic representation. Constants, reduction order and precision differ from
NumPy. Some extremely small easy-example logistic contributions also underflow
late in this replay, although the raw second moments remain positive. This is
native extended-precision evidence, not an arbitrary-precision proof or a claim
of bitwise-identical zero-epsilon trajectories. The replay report retains the
full differences and underflow diagnostics.

## What can be criticized in the paper

- **Finite-epsilon evidence has a reproducibility gap.** The paper's checklist
  item 5 explicitly states that it does not provide open access to data and code
  sufficient for reproduction. We still lack the author's arrays and executable
  treatment of small gradients. This publication statement does not establish
  that no later code release can exist.
- **The practical epsilon argument needs a measured scale comparison.** Section
  3's prose compares v_t with epsilon. Its displayed denominator instead calls
  for comparing sqrt(v_t,j) with epsilon, equivalently v_t,j with epsilon squared.
  This is a precision issue in the explanatory text, not proof of a coding error.
- **Curve similarity alone does not validate the recurrence.** Our deliberately
  incorrect scaling control produces similar curves while changing the effective
  epsilon. It establishes a possible failure mechanism without attributing it
  to the authors.

Primary URLs and related-work distinctions are in [FOLLOWUP_SOURCES.md](FOLLOWUP_SOURCES.md).
The independent multiclass and MNIST studies support investigating epsilon's
effect, but do not reproduce the target experiment. Unauthenticated review
mirrors are excluded from slide claims.

## Copy-ready conclusion

> We reproduce the zero-epsilon max-margin trend, but do not recover the paper's
> positive-epsilon overlap on our declared data. Independent implementations
> support our fixed-epsilon result. An incorrect epsilon transformation can
> imitate the reported qualitative behavior, but we have no verified evidence
> that the authors used it.

Use [two-slide copy and speaker notes](reproduction_update.md)
for integration. Keep the full audit as backup rather than expanding the main
nine-minute presentation. The live Google Slides has not been modified by this
update. The PR remains a draft for group review, and earlier evidence remains
unchanged.
