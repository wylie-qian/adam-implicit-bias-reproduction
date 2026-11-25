# Source checks for the four-rate diagnostic

Checked on 2026-11-12. These notes separate inspected primary sources, our
algebraic observations, and an unauthenticated retrieval lead. They do not
identify an error in the authors' executable implementation.

## The claim being investigated

Zhang, Zou and Cao's [NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/2ac79356a03fe5e9250e5e77ebc76e6e-Paper-Conference.pdf)
proves its main margin theorem for Adam without epsilon. Section 5 separately
reports similar finite-run behavior with and without epsilon, including four
learning-rate exponents in Figure 2. These are distinct claims. The paper itself
acknowledges that a fixed positive epsilon eventually changes the dynamics.

The four-rate experiment tests that empirical similarity on our existing dataset.
It cannot recover the authors' exact random draws or identify their code.

## A precise threshold clarification

In Section 3, immediately after the practical Adam update, the prose compares
v_t with epsilon. The stated denominator is sqrt(v_t) + epsilon, so the relevant
coordinatewise comparison is sqrt(v_t,j) / epsilon. Equivalently, the threshold
for v_t,j is epsilon squared. At epsilon = 1e-8 that threshold is 1e-16.

This follows directly from the displayed denominator. It is a clarification of
the explanatory text, not evidence that the implementation uses the wrong units.

For identical moment histories the retained update fraction is exactly

    A_j = sqrt(v_j) / (sqrt(v_j) + epsilon) = r_j / (1 + r_j).

The fractional reduction is 1 - A_j. Our diagnostics name these quantities
separately. If all r_j are small, the update direction approaches minus the
first moment. This observation alone does not prove a limiting maximum-L2-margin
theorem for every learning-rate schedule or finite experiment.

## Related primary evidence

Fan, Schmidt and Thrampoulidis, [Implicit Bias of Spectral Descent and Muon on
Multiclass Separable Data](https://arxiv.org/html/2502.04664v4), Appendix G,
Numerical Validations / Figure 6, reports a transition in positive-epsilon Adam
from max-norm preference towards 2-norm preference. Larger epsilon produces an
earlier transition. The setting is multiclass cross-entropy, with a different
dataset and model dimensions. This supports the proposed mechanism in a related
setting; it does not reproduce or falsify the 2024 binary experiment.

Wang et al., [Does Momentum Change the Implicit Regularization on Separable
Data?](https://arxiv.org/html/2110.03891v2), Section 6.1, studies deterministic
Adam under specified conditions. Its update includes bias correction and places
epsilon inside the square root. The formal result uses a sufficiently small
constant learning rate; Remark 8 discusses decaying-rate extensions. Cite it as
theoretical context rather than claiming its statement directly proves the
limit of our exact outside-square-root, no-correction implementation.

## Code search and a review-mirror lead

The inspected [official NeurIPS poster page](https://neurips.cc/virtual/2024/poster/93086)
and author publication entrypoints did not expose a verified implementation or
correction. This is a bounded search result, not proof that no code exists.

A third-party Hugging Face dataset contains a row matching the paper title and
purported review/rebuttal text. The original OpenReview forum and attachment
required a verification challenge, so the copied replies could not be
authenticated. The retrieval record preserves URLs, row number and a content
hash in `results/2026-11-12-multirate/sources/review_source_audit.json`. Full review text
is not redistributed. These replies are excluded from established findings.

The lead suggested a useful independent question: does a low-loss stopping
criterion guarantee negligible epsilon? For logistic loss, our own elementary
bound is

    ||grad R(w)||_infinity <= B R(w),  B = max_i ||x_i||_infinity,

because 1/(1+exp(z)) <= log(1+exp(-z)). This is an upper bound on current
gradients, not a lower bound ensuring epsilon is negligible; second moments also
retain gradient history. Therefore low loss alone cannot certify either
negligible epsilon or a small normalized margin gap. We report the first retained
checkpoint below loss 1e-8 and its moment ratios without treating it as the exact
first crossing or as a separate early-stopped training run.


## 2026-11-13 primary-source follow-up

The final NeurIPS paper's checklist item 5 explicitly answers **No** to open
access to data and code sufficient to reproduce the experiments. Item 6 states
that the experimental details are supplied. These are the authors' statements
at publication, not proof that no code could have been released later. A fresh
check of the arXiv version history and the authors' publication entries still
located no verified implementation or erratum. OpenReview remained behind a
verification challenge, so mirrored review/rebuttal text is still excluded.

A newer independent primary source, Gronich and Vardi,
[Watch Your Step: Adam's Margin Maximization Bias Depends on the Learning Rate
Schedule](https://arxiv.org/html/2611.01929v1), Appendix F.2 / Figure 5,
measures the relative norm difference between updates with and without epsilon
at the **same** moments. Its MNIST network experiments also show epsilon
becoming influential during finite training. This is related mechanism evidence,
not a reproduction of Zhang et al.'s binary-linear experiment, and not evidence
of an author implementation error. The model, bias correction, beta2 and other
settings differ. Its deep-model learning-rate claims are not transferred to our
linear model.

The source audit record is retained in
[primary_sources_20261113.json](primary_sources_20261113.json).
No author was contacted and no external message was sent.
