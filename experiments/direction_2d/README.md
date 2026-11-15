# Two-dimensional example: the directions of GD and Adam

This is a small synthetic example that illustrates the paper's main claim in
two dimensions. It is not part of the Section 5 reproduction, and it does not
use the authors' code. The data are the two points ((1, 2), +1) and
((−2, −1/2), −1). The loss is the mean logistic loss without an intercept, and
training starts from w₀ = 0.

The protocol in [protocol.json](protocol.json) was fixed before the run. Each
method takes 20,000 full-batch steps. GD uses a fixed learning rate of 0.25.
Adam follows the update in Section 3 of the paper, with no ε and no bias
correction, β₁ = 0.9, β₂ = 0.99 and η_t = (t + 2)^(−2/3). Seed 7 only places
the points for the finite-difference gradient check. Training itself has no
random sampling. The run had a budget of 30 CPU seconds and 60 wall-clock
seconds, and there was no tuning and no early stopping.
[Formulas: arXiv HTML, Section 3](https://arxiv.org/html/2406.10650v1#S3)

**Assumptions.** The vector u = (1, 1) separates the data with margins 3 and
2.5. The initial gradient is (−3/4, −5/8), so ρ = 25/64 satisfies Assumption
4.2. The Adam parameters satisfy β₁ ≤ β₂. The learning rates satisfy
Assumption 4.3, and Assumption 4.4 by Lemma C.1. The Hessian norm for GD is at
most Σ‖zᵢ‖²/(4n) = 37/32, so 0.25 ≤ 1/L. The code computes the loss and the
gradient in a numerically stable form and stops if a denominator becomes zero.
It never adds an ε.

| Value after 20,000 steps | GD | Adam |
|---|---:|---:|
| Loss | 5.30584e-5 | 2.24584e-32 |
| ‖w‖₂ | 5.06203 | 39.92943 |
| L2-normalized margin (optimum 1.94145) | 1.90810 | 1.80770 |
| L∞-normalized margin (optimum 2.5) | 2.36706 | 2.46252 |
| L2 distance of the unit vector to its reference direction | 0.045251 | 0.038921 |
| Gap to its own optimal margin | 0.033355 | 0.037483 |
| Both preset criteria (≤ 0.001) met | No | No |

Both methods completed the fixed number of steps without a budget or numerical
stop. Together they computed 40,000 training gradients in about 0.086 CPU
seconds. A small loss does not mean that the direction is already close to the
reference solution. The Adam direction first moves away from its reference
direction and then turns back, so a finite trajectory is no evidence of
monotone convergence. The two methods use different learning rates and norms,
so the example supports no general claim about speed or generalization.

**The size of ε matters.** At the last update, √v is about
(6.32e-32, 1.58e-32). The second value agrees with
min_sqrt_v = 1.5806837868122177e-32 in the saved CSV. With the denominator
√v + 1e-8, ε would dominate both coordinates. This ε-free trajectory is
therefore not the trajectory of Adam with the common default ε = 1e-8. We did
not run an ε comparison in this example, so we report no trajectory or end
value for it. [Paper, Section 3](https://proceedings.neurips.cc/paper_files/paper/2024/file/2ac79356a03fe5e9250e5e77ebc76e6e-Paper-Conference.pdf)

**Metrics and checks.** The raw margin fields were corrected when the run was
recorded. At the 422 checkpoints that were checked under the same protocol,
all other fields agree exactly. The folder `runs/frozen-20k-fixed-metrics`
holds the corrected data and figures. `verify.py` covers finite differences,
the first step and the moment history, the stable computation, the reference
solutions in both norms, the zero-budget and no-overwrite behavior, and an
independent three-step recursion.

Run the commands from the repository root. The output folder must not exist
yet, and plotting needs Matplotlib.

    python3 experiments/direction_2d/verify.py
    python3 experiments/direction_2d/run.py --protocol experiments/direction_2d/protocol.json --output experiments/direction_2d/runs/new-run
    python3 experiments/direction_2d/plot.py --run experiments/direction_2d/runs/new-run

The [six-panel figure](runs/frozen-20k-fixed-metrics/figures/trajectories.png)
is also available as [SVG](runs/frozen-20k-fixed-metrics/figures/trajectories.svg).
The check and the run need only the Python 3.10+ standard library. The saved
figures were made with Python 3.13.2 and Matplotlib 3.10.3. New runs are
ignored by git.
