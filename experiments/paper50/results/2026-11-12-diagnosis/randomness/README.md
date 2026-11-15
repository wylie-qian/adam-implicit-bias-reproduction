# Randomness and legacy RNG controls

All **46 runs completed one million updates**. The fixed probes do not restore
near-overlap between epsilon=0 and epsilon=1e-8. They test ordinary choices of
samples, initialization and NumPy generator, not the unknown authors' exact seeds.

All runs use full-batch mean logistic loss, n=d=50, beta1=.9, beta2=.99,
eta_t=(t+1)^(-.5) for zero-based code index t, no bias correction, and epsilon
outside sqrt. The audited frozen scaled recurrence is reused. Every planned
outcome is retained; no data or initialization is selected by optimizer results.

| Fixed family | Paired conditions | epsilon=0 relative Linfinity gap | epsilon=1e-8 gap |
|---|---:|---:|---:|
| Original ten PCG64 datasets, seeds 0–9 | 10 | 0.497870%–1.291502% | 24.190737%–42.868304% |
| Disjoint PCG64 initialization seeds 20000–20004 on dataset 2 | 5 | 0.504576%–0.752121% | 31.773023%–38.202657% |
| Original exploratory initialization seeds 0,1,3,4 on dataset 2 | 4 | 0.584895%–0.909872% | 30.429158%–38.449140% |
| Correlated initialization seed 2 on dataset 2 (disclosed control) | 1 | 0.912553% | 39.283372% |
| Legacy MT19937 root seeds 0,42,2024; one geometry-selected candidate per root | 3 | 0.683112%–0.856459% | 28.340414%–38.860347% |

PCG64 initialization seed 2 exactly reproduces the first feature row of PCG64
dataset seed 2. That control is correlated with its data and must not be called
an independent initialization. It remains visible in the original matrix. The
separate disjoint-seed matrix was added to check independent initialization
without deleting an inconvenient outcome. The archived regression check verifies
this collision explicitly.

For the legacy probes, each root seed initializes one `RandomState` stream.
Ten `(X,y)` candidates are generated sequentially, each with standard-normal
features followed by uniform +/-1 labels. The candidate with greatest Euclidean
distance between the L2-normalized L2/Linfinity reference directions is selected
before training (ties use smallest index). Initialization uses an independent
`RandomState(root_seed+10000)` stream, shared between both epsilon conditions.
Selections were root 0 -> candidate 0; root 42 -> candidate 6; root 2024 ->
candidate 2. Reference certificates and array hashes for all 30 candidates remain
in the manifest. This is a diagnostic choice, not a claim to recover author seeds.

## Compact retention and verification

One `compact_states.npz` stores six checkpoint indices from **every** run, chosen
as `linspace(0, original_checkpoint_count-1, 6, dtype=int)`. It includes initial
and final weights, scaled moments and moment scales. The scalar observations
are in `manifest.json`; endpoints and family ranges are in `summary.json`.
The same retention rule applies to every outcome, including the correlated draw.

The compact verifier regenerates all 40 distinct candidate datasets, checks
40 LP/QP reference certificates and the geometry selection, checks the fixed
condition matrix and all 46 initializations, and recomputes all **276 retained
checkpoints**. Maximum margin discrepancy is zero at stored float64 precision;
maximum reference-certificate residual is 1.681322e-12.

The manifest separately records the original full-run verification outputs and
source-manifest/file hashes. Those records refer to full histories verified
before compaction. Omitted histories are not claimed to be reverified by the
compact checker. No author data or code is distributed. Implementation and
analysis are AI-assisted.

## Commands from the repository root

Use the existing `experiments/paper50/requirements.lock.txt` environment.
Every output directory for new runs must not exist.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 experiments/paper50/verify_randomness.py test
python3 experiments/paper50/verify_randomness.py verify

# Recreate all 46 full runs in three separate fixed matrices.
python3 experiments/paper50/randomness.py run --repo . --output artifacts/randomness-new/original --jobs 2
python3 experiments/paper50/results/2026-11-12-diagnosis/randomness/provenance/independent_initialization_controls.py run --repo . --output artifacts/randomness-new/independent --jobs 2
python3 experiments/paper50/legacy_rng.py run --output artifacts/randomness-new/legacy --jobs 2

# Recreate a compact snapshot in a NEW directory.
python3 experiments/paper50/verify_randomness.py compact --original artifacts/randomness-new/original --independent artifacts/randomness-new/independent --legacy artifacts/randomness-new/legacy --clean-source experiments/paper50/results/2026-11-12-diagnosis/randomness/provenance/independent_initialization_controls.py --output artifacts/randomness-new/compact
python3 experiments/paper50/verify_randomness.py verify --output artifacts/randomness-new/compact
```

The original runner was copied byte-for-byte to `randomness.py` after the full
study; its content hash remains unchanged. The disjoint initialization runner is
preserved separately with its original hash, avoiding a misleading retroactive
change to the frozen matrix. Run-duration/platform fields naturally vary on a
rerun; numerical measurements and input-generation rules are the comparison
criteria.
