# Second diagnostic audit — 2026-11-12

**56 additional million-step conditions completed. Fixed positive epsilon still
does not reproduce the near-overlap in the paper.** No audited training formula
was changed to force visual agreement. See [DIAGNOSIS.md](../../DIAGNOSIS.md) for
measured outcomes, the mathematical scaling trap, limits and reproduction commands.

- `conventions/`: ten fixed implementation controls, complete configurations,
  scalar trajectories and endpoint weights/moments; all metrics use original
  float64 evaluation data. `INVALID_RECURRENCE` is explicitly an incorrect
  fixed-epsilon implementation retained only to test the diagnostic.
- `randomness/`: all 46 paired data, initialization and legacy RNG outcomes;
  one aggregate NPZ retains six predetermined checkpoint states per run.
  The correlated exploratory initialization is disclosed and retained.
- `precision.json`: 80/120-digit independent Decimal checks of ten next updates
  from saved states, including states beyond raw float64 moment underflow.
- `paper_source_audit.json`: primary-source URLs/hashes, confirmed settings,
  original figure comparison method and bounded code-search findings.
- `figures/`: regenerated plots of our measured data. No author figures copied.

The frozen training source, protocol, 2026-11-05 snapshot and first Figure 1
alignment snapshot remain unchanged. All code and analysis are AI-assisted.
