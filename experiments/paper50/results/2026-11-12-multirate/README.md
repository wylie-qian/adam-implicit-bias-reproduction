# Retained four-rate audit — 2026-11-12

All 12 planned conditions completed 1,000,000 updates each. Three a=.5 conditions repeat the prior convention study for cross-validation. Full interpretation, limitations and commands: [MULTIRATE.md](../../MULTIRATE.md).

- `manifest.json`: fixed matrix, inputs, source hashes, all outcomes and file hashes.
- `data.json`, `reference.json`: frozen original seed-2 inputs and certified reference data.
- `a-*.json`: 12 complete retained checkpoint histories; 61 states each. These include weights, first/second scaled moments, log scale and metrics, not every update.
- `verification.json`: saved-state metric verification and exact a=.5 endpoint comparisons.
- `transition_geometry.json`: independently recomputable same-state epsilon geometry, definitions, source/input hashes and six invariant checks.
- `figures/`: two original figures in PNG and SVG, plus machine-readable comparison metrics. The four-panel figure (`four_rate_comparison`) uses absolute gap; the single-rate figure (`slide_epsilon_comparison`) uses relative gap percent.
- `sources/review_source_audit.json`: provenance and authentication limits of a third-party review mirror. This is an investigative lead, not evidence for established author claims or for the claims in this repository.

All figures/results are AI-assisted group diagnostics, not author artifacts. INVALID_RECURRENCE is intentionally wrong and must never be reported as a repaired optimizer or successful reproduction. No original source, protocol or earlier evidence was overwritten.
