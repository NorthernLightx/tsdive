# Scope

Three lists. If a claim is not here, tsdive does not make it. Every
claim below has a provoking test in `tests/`.

## Claims

- Every retrieved window states the sampling contract it was produced
  under (calculation basis, retrieval mode, aggregate type, stepped-ness)
  and a stable digest of that contract.
- Every window reports its quality codes verbatim beside a derived
  severity of GOOD, UNCERTAIN or BAD.
- Every window reports coverage, gap classes with the rule that produced
  each one, clipping against the engineering range, a timestamp audit
  (duplicates, backwards runs, DST transitions in-window), and unit
  resolution.
- Identical inputs give identical outputs, and provenance records detect
  when the inputs changed (a backfill, a code or contract drift).
- A naive timestamp raises `SchemaError`, and a backwards index raises
  `NonMonotonicIndex`.
- A source with no quality column and no `--assume-quality` raises
  `SchemaError`.
- An unresolvable unit raises `UnresolvedUnitError` when an operation
  needs the canonical unit.
- A reference-condition unit with no declared reference state raises
  `IncomparableUnitsError`.
- Windows built under different sampling contracts raise
  `IncomparableSamplingError`.
- A censored baseline, or statistics over a window with no GOOD sample,
  raises `InsufficientQuality`. A baseline that overlaps the window it
  screens raises `ValueError` in `screen`, `spc` and `mspc`.
- A regime with too few GOOD samples raises `RegimeTooSparse`, and an
  (asset, variable) pair with too few training windows raises
  `PopulationTooSparse`.
- An under-covered multi-tag alignment raises `MspcAlignmentError`.
- A holdout group reaching two folds raises `GroupLeakage`.
- Baselines, SPC, MSPC, ML and narration inherit every refusal above.
  Control limits come from validated baseline windows, and the stage-8
  narrator sees only the serializable evidence ledger.
- Benchmark rows in `BENCHMARKS.md` come from the deterministic SYNTHETIC
  backbone unless a REAL section says otherwise (docs/DATA.md).

## Does not claim

- No fault diagnosis. Detectors report which signals fired, not causes.
- No action execution. tsdive never writes to a process, a historian or
  any other source, and the store package exposes no mutation API for an
  ingested archive.
- No alarm limits, no notification paths, no real-time posture.
- No SIL or safety-instrumented claim.
- No causal claims. No causal inference ships at any stage.
- No cross-platform numeric equality. CI pins ubuntu-latest and a locked
  environment, and fingerprints record the environment because BLAS and
  library versions move floating-point results.

## Not yet in scope

- Batch processes. Time-weighted averages, coverage math and flatline
  references are all invalid on batch data, so batch needs its own physics
  layer first.
- Alarm rationalization. It needs alarm datasets and event semantics that
  the store does not ingest.
- Causality (Granger, PCMCI, transfer entropy). It ships when a benchmark
  row and a test prove a claimed causal link on replayable data.
