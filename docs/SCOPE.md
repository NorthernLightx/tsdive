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
- A metadata file or `tsdive.meta` object carrying a key tsdive does
  not define raises `SchemaError` naming the closest known key
  (`tests/test_ingest.py`).
- At ingest, a numeric date that reads both day first and month first,
  with neither `--dayfirst` nor `--timestamp-format` stated, raises
  `SchemaError` (`tests/test_ingest.py`).
- A command that raises a typed error exits with status 3, and under
  `--json` also prints the refusal object the MCP server returns. A usage
  error or invalid input exits with status 2 (`tests/test_cli.py`).
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
- A switchback plan is a function of its window, block length, washout
  and seed, and its SHA-256 digest reads only integers and ISO 8601 UTC
  timestamps, so the same arguments give the same digest on every
  platform (`tests/test_switchback.py`).
- A switchback schedule with fewer than 20 balanced assignments, or a
  smallest two-sided p-value above 0.05, raises `DesignTooSmall`
  (`tests/test_switchback.py`).
- `switchback analyze` raises `ScheduleMismatch` on a plan whose digest,
  block times, balance or seed disagree with its blocks
  (`tests/test_switchback.py`, `tests/test_switchback_api.py`).
- Under the declared randomization, the `switchback analyze` test
  rejects a zero difference for at most 5% of the balanced assignments
  of an enumerated design, and its claim rate at a zero shift stays
  within 0.05 plus 3.5 Monte Carlo SE over 400 seeded plans
  (`tests/test_switchback.py`, `tests/test_switchback_api.py`).
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
- No causal claims from observational data. The one exception is
  `switchback analyze`: it states the effect of setting B against setting
  A under a randomized schedule that `switchback plan` drew and whose
  digest it verifies, by randomization inference, assuming the schedule
  was followed and carryover ended within the washout.
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
