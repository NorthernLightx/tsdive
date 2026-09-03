# Roadmap

Each stage is one method; a stage ships when it has a benchmark row and a
test. This page names what shipped at each stage and what it still owes.

| Stage | Theme | Status |
|-------|-------|--------|
| 1 | Store + data-physics layer | shipped (3W real, TEP real) |
| 2 | Features + splits + first published number | shipped (synthetic backbone) |
| 3 | Provisional baselines (MAD / moving-range screen) | shipped (synthetic; 3W real with caveats) |
| 4 | Refined per-tag regime baselines | shipped (synthetic; 3W real with caveats) |
| 5 | SPC | shipped (synthetic; 3W real with caveats) |
| 6 | MSPC (PCA T2/SPE) | shipped (synthetic; 3W real with caveats) |
| 7 | ML detectors + static evidence viewer | shipped (synthetic; 3W real with caveats) |
| 8 | LLM narration over the evidence ledger | shipped (offline-gated) |

Stages 1 to 6 are reachable from the CLI (`profile`, `segment`, `screen`,
`spc`, `mspc`, `compare`). Stage 7 is library-only, because it needs feature rows over
many windows and the `ml` extra. Every stage ships benchmark rows in
`BENCHMARKS.md`, computed from the deterministic synthetic backbone and
compared byte for byte in CI. Three REAL sections follow the synthetic
table, one per study in `examples/studies/`.

Real-data numbers are published under group holdout by asset; each names
its fold split. The reading rules are in each study's REPORT.md in
`examples/studies/`.

## Status

Stages 1 to 8 ship in 0.3.0, with the evaluation protocol as `tsdive.eval`.

Parked, each with the measured reason:

- Conformal martingale alarm. It lives in `tsdive.eval` and is not wired
  into `screen` or `spc`. On 3W normal records it fires on 76.3% at delta
  0.05 against a 5% bound, and the permutation check gives 0.04%, so the
  bound needs an exchangeability the records do not have. Report:
  `examples/studies/3w_conformal/REPORT.md`.
- Compression detection. A Thornhill-style compression-factor estimate
  saturates on data rounded to a lattice: the estimate stays near 2
  whether the true factor is 1 or 40. It waits for an archive with
  documented compression settings.
- Control-loop performance (Harris index, stiction). It waits for a loop
  archive with a loop schema. A candidate archive exists (Bauer 2019,
  IECR).
- Zero-shot forecasters as shipped detectors. The Chronos-Bolt study
  gives the only false-alarm rate under the floor (18.2%) and a post1
  detection rate of 27.3%. It stays a benchmark subject. No forecaster
  ships as a tool.

Next beds: a second real dataset where records return to normal after a
fault, and a row that measures what the frozen-sensor and rounding checks
change in the 3W detector numbers.

## Stage 0, sources

Shipped: parquet archives, CSV and parquet ingest, and the 3W and TEP
converters, each answering a window read under a stated contract through
`tsdive.store.source.Source`. Owed: an OPC UA history client and a
message-bus capture. A source ships with a converter or a `Source`
implementation, a benchmark row and a test (docs/SOURCES.md).

## Stage 1, store and data-physics layer

Shipped: the store, `tsdive profile`, and every check it prints. The
check-by-check table, with the error each one raises, is the "What it
checks" section of the README. A tag is `(source_id, point_id)` and the
display name is metadata. `loop_id` and `role` in `{PV, SP, OP, MODE}` are
stage-1 schema fields, not a later retrofit. A read declares its sampling
contract as a positional argument and gets a stable digest back.
Provenance fingerprints produce the replay verdicts `MATCH`, `CODE_DRIFT`,
`DATA_CHANGED` and `BOTH`. Two refusal reference cases ship with committed
outputs.

## Stage 2, features, splits and the first published number

Shipped: contract-aware feature extraction (`features/`), the synthetic
validation backbone with injected fault archetypes and ground truth
(`data/backbone.py`), and provenance-stratified splits whose proportions
are exact within each stratum (`data/splits.py`), all labelled SYNTHETIC
in `BENCHMARKS.md`. `stratified_split` cuts rows inside a stratum, so it
cannot hold a whole group out; `group_holdout` closes that. Whole groups
go to one fold, taken
largest first, placed where the per-stratum counts end up least spread. A
seeded SHA-256 rank breaks size ties. `GroupSplit.leakage_check()` raises
`GroupLeakage` if a group reaches two folds, and it runs before the split
is returned. Folds come out uneven by construction, and `fold_sizes`
reports that.
Shipped: `tsdive.eval`, which carries the evaluation protocol as an API:
`ranking_metrics`, `clock_control`, `worst_baseline_threshold`, `fires`,
`far_floor` and the group holdout re-exported beside them.

## Stage 3, provisional baselines

Shipped: MAD and moving-range screens, each object carrying its
provisional caveat, with sparse or censored history raising
`InsufficientQuality` rather than producing a wide interval. The benchmark
row states why they are provisional: the regime-blind MAD misses a level
shift that stage 4 catches.

## Stage 4, refined baselines

Shipped: MODE-keyed per-regime median and MAD baselines. A regime with too
few GOOD samples raises `RegimeTooSparse`. Benchmark: 100% regime screen
detection against stage 3's 0% on the same shift.

Shipped: `segment` (`changepoints/pelt.py`), which supplies regimes when
no MODE tag keys them, which is most tags. It runs PELT with the L2 cost
over MAD-scaled values, so one penalty means the same thing on a flow and
on a temperature. The penalty defaults to `3*log(n)`.

Shipped: cross-instance population baselines (`baselines/population.py`).
`population_baseline` summarises every other training-fold window of the
same (asset, variable). `screen_population` fires on a frozen window only
when that pair holds more than one value in at least 95% of its training
windows. A pair with fewer than `MIN_BASELINE_WINDOWS` training windows
raises `PopulationTooSparse`.

Shipped: `compare` (`compare.py`), which reads two periods of one unit
and reports what moved between them. Table 1 gives each tag the after
period's quality read against the before period's, the level shift in
before-period MAD-sigmas, the spread ratio, the share of after samples a
before-period MAD screen flags, and the first changepoint at or after the
after period starts. Table 2 gives each pair the Pearson correlation of
first differences in each period and a moving block bootstrap interval on
the change, and prints only the pairs whose interval excludes 0. Table 3
fits one PCA on the before period and reads it on the after one. A tag
changed and a pair decoupled. Neither word names a cause.

Shipped: a plan that sets `before` and `after` runs `compare` over all its
archives at once, as the last step.

Shipped: `segment --mode-out FILE` writes the segments as a MODE archive,
one label (`S1`, `S2`, ...) per sample the window read, and `screen --mode
FILE` builds one baseline per label from it.

Owed: mutual information between two tags, and per-tag attributions from
a fitted model. Both wait for a benchmark row showing them find something
the Spearman column and the SPE contributions do not.

Owed: a population baseline that generalises across assets. Under a
held-out-asset split the current one has no training windows for the pair
it is asked about and refuses (`PopulationTooSparse`).

## Stage 5, SPC

Shipped: individuals limits from baseline statistics, xbar and R charts
from the Shewhart constant tables, and a rules engine (BEYOND_3SIGMA,
RUN_9_SAMESIDE, TREND_6) reporting each rule's hits separately. Limits
come from baseline windows, never from the monitored data.

## Stage 6, MSPC

Shipped: PCA-based Hotelling T2 and SPE over aligned multi-tag matrices.
Alignment raises `MspcAlignmentError` on insufficient common-grid coverage
instead of interpolating across tags. Control limits are empirical
training percentiles, so they carry no distributional claim. SVD signs are
fixed so models replay identically. The benchmark shows variance faults
surfacing in SPE while T2 stays quiet. The two statistics are reported
separately and never summed.

## Stage 7, ML detectors and the viewer

Shipped: IsolationForest over physics-gated feature rows. Incomplete rows
are excluded and counted rather than dropped, and `random_state` is
pinned. Shipped: `tsdive report-html`, a self-contained static HTML
evidence viewer with no server and no JavaScript dependencies. Studied,
not shipped: a zero-shot time-series foundation model (Chronos-Bolt) as a
benchmark subject on the onset-aligned 3W windows, under the same
protocol as the detectors (`examples/studies/3w_chronos`, extra `tsfm`).

## Stage 8, LLM narration

Shipped as the evidence-ledger boundary. Narration consumes only the
serializable `EvidenceLedger` (profiles, benchmark rows, refusals). The
offline scripted narrator is what tests and CI use. A remote narrator
raises `NarratorUnavailable` unless the operator sets
`TSDIVE_LLM_ENDPOINT`. No endpoint, key or network call exists in this
repository's default code paths. Narration produces advisory prose over
cited evidence. It cannot write to a historian, because the store is
read-only, and it cannot override a typed refusal. `tsdive run` fills
the ledger by walking one plan over several archives. A step that raises
for one archive writes a ledger row rather than propagating.

## Deferred, pending a benchmark row and a test

- Granger causality, PCMCI and transfer entropy: no replayable causal
  benchmark exists.
- Batch machinery (EVT, batch SPC): every current tool is invalid on batch
  data, so batch needs its own physics layer first.
- Stiction detection and the Harris index: these need a control-loop
  schema at scale and a benchmark archive with known stiction cases.

## Cut

Anything that needs write access to a historian. Anything with no refusal
path.
