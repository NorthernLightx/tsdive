# DATA

Dataset roles, licences, and known upstream issues. Nothing here is
downloaded by default; `scripts/fetch_*.py` are opt-in and print total
bytes before asking.

## Roles (locked decision)

| Dataset | Role | Why |
|---------|------|-----|
| Tennessee Eastman Process (TEP) | **Validation backbone** | Fully synthetic, licence-clear, every variable's role documented; repeatable simulations give ground truth for validation numbers. |
| 3W (Petrobras) | **Provenance-stratified demo** | Real offshore data with per-fault labels and instance provenance; demonstrates the pipeline on messy reality. |

## 3W

Verified on 2026-08-28 against `main` @
`d717f0aadbe90b6dc6e15e29f6035e9630d22a5e`.

- Source: `github.com/petrobras/3W`, dataset version 2.0.0.
- **Licence: the data is CC BY 4.0** (`dataset/LICENSE-CC-BY`); the
  repository's code is Apache-2.0. Two different licences sit in one repo.
  An earlier version of this file said "MIT-style", which was wrong for
  both.
- Fetch: `python scripts/fetch_3w.py --dest data/3w --only-real [--classes 0,1] --yes`

### Layout

The parquet lives in the repository itself, with no git LFS, so raw blob
downloads work and no LFS client is needed. In `dataset/0` … `dataset/9`
the folder name is the event label of the instance:

| folder | files | folder | files |
|---|---|---|---|
| 0 | 594 | 5 | 450 |
| 1 | 128 | 6 | 221 |
| 2 | 38 | 7 | 46 |
| 3 | 106 | 8 | 95 |
| 4 | 343 | 9 | 207 |

2,228 instance parquet files, 1,873,131,359 bytes in total. Of those,
1,119 files (515,965,508 bytes) are real wells; the rest are
`SIMULATED_*` (1,089) and `DRAWN_*` (20). `instance_source` is read off
the filename prefix: `WELL-*`, `SIMULATED_*`, `DRAWN_*`.

### Schema facts that change how the data must be read

- `timestamp` is the parquet **index**, `datetime64[ns]`, **tz-naive**.
  It names no instant on its own; `scripts/convert_3w.py` localises it to
  UTC only because the caller says so (`--tz`), and records
  `timestamp_tz_assumed` in each instance's `INSTANCE.json`.
- 27 float64 process variables plus `class` and `state` (`Int16`).
- Units are in `dataset/dataset.ini`: `P-*` in Pa, `T-*` in oC, `ABER-*`
  in %, `QGL`/`QBS` in m3/s, `ESTADO-*` ∈ {0, 0.5, 1}.
- Sampling is exactly 1 s, monotonic, with no interior gaps in the
  instances sampled.
- **An absent variable is an entirely-NaN column**, not a missing column:
  9 to 23 of the 27 variables are absent per instance. Nothing in the
  file says a variable was never instrumented rather than lost.
- **There is no quality column.** A frozen sensor carries no code at all;
  the values repeat (for example `WELL-00002_20131214190030` holds `P-PDG`
  pinned at 0.0 for the whole instance). Any quality in a converted
  archive is therefore assumed, and marked as such
  (`quality_assumed=true`).
- Labels are per row, not per file. `class` ∈ 0-9, with transient periods
  coded `class + 100`, and `state` is the well status. The folder name is
  not the row label. Every real `WELL-*` instance sampled carries 3,600
  leading rows whose `class` and `state` are `<NA>`, and the real
  instances in `dataset/9` contain no row labelled 9 or 109.
- Manifests are locally-computed SHA256 over the downloaded files
  (`MANIFEST.sha256.json`), never git blob shas. `FETCH.json` records the
  upstream commit, fetch time, byte total and the filters used.

### Group holdout by well is the 3W gate

The 1,119 real instances come from 40 wells, and the distribution is
steep: `WELL-00002` alone contributes 326 of them and the bottom 20 wells
contribute one or two each. The same well appears in many event folders,
so a split that cuts inside a stratum puts one well on both sides.

Every 3W number at stage 3 and above is computed under
`tsdive.data.group_holdout(items, group_col="well",
stratum_col="folder_label", n_folds=5, seed=...)`, which assigns whole
wells to folds and refuses (`GroupLeakage`) if one ever lands in two.
Folds are uneven as a consequence: the fold holding `WELL-00002` is
several times the size of the others. The per-fold well lists are
published with the numbers, and what the split does to the scores is read
in [3w_detectors/REPORT.md](../examples/studies/3w_detectors/REPORT.md).

### Chronos-Bolt zero-shot on the onset-aligned windows

A benchmark subject, not a shipped detector: `amazon/chronos-bolt-small`
forecasts each onset-aligned window from the four hours before it, with
nothing fitted on 3W, and the forecast error is scored under the detector
study's protocol (`tsdive.eval`). It needs the detector study's aligned
cache under `data/3w_windows_aligned` and its frozen
`results/aligned_per_instance.csv` for the fold membership.

- Install: `uv sync --extra tsfm` (chronos-forecasting and a CPU torch;
  the library itself never imports either).
- Run: `uv run python examples/studies/3w_chronos/run_chronos.py`, then
  `uv run python examples/studies/3w_chronos/make_report.py`.
- Read: [3w_chronos/REPORT.md](../examples/studies/3w_chronos/REPORT.md).

### Conformal martingale on the own-history and onset-aligned windows

The conformal test martingale alarm (`tsdive.eval`) runs over both of the
detector study's caches, `data/3w_windows` and `data/3w_windows_aligned`,
reading the per-minute sub-group medians they already hold. Per instance
the first two windows are the fit set, the third the calibration window
and the rest the stream. The own-history cache splits by label into 538
normal instances (every window labelled 0), 53 mixed (three normal
baseline windows, a fault later), 54 with a fault inside the baseline,
and 468 with fewer than 4 windows, a design refusal. The aligned cache is
the 48 evaluable instances. A permutation of each instance's own scores
is scored beside it as the exchangeability check.

- Run: `uv run python examples/studies/3w_conformal/run_conformal.py`,
  then `uv run python examples/studies/3w_conformal/make_report.py
  --update-benchmarks`.
- Read: [3w_conformal/REPORT.md](../examples/studies/3w_conformal/REPORT.md).

### Baseline drift on the own-history and onset-aligned windows

Three own-history baselines scored by one code path over the same two
caches: `static` fits the median and MAD of the first three windows'
minute medians, `differenced` fits the same over the minute-to-minute
differences, and `rolling` refits over the three windows before every
scored window. The clock control sits beside them. The alarm rule needs
seven windows, so it scores 25 of the 538 normal instances and 49 of the
53 mixed ones; `rolling` is refused on the aligned cache, whose six
windows leave no predecessors for a threshold window.

- Run: `uv run python examples/studies/baseline_drift/run_drift.py`,
  then `uv run python examples/studies/baseline_drift/make_report.py
  --update-benchmarks`.
- Read: [baseline_drift/REPORT.md](../examples/studies/baseline_drift/REPORT.md).

### What the own-history checks change

The own-history design run twice over `data/3w_windows` and one code path.
`checked` is the shipped behaviour, where a pair whose baseline raises
`InsufficientQuality` or whose MAD scale is zero is dropped. `unchecked`
keeps every pair and uses a zero scale the way naive code does. On this
cache the zero-scale check drops 690 of 3,870 pairs and the quality check
drops none.

- Run: `uv run python examples/studies/3w_audit_effect/run_audit_effect.py`,
  then `uv run python examples/studies/3w_audit_effect/make_report.py
  --update-benchmarks`. It also reads
  `examples/studies/3w_profile/results/per_tag.csv` for the frozen-tag
  population.
- Read: [3w_audit_effect/REPORT.md](../examples/studies/3w_audit_effect/REPORT.md).

### Shift intervals on placebo dates and fault onsets

Before/after level and spread intervals (`naive`, `hac`, `ewc`,
`block_bootstrap`), raw and adjusted on the other tags of the instance.
The placebo bed reads `data/3w_windows`: each of the 538 instances with
no fault window and 4 consecutive windows gives 2 windows of minute
medians before a placebo date and the next 2 after it. The known-change
bed reads `data/3w_windows_aligned`: the 3 windows nearest before the
onset against the 2 after it. Nothing is fitted across instances. A
synthetic AR(1) bed with a known shift runs beside them.

- Run: `uv run python examples/studies/shift_intervals/run_shift.py`,
  then `uv run python examples/studies/shift_intervals/make_report.py
  --update-benchmarks`.
- Read: [shift_intervals/REPORT.md](../examples/studies/shift_intervals/REPORT.md).

### Switchback schedules on the placebo instances

The same 538 instances as the placebo bed above, each as one record of
240 minute medians. Settings A and B are assigned at random to 15, 30
and 60 min blocks, and the study injects a known shift into the B blocks.
The 60 min design has 6 assignments and is refused (`design_too_small`).
Nothing is fitted across instances.

- Run: `uv run python examples/studies/switchback/run_switchback.py
  --beds 3w`, 62 s on the first draws and 63 s on the fresh ones, 8
  worker processes. `--draw-offset 100000 --out
  examples/studies/switchback/results/confirm` reruns any bed on fresh
  assignments.
  Read: [switchback/REPORT.md](../examples/studies/switchback/REPORT.md).

### Known upstream issues

- The `dataset/folds` path 404s on `main` (fold splits were removed
  upstream). Stratification therefore uses `instance_source` provenance
  instead of folds.
- A HEAD request to the GitHub repository archive endpoint returns no
  `Content-Length`, so sizing a download from it is impossible. The
  fetcher enumerates `dataset/**` blob sizes through the git-trees API
  instead and prints the real byte total before asking.

## SKAB

Verified on 2026-09-03 against commit
`b2c0d46c2971dcbfe71e26087b6d231998bb91c2`.

- Source: `github.com/waico/SKAB`, the Skoltech Anomaly Benchmark. A
  water-circulation testbed (pump, valves, tank) runs 34 experiments
  of about 20 minutes in which a fault is induced and then removed, plus
  one anomaly-free run of 166 minutes. It is a testbed, not a plant.
- Role: the second real bed. Every labelled record returns to normal
  after the fault, so it measures what 3W cannot, the false-alarm rate
  after recovery and whether a score comes back under its threshold.
- Licence: GPL-3.0, repository and data. Nothing under `data/` is
  committed; the fetch script downloads the files.
- Fetch: `uv run python examples/studies/skab/fetch_skab.py --dest data/skab`.
  35 files, 4,392,581 bytes, 13 s; manifest sha256 `c0d612939333`. A
  second run downloads nothing when every file's sha256 matches the
  manifest.
- Convert: `uv run python examples/studies/skab/build_archives.py`.
  35 experiments, 280 archives, 0 refusals, 18 s.
- Run: `uv run python examples/studies/skab/run_skab.py`. 35 records,
  850 windows of 60 s, five tools, 313 s. It writes the tables to
  `examples/studies/skab/results/`.
- Read: `examples/studies/skab/REPORT.md` for the method, the tables,
  the conformal bed and what the profile found before any detector ran.
- Baseline drift: `uv run python
  examples/studies/baseline_drift/run_drift.py` scores the same 850
  windows under a static, a differenced and a rolling baseline in 4 s,
  reading the minute median of each window rather than its 1 s samples.
  Read:
  [baseline_drift/REPORT.md](../examples/studies/baseline_drift/REPORT.md).
- Shift intervals: `uv run python
  examples/studies/shift_intervals/run_shift.py --beds placebo
  known_change` splits the anomaly-free record into 8 non-overlapping
  10 min before and after pairs, splits each labelled record's pre-onset
  rows in half, and sets each labelled record's pre-onset rows against
  its anomaly span, on the 1 s rows as recorded. Both beds, with the 3W
  ones, run in 9 s. Read:
  [shift_intervals/REPORT.md](../examples/studies/shift_intervals/REPORT.md).
- Switchback schedules: `uv run python
  examples/studies/switchback/run_switchback.py --beds skab` assigns
  settings A and B to 5 and 10 min calendar blocks of the anomaly-free
  record, 500 draws, 73 s on the first draws and 72 s on the fresh ones.
  Read: [switchback/REPORT.md](../examples/studies/switchback/REPORT.md).

### Layout

The files sit under `data/<folder>/<name>.csv` upstream and land under
`data/skab/raw/<folder>/<name>.csv`:

| folder | files | rows | labels |
|---|---|---|---|
| valve1 | 16 | 18,160 | yes |
| valve2 | 4 | 4,312 | yes |
| other | 14 | 14,929 | yes |
| anomaly-free | 1 | 9,405 | no |

`build_archives.py` writes one archive per sensor to
`data/skab_archives/<folder>__<name>/<sensor>.parquet`, the labels to
`labels.parquet` beside them, and the wide CSV and metadata templates it
ingests from to `data/skab_work/<folder>__<name>/`. Every ingest refusal
goes to `data/skab_archives/refusals.json` with its error class and
message, and the run continues with the next file.

### Schema facts that change how the data is read

- The separator is `;`. The header is `datetime`, eight sensor columns
  (`Accelerometer1RMS`, `Accelerometer2RMS`, `Current`, `Pressure`,
  `Temperature`, `Thermocouple`, `Voltage`, `Volume Flow RateRMS`),
  `anomaly` and `changepoint`.
- `datetime` is naive, `2020-03-09 10:14:33`, with no offset anywhere
  in the files. The study passes `tz="UTC"` at ingest. The true offset
  is unknown and changes nothing in a within-record analysis, which
  reads only differences between stamps.
- The step is 1 s and rows are missing. Most records carry 2 s steps,
  and seven records carry one longer gap: 247 s in `other/2`, 76 s in
  `valve1/2`, 65 s in `valve1/7`, 64 s in `valve2/1`, 54 s in
  `valve1/4`, 33 s in `other/13`, 5 s in `other/12`. A fixed 60 s
  window inside such a gap holds no rows and is a refusal.
- There is no quality column. Every sample is ingested with
  `assume_quality="GOOD"` and every archive says `quality_assumed`.
- Units are not documented upstream, so `unit_raw` stays null, and no
  `sample_rate_s` is declared, so `mspc` is called with `rate_s=1`.
- Labels are per row: `anomaly` and `changepoint`, both 0 or 1. Each
  labelled record holds one anomaly run with normal rows before and
  after it; in `valve1/0` the anomaly rows are 573 to 973 of 1,147 and
  `changepoint` marks 4 rows. The anomaly-free file has no label
  columns. Labels stay in `labels.parquet` and never enter an archive.
- Timestamps are unique in every file. A duplicated stamp is refused
  before ingest (`SchemaError` from `require_unique`), because
  `ingest_wide` accepts it and only the read audit reports it.

## Turbine Upgrade

Verified on 2026-09-28 against Zenodo record 5516556.

- Source: Zenodo, DOI
  [10.5281/zenodo.5516556](https://doi.org/10.5281/zenodo.5516556),
  "Turbine Upgrade Dataset" (Yu Ding, 2021), companion data to the book
  *Data Science for Wind Energy*. Two pairs of turbines from one inland
  wind farm with a met mast: a vortex generator retrofit on one turbine
  of the first pair (effective 2011-06-20) and a pitch angle adjustment
  simulated on one turbine of the second pair (effective 2011-04-25).
- Role: a real bed with a known injected shift. The pitch pair's upgrade
  is a data modification, so its size is known exactly.
- Licence: CC BY 4.0. Nothing under `data/` is committed.
- Citation, as the Zenodo record gives it: Ding, Y. (2021). Turbine
  Upgrade Dataset [Dataset]. Zenodo.
  https://doi.org/10.5281/zenodo.5516556. The record lists as its
  reference Ding, Y. (2019) Data Science for Wind Energy, Chapman &
  Hall/CRC Press, Boca Raton, FL.
- Fetch: `uv run python examples/studies/shift_intervals/fetch_turbine.py
  --dest data/turbine_upgrade`. One zip, `Turbine_Upgrade_Dataset.zip`,
  4,891,261 bytes, sha256
  `c302fbdc5e68989dc0792d7125277dd76a3ca20f27a96959d1238a294146511e`,
  3 s. The script checks the size and sha256 before keeping the zip,
  extracts the three CSVs into `raw/` and writes `MANIFEST.sha256.json`
  (sha256 `31b2b0c7e33a`) and `FETCH.json`.

### Schema facts that change how the data is read

- Three CSVs: `Turbine Upgrade Dataset(VG Pair).csv` (45,114 rows,
  2010-04-29 to 2011-08-15, 5,000 upgraded rows),
  `Turbine Upgrade Dataset(Pitch Angle Pair).csv` (28,486 rows,
  2010-07-30 to 2011-06-25, 7,000 upgraded rows from 2011-04-25 21:50)
  and `Turbine Upgrade Dataset(Pitch Angle Pair, Table7.3).csv` (the same
  rows with `y_test` at r = 0.02 to 0.09).
- Header spellings differ between files: `upgrade status` and
  `upgrade.status`; `y_test (normalized)`, `y_test(normalized)` and
  `y_test(r=0.05, normalized)`. The two pair files carry an unnamed row
  number column. Time is `m/d/Y H:M` with no time zone.
- The step is 10 min, but rows with V under 3.5 m/s are absent and the
  record has holes: 96.0% (VG) and 96.4% (pitch) of steps are 10 min,
  and the longest holes are 29.9 and 57.0 days.
- The pitch pair's `y_test` is multiplied by 1 + r on upgraded rows with
  V > 9 (V is recorded to 0.01): 2,666 rows. 7 upgraded rows at exactly
  V = 9.00 are unmodified. The pair file's `y_test` equals the r = 0.05
  column, and y_r / (1 + r) agrees across r to within 0.0001.
- Shift intervals: `uv run python
  examples/studies/shift_intervals/run_shift.py --beds turbine`, 3 s.
  Read: [shift_intervals/REPORT.md](../examples/studies/shift_intervals/REPORT.md).
- Switchback schedules: `uv run python
  examples/studies/switchback/run_switchback.py --beds turbine` assigns
  settings A and B to 1 and 3 day calendar blocks of each pair's rows
  before its upgrade. The holes leave some blocks empty. 250 draws per
  pair, 56 s on the first draws and 56 s on the fresh ones.
  Read: [switchback/REPORT.md](../examples/studies/switchback/REPORT.md).

## TEP

Verified on 2026-08-28; testing files verified on 2026-09-28.

- Source: Harvard Dataverse, DOI
  [10.7910/DVN/6C3JR1](https://doi.org/10.7910/DVN/6C3JR1) (Rieth,
  Amsel, Tran, Cook, "Additional Tennessee Eastman Process Simulation
  Data for Anomaly Detection Evaluation").
- Licence: the Dataverse `license` field is **null**; the dataset instead
  carries Harvard Dataverse's *User Agreement, Public Domain Dedication,
  and Disclaimer of Liability* terms-of-use text, which waives copyright
  and related rights worldwide. Attribution is requested, not required.
- Four files, 1,402,950,911 bytes total:

  | file | datafile id | bytes |
  |---|---|---|
  | `TEP_FaultFree_Training.RData` | 3031241 | 24,678,017 |
  | `TEP_FaultFree_Testing.RData` | 3031240 | 47,327,663 |
  | `TEP_Faulty_Training.RData` | 3031242 | 494,063,194 |
  | `TEP_Faulty_Testing.RData` | 3031243 | 836,882,037 |

- Unauthenticated download works via
  `https://dataverse.harvard.edu/api/access/datafile/<id>`; it answers
  `303 See Other` to a signed S3 URL, so a HEAD against it states no
  `Content-Length` and sizes must come from the dataset metadata API.
  **The metadata API answers 403 to a request that sends no
  `User-Agent`**, so the fetcher names itself in every call.
- The API publishes an MD5 per datafile. `scripts/fetch_tep.py` streams
  each file to a `.part`, checks that MD5 before renaming, and records
  both the MD5 and a locally-computed SHA256 in `MANIFEST.sha256.json`;
  `FETCH.json` records the DOI, dataset version, datafile ids, byte total
  and fetch time. A file the API does not checksum is refused rather than
  downloaded unverified.
- The files are R `.RData`, so reading them needs `pyreadr`
  (`uv sync --extra tep`). The library itself never imports it; only
  `scripts/convert_tep.py` does, lazily.

- Fetch: `python scripts/fetch_tep.py --dest data/tep [--all] [--yes]`.
  The default is the two training files (519 MB); `--all` adds the two
  testing files, `--files A,B` names them explicitly.

### Executed fetch

- Fetched 2026-08-28: `TEP_FaultFree_Training.RData` and
  `TEP_Faulty_Training.RData`, **518,741,211 bytes**, 95 s wall, both
  MD5-verified against the API.
  Manifest sha256 `95f369c2b2b8eb6abc34eac3d003d312fcb5d8b1105d7c42876044c69e27cd0f`.
- Fetched 2026-09-28 with `--all`: all four files, **1,402,950,911
  bytes**, 215 s wall, every file MD5-verified against the API.
  Manifest sha256
  `f57ab3ee3443032592c99004172999b21c85090fa13a70aadf065ab56be7da6e`.
  The profile study reads the training pair; the shift interval study
  reads the testing pair.
- Converted subset: `--runs-per-fault 20` over all 21 conditions (fault
  free plus faults 1-20) = **420 runs, 22,260 archives, 210,000 rows**,
  191 MB on disk, 43 s wall. The full training pair would be 556,500
  archives, which is more small files than the exercise justifies; the
  cap is a documented subset, printed by the converter and recorded in
  the study's `run.json`.

### Schema facts that change how the data must be read

- 55 columns: `faultNumber`, `simulationRun`, `sample`, `xmeas_1`
  … `xmeas_41`, `xmv_1` … `xmv_11`. `xmv_12` (the agitator speed, the one
  manipulated variable the base-case controller never moves) is not in
  this distribution. Nothing here is constant by design.
- Sampling is every 180 s. Training runs are 500 samples (25 h) and the
  first faulty sample is 21. Testing runs are 960 samples (48 h) and the
  first faulty sample is 161. Fault 6 zeroes `xmeas_1` from sample 21 in
  all 500 training runs. 500 runs per fault in each split.
- TEP has no clock. A row is numbered, not stamped. tsdive stores UTC
  only, so `scripts/convert_tep.py` synthesises a timestamp
  (`2000-01-01T00:00:00Z + 180 s × (sample - 1)`) and writes
  `timestamp_synthetic: true` into every run's `RUN.json`. Every run
  shares that clock, because each is an independent simulation from its
  own t=0: the instants are an addressing scheme, not a history. Any
  statement about time over TEP archives is a statement about the sample
  grid.
- **TEP has no quality column**, because a simulator has nothing to
  report. Every row is written `SIMULATED`, the tag declares
  `{"SIMULATED": "GOOD"}`, and `quality_assumed` is set so every profile
  says so. A run holding a non-finite value is refused rather than coded
  GOOD; no run in the converted subset holds one.
- **TEP publishes no engineering ranges**, so `eng_range` is `None` and
  no clipping verdict is possible.
- Units come from Downs & Vogel (1993), carried in `scripts/convert_tep.py`
  verbatim. Four spellings do not resolve against the shipped alias table
  and cover 26 of the 52 variables: `kPa gauge` (the gauge-datum refusal
  `docs/SCHEMA.md` documents), `mol%` (a composition basis, not a bare
  percent), `kscmh` (a reference-condition flow, like `Nm3/h`) and `kW`
  (a plain table gap).
- Ground truth is per row: `LABEL_fault` is a `role=MODE` archive. It holds
  `"0"` on every sample before the first faulty sample, and the fault
  number from sample 21 (training) or 161 (testing) to the end of the run.

Findings from the executed profile study are in
[examples/studies/tep_profile/REPORT.md](../examples/studies/tep_profile/REPORT.md).

### Shift intervals on the testing runs

- Cache: `uv run python examples/studies/shift_intervals/build_tep_cache.py`
  reads both testing files through `pyreadr` and keeps runs 1-350 and
  samples 1-320 of fault-free and faults 1-20 as one float32 parquet
  under `data/tep_shift_cache/` (2,352,000 rows, 205 MB, cache sha256
  `ee6e1d65cb2c`, 56 s). Reading `TEP_Faulty_Testing.RData` takes the
  process to a peak working set of 15.4 GiB.
- Onset: fault 6 zeroes `xmeas_1` from sample 161, so samples 1-160 of a
  testing run precede the fault. `scripts/convert_tep.py` labels sample
  161 as the first fault sample.
- The fault-free testing runs start with a lower spread: over runs 1-250
  the log SD ratio of samples 161-320 over 1-160 has a median of 0.038
  over the 52 variables and reaches 0.45.
- Run: `uv run python examples/studies/shift_intervals/run_shift.py
  --beds tep`, 167 s. Per-row results stay in the cache directory.
  Read: [shift_intervals/REPORT.md](../examples/studies/shift_intervals/REPORT.md).

### Switchback schedules on the fault-free testing runs

- Cache: `uv run python examples/studies/switchback/build_tep_cache.py`
  reads `TEP_FaultFree_Testing.RData` only and keeps runs 251-350, all
  960 samples, as one float32 parquet under `data/tep_switchback_cache/`
  (96,000 rows, cache sha256 `49ffa27703fe`, 3 s).
- Run: `uv run python examples/studies/switchback/run_switchback.py
  --beds tep`, 430 s on the first draws and 436 s on the fresh ones, 8
  worker processes. Settings A and B go to 2 h and 4 h blocks, every
  variable is the target in turn, and the adjusted arm regresses on the
  other 51. Per-row results stay in `data/switchback_rows/`.
  Read: [switchback/REPORT.md](../examples/studies/switchback/REPORT.md).

## Verification

Every fetched byte is hashed locally and recorded in a manifest next to
the data. Any number this repo publishes names its split, its manifest,
and the code version that produced it.
