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

## TEP

Verified on 2026-08-28.

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
- The two testing files (884 MB) were **not** fetched. Profiling is
  descriptive and scores nothing, so a held-out split buys it nothing;
  `--all` fetches them when a stage-3-and-above study needs them.
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
- Sampling is every 180 s. Training runs are 500 samples (25 h) with the
  fault stepped in at sample 20; testing runs are 960 samples (48 h) with
  the fault stepped in at sample 160. 500 runs per fault in each split.
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
- Ground truth is per row: `LABEL_fault` is a `role=MODE` archive holding
  `"0"` before the onset sample and the fault number after it.

Findings from the executed profile study are in
[examples/studies/tep_profile/REPORT.md](../examples/studies/tep_profile/REPORT.md).

## Verification

Every fetched byte is hashed locally and recorded in a manifest next to
the data. Any number this repo publishes names its split, its manifest,
and the code version that produced it.
