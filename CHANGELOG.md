# Changelog

Releases, newest first. While the version is 0.x a minor release can
change any interface, and the entries say which ones moved.

## 0.10.0 - 2026-09-30

### Added

- `tsdive profile` reports the longest constant run of the window: the
  longest stretch of GOOD samples that hold one value, its first and
  last sample, the time between them and the sample count. The Values
  section prints it as
  `constant run <start> -> <end>   <duration>   n=<samples>`.
  `--json`, `Profile.to_dict()` and the MCP `profile`
  answer carry it as `values.constant_run` with `start`, `end`,
  `duration_s` and `samples`, or `null` with no GOOD sample. A freeze
  that ended before the window end reads `stall 0 s` and shows here.
  The constant run sets no flag, because a tag archived on exception or
  compression settings also holds one value for hours. A SYNTHETIC row
  of `BENCHMARKS.md` plants a 400-sample freeze in 7 days at 300 s, and
  the run `profile` reports equals it.
- The tag table of `tsdive run`, in `ledger.txt` and `report.html`,
  adds `constant`, the longest constant run as its duration and sample
  count, and `errors`, the steps that filed an error row for the tag.
  Each `tags` row of `ledger.json` carries `constant_run_s`,
  `constant_run_samples` and `errors`.
- The MCP `screen` and `spc` tools take `max_runs`, the runs to list per
  rule, 40 when left out. `runs_dropped` counts the runs left out, at
  the top of a `screen` answer and on each rule of an `spc` answer.

### Changed

- Breaking: `WindowStats` in `tsdive.features` takes a required
  `constant_run` field, a `ConstantRun` or `None`. `TAG_COLUMNS` in
  `tsdive.narrate.ledger` adds `constant` and `errors`.

### Fixed

- A single-tag ingest of an export whose tags sample on offset clocks,
  such as tag A at :00, B at :20 and C at :40, raises `SchemaError`
  naming the tag column and `--tag-col`. It wrote one archive of every
  tag's rows before, because no timestamp repeats and none runs
  backwards. The check takes an unread column whose series overlap in
  time, are each in time order, and alternate in more than half of the
  consecutive rows. A shift, batch or row id column does not qualify,
  and neither does a column of floats. `ingest_long` runs the same
  check on each tag's rows.
- The MCP `screen` and `spc` answers list at most 40 runs per rule
  when `max_runs` is left out. They listed every run before. On two
  days at 60 s with a lone spike every fourth sample, the `spc` answer
  holds 17,278 characters instead of 188,754, and the `screen` answer
  9,376 instead of 142,808. On the switchback demo temperature, a 12 h
  baseline and a two-day window, the `spc` answer holds 24,858
  characters instead of 32,884 and lists 40 of its 74 `TREND_6` runs.
  `tsdive screen --json` and `tsdive spc --json` keep every run.

## 0.9.0 - 2026-09-29

### Added

- CI tests Python 3.12 on Linux, Windows and macOS, and Python 3.13 and
  3.14 on Linux. The reference case and the benchmarks run on Python
  3.12 on Linux. The package classifiers list the three versions.
- A CI job installs the lowest version of each direct dependency that
  its range allows and runs the tests. It installs pandas 2.2.2, numpy
  2.0.0, pyarrow 23.0.1 and pint 0.24.4.
- A weekly workflow locks the newest versions the ranges allow, then
  runs the tests, the reference case and the benchmarks. A failed run is
  the report.
- The release workflow installs the built wheel in a clean environment
  and runs `tsdive demo` and `tsdive profile` before it uploads.

### Changed

- tsdive accepts pandas 2.2 to 3.x (`pandas>=2.2,<4`) and pyarrow 23.0.1
  to 25.x (`pyarrow>=23.0.1,<26`). The lock file installs pandas 3.0.6
  and pyarrow 25.0.1. Under pandas 3, `ingest` reads the date order, the
  decimal mark and each tag's value type from text columns as under
  pandas 2.
- pint needs 0.24.4 or newer (`pint>=0.24.4,<1`). pint 0.24 to 0.24.3
  fail to import next to flexparser 0.4.
- Breaking: the `tsfm` and `tep` extras are removed.
  `pip install "tsdive[tsfm]"` and `pip install "tsdive[tep]"` install
  tsdive alone. In a clone, `uv sync --group tsfm` installs
  chronos-forecasting for `examples/studies/3w_chronos`, and
  `uv sync --group tep` installs pyreadr for `scripts/convert_tep.py`.
  The library imports neither.
- Breaking: `Backbone.tag_path`, `severity_floor_note` in
  `tsdive.features.window_features`, and `dim`, `green` and `yellow` in
  `tsdive.ui.term` are removed. Nothing in the package called them.
- The documentation build runs the switchback transcript of
  `docs/SWITCHBACK.md` on the demo data. A test runs each README
  transcript command on the demo data and fails when the output drops a
  line the README shows.

### Fixed

- An archive with timestamps stored in microseconds or milliseconds
  gives the same results as one stored in nanoseconds. `profile
  --flatline` found no reference history on such an archive, `compare`
  read its change intervals 1000 times too short and could report a
  moving tag as `frozen`, and `report-html` drew its samples outside the
  plot. Under pandas 3, `ingest` writes microsecond timestamps.
- With no reference history, the flatline `time_since_last_actual_change`
  finding of a window whose value changed reads `stall 120s; no reference
  history provided` instead of `no GOOD samples; signal not evaluable; no
  reference history provided`. A window with one GOOD sample reads `fewer
  than 2 GOOD samples; signal not evaluable`.

## 0.8.0 - 2026-09-29

### Added

- Every JSON document opens with `result_kind` and `tsdive_version`:
  the `--json` output of each command that takes the flag, the refusal
  object, each MCP answer and `ledger.json`. `result_kind` is `evidence` for an analysis,
  `refusal`, `ingest`, `plan` for `switchback plan --json`, or `ledger`.
  `to_dict()` returns the document without the two keys.
- `tsdive ingest --json` prints one object: `form` (`single`, `wide` or
  `long`) and `archives`, one entry per archive written with `path`,
  `tag`, `identity`, `rows`, `first`, `last`, `quality_source` (`column`
  or `assumed`) and `assumed_quality`. Under `--init-meta` the object
  lists `templates`. A refused ingest prints the refusal object and exits
  3.
- `tsdive run --strict` exits 2 when a step filed an error row, else 3
  when a step filed a refusal row, else 0.
- `ledger.json` carries `tags`, one row per archive read off its
  profile: `coverage`, `good_share`, `censored`, `gaps`,
  `longest_gap_s`, `flatline` and the steps `refused` for it. Each
  finding carries `data`, the document the step prints under `--json`.
  `ledger.txt` and `report.html` open with the same table, and
  `ledger.txt` holds every profile.
- `ScreenAnalysis.to_dict()` and `SpcAnalysis.to_dict()` carry `runs`:
  each stretch of consecutive flagged GOOD samples as `start`, `end` and
  `n`, with its `rule` in `spc`.
- The MCP `screen` and `spc` tools take `max_events`, the flagged
  timestamps or hits per rule to list, 0 when left out. `flagged_dropped`
  and `hits_dropped` count the entries left out, and `n_flagged`,
  `n_hits` and each rule's `n` count every sample. On the switchback demo
  temperature, a 12 h baseline and a two-day window, the `spc` answer
  holds 32,828 characters instead of 223,070 and the `screen` answer
  8,094 instead of 53,974.
- `make api-diff` lists the public API breaks since the last `v*` tag
  with griffe.

### Changed

- `ledger.json` holds each refusal as an object (`step`, `tags`,
  `error_type`, `cause`) instead of a `[ErrorName] step tags: message`
  line. A step that raises `ValueError` or `OSError` (a rejected option,
  overlapping windows, a file the OS cannot read) files a row of the
  `errors` list instead of a refusal. `tsdive run` prints the row as
  `ERROR` and adds `errors N` to its headline. The default exit rule of
  `tsdive run` stays as it was.
- `tsdive screen` and `tsdive spc` print consecutive flagged samples and
  rule hits as one line per run, with its span and sample count. Each
  section lists five runs and counts the rest as `(+N more runs)`. A
  lone flag or hit prints as before. `RUNS_SHOWN` replaces
  `FLAGGED_SHOWN` in `tsdive.analyses_render`.
- Breaking: the optional fields of these public dataclasses are
  keyword-only, so a positional call raises `TypeError`: `TagMeta` from
  `unit_raw` on, `SamplingContract` (`aggregate_type`, `stepped`),
  `Profile` (`flatline`), `ScreenAnalysis` (`mode_path`, `provisional`,
  `regimes`, `alignment`), `CompareAnalysis` (`top`),
  `SwitchbackAnalysis` (`assumptions`, `covariate_checks`),
  `SwitchbackEstimate` (`estimate`, `p_value`, `lo`, `hi`, `reason`,
  `detail`) and `SwitchbackPlan` (`power`). Pass each by name. A field
  added to one of them no longer moves another.

### Fixed

- A path the OS cannot read, such as a directory passed as an archive,
  exits 2 with one `error:` line that names the path and carries the OS
  message, instead of a traceback and exit 1. The MCP server returns the
  same message as a tool error instead of `Error executing tool`.
  `report-html` names the error class of each unreadable archive instead
  of `FileNotFoundError` for every one.
- `mspc` and the joint table of `compare` do not assess SPE for a model
  that keeps as many components as tags. Its residual is zero up to
  rounding, and its empirical limit flagged rounding error: four
  independent tags reported 3 SPE breaches in 200 rows. The report
  prints `SPE  NOT ASSESSED` with the reason, and `mspc` adds the
  `--variance` value that keeps one component fewer. The JSON carries
  `spe_breaches: null` and `spe_not_assessed`, the SPE limit is null,
  and `mspc` ranks no contributors for such a model.

## 0.7.0 - 2026-09-29

### Added

- `tsdive ingest --tag-col COL` reads a long export, one row per tag and
  timestamp with the tag in `COL`, and writes one archive per tag into
  `--out`. Metadata comes from `--meta-dir`, `--init-meta DIR` writes one
  template per tag, `--tags` picks a subset, and every check runs before
  the first archive is written. `tsdive.ingest_long` and
  `tsdive.init_long_meta` are the Python forms.
- `tsdive ingest --sep`, `--decimal` and `--encoding`, and the `sep`,
  `decimal` and `encoding` keywords of every ingest and template
  function, read a CSV with another column separator, decimal mark or
  text encoding, such as the `;`-separated, comma-decimal cp1252 file of
  a German Excel. Given for a parquet file, they are a usage error (exit
  2).
- `ZeroSpreadBaseline`, a refusal for a baseline whose GOOD values do not
  spread. The message names the tag, the baseline window, the count of
  distinct values and the most common value with its share.
- A metadata template lists each string of a value column that holds
  numbers and strings, such as a PI digital state, under
  `quality_codes` with no severity.

### Changed

- The sdist lists the paths it holds: source, tests, docs, examples,
  scripts and the top-level project files. A local build leaves out
  files that one clone excludes from git in `.git/info/exclude`.
- Ingest parses a timestamp column in one pass. It parses one row at a
  time only a column whose rows carry different UTC offsets, and the rows
  the one pass cannot read. A 129,600-row export of one-minute samples
  ingests in 1.9 s instead of 99.6 s, and every archive the test suite
  writes is byte-identical.
- The `SchemaError` for strings in the value column lists every string
  with its row count, and its example maps each of them.
- `mad_baseline`, `moving_range_baseline` and `regime_baselines` take
  `label`, the tag and window their refusal names.
- `mspc` and the joint table of `compare` divide each tag by its
  baseline standard deviation before the PCA, so a tag's unit no longer
  sets its weight. On the demo `mspc` reports 70 T2 breaches of 121
  rows instead of 22 (SPE stays at 108), explained variance 0.9767
  instead of 0.9839, and limits T2 3.759 and SPE 0.2859 instead of 4.052
  and 0.03662. `compare` keeps 0.5129 of the after variance instead of
  0.2643, and its two SPE contributors carry 50% each instead of 79% and
  21%. `PcaModel` carries the `scale` it applies.

### Fixed

- A single-tag ingest of an export that holds several tags raises
  `SchemaError` (exit 3) naming the tag column and `--tag-col`, instead
  of writing one archive that mixes every tag. The refusal applies when
  timestamps repeat or run backwards and a column the ingest leaves
  behind splits the rows into overlapping series, each in time order.
- Ingest raises `NonMonotonicIndex` (exit 3) for rows whose timestamps
  run backwards, instead of writing an archive every read refuses.
- Ingest places the hour the clocks repeat in autumn by row order, first
  pass at the summer offset, instead of refusing every export in local
  time that crosses the change. A time in that hour the export holds
  once, rows of it out of time order, and a spring time the clocks skip
  raise `SchemaError` with a message for each case.
- `screen`, `spc` and `screen --mode` raise `ZeroSpreadBaseline` (exit
  3) for a baseline whose MAD or moving-range scale is 0, instead of
  limits of zero width that flag every sample off the center. `mspc`
  raises it for a tag whose baseline standard deviation is 0, and
  `individuals_limits` for a sigma of 0. In `compare`, the tag table
  prints `no spread before` as the flagged reason, and the joint table
  is refused with the error as its reason.

## 0.6.0 - 2026-09-29

### Added

- `tsdive demo [DIR]` and `tsdive.write_demo_data` write the demo
  archives, two tags for the walkthrough and three for the switchback
  trial, into a directory, `tsdive-demo/` by default. An archive that
  exists already is refused, and nothing is written.
- The documentation site gains a getting-started tutorial, eleven
  concept pages, the annotated output of profile, screen, spc, mspc,
  compare and switchback analyze, six how-to guides, an errors and
  refusals reference, a glossary and a home page that starts from the
  reader's question. Every command on those pages runs on the demo data
  when the site is built, and every Python snippet runs as a test.

### Changed

- `scripts/make_demo_archive.py` and `examples/switchback/make_trial.py`
  call the builders in `tsdive.demo` and write the same bytes as before.
- The README installs with `pip install` and writes the demo data with
  `tsdive demo data`, so the transcripts need no clone.

### Fixed

- The `SchemaError` for a date that reads day first and month first
  suggests a strptime format with the shape of the value it quotes, such
  as `%d/%m/%Y %H:%M` for `01/02/2026 00:00`, instead of always
  `%d/%m/%Y %H:%M:%S`.
- `tsdive-mcp` without the `mcp` extra names the git and release-wheel
  installs with the extra, instead of a command that works only in a
  clone.

## 0.5.0 - 2026-09-28

### Added

- `tsdive ingest --dayfirst` and `--timestamp-format`, and the `dayfirst`
  and `timestamp_format` keywords of `ingest` and `ingest_wide`, state
  the date order of an export.
- `tsdive ingest <export> --init-meta META.json` and
  `tsdive.init_tag_meta` write a metadata template for a single-tag
  export.
- `retrieval_mode` in tag metadata, `RECORDED` or `INTERPOLATED`, is
  stated in the sampling contract of every read of the archive.
- `Profile.to_dict()` returns the document `tsdive profile --json`
  prints.
- `switchback analyze` tests each covariate's own difference between the
  settings and flags one the setting moves; `--json` lists the tests
  under `covariate_checks`.
- `compare` prints each refused pair or joint table under the headline,
  and `--json` maps them to their reasons under `refused_tables`.

### Changed

- A refusal (a typed `TSDiveError`) exits with status 3 instead of 2.
  Usage errors and invalid input still exit 2, and `tsdive run` keeps
  0 and 2.
- Under `--json`, a refusal also prints the object the MCP server
  returns on stdout: `result_kind`, `error_type` and `cause`.
- `switchback analyze` exits with status 3 when its difference in means
  is refused.
- A metadata key tsdive does not define, one end of the engineering
  range without the other, or a span of 0 or less raises `SchemaError`
  instead of being dropped.
- A date that reads both day first and month first raises `SchemaError`
  at ingest unless the order is stated.
- Errors raised from Python name keywords such as `rate_s=` where the
  CLI names flags such as `--rate-s`.
- The profile, segment, screen, spc and switchback `--json` documents
  carry `range_known` beside `censored`.

### Fixed

- Reports print `censored unknown`, and `--json` prints `censored: null`,
  when no engineering range is declared and nothing is flagged, instead
  of `censored no`.
- A profile of an empty window prints `clipped null (no samples)` when
  the range is declared, instead of `eng range unknown`.
- A value-column string the tag's `quality_codes` names, such as PI's
  `I/O Timeout`, is nulled at its declared severity instead of raising
  `SchemaError`.
- A file that is not a tsdive archive raises `SchemaError` naming
  `tsdive ingest` instead of a pyarrow error.
- `ScheduleMismatch` on an edited plan says to restore the plan file
  with the recorded digest instead of planning the trial again.
- A refused power readout prints the history window that was passed and
  the span the schedule needs, instead of a span past the window.
- The `switchback plan --window` and `--history-window` help states that
  bounds with a UTC offset are converted.

## 0.4.0 - 2026-09-28

### Added

- `tsdive.switchback_plan` and `tsdive.switchback_analyze`: a balanced
  random schedule of settings A and B over one window, written as a plan
  file with a SHA-256 digest, and the difference between the settings on
  a target under that plan by randomization inference, with an adjusted
  estimate on declared covariates. `SwitchbackPlan`, `SwitchbackAnalysis`,
  `SwitchbackEstimate`, `DesignTooSmall` and `ScheduleMismatch` are
  exported beside them.
- `tsdive switchback plan` and `tsdive switchback analyze`, with
  `--history` and `--history-window` for a power readout, repeatable
  `--covariate`, and `--json`. A `tsdive run` plan can list `switchback`
  as a step.
- `tsdive-mcp` serves `switchback_analyze` as a sixth tool.
- `docs/SWITCHBACK.md` and `examples/switchback/make_trial.py`, a
  synthetic trial for its transcripts.
- Two benchmark rows: the switchback claim rate at a zero shift and its
  detection of a 0.5 sigma shift on seeded AR(1) records.

### Changed

- `docs/SCOPE.md` states no causal claims from observational data, with
  `switchback analyze` as the one exception, instead of no causal
  inference at all.

### Fixed

- `compare` pair intervals resample both periods with 1000 replicates
  instead of the after period with 200, so they are wider and fewer pairs
  clear.
- `scripts/convert_tep.py` labels the first faulty sample 21 (training)
  and 161 (testing) instead of 20 and 160.

## 0.3.0 - 2026-09-03

### Added

- `tsdive.eval`: the evaluation protocol as an API. `ranking_metrics`
  returns ROC-AUC, PR-AUC and precision at a recall floor and reports a
  one-class fold as a refusal; `clock_control` scores a window by its
  position in its record; `worst_baseline_threshold`, `fires`, `far_floor`
  and `over_floor` carry the alarm rule and its false-alarm floor;
  `group_holdout`, `GroupSplit` and `GroupLeakage` are re-exported.
- `tsdive.eval`: `conformal_p_values`, `power_martingale`,
  `mixture_martingale` and `martingale_alarm` build a conformal test
  martingale alarm whose false-alarm probability over a whole record is
  bounded by `delta` when the calibration and stream scores are
  exchangeable.

### Changed

- `tsdive.analyses_render.render_lines` replaces the private line splitter
  the plan runner imported from the CLI module.

## 0.2.0 - 2026-09-02

### Added

- `tsdive.segment`, `tsdive.screen`, `tsdive.spc`, `tsdive.mspc` and
  `tsdive.compare` return result objects with `render()`, `to_dict()` and
  `frame`. The CLI prints the same text and JSON.
- `tsdive ingest --wide` reads an export with one column per tag into one
  archive per tag. `--quality-suffix` names the per-tag quality columns,
  `--tags` picks columns, and `--init-meta DIR --source-id ID` writes one
  metadata template per column.
- Window arguments accept `START/DURATION`, `DURATION/END` and a bare
  date standing for one UTC day, alongside `START/END`.
- `report-html` and `run` draw one SVG plot per tag: values, samples
  below GOOD, gaps and the engineering range. In `run` the plot also
  carries the screen limits, the flagged samples and the SPC rule hits.
- `tsdive segment --mode-out FILE` writes the segments as a MODE archive
  that `screen --mode FILE` reads as regimes.
- A plan can set `before` and `after` and list `compare` as a step.
- `tsdive-mcp`, a read-only MCP server over `profile`, `segment`,
  `screen`, `spc` and `compare`, installed with the `mcp` extra.

### Changed

- `--json` and `--no-color` are accepted before the command name as well
  as after it.

### Fixed

- The refusal messages of regime baselines, unit resolution and
  time-weighted averaging name the defect they found.

## 0.1.0 - 2026-09-01

First release: parquet archives carrying `tsdive.meta`, the commands
`ingest`, `profile`, `segment`, `screen`, `spc`, `mspc`, `compare`, `run`
and `report-html`, the 3W and TEP converters, the benchmark rows and the
3W study reports.
