# Changelog

Releases, newest first. While the version is 0.x a minor release can
change any interface, and the entries say which ones moved.

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
