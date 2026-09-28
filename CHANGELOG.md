# Changelog

Releases, newest first. While the version is 0.x a minor release can
change any interface, and the entries say which ones moved.

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
