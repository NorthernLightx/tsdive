# Changelog

Releases, newest first. While the version is 0.x a minor release can
change any interface, and the entries say which ones moved.

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
