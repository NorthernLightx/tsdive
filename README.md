# tsdive

[![ci](https://github.com/NorthernLightx/tsdive/actions/workflows/ci.yml/badge.svg)](https://github.com/NorthernLightx/tsdive/actions/workflows/ci.yml)

tsdive is a Python library and command-line tool that checks process
time series (historian and sensor archives) for gaps, bad quality codes,
clipped ranges, timestamp faults and unit mismatches, then runs baseline,
control-chart and PCA monitoring on the same window. It reads single-tag
parquet archives, built from CSV or parquet exports with `ingest` or
through a source adapter ([docs/SOURCES.md](docs/SOURCES.md)), and writes
text, JSON, Python objects or a static HTML page. A check with no answer
raises a typed error naming the check.

## Install

tsdive needs Python 3.12 or newer. To try it on the demo data, clone the
repository:

```console
git clone https://github.com/NorthernLightx/tsdive.git && cd tsdive
pip install -e .
python scripts/make_demo_archive.py
```

To use it on your own archives without a clone:
`pip install "git+https://github.com/NorthernLightx/tsdive.git"`. Built
wheels are on the
[releases page](https://github.com/NorthernLightx/tsdive/releases). Add
`[ml]` to the install for the IsolationForest detector (scikit-learn) or
`[mcp]` for the MCP server. The package is not on PyPI.

The library and CLI make no network calls. The dataset fetch scripts
under `scripts/` download on request and print their size first.

## Use

Commands in the order you run them on an archive. A window is `START/END`,
`START/PT5H`, `PT5H/END`, or a date for one UTC day. Every analysis
command takes `--json` after the command name and prints one JSON object
instead of text. `--no-color` and the `NO_COLOR` environment variable turn
colour off. The transcripts below are trimmed to the lines the text reads.

### profile

Data checks and statistics for one window. The headline carries the
numbers you read first, then one section per group of findings.

```console
$ tsdive profile data/demo/fic101_demo.parquet \
    --window "2024-03-30T20:00:00Z/2024-03-31T06:00:00Z" \
    --tz Europe/London --flatline
demo:FIC101.PV  FIC-101 flow
coverage 0.933   GOOD 561/562   censored yes   gaps 1   flatline NOT ASSESSED

window    2024-03-30 20:00:00Z -> 2024-03-31 06:00:00Z  (10 h)
contract  TIME_WEIGHTED  RECORDED  NONE  stepped no  digest d59d433d9c62
units     m3/h -> cubic meters per hour

Coverage
  coverage 0.933   valid 0.998   gaps 1   data-loss gaps 1   longest 40 min
  2024-03-30 23:00:00Z -> 23:40:00Z   40 min   unknown (no rule matched)

Quality
  GOOD 561   UNCERTAIN 1   BAD 0
  unmapped codes, treated UNCERTAIN: SENSOR DRIFT

Range
  clipped 0.0516   censored yes

Values  GOOD n=561
  min 60.86   p05 61.20   median 62.32   p95 100.0   max 100.0
  mean 63.95 (time-weighted)   std 8.399   mad 0.4017

Flatline
  NOT ASSESSED (window censored; saturated != frozen)
```

Omit `--window` to profile the whole archive. Statistics cover GOOD rows
only. The transmitter sat at full scale for half an hour.

### segment

Finds where the tag's level changed, so you can pick a baseline window.

```console
$ tsdive segment data/demo/fic101_demo.parquet
demo:FIC101.PV  6 segments   5 breakpoints   censored yes

window    2024-03-30 20:00:00Z -> 2024-03-31 06:00:00Z   usable 561
method    PELT L2 on MAD-scaled values   penalty 18.99 (3*log n)   min-size 10

  #  start                 end                     n   median     mad
  3  2024-03-31 00:10:00Z  2024-03-31 02:00:00Z  111    62.56  0.1826
  4  2024-03-31 02:01:00Z  2024-03-31 02:29:00Z   29    100.0       0
  5  2024-03-31 02:30:00Z  2024-03-31 03:56:00Z   86    61.37  0.1696
```

Segment 4 is the half hour at full scale. A larger `--penalty` gives
fewer segments.

### screen

Flags samples outside a baseline built from another window.

```console
$ tsdive screen data/demo/fic101_demo.parquet \
    --baseline "2024-03-30T20:00:00Z/2024-03-31T01:00:00Z" \
    --window "2024-03-31T01:00:00Z/2024-03-31T06:00:00Z"
demo:FIC101.PV  flagged 29 of 300 (9.7%)

baseline  2024-03-30 20:00:00Z -> 2024-03-31 01:00:00Z   GOOD 261   censored no
window    2024-03-31 01:00:00Z -> 06:00:00Z
method    MAD   center 62.32   scale 0.5488   k 3.0   limits [60.67, 63.97]

Flagged
  2024-03-31 02:01:00Z
  2024-03-31 02:02:00Z
  2024-03-31 02:03:00Z
  (+26 more)
```

`--mode <parquet>` computes one baseline per regime of a MODE tag, such
as the archive `segment --mode-out` writes. `--method moving-range`
replaces the MAD scale. A censored baseline, or one that overlaps the
window, raises an error in `screen`, `spc` and `mspc`.

### spc

Individuals control chart with three Nelson rules on the same baseline
as `screen`. A rule with no hits prints zero.

```console
$ tsdive spc data/demo/fic101_demo.parquet \
    --baseline "2024-03-30T20:00:00Z/2024-03-31T01:00:00Z" \
    --window "2024-03-31T01:00:00Z/2024-03-31T06:00:00Z"
demo:FIC101.PV  37 rule hits in 300 samples

baseline  2024-03-30 20:00:00Z -> 2024-03-31 01:00:00Z   GOOD 261   censored no
window    2024-03-31 01:00:00Z -> 06:00:00Z
limits    center 62.32   sigma 0.5488   lcl 60.67   ucl 63.97
basis     individuals 3-sigma

BEYOND_3SIGMA  29
  2024-03-31 02:01:00Z   value 100 outside [60.67, 63.97]

RUN_9_SAMESIDE  6
  2024-03-31 01:08:00Z   9 consecutive points on one side of center 62.32

TREND_6  2
  2024-03-31 01:25:00Z   6 consecutively increasing/decreasing points
```

### mspc

PCA T2 and SPE over two or more archives on one grid.

```console
$ tsdive mspc \
    data/demo/fic101_demo.parquet data/demo/tic101_demo.parquet \
    --baseline "2024-03-30T20:00:00Z/2024-03-30T23:00:00Z" \
    --window "2024-03-31T04:00:00Z/2024-03-31T06:00:00Z"
demo:FIC101.PV, demo:TIC101.PV  T2 breaches 22   SPE breaches 108   of 121 rows

baseline  2024-03-30 20:00:00Z -> 23:00:00Z   rows 181   coverage 1.000
window    2024-03-31 04:00:00Z -> 06:00:00Z   rows 121   coverage 1.000
model     rate 60 s (declared)   components 1 of 2   explained 0.9839
limits    T2 4.052   SPE 0.03662   (empirical q0.99)
contributors  not ranked (2 tags; top-3 would list every one)

T2 breaches  22
  2024-03-31 04:10:00Z

SPE breaches  108
  2024-03-31 04:00:00Z
```

The demo temperature stops tracking the flow at 04:00Z. SPE carries the
residual.

### compare

Reports what changed between two periods, over every tag of a unit at
once. The tables say that a tag changed and that a pair decoupled; they
do not say why.

```console
$ tsdive compare \
    data/demo/fic101_demo.parquet data/demo/tic101_demo.parquet \
    --before "2024-03-30T20:00:00Z/2024-03-30T23:00:00Z" \
    --after "2024-03-31T04:00:00Z/2024-03-31T06:00:00Z"
demo  2 tags   refused 0   pairs 1 of 1 clearing

before    2024-03-30 20:00:00Z -> 23:00:00Z   (3 h)
after     2024-03-31 04:00:00Z -> 06:00:00Z   (2 h)
grid      rate 60 s (declared)   coverage 1.000 -> 1.000

Tags that changed
  tag        quality   sigma  spread  flagged  changed at
  TIC101.PV  ok         +0.8    x6.0      62%  2024-03-31 04:18:00Z
  FIC101.PV  ok         +0.4    x0.5       0%  2024-03-31 04:00:00Z

Pairs that decoupled  (Pearson on first differences, 95% block bootstrap)
  pair                   before   after   delta  interval
  FIC101.PV ~ TIC101.PV    0.69    0.17   -0.53  [-0.74, -0.35]

Joint structure  (PCA fitted on before, 2 of 2 tags aligned)
  components 1 of 2   explained 0.9839 -> 0.2643 on after
  rows 121   T2 breaches 22   SPE breaches 108
  SPE contributors   TIC101.PV 79%   FIC101.PV 21%
```

The temperature's spread is 6 times the before period's, 62% of its
samples fall outside its own before-period baseline, and its coupling to
the flow drops from 0.69 to 0.17 with an interval that excludes 0. The
PCA model fitted on the before period keeps 26% of the after period's
variance, and 79% of the residual sits on the temperature. `--top N`
sets how many rows each table prints; `--json` carries every column and
every pair.

### run

Runs one TOML plan over many archives and writes `ledger.json`,
`ledger.txt` and `report.html` (tables and one plot per tag) to
`-o DIR` (default `tsdive-run/` beside the plan). Options are the
step's own flags: a list repeats the flag, `true` is the bare flag.
Globs resolve against the plan file. `examples/plans/demo.toml`:

```toml
archives = ["../../data/demo/*.parquet"]
window   = "2024-03-31T01:00:00Z/2024-03-31T06:00:00Z"
baseline = "2024-03-30T20:00:00Z/2024-03-31T01:00:00Z"
before   = "2024-03-30T20:00:00Z/2024-03-30T23:00:00Z"
after    = "2024-03-31T04:00:00Z/2024-03-31T06:00:00Z"
steps    = ["profile", "segment", "screen", "spc", "mspc", "compare"]

[options.profile]
flatline = true
tz = ["Europe/London"]
```

```console
$ tsdive run examples/plans/demo.toml
run demo.toml   2 archives   6 steps   profiles 2   findings 7   refusals 1

REFUSAL  mspc   demo:FIC101.PV, demo:TIC101.PV
  [MspcAlignmentError] aligned coverage 0.867 below required 0.95; refusing to
  interpolate across tags

wrote     examples/plans/tsdive-run/ledger.json
          examples/plans/tsdive-run/ledger.txt
          examples/plans/tsdive-run/report.html
```

Here the baseline holds the 40-minute outage, so the aligned grid covers
0.867 and `mspc` raises `MspcAlignmentError`. `compare` reads `before` and
`after`. `ledger.txt` opens with this text, then every finding in full.
`tsdive report-html <parquet...> --window START/END -o report.html` writes
the same HTML page for bare archives without a plan.

### ingest

A wide export, one column per tag, takes two commands:

```console
tsdive ingest export.csv --wide --timestamp-col ts --quality-suffix _q \
    --init-meta meta/ --source-id plant1
tsdive ingest export.csv --wide --timestamp-col ts --quality-suffix _q \
    --out archive/plant1/ --meta-dir meta/ --tz Europe/London
```

The first writes one template per tag with the optional keys of
[docs/SCHEMA.md](docs/SCHEMA.md) left null. The second writes one
archive per tag, reading quality from `<tag>_q`. A filled template:

```json
{
  "identity": {"source_id": "plant1", "point_id": "FIC101.PV"},
  "name": "FIC-101 flow",
  "unit_raw": "m3/h",
  "eng_range_zero": 0.0,
  "eng_range_span": 100.0,
  "sample_rate_s": 3.0,
  "role": "PV"
}
```

A single-tag export uses `--out FILE --meta FILE` (see `tsdive ingest --help`).

### Python

```python
import tsdive

p = tsdive.profile("data/demo/fic101_demo.parquet",
                   "2024-03-30T20:00:00Z/2024-03-31T06:00:00Z")
p.physics.coverage.coverage      # 0.933
p.render()                       # the plain text the CLI prints, no colour
s = tsdive.screen("data/demo/fic101_demo.parquet",
                  "2024-03-30T20:00:00Z/2024-03-31T01:00:00Z",   # baseline
                  "2024-03-31T01:00:00Z/2024-03-31T06:00:00Z")   # window
s.to_dict()["n_flagged"]         # 29; s.render() is the text tsdive screen prints
```

### MCP

`tsdive-mcp` serves `profile`, `segment`, `screen`, `spc` and `compare`
to an MCP client over stdio, with the same arguments and fields as
`--json`. It needs the `[mcp]` extra. Tools, arguments and result shapes
are in [docs/MCP.md](docs/MCP.md).

## What it checks

Every read runs these checks.

| Check | Reported as | Error |
|---|---|---|
| tag identity is `(source_id, point_id)`, not the display name | the headline: identity, then name | `SchemaError`: no `tsdive.meta` or a missing key |
| timestamps are time-zone aware, stored in UTC | `Timestamps`: `audited`, `duplicates`, `non-monotonic` | `SchemaError`: naive timestamps at ingest; `NonMonotonicIndex`: index runs backwards |
| DST transition inside the window | `DST (<zone>)`, with `--tz` | none |
| vendor quality codes kept verbatim, severity derived | `Quality`: the severity counts and `unmapped codes` | `SchemaError`: no quality column and no `--assume-quality` |
| digital state codes are states, not values | `value` nulled, code kept in `quality` | `SchemaError`: string value on a tag that is not `role: MODE` |
| data loss separated from compression silence | `Coverage`: `coverage`, `gaps`, `longest`, per-gap class and rule | none; an unmatched gap is `unknown` |
| values pinned at the engineering range | `Range`: `clipped`, `censored` | `InsufficientQuality`: censored window used as a baseline |
| unit spelling resolves through one alias table | `units <raw> -> <canonical>` or `unresolved (null)` | `UnresolvedUnitError`; `IncomparableUnitsError`: reference-condition unit with no declared state |
| statistics use GOOD rows only | `Values  GOOD n=...` | `InsufficientQuality`: no GOOD sample |

## How it compares

General anomaly-detection libraries score the values they are given.
tsdive checks what the archive did to those values first.

| Tool | What it does | What tsdive does differently |
|---|---|---|
| PyOD | outlier detectors over numeric feature arrays | reads timestamps and quality codes before any detector runs, and raises `SchemaError` on an archive with no metadata, naive timestamps or no quality column |
| ADTK | rule-based and unsupervised anomaly detection on pandas series | reports coverage, gap classes and clipping for the window, and raises `InsufficientQuality` instead of scoring a censored one |
| Darts | forecasting models with anomaly scores from forecast residuals | takes baselines from validated reference windows and raises `ValueError` when a baseline overlaps the window it screens |
| aeon | machine-learning toolkit for time-series tasks | states the sampling contract of every window and raises `IncomparableSamplingError` across contracts |
| Seeq, TrendMiner | commercial analytics servers connected to a live historian | runs offline on exported archives and replays identically under a recorded provenance fingerprint |

## Results on real data

The checks ran over the 3W dataset (Petrobras, 1,119 offshore well
instances, 14,347 tags) with zero refusals. The study reports:
[data-physics profile](examples/studies/3w_profile/REPORT.md),
[detector evaluation](examples/studies/3w_detectors/REPORT.md).
Protocol in [docs/EVAL.md](docs/EVAL.md), reproduction in [docs/DATA.md](docs/DATA.md).

## Limits

No fault diagnosis and no causal inference. No writes to any process or
historian. No alarm limits, notifications or real-time path. Deferred
work is in [docs/SCOPE.md](docs/SCOPE.md) and
[docs/ROADMAP.md](docs/ROADMAP.md).

## Development

```console
uv sync          # editable install plus the dev group
make test        # uv run pytest, offline
make lint        # uv run ruff check .
make reference   # refusal cases, byte for byte
make bench       # BENCHMARKS.md, byte for byte
make check       # all of the above
```

## License

Apache-2.0. The runtime dependencies (numpy, pandas, pyarrow, pint) carry
BSD or Apache licences; the `tep` extra's pyreadr is AGPL-3.0.
