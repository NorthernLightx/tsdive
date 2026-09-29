# Getting started

This tutorial takes you from an empty Python environment to a profile of
your own historian export in about 15 minutes. Every command on this page
ran when the page was built, and the block under it shows what it printed.

## Install

tsdive needs Python 3.12 or newer. Check the version you have:

```console
python --version
```

If it prints 3.11 or older, let [uv](https://docs.astral.sh/uv/) fetch
Python 3.12 into a project folder. It needs no administrator rights and
leaves the system Python alone:

```console
uv venv --python 3.12
.venv\Scripts\activate
uv pip install https://github.com/NorthernLightx/tsdive/releases/download/vX.Y.Z/tsdive-X.Y.Z-py3-none-any.whl
```

On Linux and macOS the second line is `source .venv/bin/activate`. With
Python 3.12 already installed, `pip install` takes the same wheel URL.
With git installed, `pip install "git+https://github.com/NorthernLightx/tsdive.git"`
builds the current main branch instead. The package is not on PyPI.

Check the install:

```tsdive
tsdive --version
```

## Get the demo data

`tsdive demo` writes five small archives into a new folder. They are
synthetic, so every number on this site comes out the same on your
machine:

```tsdive
tsdive demo
```

The two archives under `demo/` hold ten hours of one flow loop, FIC-101
in m3/h, and one temperature, TIC-101 in degC, one sample a minute. The
flow has three faults built in: a 40-minute collection outage, half an
hour pinned at the top of its 0 to 100 m3/h range, and one sample whose
quality code the site never defined. The three archives under
`switchback_demo/` belong to the [switchback trial how-to](guide/howto/switchback-trial.md).

An archive is one parquet file per [tag](reference/glossary.md#tag): the
samples, their raw quality codes, and the tag's metadata. The last line
of the output is the command to try next.

## Profile a window

`tsdive profile` reads one tag over a window and reports what the
historian did to it:

```tsdive
tsdive profile tsdive-demo/demo/fic101_demo.parquet
```

With no `--window`, the window is the archive's own extent, 20:00 to
06:00 UTC. Read the top of the report line by line:

- `demo:FIC101.PV  FIC-101 flow` is the tag's identity, the historian
  (`demo`) and the point (`FIC101.PV`), then its display name.
- `coverage 0.933` is the share of the 10-hour window that is not lost
  to a data-loss [gap](guide/concepts/coverage-and-gaps.md). The
  40-minute outage takes 40 of 600 minutes, so coverage is 1 - 40/600.
- `GOOD 561/562` counts the samples whose [quality code](guide/concepts/quality-and-severity.md)
  reads GOOD. The one other sample carries the undefined code
  `SENSOR DRIFT`, which tsdive counts as UNCERTAIN.
- `censored yes` says some samples sit at the end of the engineering
  range, where the transmitter stops measuring. Those samples say only
  that the flow was at least 100 m3/h. See
  [clipping and censoring](guide/concepts/clipping-and-censoring.md).
- `gaps 1` counts every hole in the timestamps longer than the gap
  threshold, here the outage.
- `window`, `contract` and `units` state what was read: the window in
  UTC, the [sampling contract](guide/concepts/sampling-contract.md) the
  statistics assume, and the unit resolved from the metadata.

The sections under it give the evidence behind each number.
[Reading the profile](guide/output/profile.md) explains every line.

## Screen a later window

`tsdive screen` asks whether a window behaves like a baseline. The
[baseline](reference/glossary.md#baseline) is a stretch of history you
trust. Here it is the five hours from 20:00 to 01:00, and the window is
the five hours after it:

```tsdive
tsdive screen tsdive-demo/demo/fic101_demo.parquet \
    --baseline 2024-03-30T20:00:00Z/2024-03-31T01:00:00Z \
    --window 2024-03-31T01:00:00Z/2024-03-31T06:00:00Z
```

The baseline's median is 62.32 m3/h. Its spread, 1.4826 times the
median absolute deviation, is 0.5488 m3/h, and that is the sigma the
screen works in. A sample more than 3 sigma from the median is flagged,
so the limits are 60.67 and 63.97 m3/h. The 29 flagged samples are the
half hour the transmitter sat at 100 m3/h.
[Baselines and the MAD screen](guide/concepts/baselines.md) explains the
method and the `provisional` caveat.

A baseline has to be clean. Move it over the saturated half hour and
tsdive stops with a typed error instead of computing limits from it:

```tsdive exit=3
tsdive screen tsdive-demo/demo/fic101_demo.parquet \
    --baseline 2024-03-31T00:00:00Z/2024-03-31T03:00:00Z \
    --window 2024-03-31T03:00:00Z/2024-03-31T06:00:00Z
```

This is a [refusal](guide/concepts/refusals.md): the question has no
answer on this data, and the message names the check and the reason. The
exit status is 3. Pick a baseline window without the clipped stretch.
[Choose a clean baseline window](guide/howto/baseline-window.md) shows
how to find one.

## Now your own data

Say your historian exports one tag to CSV, with local timestamps in
day-first order and its own quality words:

```csv file=FI2201.csv
Timestamp,Value,Quality
02/03/2026 08:00,41.30,Good
02/03/2026 08:05,41.17,Good
02/03/2026 08:10,41.52,Good
02/03/2026 08:15,42.32,Good
02/03/2026 08:20,42.06,Good
02/03/2026 08:25,42.22,Good
02/03/2026 08:30,42.80,Good
02/03/2026 08:35,42.28,Questionable
02/03/2026 08:40,42.16,Good
02/03/2026 08:45,42.47,Good
02/03/2026 08:50,41.70,Good
02/03/2026 08:55,41.37,Good
02/03/2026 09:00,41.51,Good
02/03/2026 09:05,40.64,Good
02/03/2026 09:30,39.83,Bad
02/03/2026 09:35,39.30,Good
02/03/2026 09:40,39.36,Good
02/03/2026 09:45,40.01,Good
02/03/2026 09:50,39.74,Good
02/03/2026 09:55,40.04,Good
02/03/2026 10:00,40.88,Good
```

tsdive reads an archive, not a CSV, so the first step is
[`tsdive ingest`](reference/cli/ingest.md). An archive carries the tag's
metadata, and ingest writes a template for it:

```tsdive
tsdive ingest FI2201.csv --init-meta FI2201.json \
    --timestamp-col Timestamp --value-col Value --quality-col Quality
```

```json show=FI2201.json
```

The `_comments` block explains each key and ingest ignores it. The
quality codes come from the file: `Good` and `Bad` are mapped already,
and `Questionable` waits for you to name its severity. Fill in what you
know about the tag and delete what you do not:

```json file=FI2201.json
{
  "identity": {"source_id": "plant1", "point_id": "FI2201.PV"},
  "name": "FI-2201 cooling water flow",
  "unit_raw": "m3/h",
  "eng_range_zero": 0.0,
  "eng_range_span": 80.0,
  "sample_rate_s": 300,
  "role": "PV",
  "quality_codes": {"Good": "GOOD", "Questionable": "UNCERTAIN", "Bad": "BAD"}
}
```

The engineering range turns on the clipping check, and the sample rate
of 300 s tells tsdive that a 5-minute spacing is normal. The timestamps
carry no offset, so `--tz` states the zone they were written in:

```tsdive exit=3
tsdive ingest FI2201.csv --out FI2201.parquet --meta FI2201.json \
    --timestamp-col Timestamp --value-col Value --quality-col Quality \
    --tz Europe/Berlin
```

`02/03/2026` is 2 March in Europe and 3 February in the US, and tsdive
does not pick one. State the order with `--dayfirst`:

```tsdive
tsdive ingest FI2201.csv --out FI2201.parquet --meta FI2201.json \
    --timestamp-col Timestamp --value-col Value --quality-col Quality \
    --tz Europe/Berlin --dayfirst
```

The archive holds UTC timestamps. 08:00 in Berlin in March is 07:00 UTC.
Profile it:

```tsdive
tsdive profile FI2201.parquet
```

Coverage is 0.792, because the 25 minutes between 09:05 and 09:30 local
time hold no sample, where a 5-minute tag should have four. One sample
is UNCERTAIN and one BAD, so 19 of 21 are GOOD, and only those 19 feed
the statistics. `censored no` is now a real answer: the range is
declared and no sample reached 0 or 80 m3/h.

## Next

- [Reading the output](guide/output/profile.md) explains every line of
  profile, screen, spc, mspc, compare and switchback analyze.
- The [how-to guides](guide/howto/historian-export.md) cover real
  exports, baseline choice, many tags at once, Python, AI assistants and
  controller trials.
- [Errors and refusals](reference/errors.md) lists every typed error and
  what to do about it.
