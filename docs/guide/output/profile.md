# Reading the profile

`tsdive profile` reports what the historian did to one tag over one
window, before any statistic is trusted. This is the full report on the
demo flow, then every line in plant terms.

```tsdive
tsdive profile data/demo/fic101_demo.parquet \
    --window 2024-03-30T20:00:00Z/2024-03-31T06:00:00Z \
    --tz Europe/London --flatline
```

## Headline

- `demo:FIC101.PV  FIC-101 flow`: the tag's
  [identity](../../reference/glossary.md#identity), historian and point,
  then its display name.
- `coverage 0.933`: the share of the window not lost to a data-loss gap.
- `GOOD 561/562`: samples whose quality reads GOOD, out of all samples.
- `censored yes`: at least one sample sits at an end of the engineering
  range. `no` means none does; `unknown` means no range is declared.
- `gaps 1`: gaps of any class.
- `flatline NOT ASSESSED`: the frozen-sensor check did not run; the
  Flatline section says why. Without `--flatline` the field is absent.

Read the headline first. When it reads well (coverage near 1, nearly
every sample GOOD, `censored no`, no gaps, `flatline none`), the
statistics below describe the process and not the historian.
`flatline SUSPECTED` means one of the flatline signals fired.

## What was read

- `window`: the span read, in UTC, and its length.
- `contract`: the [sampling contract](../concepts/sampling-contract.md),
  four fields and a digest of them.
- `units    m3/h -> cubic meters per hour`: `unit_raw` from the metadata
  and what it resolved to. `unresolved (null)` means the spelling is not
  in the alias table.

## Coverage

- `coverage 0.933   valid 0.998`: time not lost to gaps, and rows that
  are GOOD and a number. See
  [coverage and gap classes](../concepts/coverage-and-gaps.md).
- `gaps 1   data-loss gaps 1   longest 40 min`: every gap, the data-loss
  ones, and the longest.
- One line per gap: start, end, length, class and rule.
  `unknown (no rule matched)` is a data-loss class.
- `edge slack`, when present: time at the window's edges shorter than
  the gap threshold, counted as covered.

## Quality

- `GOOD 561   UNCERTAIN 1   BAD 0`: samples per
  [severity](../../reference/glossary.md#severity).
- `unmapped codes, treated UNCERTAIN: SENSOR DRIFT`: codes no rule
  knows. Declare each in the tag's `quality_codes`. See
  [quality codes and severity](../concepts/quality-and-severity.md).

## Range

- `clipped 0.0516`: the share of samples at an end of the engineering
  range, 29 of 562.
- `censored yes`: the verdict the headline repeats. See
  [clipping and censoring](../concepts/clipping-and-censoring.md).

## Timestamps

- `audited 562   duplicates 0   non-monotonic 0`: samples checked,
  repeated timestamps, and timestamps that go backwards. A backwards
  timestamp stops the read with `NonMonotonicIndex` instead of reaching
  this line.
- `DST (Europe/London) 2024-03-31 01:00:00Z  +0h -> +1h`: a clock change
  inside the window in a zone `--tz` named. The archive is in UTC, so no
  sample moved; a local shift report has an hour less that night.

## Values

Statistics over the GOOD samples only, `n=561` here.

- `min 60.86   p05 61.20   median 62.32   p95 100.0   max 100.0`: the
  range and the 5th, 50th and 95th percentiles in m3/h. A p95 at the
  range end says at least 5% of the samples are clipped.
- `mean 63.95 (time-weighted)`: each sample counts for the time until the
  next one.
- `std 8.399   mad 0.4017`: standard deviation and median absolute
  deviation. The half hour at 100 m3/h inflates the standard deviation
  to 8.4 m3/h, while 1.4826 times the MAD is 0.60 m3/h.
- `distinct 533`: how many different values the GOOD samples hold. A
  handful on a measurement points at heavy rounding or a frozen value.
- `stall 0 s`: time since the value last changed, at the end of the
  window. Hours on a measurement point at a frozen sensor.
- `changes/h 53.20`: value changes per hour. A 60 s tag that changes
  every sample makes about 60.
- `interval 60 s (p05 60 s, p95 60 s)   declared 60 s`: the median
  spacing between samples, its 5th and 95th percentiles, and
  `sample_rate_s`. `(differs from declared)` follows when they disagree.

## Flatline

- `NOT ASSESSED (window censored; saturated != frozen)`: the
  [flatline](../../reference/glossary.md#flatline) check compares the
  window's stall time and distinct count with the tag's own history in
  earlier windows of the same length, and reports each signal on its
  own. A censored window is not assessed, because a transmitter pinned at
  full scale also reads flat. A MODE tag, a window with no valid sample
  and a window with no history to compare against are not assessed
  either.

## What to do next

- Coverage below 1: find the gap lines and check the collector for
  those times.
- Unmapped codes: declare them, then profile again.
- `censored yes`: keep the clipped stretch out of any baseline.
- `censored unknown`: declare the engineering range.
- The whole report as one JSON object: `tsdive profile --json`, or
  `Profile.to_dict()` in Python.
