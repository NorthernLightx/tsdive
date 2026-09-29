# Clipping and censoring

A transmitter measures between the two ends of its
[engineering range](../../reference/glossary.md#engineering-range).
Past an end it keeps reporting the end value: the flow is 130 m3/h and
the historian stores 100. That sample is
[clipped](../../reference/glossary.md#clipped), and a window holding one
is [censored](../../reference/glossary.md#censored). The plant words are
saturated, pegged and over range.

## How tsdive flags a sample

A sample is clipped when its value sits at either end of the declared
range, within 1e-9 times the span, or when its quality code is an Over
Range or Under Range state. The range comes from `eng_range_zero` and
`eng_range_span` in the metadata. FIC-101 declares 0 to 100 m3/h, and
from 02:00 to 02:30 it sat at 100:

```tsdive lines=20
tsdive profile data/demo/fic101_demo.parquet
```

`clipped 0.0516` in the Range section is the share of samples at an end:
29 of 562. `censored yes` in the headline says the window holds at least
one.

## Yes, no and unknown

| verdict | means |
|---|---|
| `censored yes` | at least one sample is clipped |
| `censored no` | the range is declared and no sample reached either end |
| `censored unknown` | the metadata declares no engineering range, so tsdive cannot tell |

`censored unknown` is common on a first ingest. A tag pegged at its top
for a whole shift reads as a flat line at a normal-looking number until
the range is declared. Declare it, and the same data reads
`censored yes`.

## Why a censored window matters

A clipped sample is a bound, not a measurement: the flow was 100 m3/h
or more, by an unknown amount. The mean and the spread over it are
wrong by that unknown amount, and a baseline built on it would call the
real excursion normal. So a censored window may never serve as a
[baseline](baselines.md): `screen`, `spc` and `mspc` refuse one with
[`InsufficientQuality`](../../reference/errors.md#insufficientquality),
and `compare` leaves that tag's `flagged` column empty with the reason.
It takes one clipped sample, not a clipped majority. The flatline check
also stops: a transmitter pinned at full scale reads flat, and frozen
and saturated are different faults. The profile prints
`flatline NOT ASSESSED (window censored; saturated != frozen)`.

A window of `censored unknown` is accepted as a baseline, and the report
prints the verdict beside it.

## What to do

- Declare the engineering range of every tag. Take it from the
  transmitter data sheet or the historian's zero and span attributes.
- When a baseline is refused as censored, move it off the clipped
  stretch. [Choose a clean baseline window](../howto/baseline-window.md)
  shows how to find one with `segment`.
- When a window you monitor is censored, treat the excursion as real and
  larger than the numbers show.
