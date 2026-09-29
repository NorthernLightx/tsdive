# Coverage and gap classes

A historian that stops collecting leaves a hole in the timestamps.
tsdive finds each hole, gives it a
[class](../../reference/glossary.md#gap-class) with the rule behind it,
and subtracts the holes that are real data loss from
[coverage](../../reference/glossary.md#coverage).

## What counts as a gap

A [gap](../../reference/glossary.md#gap) is a spacing between two
samples longer than the
[gap threshold](../../reference/glossary.md#gap-threshold): 2.5 times
the median interval of the window, and at least 60 s. The demo flow
samples every 60 s, so its threshold is 150 s. The start and end of the
window count too: if the samples stop 19 minutes before the window ends,
those 19 minutes are a gap. A shorter stretch at an edge is
[edge slack](../../reference/glossary.md#edge-slack) and counts as
covered.

## Gap classes

The first rule that matches gives the class:

| class | rule | lowers coverage |
|---|---|---|
| `scan_off` | the hole lies inside a declared scan-off period | yes |
| `comm_outage_multitag` | a peer tag of the same source has the same hole, so collection failed | yes |
| `compression_steady` | the contract is `stepped` and the hole is at most 50 times the median interval, so the value held and the historian stored nothing | no |
| `sparse_by_design` | the hole is at most 3 times the declared `sample_rate_s`, the normal spacing of a slow tag | no |
| `unknown` | no rule matched | yes |

`unknown` is an answer. It says tsdive could not name a benign cause,
so the time counts as lost. The classes `scan_off`,
`comm_outage_multitag` and `unknown` are the
[data-loss](../../reference/glossary.md#data-loss-gap) classes.

## Reading the Coverage section

```tsdive lines=10
tsdive profile data/demo/fic101_demo.parquet
```

The Coverage section of the report says:

- `coverage 0.933`: 40 of the 600 minutes are lost, and 1 - 40/600 is
  0.933.
- `valid 0.998`: the share of rows that are GOOD and a number, a
  different question from coverage. See
  [quality codes and severity](quality-and-severity.md).
- `gaps 1   data-loss gaps 1`: one gap in total, and it is a data-loss
  gap.
- `longest 40 min`: the longest gap.
- `2024-03-30 23:00:00Z -> 23:40:00Z   40 min   unknown (no rule
  matched)`: the gap itself, its class and the rule. The two counts and
  this line describe the same hole: its class is `unknown`, and
  `unknown` is a data-loss class.

## Turning `unknown` into a benign class

A benign class needs a fact tsdive can check. Give it the fact:

- Declare `sample_rate_s` in the metadata, so a slow tag's normal
  spacing reads `sparse_by_design`.
- Read a tag stored on change with `--stepped`, so a hole of up to 50
  times the median interval reads `compression_steady`. A longer hole
  stays `unknown`.

Pass `--stepped` only for a tag the historian really stores on change.
The demo flow is stored every scan, so its 40-minute hole is lost data,
and coverage says so.

## What to do

- Check the collector logs for a window whose coverage is below 0.95,
  the coverage `mspc` asks for by default, before you trust a statistic
  over it.
- Look up each `unknown` gap: an outage in several tags at once points
  at the collector, one tag alone at its transmitter or scan class.
