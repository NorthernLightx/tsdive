# Compare's three tables

`tsdive compare` answers "something changed on this unit, what?". You
name every tag of the unit, a period before and a period after, and it
prints three tables. Each is a statement about the samples. None names a
cause: a recalibrated transmitter, a rate change and a sticking valve
can give the same numbers.

```tsdive
tsdive compare data/demo/fic101_demo.parquet data/demo/tic101_demo.parquet \
    --before 2024-03-30T20:00:00Z/2024-03-30T23:00:00Z \
    --after 2024-03-31T04:00:00Z/2024-03-31T06:00:00Z
```

The headline `demo  2 tags   refused 0   pairs 1 of 1 clearing` counts
the tags, the tags whose read was refused, and the pairs that
[clear](../../reference/glossary.md#clearing). `grid` is the rate the
pair and joint tables align on.

## Table 1: tags that changed

One row per tag, every tag listed, ranked by the size of its shift:

| column | means | demo TIC-101 |
|---|---|---|
| `quality` | the after period's quality against the before period's: `ok`, a severity share, `censored` or `frozen` | `ok` |
| `sigma` | the [level shift](../../reference/glossary.md#level-shift): after median minus before median, in before-period sigmas | `+0.8` |
| `spread` | the [spread ratio](../../reference/glossary.md#spread-ratio): after MAD over before MAD | `x6.0`, six times noisier |
| `flagged` | the share of after samples a [MAD screen](baselines.md) built on the before period flags | `62%` |
| `changed at` | the first changepoint at or after the after period starts, from `segment` over both periods | `04:18` |

The table lists every tag, so a row with a small shift and 0% flagged,
such as FIC-101's `+0.4  x0.5  0%`, is a tag that did not change much. A
column that cannot be computed stays empty and `--json` gives the
reason, such as `no spread before` for a tag whose MAD was 0.

## Table 2: pairs that decoupled

One row per pair of tags. For each period, tsdive takes the Pearson
correlation of the first differences, the change from one sample to the
next, which removes the slow drift both tags share. `delta` is after
minus before, and `interval` is a 95% moving block bootstrap interval on
that delta. A pair clears when the interval excludes 0: the change is
larger than resampling noise.

`FIC101.PV ~ TIC101.PV    0.69    0.17   -0.53  [-0.76, -0.26]` says the
temperature followed the flow minute by minute before, barely after,
and the drop is well outside the noise. That is a
[decoupled pair](../../reference/glossary.md#decoupled-pair).

## Table 3: joint structure

A PCA fitted on the before period, read on the after period, as `mspc`
does. `explained 0.9839 -> 0.2643 on after` says one component carried
98% of the before variance and 26% of the after variance: the structure
the tags shared is gone. The T2 and SPE breaches count after rows over
the before period's limits, and the SPE contributors name the tag that
carries the break, TIC-101 with 79%. See [MSPC](mspc.md).

## When a table is refused

The pair and joint tables align every tag on one grid, so they need a
`sample_rate_s` on every archive, or `--rate-s`. Without one they are
refused, and the headline lists each refused table with its reason.
The tag table still prints, because it reads each tag alone.

## What to do

- Name every tag of the unit, not only the suspect. A tag that changed
  alone points at that tag; a broken pair points at the relationship.
- Pick a before period of normal running at the same operating point,
  clean of clipping. The after period may hold the fault.
- Use the `changed at` time to line the change up with the shift log and
  the operator actions.
