# Choose a clean baseline window

`screen`, `spc`, `mspc` and `compare` judge a window against a
[baseline](../../reference/glossary.md#baseline). A bad baseline gives
confident wrong answers, so this guide picks one step by step on the
demo flow, to judge the two hours from 04:00 to 06:00.

## 1. Profile the history

```tsdive lines=16
tsdive profile data/demo/fic101_demo.parquet
```

Two stretches are off limits: the 40-minute outage from 23:00, and the
half hour pinned at 100 m3/h from 02:00. A baseline over the clipped
stretch is refused, and one over the outage is refused by `mspc`.

## 2. Segment it

`tsdive segment` cuts the archive into stretches of steady level, with
each stretch's median and MAD:

```tsdive
tsdive segment data/demo/fic101_demo.parquet
```

Segment 4 is the clipped half hour: median 100.0, MAD 0. The window
from 04:00 lies in segment 6, median 62.52 m3/h.

## 3. Pick a segment at the window's operating point

The nearest clean segment is 5, right before the window. Its median is
61.37 m3/h, more than one m3/h below the window's level:

```tsdive lines=6
tsdive screen data/demo/fic101_demo.parquet \
    --baseline 2024-03-31T02:30:00Z/2024-03-31T03:56:00Z \
    --window 2024-03-31T04:00:00Z/2024-03-31T06:00:00Z
```

84% of the window is flagged. Nothing is wrong with the flow: segment 5
ran at a lower level, and its sigma of 0.2514 m3/h makes the step up to
62.5 look like a fault. Segment 1 ran at 62.54 m3/h, the window's level:

```tsdive lines=6
tsdive screen data/demo/fic101_demo.parquet \
    --baseline 2024-03-30T20:00:00Z/2024-03-30T21:55:00Z \
    --window 2024-03-31T04:00:00Z/2024-03-31T06:00:00Z
```

0 of 121 flagged. The flow from 04:00 behaves like the flow at 20:00.

## Rules of thumb

- Same operating point: throughput, grade, recipe and controller mode
  as the window. The segment medians show the levels.
- Clean: no clipped sample, which `censored no` confirms, and no outage
  when you run `mspc` or `compare`'s joint tables.
- Long enough: at least 30 GOOD samples; a few hundred give steadier
  limits. Segment 1 holds 116.
- Normal: a period the operators call normal, not the week of a known
  upset.

When the window visits several operating points, a single baseline
cannot fit it. Write the segments as a MODE archive with
`segment --mode-out` and screen per regime with `--mode`, as
[baselines and the MAD screen](../concepts/baselines.md#screening-per-regime)
shows.

## If the baseline is refused

- `window is censored`: the baseline holds a clipped sample. Move it off
  the stretch segment shows at the range end.
- `only 20 GOOD history samples; 30 required`: widen it, and check it
  lies inside the archive's time range.
- `aligned coverage 0.867 below required 0.95` from `mspc`: move it off
  the outage.
