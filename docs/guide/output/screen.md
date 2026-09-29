# Reading the screen

`tsdive screen` asks whether a window behaves like a baseline. This is
the report on the demo flow, baseline 20:00 to 01:00, window 01:00 to
06:00, then every line.

```tsdive
tsdive screen data/demo/fic101_demo.parquet \
    --baseline 2024-03-30T20:00:00Z/2024-03-31T01:00:00Z \
    --window 2024-03-31T01:00:00Z/2024-03-31T06:00:00Z
```

## Line by line

- `demo:FIC101.PV  flagged 29 of 300 (9.7%)`: 29 of the window's 300
  GOOD samples lie outside the limits.
- `baseline ... GOOD 261   censored no`: the baseline span, its GOOD
  sample count and its censoring verdict. A baseline needs 30 GOOD
  samples and `censored no` or `unknown`.
- `window`: the span screened. The date is left out when it repeats.
- `method    MAD   center 62.32   scale 0.5488   k 3.0   limits [60.67, 63.97]`:
  the baseline's median, its sigma (1.4826 times the MAD), the number of
  sigmas `k`, and the limits `center - k x scale` and
  `center + k x scale`, all in m3/h. See
  [baselines and the MAD screen](../concepts/baselines.md).
- `caveat    provisional: one baseline for every regime in the window`:
  the screen assumed one operating point. If the regime changed inside
  the window, the flags say so and not more.
- `Flagged`: the first flagged timestamps, and `(+26 more)` for the
  rest. All 29 here are the half hour at 100 m3/h from 02:01.

With `--mode`, two more lines appear: `alignment`, the baseline and
window rows that carried a mode sample of their own, and a `Regimes`
section with one center, scale and count per regime.

## What to do next

- A flagged share near 0 with a clean baseline: the window behaves like
  the baseline.
- Flags bunched in one stretch: profile that stretch. Here the Range
  section shows it is clipped, a transmitter fault rather than a process
  change.
- Flags spread over the whole window after a regime change: screen per
  regime with `--mode`, or pick a baseline at the window's operating
  point.
- The flagged rows as data: `--json`, or `ScreenAnalysis.frame` in
  Python.
